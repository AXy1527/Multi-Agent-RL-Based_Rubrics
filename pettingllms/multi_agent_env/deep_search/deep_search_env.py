import logging
import copy
from typing import Any, Dict, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

from pettingllms.multi_agent_env.base. env import Env
from pettingllms. multi_agent_env.deep_search.deep_search_utils import (
    load_deep_search_problem_batch,
    perform_web_search_with_summary,
    format_docs_for_r_agent,
    extract_q_agent_response,
    extract_r_agent_response,
    extract_a_agent_response,
    evaluate_deep_search_answer,
    format_history_for_q_agent,
    format_context_for_r_agent,
    format_context_for_a_agent,
    format_trajectory_for_rm_q_scoring,
    format_round_for_rm_r_scoring,
)

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in deep search"""
    Q_AGENT = "q_agent"      # Planner: query decomposition, keyword extraction
    R_AGENT = "r_agent"      # Ranker: document selection and filtering
    A_AGENT = "a_agent"      # Solver: evidence synthesis and answer generation
    RM_AGENT = "rm_agent"    # Critic: reward shaping (no gradient update)


class QAgentActionType(Enum):
    """Actions available to Q-Agent"""
    SEARCH = "search"                    # Generate sub-queries with goals
    GENERATE_ANSWER = "generate_answer"  # Decide to generate final answer


@dataclass
class SubQueryInfo:
    """Sub-query information (without storing full docs)"""
    query: str
    goal: str


@dataclass
class SelectedDocument:
    """Selected document with content"""
    title: str
    url: str
    snippet: str
    content: str  # Evidence + Summary formatted content


@dataclass
class RoundTrajectory:
    """
    Single round trajectory in the Q-R loop. 
    
    Only stores:
    - Q-Agent response (think + tool_call)
    - R-Agent response (think + tool_call)
    - Selected documents (not all retrieved docs)
    """
    round_idx: int
    
    # Q-Agent output (stored in trajectory)
    q_agent_think: str = ""
    q_agent_tool_call: Dict = field(default_factory=dict)
    sub_queries: List[SubQueryInfo] = field(default_factory=list)
    
    # R-Agent output (stored in trajectory)
    r_agent_think: str = ""
    r_agent_tool_call: Dict = field(default_factory=dict)
    selected_docs: List[SelectedDocument] = field(default_factory=list)
    
    # Temporary storage for current round (not part of final trajectory)
    # These are cleared after R-Agent step
    _current_round_all_docs: List[Dict] = field(default_factory=list, repr=False)


@dataclass
class DeepSearchEnvState:
    """State class for Deep Search environment"""
    # Problem definition
    question: str = None
    ground_truth_answer: str = None
    search_hint: str = None
    
    # Complete trajectory: list of Q-R rounds
    trajectory: List[RoundTrajectory] = field(default_factory=list)
    current_round: int = 0
    
    # Current agent turn
    current_agent: AgentRole = None
    is_terminated: bool = False
    
    # URL caching to avoid re-crawling
    visited_urls: Set[str] = field(default_factory=set)
    url_raw_content_cache: Dict[str, str] = field(default_factory=dict)
    
    # A-Agent output (final)
    a_agent_think: str = ""
    a_agent_answer: str = None
    a_agent_is_correct: bool = False
    a_agent_reward: float = 0. 0  # Binary reward: 0 or 1
    
    # RM-Agent scores (computed after rollout completion)
    # Q-Agent: one score dict per round, computed from full trajectory
    # R-Agent: one score dict per round, computed per-round
    q_agent_scores: List[Dict[str, Any]] = field(default_factory=list)
    r_agent_scores: List[Dict[str, Any]] = field(default_factory=list)
    
    # RM-Agent round data for R-Agent scoring (stores all_docs per round)
    rm_r_agent_round_data: List[Dict] = field(default_factory=list)


class DeepSearchEnv(Env):
    """
    Environment for deep search with multi-agent collaboration.
    
    Agent Flow: Q-Agent → R-Agent → Q-Agent → R-Agent → ...  → Q-Agent(generate_answer) → A-Agent
    
    Input/Output specifications:
    
    Q-Agent (Planner):
        - Input: initial question + history (previous rounds' Q-response, R-response, selected docs)
        - Output: <think>... </think><tool_call>{"name": "search"/"generate_answer", ... }</tool_call>
        - Reward: RM-Agent scores (from full trajectory)
    
    R-Agent (Ranker):
        - Input: initial question + current round Q-response + all_summarized_docs from search
        - Output: <think>...</think><tool_call>{"name": "select_documents", ...}</tool_call>
        - Reward: RM-Agent scores (per-round scoring)
    
    A-Agent (Solver):
        - Input: initial question + complete history trajectory
        - Output: <think>...</think><answer>...</answer>
        - Reward: Binary 0/1 based on answer correctness (NO RM scoring)
    
    RM-Agent (Critic) - does NOT score A-Agent:
        - Q-Agent scoring: initial question + full trajectory → score each round's Q-Agent action
        - R-Agent scoring: per-round, initial question + round's (Q-response, all_docs, R-response, selected_docs)
    """
    
    def __init__(
        self,
        env_idx: int,
        rollout_idx: int,
        max_turns: int,
        config: dict | None = None,
    ):
        """Initialize the deep search environment."""
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, max_turns=max_turns, config=config)
        self.state = DeepSearchEnvState()
        self.config = config or {}
        
        # Configuration
        self.max_rounds = self.config. get("max_rounds", 10)
        self.max_docs_per_query = self.config. get("max_docs_per_query", 5)
        self. summary_model = self.config.get("summary_model", "gpt-3.5-turbo")
    
    def reset(self):
        """Reset the deep search environment state."""
        question = self.state. question
        ground_truth = self.state.ground_truth_answer
        search_hint = self. state.search_hint
        
        self.state = DeepSearchEnvState(
            question=question,
            ground_truth_answer=ground_truth,
            search_hint=search_hint,
            current_agent=AgentRole. Q_AGENT,
            current_round=0,
        )
    
    async def step(self, role: str, action: str, env_worker: Any = None):
        """
        Execute an action in the deep search environment.
        
        Args:
            role: Agent role ("q_agent", "r_agent", "a_agent")
            action: Action/response from the agent (raw LLM output)
            env_worker: Optional environment worker for parallel execution
        """
        role_enum = AgentRole(role)
        
        if role_enum == AgentRole.Q_AGENT:
            await self._q_agent_step(action, env_worker)
        elif role_enum == AgentRole. R_AGENT:
            await self._r_agent_step(action, env_worker)
        elif role_enum == AgentRole.A_AGENT:
            await self._a_agent_step(action, env_worker)
        else:
            raise ValueError(f"Invalid role: {role}")
    
    async def _q_agent_step(self, action: str, env_worker: Any = None):
        """
        Execute Q-Agent step. 
        
        Q-Agent can:
        1.  Generate sub-queries with goals → triggers web search → next is R-Agent
        2. Transition to A-Agent → next is A-Agent
        """
        # Parse Q-Agent response
        parsed = extract_q_agent_response(action)
        
        action_type = parsed.get("action_type", "search")
        
        if action_type == "generate_answer":
            # Q-Agent decides to generate final answer, transition to A-Agent
            self.state. current_agent = AgentRole.A_AGENT
            logger.info(f"Q-Agent decided to generate answer at round {self.state.current_round}")
            return
        
        # Create new round trajectory for search action
        round_traj = RoundTrajectory(round_idx=self. state.current_round)
        round_traj.q_agent_think = parsed.get("think", "")
        round_traj. q_agent_tool_call = parsed. get("tool_call", {})
        
        # Store sub-query info (without docs)
        sub_queries = []
        all_summarized_docs = []  # Temporary storage for R-Agent
        
        for sq_data in parsed.get("queries", []):
            query_text = sq_data.get("query", "") if isinstance(sq_data, dict) else str(sq_data)
            goal_text = sq_data.get("goal", query_text) if isinstance(sq_data, dict) else query_text
            
            sub_queries. append(SubQueryInfo(query=query_text, goal=goal_text))
            
            # Perform web search and summarization
            search_results, summarized_docs = await perform_web_search_with_summary(
                query=query_text,
                goal=goal_text,
                max_results=self.max_docs_per_query,
                summary_model=self.summary_model,
                env_worker=env_worker,
                visited_urls=self. state.visited_urls,
                url_content_cache=self. state.url_raw_content_cache,
            )
            
            # Add source query info to each doc
            for doc in summarized_docs:
                doc["source_query"] = query_text
                doc["source_goal"] = goal_text
            
            all_summarized_docs. extend(summarized_docs)
            
            # Update visited URLs
            for result in search_results:
                url = result.get("url", "")
                if url:
                    self. state.visited_urls.add(url)
        
        round_traj.sub_queries = sub_queries
        round_traj._current_round_all_docs = all_summarized_docs  # Temporary for R-Agent
        
        self.state.trajectory.append(round_traj)
        self.state.current_agent = AgentRole. R_AGENT
        
        logger.info(f"Q-Agent generated {len(sub_queries)} sub-queries with {len(all_summarized_docs)} total docs at round {self.state.current_round}")
    
    async def _r_agent_step(self, action: str, env_worker: Any = None):
        """
        Execute R-Agent step.
        
        R-Agent selects relevant documents from the retrieved docs. 
        After selection, all_docs are stored for RM scoring but cleared from trajectory.
        """
        if not self.state.trajectory:
            raise RuntimeError("R-Agent step called without Q-Agent trajectory")
        
        current_round = self.state.trajectory[-1]
        all_docs = current_round._current_round_all_docs
        
        # Build URL to doc mapping
        url_to_doc = {doc. get("url", ""): doc for doc in all_docs}
        
        # Parse R-Agent response
        parsed = extract_r_agent_response(action, all_docs)
        
        current_round.r_agent_think = parsed.get("think", "")
        current_round.r_agent_tool_call = parsed.get("tool_call", {})
        
        # Get selected documents
        selected_docs = []
        selected_urls = parsed.get("selected_urls", [])
        
        for url in selected_urls:
            if url in url_to_doc:
                doc = url_to_doc[url]
                selected_docs. append(SelectedDocument(
                    title=doc.get("title", ""),
                    url=doc.get("url", ""),
                    snippet=doc. get("snippet", ""),
                    content=doc.get("content", ""),
                ))
        
        # Fallback: try index-based selection
        if not selected_docs and parsed.get("selected_indices"):
            for idx in parsed["selected_indices"]:
                if 0 <= idx < len(all_docs):
                    doc = all_docs[idx]
                    selected_docs.append(SelectedDocument(
                        title=doc.get("title", ""),
                        url=doc. get("url", ""),
                        snippet=doc.get("snippet", ""),
                        content=doc.get("content", ""),
                    ))
        
        current_round.selected_docs = selected_docs
        
        # Store data for RM R-Agent scoring (includes all_docs)
        self.state.rm_r_agent_round_data. append({
            "round_idx": current_round.round_idx,
            "q_agent_think": current_round.q_agent_think,
            "q_agent_tool_call": current_round. q_agent_tool_call,
            "sub_queries": [{"query": sq.query, "goal": sq.goal} for sq in current_round.sub_queries],
            "all_docs": all_docs,  # Full docs for RM scoring
            "r_agent_think": current_round.r_agent_think,
            "r_agent_tool_call": current_round. r_agent_tool_call,
            "selected_docs": [
                {"title": sd.title, "url": sd.url, "snippet": sd. snippet, "content": sd.content}
                for sd in selected_docs
            ],
        })
        
        # Clear temporary docs storage (not part of trajectory)
        current_round._current_round_all_docs = []
        
        # Move to next round
        self.state.current_round += 1
        
        if self.state.current_round >= self. max_rounds:
            self.state.current_agent = AgentRole. A_AGENT
            logger.info(f"Max rounds ({self.max_rounds}) reached, transitioning to A-Agent")
        else:
            self.state.current_agent = AgentRole.Q_AGENT
        
        logger.info(f"R-Agent selected {len(selected_docs)} documents at round {self.state.current_round - 1}")
    
    async def _a_agent_step(self, action: str, env_worker: Any = None):
        """
        Execute A-Agent step.
        
        A-Agent synthesizes evidence and generates final answer.
        Reward is binary 0/1 based on answer correctness (NO RM scoring). 
        """
        parsed = extract_a_agent_response(action)
        
        self.state.a_agent_think = parsed.get("think", "")
        self.state.a_agent_answer = parsed.get("answer", action)
        self.state.is_terminated = True
        self.state.current_agent = None
        
        # Evaluate correctness and set binary reward
        if self.state.a_agent_answer and self.state.ground_truth_answer:
            self.state.a_agent_is_correct = evaluate_deep_search_answer(
                self.state.a_agent_answer, self.state.ground_truth_answer
            )
            # Binary reward: 1. 0 if correct, 0. 0 if incorrect
            self.state. a_agent_reward = 1.0 if self.state.a_agent_is_correct else 0.0
        else:
            self.state.a_agent_is_correct = False
            self.state.a_agent_reward = 0.0
        
        logger.info(f"A-Agent generated answer.  Correct: {self.state.a_agent_is_correct}, Reward: {self.state.a_agent_reward}")
    
    def get_current_agent(self) -> Optional[AgentRole]:
        """Get the current agent that should act."""
        return self.state.current_agent
    
    def is_done(self) -> bool:
        """Check if the episode is finished."""
        return self.state.is_terminated
    
    def get_observation(self, agent_role: AgentRole) -> Dict[str, Any]:
        """
        Get observation for the specified agent. 
        
        Q-Agent: initial question + history (Q-response, R-response, selected docs per round)
        R-Agent: initial question + current round Q-response + all_summarized_docs
        A-Agent: initial question + complete history
        """
        if agent_role == AgentRole.Q_AGENT:
            # Q-Agent sees: question + history of (Q-response, R-response, selected docs)
            history = format_history_for_q_agent(self.state.trajectory)
            return {
                "question": self.state. question,
                "history": history,
                "current_round": self.state. current_round,
                "max_rounds": self. max_rounds,
            }
        
        elif agent_role == AgentRole.R_AGENT:
            # R-Agent sees: question + current Q-response + all retrieved docs
            if not self.state. trajectory:
                raise RuntimeError("R-Agent observation requested but no trajectory exists")
            
            current_round = self. state.trajectory[-1]
            all_docs = current_round._current_round_all_docs
            
            context = format_context_for_r_agent(
                question=self.state. question,
                q_agent_think=current_round.q_agent_think,
                q_agent_tool_call=current_round.q_agent_tool_call,
                sub_queries=current_round.sub_queries,
                all_docs=all_docs,
            )
            
            return {
                "question": self.state.question,
                "context": context,
                "q_agent_think": current_round.q_agent_think,
                "queries": [{"query": sq.query, "goal": sq.goal} for sq in current_round. sub_queries],
                "docs": all_docs,
                "docs_formatted": format_docs_for_r_agent(all_docs),
            }
        
        elif agent_role == AgentRole.A_AGENT:
            # A-Agent sees: question + complete history trajectory
            context = format_context_for_a_agent(
                question=self.state.question,
                trajectory=self.state.trajectory,
            )
            
            # Collect all selected docs
            all_selected_docs = []
            for rt in self.state. trajectory:
                for sd in rt.selected_docs:
                    all_selected_docs.append({
                        "title": sd.title,
                        "url": sd.url,
                        "snippet": sd.snippet,
                        "content": sd.content,
                    })
            
            return {
                "question": self.state.question,
                "context": context,
                "all_selected_docs": all_selected_docs,
            }
        
        else:
            raise ValueError(f"Invalid agent role: {agent_role}")
    
    def get_trajectory_for_rm_q_agent(self) -> Dict[str, Any]:
        """
        Get trajectory data for RM-Agent Q-Agent scoring. 
        
        RM scores all Q-Agent actions at once using full trajectory.
        
        Returns:
            Dict with question and full trajectory (Q-response, R-response, selected docs per round)
        """
        return {
            "question": self.state.question,
            "ground_truth": self.state.ground_truth_answer,
            "trajectory": [
                {
                    "round_idx": rt.round_idx,
                    "q_agent_think": rt.q_agent_think,
                    "q_agent_tool_call": rt.q_agent_tool_call,
                    "sub_queries": [{"query": sq.query, "goal": sq.goal} for sq in rt.sub_queries],
                    "r_agent_think": rt.r_agent_think,
                    "r_agent_tool_call": rt.r_agent_tool_call,
                    "selected_docs": [
                        {"title": sd.title, "url": sd.url, "content": sd.content}
                        for sd in rt.selected_docs
                    ],
                }
                for rt in self. state.trajectory
            ],
            "final_answer": self. state.a_agent_answer,
            "is_correct": self. state.a_agent_is_correct,
        }
    
    def get_round_data_for_rm_r_agent(self, round_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get per-round data for RM-Agent R-Agent scoring.
        
        Each round is scored independently with:
        - question
        - Q-response (think + tool_call)
        - all_summarized_docs from search
        - R-response (think + tool_call)
        - selected_docs
        
        Args:
            round_idx: Round index to get data for
            
        Returns:
            Dict with round-specific data for R-Agent scoring
        """
        if round_idx >= len(self.state.rm_r_agent_round_data):
            return None
        
        round_data = self. state.rm_r_agent_round_data[round_idx]
        return {
            "question": self.state.question,
            "ground_truth": self. state.ground_truth_answer,
            **round_data,
        }
    
    def get_all_round_data_for_rm_r_agent(self) -> List[Dict[str, Any]]:
        """Get all round data for RM-Agent R-Agent scoring."""
        return [
            {
                "question": self.state. question,
                "ground_truth": self.state.ground_truth_answer,
                **round_data,
            }
            for round_data in self.state. rm_r_agent_round_data
        ]
    
    def get_a_agent_reward(self) -> float:
        """
        Get A-Agent's binary reward. 
        
        Returns:
            1.0 if answer is correct, 0.0 otherwise
        """
        return self.state.a_agent_reward
    
    def get_all_rewards(self) -> Dict[str, Any]:
        """
        Get all rewards for the episode.
        
        Returns:
            Dict containing:
            - q_agent_scores: List of per-round RM scores for Q-Agent
            - r_agent_scores: List of per-round RM scores for R-Agent
            - a_agent_reward: Binary 0/1 reward for A-Agent
        """
        return {
            "q_agent_scores": self.state.q_agent_scores,
            "r_agent_scores": self.state.r_agent_scores,
            "a_agent_reward": self. state.a_agent_reward,
            "is_correct": self.state.a_agent_is_correct,
        }
    
    def get_full_trajectory_for_logging(self) -> Dict[str, Any]:
        """
        Get complete trajectory for logging/saving.
        
        Returns full trajectory data including final answer. 
        """
        trajectory_data = []
        
        for rt in self.state.trajectory:
            # Q-Agent step
            q_step = {
                "agent": "Q-Agent",
                "round": rt.round_idx + 1,
                "think": rt.q_agent_think,
                "tool_call": rt.q_agent_tool_call,
            }
            trajectory_data.append(q_step)
            
            # R-Agent step
            r_step = {
                "agent": "R-Agent",
                "round": rt.round_idx + 1,
                "think": rt.r_agent_think,
                "tool_call": rt.r_agent_tool_call,
                "selected_docs": [
                    {"title": sd.title, "url": sd.url, "content": sd.content}
                    for sd in rt.selected_docs
                ],
            }
            trajectory_data.append(r_step)
        
        # A-Agent step
        a_step = {
            "agent": "A-Agent",
            "think": self.state. a_agent_think,
            "answer": self.state. a_agent_answer,
        }
        trajectory_data. append(a_step)
        
        return {
            "question": self.state.question,
            "ground_truth": self. state.ground_truth_answer,
            "trajectory": trajectory_data,
            "final_answer": self.state.a_agent_answer,
            "is_correct": self.state. a_agent_is_correct,
            "rewards": {
                "q_agent_scores": self.state.q_agent_scores,
                "r_agent_scores": self. state.r_agent_scores,
                "a_agent_reward": self.state.a_agent_reward,
            },
        }
    
    def set_rm_scores(self, q_scores: List[Dict], r_scores: List[Dict]):
        """
        Set RM-Agent computed scores for Q-Agent and R-Agent.
        
        Note: A-Agent does NOT receive RM scores, only binary reward.
        
        Args:
            q_scores: Per-round scores for Q-Agent (from single RM call with full trajectory)
            r_scores: Per-round scores for R-Agent (from per-round RM calls)
        """
        self.state. q_agent_scores = q_scores
        self.state. r_agent_scores = r_scores
    
    def render(self, mode=None):
        """Render the environment state."""
        total_selected = sum(len(rt.selected_docs) for rt in self. state.trajectory)
        return (
            f"Question: {self.state.question}\n"
            f"Ground Truth: {self.state.ground_truth_answer}\n"
            f"Current Agent: {self.state.current_agent}\n"
            f"Current Round: {self. state.current_round}\n"
            f"Trajectory Rounds: {len(self.state.trajectory)}\n"
            f"Total Selected Docs: {total_selected}\n"
            f"Is Terminated: {self.state.is_terminated}\n"
            f"A-Agent Reward: {self.state. a_agent_reward}"
        )
    
    def close(self):
        """Close the environment."""
        pass


class DeepSearchEnvBatch:
    """Batch environment manager for Deep Search environments."""
    
    def __init__(
        self,
        env_idx_list: List[int],
        env_indices: List[int],
        rollout_idx_list: List[int],
        samples: int,
        max_turns: int,
        config: dict,
        mode: str = "train",
        *,
        env_workers: List = None
    ):
        """
        Initialize batch of Deep Search environments. 
        """
        self.mode = mode
        self.env_list = []
        self.config = config
        
        benchmark_name = getattr(config, "benchmark", "bamboogle") if hasattr(config, "benchmark") else "bamboogle"
        self.problem_list = load_deep_search_problem_batch(
            env_indices, dataset_name=benchmark_name, mode=mode, config=config
        )
        
        if mode == "validate":
            rollout_idx_list = range(len(self. problem_list) * samples)
            samples = 1
        
        if not self.problem_list:
            raise ValueError(
                f"Failed to load problems from deep search dataset.  "
                f"Please check if the dataset is available and accessible."
            )
        
        for i, problem in enumerate(self. problem_list):
            state = DeepSearchEnvState(
                question=problem["question"],
                ground_truth_answer=problem["ground_truth"],
                search_hint=problem. get("search_hint", ""),
                current_agent=AgentRole.Q_AGENT,
            )
            
            for s in range(samples):
                env = DeepSearchEnv(
                    env_idx=i,
                    rollout_idx=rollout_idx_list[i * samples + s],
                    max_turns=max_turns,
                    config=config,
                )
                env. state = copy.deepcopy(state)
                self.env_list. append(env)
        
        if len(self.env_list) != len(rollout_idx_list):
            raise ValueError(
                f"len(self.env_list)!=len(rollout_idx_list), "
                f"{len(self.env_list)}!={len(rollout_idx_list)}"
            )
    
    def get_env_by_rollout_idx(self, rollout_idx: int) -> Optional[DeepSearchEnv]:
        """Get environment by rollout index."""
        for env in self.env_list:
            if env.rollout_idx == rollout_idx:
                return env
        return None
    
    def get_all_current_agents(self) -> Dict[int, AgentRole]:
        """Get current agent for all environments."""
        return {
            env.rollout_idx: env.get_current_agent()
            for env in self.env_list
            if not env.is_done()
        }