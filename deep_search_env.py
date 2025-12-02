"""
Deep Search Environment with proper input/output flow for training alignment.
"""
import logging
import copy
import json
from typing import Any, Dict, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    Q_AGENT = "q_agent"
    R_AGENT = "r_agent"
    A_AGENT = "a_agent"


@dataclass
class SubQueryInfo:
    query: str
    goal: str


@dataclass
class SelectedDocument:
    title: str
    url: str
    snippet: str
    content: str
    source_query: str = ""
    source_goal: str = ""


@dataclass
class RoundTrajectory:
    """Single round trajectory in the Q-R loop."""
    round_idx: int

    # Q-Agent output
    q_agent_think: str = ""
    q_agent_tool_call: Dict = field(default_factory=dict)
    sub_queries: List[SubQueryInfo] = field(default_factory=list)

    # R-Agent output
    r_agent_think: str = ""
    r_agent_tool_call: Dict = field(default_factory=dict)
    selected_docs: List[SelectedDocument] = field(default_factory=list)

    # Temporary storage for current round (cleared after R-Agent step)
    _current_round_all_docs: List[Dict] = field(default_factory=list, repr=False)

    # Flag: is this Q-Agent action scored?  (generate_answer is not scored)
    q_agent_should_score: bool = True


@dataclass
class DeepSearchEnvState:
    """State class for Deep Search environment."""
    question: str = ""
    ground_truth_answer: str = ""

    trajectory: List[RoundTrajectory] = field(default_factory=list)
    current_round: int = 0

    current_agent: AgentRole = None
    is_terminated: bool = False

    # URL caching
    visited_urls: Set[str] = field(default_factory=set)
    url_raw_content_cache: Dict[str, str] = field(default_factory=dict)

    # Final Q-Agent generate_answer response (stored separately, not scored)
    final_q_agent_think: str = ""
    final_q_agent_tool_call: Dict = field(default_factory=dict)

    # A-Agent output
    a_agent_think: str = ""
    a_agent_answer: str = ""
    a_agent_is_correct: bool = False
    a_agent_reward: float = 0.
    0

    # RM-Agent scores
    q_agent_scores: List[Dict] = field(default_factory=list)
    r_agent_scores: List[Dict] = field(default_factory=list)

    # RM R-Agent round data (includes all_docs for scoring)
    rm_r_agent_round_data: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "current_round": self.current_round,
            "current_agent": self.current_agent.value if self.current_agent else None,
            "is_done": self.is_terminated,
        }

    def to_dict_compact(self) -> Dict:
        return self.to_dict()


class DeepSearchEnv(Env):
    """
    Deep Search Environment with proper training alignment.

    Key flows:
    - Q-Agent: sees history, outputs search/generate_answer
    - R-Agent: sees current round Q-output + all_docs, outputs selection
    - A-Agent: sees complete history + final Q generate_answer, outputs answer

    Reward:
    - Q-Agent: RM scores (except generate_answer call)
    - R-Agent: RM scores (per-round, per-document)
    - A-Agent: Binary 0/1 (no RM scoring)
    """

    def __init__(
            self,
            env_idx: int = 0,
            rollout_idx: int = 0,
            max_turns: int = 20,
            config: Any = None,
            **kwargs
    ):
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, max_turns=max_turns, config=config)
        self.state = DeepSearchEnvState()
        self.config = config or {}

        env_config = getattr(config, 'env', config) if config else {}
        self.max_rounds = getattr(env_config, 'max_rounds', 10)
        self.max_docs_per_query = getattr(env_config, 'max_docs_per_query', 5)
        self.summary_model = getattr(env_config, 'summary_model', 'gpt-3.5-turbo')

        self.done = False

    def reset(self):
        """Reset environment state."""
        question = self.state.question
        ground_truth = self.state.ground_truth_answer

        self.state = DeepSearchEnvState(
            question=question,
            ground_truth_answer=ground_truth,
            current_agent=AgentRole.Q_AGENT,
            current_round=0,
        )
        self.done = False

    def get_current_agent(self) -> Optional[AgentRole]:
        """Get which agent should act next."""
        if self.state.is_terminated:
            return None
        return self.state.current_agent

    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.state.is_terminated or self.done

    async def step(self, role: str, action: str, env_worker: Any = None):
        """Execute a step in the environment."""
        from pettingllms.multi_agent_env.deep_search.deep_search_utils import (
            extract_q_agent_response,
            extract_r_agent_response,
            extract_a_agent_response,
        )
        from pettingllms.multi_agent_env.deep_search.deep_search_worker import get_search_results

        if role == "q_agent":
            await self._handle_q_agent_step(action, env_worker)
        elif role == "r_agent":
            await self._handle_r_agent_step(action, env_worker)
        elif role == "a_agent":
            await self._handle_a_agent_step(action, env_worker)

    async def _handle_q_agent_step(self, action: str, env_worker: Any):
        """Handle Q-Agent step."""
        from pettingllms.multi_agent_env.deep_search.deep_search_utils import extract_q_agent_response
        from pettingllms.multi_agent_env.deep_search.deep_search_worker import get_search_results

        parsed = extract_q_agent_response(action)
        action_type = parsed.get("action_type", "search")

        if action_type == "generate_answer":
            # Store final Q-Agent response (not scored, but included in A-Agent history)
            self.state.final_q_agent_think = parsed.get("think", "")
            self.state.final_q_agent_tool_call = parsed.get("tool_call", {})

            # Transition to A-Agent
            self.state.current_agent = AgentRole.A_AGENT
            logger.info(f"Q-Agent decided to generate answer at round {self.state.current_round}")
            return

        # Search action - create new round
        round_traj = RoundTrajectory(
            round_idx=self.state.current_round,
            q_agent_should_score=True  # Search actions are scored
        )
        round_traj.q_agent_think = parsed.get("think", "")
        round_traj.q_agent_tool_call = parsed.get("tool_call", {})

        # Process sub-queries and perform searches
        sub_queries = []
        all_summarized_docs = []

        for sq_data in parsed.get("queries", []):
            query_text = sq_data.get("query", "") if isinstance(sq_data, dict) else str(sq_data)
            goal_text = sq_data.get("goal", query_text) if isinstance(sq_data, dict) else query_text

            sub_queries.append(SubQueryInfo(query=query_text, goal=goal_text))

            # Perform web search
            try:
                result = await get_search_results(
                    query=query_text,
                    goal=goal_text,
                    max_results=self.max_docs_per_query,
                    ray_actor=env_worker,
                    summary_model=self.summary_model,
                )

                docs = result.get("summarized_docs", [])
                for doc in docs:
                    doc["source_query"] = query_text
                    doc["source_goal"] = goal_text
                all_summarized_docs.extend(docs)

            except Exception as e:
                logger.error(f"Search failed for query '{query_text}': {e}")

        round_traj.sub_queries = sub_queries
        round_traj._current_round_all_docs = all_summarized_docs

        self.state.trajectory.append(round_traj)
        self.state.current_agent = AgentRole.R_AGENT

        logger.info(
            f"Q-Agent generated {len(sub_queries)} queries with {len(all_summarized_docs)} docs at round {self.state.current_round}")

    async def _handle_r_agent_step(self, action: str, env_worker: Any):
        """Handle R-Agent step."""
        from pettingllms.multi_agent_env.deep_search.deep_search_utils import extract_r_agent_response

        if not self.state.trajectory:
            logger.warning("R-Agent step without Q-Agent trajectory")
            return

        current_round = self.state.trajectory[-1]
        all_docs = current_round._current_round_all_docs

        # Parse R-Agent response
        parsed = extract_r_agent_response(action, all_docs)

        current_round.r_agent_think = parsed.get("think", "")
        current_round.r_agent_tool_call = parsed.get("tool_call", {})

        # Select documents
        url_to_doc = {doc.get("url", ""): doc for doc in all_docs}
        selected_docs = []

        for url in parsed.get("selected_urls", []):
            if url in url_to_doc:
                doc = url_to_doc[url]
                selected_docs.append(SelectedDocument(
                    title=doc.get("title", ""),
                    url=doc.get("url", ""),
                    snippet=doc.get("snippet", ""),
                    content=doc.get("content", ""),
                    source_query=doc.get("source_query", ""),
                    source_goal=doc.get("source_goal", ""),
                ))

        # Fallback: index-based selection
        if not selected_docs and parsed.get("selected_indices"):
            for idx in parsed["selected_indices"]:
                if 0 <= idx < len(all_docs):
                    doc = all_docs[idx]
                    selected_docs.append(SelectedDocument(
                        title=doc.get("title", ""),
                        url=doc.get("url", ""),
                        snippet=doc.get("snippet", ""),
                        content=doc.get("content", ""),
                        source_query=doc.get("source_query", ""),
                        source_goal=doc.get("source_goal", ""),
                    ))

        current_round.selected_docs = selected_docs

        # Store data for RM R-Agent scoring
        self.state.rm_r_agent_round_data.append({
            "round_idx": current_round.round_idx,
            "question": self.state.question,
            "q_agent_think": current_round.q_agent_think,
            "q_agent_tool_call": current_round.q_agent_tool_call,
            "sub_queries": [{"query": sq.query, "goal": sq.goal} for sq in current_round.sub_queries],
            "all_docs": all_docs,
            "r_agent_think": current_round.r_agent_think,
            "r_agent_tool_call": current_round.r_agent_tool_call,
            "selected_docs": [
                {"title": sd.title, "url": sd.url, "content": sd.content}
                for sd in selected_docs
            ],
        })

        # Clear temporary docs
        current_round._current_round_all_docs = []

        # Move to next round
        self.state.current_round += 1

        if self.state.current_round >= self.max_rounds:
            self.state.current_agent = AgentRole.A_AGENT
            logger.info(f"Max rounds reached, transitioning to A-Agent")
        else:
            self.state.current_agent = AgentRole.Q_AGENT

        logger.info(f"R-Agent selected {len(selected_docs)} documents")

    async def _handle_a_agent_step(self, action: str, env_worker: Any):
        """Handle A-Agent step."""
        from pettingllms.multi_agent_env.deep_search.deep_search_utils import extract_a_agent_response

        parsed = extract_a_agent_response(action)

        self.state.a_agent_think = parsed.get("think", "")
        self.state.a_agent_answer = parsed.get("answer", action)
        self.state.is_terminated = True
        self.state.current_agent = None
        self.done = True

        # Evaluate and set binary reward
        if self.state.a_agent_answer and self.state.ground_truth_answer:
            self.state.a_agent_is_correct = self._evaluate_answer(
                self.state.a_agent_answer,
                self.state.ground_truth_answer
            )
            self.state.a_agent_reward = 1.0 if self.state.a_agent_is_correct else 0.0
        else:
            self.state.a_agent_is_correct = False
            self.state.a_agent_reward = 0.0

        logger.info(f"A-Agent: Correct={self.state.a_agent_is_correct}, Reward={self.state.a_agent_reward}")

    def _evaluate_answer(self, answer: str, ground_truth: str) -> bool:
        """Simple answer evaluation."""
        if not ground_truth:
            return False
        return ground_truth.lower() in answer.lower()

    def get_a_agent_reward(self) -> float:
        """Get binary A-Agent reward."""
        return self.state.a_agent_reward

    def set_rm_scores(self, q_scores: List[Dict], r_scores: List[Dict]):
        """Set RM-Agent scores after episode completion."""
        self.state.q_agent_scores = q_scores
        self.state.r_agent_scores = r_scores

    def get_trajectory_for_rm_q_agent(self) -> Dict:
        """Get trajectory data for RM Q-Agent scoring."""
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
                    "should_score": rt.q_agent_should_score,
                }
                for rt in self.state.trajectory
            ],
            "final_answer": self.state.a_agent_answer,
            "is_correct": self.state.a_agent_is_correct,
        }

    def get_all_round_data_for_rm_r_agent(self) -> List[Dict]:
        """Get all round data for RM R-Agent scoring."""
        return self.state.rm_r_agent_round_data

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, value: bool):
        self._done = value


class DeepSearchEnvBatch:
    """Batch of Deep Search environments."""

    def __init__(
            self,
            env_idx_list: List[int],
            env_indices: List[int],
            rollout_idx_list: List[int],
            samples: int,
            max_turns: int,
            config: Any,
            mode: str = "train",
            **kwargs
    ):
        self.env_list = []
        self.config = config
        self.mode = mode

        # Load dataset
        from pettingllms.multi_agent_env.deep_search.deep_search_utils import load_deep_search_problem_batch

        env_config = getattr(config, 'env', config) if config else {}
        benchmark = getattr(env_config, 'benchmark', 'bamboogle')

        problem_list = load_deep_search_problem_batch(
            env_indices, dataset_name=benchmark, mode=mode, config=config
        )

        if not problem_list:
            # Fallback mock data
            problem_list = [
                               {"question": "What is the capital of France?", "ground_truth": "Paris"},
                           ] * len(env_indices)

        for i, problem in enumerate(problem_list):
            state = DeepSearchEnvState(
                question=problem.get("question", ""),
                ground_truth_answer=problem.get("ground_truth", problem.get("answer", "")),
                current_agent=AgentRole.Q_AGENT,
            )

            for s in range(samples):
                rollout_idx = rollout_idx_list[i * samples + s] if i * samples + s < len(
                    rollout_idx_list) else i * samples + s

                env = DeepSearchEnv(
                    env_idx=i,
                    rollout_idx=rollout_idx,
                    max_turns=max_turns,
                    config=config,
                )
                env.state = copy.deepcopy(state)
                env._done = False
                self.env_list.append(env)