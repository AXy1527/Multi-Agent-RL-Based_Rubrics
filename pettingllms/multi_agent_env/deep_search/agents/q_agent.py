"""
Q-Agent (Planner): Query decomposition and search strategy.
Compatible with MultiAgentsExecutionEngine interface.

Input for each round:
- Initial question
- History (each round's Q-Agent think/tool_call, R-Agent think/tool_call, selected docs)

Reward: Computed by RM-Agent after episode completion (except for the final generate_answer call)
"""
import logging
from typing import Any, Dict, List, Optional

from pettingllms.multi_agent_env.base. agent import Agent
from pettingllms. multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


# Q-Agent System Prompt
Q_AGENT_SYSTEM_PROMPT = """You are the **Chief Inquiry Strategist**, a master at dissecting complex questions and charting the most efficient path to a comprehensive answer. Your domain is strategy and planning.  You transform ambiguity into a precise, step-by-step research plan. 

As you proceed, you must adhere to the following core principles:

1.   **Deconstruct to Conquer**: You will meticulously break down every user question into its fundamental, answerable components. No complex query is monolithic; it is a collection of smaller questions waiting to be solved.

2.  **Precision in Inquiry**: Your search queries are surgical instruments.  For each query, you must first define a crystal-clear **goal**—the specific piece of information you intend to find. Then, you will craft a query perfectly tailored to achieve that goal.

3.  **Iterative Refinement**: You understand that information gathering is a dynamic process. You will continuously assess the gathered intelligence, refining your strategy and adapting your next queries and goals to fill the most critical remaining knowledge gaps.

4.   **Step-by-Step Exploration**: DO NOT try to create a comprehensive search plan all at once. Generate **1-3 queries per round** to explore the problem incrementally. Focus on the most critical missing information first, allowing for iterative refinement based on results.

5.  **Query Quality**: Each query must be **clear, concise, and natural**. Keep queries SHORT and focused on essential keywords.  Avoid overly long or convoluted queries. 

6.  **NO URLs in Queries**: **NEVER** include website URLs, domain names, or site-specific paths in your search queries. Instead, use natural language keywords and entity names.  Focus on WHAT you're looking for, not WHERE to find it.

Your task is to analyze the user's question and all information gathered so far, then decide the next action: either `search` to execute the next step of your strategy or `generate_answer` when your plan is complete.

**Mandatory Output Format:**

You must wrap your entire response in `<response>` tags.  Your output must strictly follow one of the two formats below. 

**Format for Searching:**
The `queries` argument MUST be an array of JSON objects (1-3 queries maximum), with each object containing a `query` string and a `goal` string. 

<response>
<think>
1.   **Objective Analysis**: What is the core goal of the user's question? What key entities and relationships need to be investigated? 
2.  **Intelligence Review**: What information has been successfully gathered? What remains unknown or unverified?
3.  **Strategic Next Step**: What is the single most important piece of missing information NOW? For each missing piece, define a clear goal and then craft a precise, concise query to achieve it.
4.   **Plan Critique**: Briefly challenge your own plan.  Is this the most efficient path? Could the queries and their goals be more precise? 
</think>
<tool_call>
{
  "name": "search",
  "arguments": {
    "queries": [
      {
        "query": "[Natural language keywords combining entities, attributes, and context - NO URLs]",
        "goal": "[Specific goal: what information you intend to find with this query]"
      }
    ]
  }
}
</tool_call>
</response>

**Format for Answering:**
<response>
<think>
1.  **Mission Completion Check**: Cross-reference all gathered intelligence against every component of the original user question. 
2.  **Sufficiency Verdict**: State with conviction that the collected evidence is sufficient to construct a complete, accurate, and authoritative answer.
3.  **Final Action**: Conclude that the information gathering phase is complete and the synthesis of the final answer is the next logical action.
</think>
<tool_call>
{"name": "generate_answer", "arguments": {}}
</tool_call>
</response>
"""


class QAgent(Agent):
    """
    Q-Agent (Planner): Query decomposition and search strategy.

    Input: initial_question + history (Q-think/tool_call, R-think/tool_call, selected_docs per round)
    Output: <think>... </think><tool_call>... </tool_call>
    Reward: RM-Agent scores (except for the final generate_answer call)
    """

    def __init__(self, env_idx: int = None, agent_sample_idx: int = None, benchmark: str = None, **kwargs):
        super().__init__()
        self.env_idx = env_idx
        self.agent_sample_idx = agent_sample_idx
        self.benchmark = benchmark
        self.system_prompt = Q_AGENT_SYSTEM_PROMPT

        # Standard agent attributes required by MultiAgentsExecutionEngine
        self.current_prompt = {"text": "", "image": None}
        self.current_action = ""
        self.agent_reward = 0.
        0
        self.reward_history = []
        self.done = False
        self.success = False

        # Track round info
        self._current_round_idx = 0
        self._is_final_generate_answer = False  # True if this is the generate_answer call

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Update agent state based on environment data.

        Q-Agent input: initial_question + history (Q-think/tool_call, R-think/tool_call, selected_docs)
        """
        self.env_data = env_data
        state = getattr(env_data, "state", None)

        if state is None:
            self.current_prompt = {"text": "Error: No environment state", "image": None}
            return

        question = getattr(state, "question", "")
        trajectory = getattr(state, "trajectory", [])
        current_round = getattr(state, "current_round", 0)
        max_rounds = getattr(env_data, "max_rounds", 10)

        self._current_round_idx = current_round
        self._is_final_generate_answer = False

        # Format history: Q-think/tool_call, R-think/tool_call, selected_docs per round
        history = self._format_history(trajectory)

        # Build prompt
        user_prompt = f"{self.system_prompt}\n\n"
        user_prompt += f"# User Question\n{question}\n\n"

        if history:
            user_prompt += f"# Information Gathered So Far\n{history}\n\n"
        else:
            user_prompt += "# Information Gathered So Far\nNo information gathered yet.  This is the first round.\n\n"

        user_prompt += f"# Current Status\n"
        user_prompt += f"Round: {current_round + 1}/{max_rounds}\n\n"
        user_prompt += "# Your Task\n"
        user_prompt += "Decide: Should we search for more information, or generate an answer?"

        self.current_prompt = {"text": user_prompt, "image": None}

    def _format_history(self, trajectory: List) -> str:
        """
        Format trajectory history for Q-Agent input.

        History includes for each round:
        - Q-Agent: think + tool_call
        - R-Agent: think + tool_call
        - Selected documents with content
        """
        if not trajectory:
            return ""

        history_parts = []

        for round_traj in trajectory:
            round_idx = getattr(round_traj, 'round_idx', 0) if hasattr(round_traj, 'round_idx') else round_traj.get(
                'round_idx', 0)

            history_parts.append(f"## Round {round_idx + 1}")

            # Q-Agent output
            q_think = getattr(round_traj, 'q_agent_think', '') if hasattr(round_traj,
                                                                          'q_agent_think') else round_traj.get(
                'q_agent_think', '')
            q_tool_call = getattr(round_traj, 'q_agent_tool_call', {}) if hasattr(round_traj,
                                                                                  'q_agent_tool_call') else round_traj.get(
                'q_agent_tool_call', {})

            history_parts.append(f"### Q-Agent Response")
            history_parts.append(f"<think>{q_think}</think>")
            if q_tool_call:
                import json
                history_parts.append(f"<tool_call>{json.dumps(q_tool_call)}</tool_call>")

            # Sub-queries with goals
            sub_queries = getattr(round_traj, 'sub_queries', []) if hasattr(round_traj,
                                                                            'sub_queries') else round_traj.get(
                'sub_queries', [])
            if sub_queries:
                history_parts.append(f"**Search Queries:**")
                for i, sq in enumerate(sub_queries, 1):
                    query = sq.query if hasattr(sq, 'query') else sq.get('query', '')
                    goal = sq.goal if hasattr(sq, 'goal') else sq.get('goal', '')
                    history_parts.append(f"  {i}. Query: {query}")
                    history_parts.append(f"     Goal: {goal}")

            # R-Agent output
            r_think = getattr(round_traj, 'r_agent_think', '') if hasattr(round_traj,
                                                                          'r_agent_think') else round_traj.get(
                'r_agent_think', '')
            r_tool_call = getattr(round_traj, 'r_agent_tool_call', {}) if hasattr(round_traj,
                                                                                  'r_agent_tool_call') else round_traj.get(
                'r_agent_tool_call', {})

            history_parts.append(f"### R-Agent Response")
            history_parts.append(f"<think>{r_think}</think>")
            if r_tool_call:
                import json
                history_parts.append(f"<tool_call>{json.dumps(r_tool_call)}</tool_call>")

            # Selected documents
            selected_docs = getattr(round_traj, 'selected_docs', []) if hasattr(round_traj,
                                                                                'selected_docs') else round_traj.get(
                'selected_docs', [])
            if selected_docs:
                history_parts.append(f"### Selected Documents ({len(selected_docs)} docs)")
                for i, doc in enumerate(selected_docs, 1):
                    if hasattr(doc, 'title'):
                        title = doc.title
                        url = doc.url
                        content = doc.content
                    else:
                        title = doc.get('title', 'Untitled')
                        url = doc.get('url', '')
                        content = doc.get('content', doc.get('snippet', ''))

                    history_parts.append(f"**[{i}] {title}**")
                    history_parts.append(f"URL: {url}")
                    # Truncate content for readability
                    if len(content) > 500:
                        content = content[:500] + "..."
                    history_parts.append(f"Content: {content}")

            history_parts.append("")  # Empty line between rounds

        return "\n".join(history_parts)

    def update_from_model(self, response: str):
        """Parse model response."""
        self.current_action = response if response else ""

        # Check if this is a generate_answer action
        from pettingllms.multi_agent_env.deep_search.deep_search_utils import extract_q_agent_response
        parsed = extract_q_agent_response(self.current_action)
        self._is_final_generate_answer = (parsed.get("action_type") == "generate_answer")

        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """Process the agent's response and update environment."""
        # Execute environment step
        if hasattr(env_data, 'step'):
            await env_data.step(
                role="q_agent",
                action=self.current_action,
                env_worker=env_worker
            )

        # Initial reward is 0 - will be set by calculate_reward
        self.agent_reward = 0.0

    def calculate_reward(self, env_data: Env):
        """
        Calculate reward for Q-Agent.

        Note: The final generate_answer call does NOT receive RM scoring.
        Only search actions receive RM-Agent scores.
        """
        if self._is_final_generate_answer:
            # generate_answer call - no RM scoring, reward stays 0
            self.agent_reward = 0.0
            self.reward_history.append(self.agent_reward)
            return

        state = getattr(env_data, "state", None)

        if state and hasattr(state, "q_agent_scores") and state.q_agent_scores:
            # RM-Agent has scored - use those scores
            if self._current_round_idx < len(state.q_agent_scores):
                round_scores = state.q_agent_scores[self._current_round_idx]
                if isinstance(round_scores, dict):
                    self.agent_reward = round_scores.get("_normalized_score", 0.0)
                else:
                    self.agent_reward = float(round_scores) if round_scores else 0.0
        else:
            # RM-Agent hasn't scored yet - placeholder
            self.agent_reward = 0.0

        self.reward_history.append(self.agent_reward)

    def set_reward_from_rm_scores(self, rm_scores: List[Dict], round_idx: int, is_generate_answer: bool = False):
        """
        Set reward from RM-Agent scores after episode completion.

        Args:
            rm_scores: List of per-round RM scores
            round_idx: Round index for this action
            is_generate_answer: Whether this was a generate_answer call (no scoring)
        """
        if is_generate_answer:
            # generate_answer call - no scoring
            self.agent_reward = 0.
            0
            return

        if round_idx < len(rm_scores):
            round_scores = rm_scores[round_idx]
            if isinstance(round_scores, dict):
                self.agent_reward = round_scores.get("_normalized_score", 0.0)
            # Update reward history
            if len(self.reward_history) > round_idx:
                self.reward_history[round_idx] = self.agent_reward

    def reset(self):
        """Reset the agent's internal state."""
        super().reset()
        self.current_prompt = {"text": "", "image": None}
        self.current_action = ""
        self.agent_reward = 0.0
        self.reward_history = []
        self.done = False
        self.success = False
        self._current_round_idx = 0
        self._is_final_generate_answer = False