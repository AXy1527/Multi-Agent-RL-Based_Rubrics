"""
R-Agent (Ranker): Document selection and relevance filtering.
Compatible with MultiAgentsExecutionEngine interface.

Input for each round:
- Initial question
- Current round's Q-Agent think and tool_call
- All docs from the search (before filtering)

Reward: Computed by RM-Agent after episode completion
"""
import logging
from typing import Any, Dict, List, Optional

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


# R-Agent System Prompt
R_AGENT_SYSTEM_PROMPT = """You are the **Master Information Curator**, the discerning gatekeeper of knowledge for this system. Your purpose is not merely to select documents, but to sculpt a high-fidelity information base from raw, noisy search results. Quality, relevance, and utility are your only metrics. 

As you proceed, you will be guided by these inviolable principles:

1.   **Unwavering Relevance**: You will fixate on the user's original question and the current search goal. Any document that does not directly and substantially contribute to answering it is noise, and you will discard it without hesitation.

2.  **Aggressive Noise Filtration**: You are the bulwark against information overload. You will ruthlessly filter out redundant, low-quality, outdated, and tangentially related documents to protect the integrity of the final answer.

3.   **Maximization of Insight**: Your selection is not a collection, but a portfolio. You will choose documents that, together, provide a complementary and multi-faceted view, prioritizing sources that offer specific data, deep insights, or authoritative evidence over superficial summaries.

Your mission is to evaluate the candidate documents from the latest search and select a small, elite set that forms the strongest possible foundation for the final answer. 

**Mandatory Output Format:**

You must wrap your entire response in `<response>` tags. Your output must strictly adhere to the following structure. 

<response>
<think>
1.  **Curatorial Objective**: Restate the specific information goal for this round, as defined by the Chief Inquiry Strategist's plan.
2.  **Document Triage**:
    -   **Document 1 ([Title])**: [Assess its value.  State "SELECT" or "REJECT".  Justify your decision based on its direct relevance, information density, and novelty.  e.g., "SELECT: Provides the exact statistical data requested in the query. "]
    -   **Document 2 ([Title])**: [e.g., "REJECT: Superficial overview with no new information compared to Document 1."]
    -   ...  continue for all candidate documents. 
3.  **Final Collection Rationale**: Conclude with a summary of why the selected documents form a powerful, synergistic set.  Explain how their combined information will enable a comprehensive answer. 
</think>
<tool_call>
{"name": "select_documents", "arguments": {"selected_urls": ["url_of_selected_doc_1", "url_of_selected_doc_2", ...]}}
</tool_call>
</response>
"""


class RAgent(Agent):
    """
    R-Agent (Ranker): Document selection and relevance filtering.

    Input: initial_question + current_round Q-Agent think/tool_call + all_docs from search
    Output: <think>...</think><tool_call>... </tool_call>
    Reward: RM-Agent scores (per-round, per-document)
    """

    def __init__(self, env_idx: int = None, agent_sample_idx: int = None, benchmark: str = None, **kwargs):
        super().__init__()
        self.env_idx = env_idx
        self.agent_sample_idx = agent_sample_idx
        self.benchmark = benchmark
        self.system_prompt = R_AGENT_SYSTEM_PROMPT

        self.current_prompt = {"text": "", "image": None}
        self.current_action = ""
        self.agent_reward = 0.0
        self.reward_history = []
        self.done = False
        self.success = False

        self._current_round_idx = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Update agent state based on environment data.

        R-Agent input: initial_question + current_round Q-Agent think/tool_call + all_docs
        """
        self.env_data = env_data
        state = getattr(env_data, "state", None)

        if state is None:
            self.current_prompt = {"text": "Error: No environment state", "image": None}
            return

        question = getattr(state, "question", "")
        trajectory = getattr(state, "trajectory", [])

        if not trajectory:
            self.current_prompt = {"text": "Error: No trajectory available", "image": None}
            return

        # Get current round data
        current_round_traj = trajectory[-1]
        self._current_round_idx = getattr(current_round_traj, 'round_idx', len(trajectory) - 1)

        # Get Q-Agent's output for this round
        q_think = getattr(current_round_traj, 'q_agent_think', '')
        q_tool_call = getattr(current_round_traj, 'q_agent_tool_call', {})
        sub_queries = getattr(current_round_traj, 'sub_queries', [])

        # Get all docs retrieved this round
        all_docs = getattr(current_round_traj, '_current_round_all_docs', [])
        if not all_docs and hasattr(current_round_traj, 'all_docs'):
            all_docs = current_round_traj.all_docs

        # Build prompt
        user_prompt = f"{self.system_prompt}\n\n"
        user_prompt += f"# User Question\n{question}\n\n"

        # Q-Agent's analysis for this round
        user_prompt += f"# Q-Agent's Analysis for This Round\n"
        user_prompt += f"<think>{q_think}</think>\n"
        if q_tool_call:
            user_prompt += f"<tool_call>{json.dumps(q_tool_call)}</tool_call>\n\n"

        # Sub-queries with goals
        if sub_queries:
            user_prompt += f"# Search Queries\n"
            for i, sq in enumerate(sub_queries, 1):
                query = sq.query if hasattr(sq, 'query') else sq.get('query', '')
                goal = sq.goal if hasattr(sq, 'goal') else sq.get('goal', '')
                user_prompt += f"{i}. Query: {query}\n   Goal: {goal}\n"
            user_prompt += "\n"

        # All candidate documents
        user_prompt += f"# Candidate Documents ({len(all_docs)} documents)\n"
        user_prompt += self._format_docs(all_docs)
        user_prompt += "\n# Your Task\n"
        user_prompt += "Select the most relevant and useful documents by their URLs."

        self.current_prompt = {"text": user_prompt, "image": None}

    def _format_docs(self, docs: List) -> str:
        """Format documents for R-Agent display."""
        if not docs:
            return "No documents available.\n"

        parts = []
        for i, doc in enumerate(docs, 1):
            if isinstance(doc, dict):
                title = doc.get('title', 'Untitled')
                url = doc.get('url', '')
                snippet = doc.get('snippet', '')
                content = doc.get('content', '')
                source_query = doc.get('source_query', '')
                source_goal = doc.get('source_goal', '')
            else:
                title = getattr(doc, 'title', 'Untitled')
                url = getattr(doc, 'url', '')
                snippet = getattr(doc, 'snippet', '')
                content = getattr(doc, 'content', '')
                source_query = getattr(doc, 'source_query', '')
                source_goal = getattr(doc, 'source_goal', '')

            parts.append(f"## Document {i}")
            parts.append(f"**Title**: {title}")
            parts.append(f"**URL**: {url}")
            if source_query:
                parts.append(f"**From Query**: {source_query}")
            if source_goal:
                parts.append(f"**Query Goal**: {source_goal}")
            parts.append(f"**Snippet**: {snippet}")
            if content:
                # Show full content for selection decision
                parts.append(f"**Full Content**:\n{content}")
            parts.append("")

        return "\n".join(parts)

    def update_from_model(self, response: str):
        """Parse model response."""
        self.current_action = response if response else ""
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """Process the agent's response and update environment."""
        # Execute environment step
        if hasattr(env_data, 'step'):
            await env_data.step(
                role="r_agent",
                action=self.current_action,
                env_worker=env_worker
            )

        self.agent_reward = 0.0

    def calculate_reward(self, env_data: Env):
        """
        Calculate reward from RM-Agent scores.
        R-Agent receives per-round, per-document scores.
        """
        state = getattr(env_data, "state", None)

        if state and hasattr(state, "r_agent_scores") and state.r_agent_scores:
            # Find scores for current round
            for round_score in state.r_agent_scores:
                if round_score.get("_round_idx") == self._current_round_idx:
                    # Average document scores
                    doc_scores = [
                        v.get("_present_ratio", 0.0)
                        for k, v in round_score.items()
                        if k.startswith("doc_") and isinstance(v, dict)
                    ]
                    self.agent_reward = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
                    break
        else:
            self.agent_reward = 0.0

        self.reward_history.append(self.agent_reward)

    def set_reward_from_rm_scores(self, rm_scores: List[Dict], round_idx: int):
        """Set reward from RM-Agent scores after episode completion."""
        for round_score in rm_scores:
            if round_score.get("_round_idx") == round_idx:
                doc_scores = [
                    v.get("_present_ratio", 0.0)
                    for k, v in round_score.items()
                    if k.startswith("doc_") and isinstance(v, dict)
                ]
                self.agent_reward = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
                if len(self.reward_history) > round_idx:
                    self.reward_history[round_idx] = self.agent_reward
                break

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