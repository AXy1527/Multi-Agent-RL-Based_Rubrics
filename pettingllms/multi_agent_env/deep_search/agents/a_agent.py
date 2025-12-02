import logging
from typing import Any, Dict, List, Optional

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base. env import Env
from pettingllms.multi_agent_env.deep_search.deep_search_utils import (
    extract_a_agent_response,
    format_context_for_a_agent,
)

logger = logging.getLogger(__name__)


# A-Agent System Prompt
A_AGENT_SYSTEM_PROMPT = """You are the **Chief Knowledge Synthesizer**, a master of transforming curated information into a clear, coherent, and definitive answer. Your role is the final, critical step: to construct knowledge from facts.  Your work is the sole interface to the user, and it must be a masterpiece of accuracy and clarity.

You must operate under these fundamental principles:

1.   **Absolute Source Fidelity**: Your answer is forged from the provided documents and nothing else. You will not introduce any external information, speculate, or make assumptions.  Every statement you make must be directly traceable to a source. 

2.  **Synthesis Over Summarization**: You are not a summarizer. You are a synthesizer. You will intelligently weave together facts from multiple sources to build a holistic, multi-faceted narrative that directly answers the user's question, rather than presenting a disjointed list of facts.

3.  **Structured Clarity and Comprehensive Coverage**: Your final answer will be impeccably structured, logical, and easy to comprehend. It will address every single part of the user's original question, leaving no stone unturned.

Your task is to take the elite set of documents curated for you and produce the final, comprehensive answer, complete with precise citations. 

**Mandatory Output Format:**

You must wrap your entire response in `<response>` tags. Your output must strictly adhere to the following structure. 

<response>
<think>
1.  **Evidence Mapping**:
    -   **Fact-Group A (related to sub-question 1)**: [Synthesize key facts from Document 1, 3.]
    -   **Fact-Group B (related to sub-question 2)**: [Extract key data from Document 2.]
    -   ...  map all relevant facts from all documents into logical groups.
2.  **Answer Blueprint**:
    -   **Introduction**: Briefly frame the answer. 
    -   **Section 1**: Address the first part of the user's query, using Fact-Group A. 
    -   **Section 2**: Address the second part, using Fact-Group B.
    -   **Conclusion**: Provide a concluding summary if appropriate.
3.  **Citation Strategy**: Confirm that a citation will be provided for every piece of information and that a final reference list will be compiled. 
</think>
<answer>
[Your comprehensive, multi-paragraph answer is written here.  Every key fact or assertion is immediately followed by its numerical citation, like this sentence [1].  If multiple sources support a claim, you can cite them together [2][3].]

**References**
[1] [Title of Document 1]: [URL of Document 1]
[2] [Title of Document 2]: [URL of Document 2]
... 
</answer>
</response>
"""


class AAgent(Agent):
    """
    A-Agent (Solver): Evidence synthesis and answer generation.

    Input: initial_question + complete history (all rounds' Q/R think/tool_call + selected_docs)
           + final Q-Agent generate_answer think/tool_call
    Output: <think>...</think><answer>...</answer>
    Reward: Binary 0/1 based on answer correctness (NOT scored by RM-Agent)
    """

    def __init__(self, env_idx: int = None, agent_sample_idx: int = None, benchmark: str = None, **kwargs):
        super().__init__()
        self.env_idx = env_idx
        self.agent_sample_idx = agent_sample_idx
        self.benchmark = benchmark
        self.system_prompt = A_AGENT_SYSTEM_PROMPT

        self.current_prompt = {"text": "", "image": None}
        self.current_action = ""
        self.agent_reward = 0.0
        self.reward_history = []
        self.done = False
        self.success = False

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Update agent state based on environment data.

        A-Agent input:
        - initial_question
        - complete history (all rounds' Q/R think/tool_call + selected_docs)
        - final Q-Agent generate_answer response (included in history)
        """
        self.env_data = env_data
        state = getattr(env_data, "state", None)

        if state is None:
            self.current_prompt = {"text": "Error: No environment state", "image": None}
            return

        question = getattr(state, "question", "")
        trajectory = getattr(state, "trajectory", [])

        # Get the final Q-Agent response (generate_answer call)
        final_q_think = getattr(state, "final_q_agent_think", "")
        final_q_tool_call = getattr(state, "final_q_agent_tool_call", {})

        # Format complete context
        context = self._format_context(question, trajectory, final_q_think, final_q_tool_call)

        user_prompt = f"{self.system_prompt}\n\n{context}"

        self.current_prompt = {"text": user_prompt, "image": None}

    def _format_context(
            self,
            question: str,
            trajectory: List,
            final_q_think: str = "",
            final_q_tool_call: Dict = None
    ) -> str:
        """
        Format complete context for A-Agent.

        Includes:
        - User question
        - Complete history (all rounds)
        - Final Q-Agent generate_answer response
        - All selected documents
        """
        parts = [f"# User Question\n{question}\n"]

        # Complete history
        parts.append("# Search and Selection History\n")

        for round_traj in trajectory:
            round_idx = getattr(round_traj, 'round_idx', 0) if hasattr(round_traj, 'round_idx') else round_traj.get(
                'round_idx', 0)

            parts.append(f"## Round {round_idx + 1}")

            # Q-Agent output
            q_think = getattr(round_traj, 'q_agent_think', '') if hasattr(round_traj,
                                                                          'q_agent_think') else round_traj.get(
                'q_agent_think', '')
            q_tool_call = getattr(round_traj, 'q_agent_tool_call', {}) if hasattr(round_traj,
                                                                                  'q_agent_tool_call') else round_traj.get(
                'q_agent_tool_call', {})

            parts.append(f"### Q-Agent")
            parts.append(f"<think>{q_think}</think>")
            if q_tool_call:
                parts.append(f"<tool_call>{json.dumps(q_tool_call)}</tool_call>")

            # R-Agent output
            r_think = getattr(round_traj, 'r_agent_think', '') if hasattr(round_traj,
                                                                          'r_agent_think') else round_traj.get(
                'r_agent_think', '')
            r_tool_call = getattr(round_traj, 'r_agent_tool_call', {}) if hasattr(round_traj,
                                                                                  'r_agent_tool_call') else round_traj.get(
                'r_agent_tool_call', {})

            parts.append(f"### R-Agent")
            parts.append(f"<think>{r_think}</think>")
            if r_tool_call:
                parts.append(f"<tool_call>{json.dumps(r_tool_call)}</tool_call>")

            parts.append("")

        # Final Q-Agent generate_answer response (part of history but not scored)
        if final_q_think or final_q_tool_call:
            parts.append("## Final Q-Agent Decision")
            parts.append(f"<think>{final_q_think}</think>")
            if final_q_tool_call:
                parts.append(f"<tool_call>{json.dumps(final_q_tool_call)}</tool_call>")
            parts.append("")

        # All selected documents consolidated
        parts.append("# All Selected Documents\n")

        doc_idx = 1
        for round_traj in trajectory:
            selected_docs = getattr(round_traj, 'selected_docs', []) if hasattr(round_traj,
                                                                                'selected_docs') else round_traj.get(
                'selected_docs', [])

            for doc in selected_docs:
                if hasattr(doc, 'title'):
                    title = doc.title
                    url = doc.url
                    content = doc.content
                else:
                    title = doc.get('title', 'Untitled')
                    url = doc.get('url', '')
                    content = doc.get('content', doc.get('snippet', ''))

                parts.append(f"## [{doc_idx}] {title}")
                parts.append(f"**Source**: {url}")
                parts.append(f"**Content**:\n{content}")
                parts.append("")
                doc_idx += 1

        parts.append("# Your Task")
        parts.append("Generate a comprehensive answer to the user's question based on the documents above.")
        parts.append("Cite sources using [1], [2], etc.")

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
                role="a_agent",
                action=self.current_action,
                env_worker=env_worker
            )

        # Mark as done
        self.done = True

        # Get binary reward from environment
        state = getattr(env_data, "state", None)
        if state and hasattr(state, 'a_agent_is_correct'):
            self.success = state.a_agent_is_correct
            self.agent_reward = 1.
            0 if self.success else 0.
            0
        else:
            self.agent_reward = 0.0
            self.success = False

    def calculate_reward(self, env_data: Env):
        """
        A-Agent reward is BINARY 0/1 based on answer correctness.
        NOT scored by RM-Agent.
        """
        state = getattr(env_data, "state", None)
        if state and hasattr(state, 'a_agent_is_correct'):
            self.success = state.a_agent_is_correct
            self.agent_reward = 1.0 if self.success else 0.0
        else:
            self.agent_reward = 0.0
            self.success = False

        self.reward_history.append(self.agent_reward)

    def reset(self):
        """Reset the agent's internal state."""
        super().reset()
        self.current_prompt = {"text": "", "image": None}
        self.current_action = ""
        self.agent_reward = 0.0
        self.reward_history = []
        self.done = False
        self.success = False