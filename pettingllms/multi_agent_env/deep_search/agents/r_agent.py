import logging
from typing import Any, Dict, List, Optional

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base. env import Env
from pettingllms.multi_agent_env.deep_search.deep_search_utils import (
    extract_r_agent_response,
    truncatefn,
)

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
    R-Agent (Ranker): Responsible for multi-source result aggregation, 
    relevance filtering, source grading, and document selection.
    
    Input: Initial question + current round Q-response + all_summarized_docs
    Output: think + selected document list (by URL)
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """Initialize the R-Agent."""
        super().__init__()
        self.rollout_idx = rollout_idx
        self. system_prompt = R_AGENT_SYSTEM_PROMPT
        
        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Update agent state based on environment data and generate appropriate prompt.
        
        Args:
            turn_idx: Current turn index
            env_data: Environment data containing current state
        """
        self.env_data = env_data
        
        # Get observation from environment
        from pettingllms.multi_agent_env. deep_search.deep_search_env import AgentRole
        observation = env_data.get_observation(AgentRole.R_AGENT)
        
        question = observation.get("question", "")
        context = observation.get("context", "")
        q_agent_thinking = observation.get("q_agent_think", "")
        queries = observation.get("queries", [])
        docs_formatted = observation.get("docs_formatted", "")
        
        # Build user prompt
        user_prompt = context  # Already formatted by format_context_for_r_agent
        
        # Store as prompt with system and user parts
        self.current_prompt = {
            "text": user_prompt,
            "system": self.system_prompt,
            "image": None
        }

    def update_from_model(self, response: str):
        """
        Parse model response and update agent state. 
        
        Args:
            response: Raw response from the language model
            
        Returns:
            Processed response
        """
        self.current_action = response
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Process the R-Agent's response and execute environment step.
        
        The environment handles:
        - Parsing the response to extract selected URLs
        - Storing selected documents
        - Transitioning to Q-Agent or A-Agent
        
        Args:
            env_data: Environment data
            env_worker: Optional environment worker for parallel execution
        """
        from pettingllms.multi_agent_env. deep_search.deep_search_env import AgentRole
        
        # Execute step in environment
        await env_data.step(
            role=AgentRole.R_AGENT.value,
            action=self.current_action,
            env_worker=env_worker
        )
        
        # Parse response for logging
        parsed = extract_r_agent_response(self.current_action)
        num_selected = len(parsed. get("selected_urls", [])) or len(parsed.get("selected_indices", []))
        
        logger.info(f"R-Agent selected {num_selected} documents")
        
        # Store action in history
        self.action_history.append({
            "think": parsed.get("think", ""),
            "selected_urls": parsed.get("selected_urls", []),
            "selected_indices": parsed. get("selected_indices", []),
        })

    def reset(self):
        """Reset the agent's internal state for a new episode."""
        super().reset()
        self.action_history = []