import logging
from typing import Any, Dict, List, Optional

from pettingllms.multi_agent_env. base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.deep_search.deep_search_utils import (
    extract_a_agent_response,
    truncatefn,
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
    A-Agent (Solver): Responsible for evidence synthesis and answer generation.
    
    Input: Initial question + complete history trajectory
    Output: think + final answer
    
    Reward: Binary 0/1 based on answer correctness (NO RM scoring)
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """Initialize the A-Agent."""
        super().__init__()
        self.rollout_idx = rollout_idx
        self. system_prompt = A_AGENT_SYSTEM_PROMPT
        
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
        observation = env_data.get_observation(AgentRole.A_AGENT)
        
        question = observation.get("question", "")
        context = observation.get("context", "")
        
        # Build user prompt (context already formatted by format_context_for_a_agent)
        user_prompt = context
        
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
        Process the A-Agent's response and execute environment step.
        
        The environment handles:
        - Parsing the response to extract final answer
        - Evaluating correctness against ground truth
        - Setting binary reward (0 or 1)
        - Terminating the episode
        
        Args:
            env_data: Environment data
            env_worker: Optional environment worker for parallel execution
        """
        from pettingllms.multi_agent_env. deep_search.deep_search_env import AgentRole
        
        # Execute step in environment
        await env_data.step(
            role=AgentRole.A_AGENT.value,
            action=self.current_action,
            env_worker=env_worker
        )
        
        # Get binary reward from environment
        self.agent_reward = env_data.get_a_agent_reward()
        self.reward_history.append(float(self.agent_reward))
        
        # Set done and success flags
        self.done = True
        self.success = env_data.state.a_agent_is_correct
        
        # Parse response for logging
        parsed = extract_a_agent_response(self.current_action)
        
        logger.info(f"A-Agent generated answer.  Correct: {self.success}, Reward: {self.agent_reward}")
        
        # Store action in history
        self.action_history.append({
            "think": parsed.get("think", ""),
            "answer": parsed.get("answer", ""),
            "is_correct": self. success,
            "reward": self.agent_reward,
        })

    def reset(self):
        """Reset the agent's internal state for a new episode."""
        super().reset()
        self.action_history = []
        self.done = False
        self.success = False