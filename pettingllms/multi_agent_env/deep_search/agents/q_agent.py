import logging
from typing import Any, Dict, List, Optional

from pettingllms.multi_agent_env.base. agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.deep_search.deep_search_utils import (
    extract_q_agent_response,
    truncatefn,
)

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
    Q-Agent (Planner): Responsible for query decomposition, keyword extraction,
    and deciding when to transition to A-Agent for final answer generation.
    
    Actions:
    1. Generate sub-queries with goals
    2.  Decide to transition to A-Agent
    
    Input: Initial question + history trajectory (Q-response, R-response, selected docs per round)
    Output: think + tool_call (search or generate_answer)
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """Initialize the Q-Agent."""
        super().__init__()
        self.rollout_idx = rollout_idx
        self.system_prompt = Q_AGENT_SYSTEM_PROMPT
        
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
        from pettingllms.multi_agent_env.deep_search.deep_search_env import AgentRole
        observation = env_data.get_observation(AgentRole.Q_AGENT)
        
        question = observation.get("question", "")
        history = observation.get("history", "")
        current_round = observation.get("current_round", 0)
        max_rounds = observation. get("max_rounds", 10)
        
        # Build user prompt
        user_prompt = f"# User Question\n{question}\n\n"
        user_prompt += history
        
        user_prompt += f"\n\n# Current Status\n"
        user_prompt += f"Round: {current_round + 1}/{max_rounds}\n"
        
        user_prompt += "\n# Your Task\n"
        user_prompt += "Decide: Should we search for more information, or do we have enough to generate an answer?"
        
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
        Process the Q-Agent's response and execute environment step.
        
        The environment handles:
        - Parsing the response (search vs generate_answer)
        - Executing web search if search action
        - Transitioning to R-Agent or A-Agent
        
        Args:
            env_data: Environment data
            env_worker: Optional environment worker for parallel execution
        """
        from pettingllms.multi_agent_env.deep_search.deep_search_env import AgentRole
        
        # Execute step in environment
        await env_data.step(
            role=AgentRole. Q_AGENT. value,
            action=self.current_action,
            env_worker=env_worker
        )
        
        # Parse response to determine action type for logging
        parsed = extract_q_agent_response(self. current_action)
        action_type = parsed. get("action_type", "search")
        
        if action_type == "generate_answer":
            logger.info(f"Q-Agent decided to generate final answer")
        else:
            num_queries = len(parsed.get("queries", []))
            logger.info(f"Q-Agent generated {num_queries} sub-queries")
        
        # Store action in history
        self. action_history.append({
            "action_type": action_type,
            "think": parsed.get("think", ""),
            "queries": parsed.get("queries", []),
        })

    def reset(self):
        """Reset the agent's internal state for a new episode."""
        super(). reset()
        self.action_history = []