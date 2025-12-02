import logging
import json
import os
import asyncio
from typing import Any, Dict, List, Optional, Union
import requests

from pettingllms.multi_agent_env.base. agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms. multi_agent_env.deep_search.deep_search_utils import (
    format_trajectory_for_rm_q_scoring,
    format_round_for_rm_r_scoring,
)

logger = logging. getLogger(__name__)

# Default Rubrics for Q-Agent
DEFAULT_Q_AGENT_RUBRICS = '''
1.Decomposition Correctness : The query plan should accurately break down the original question, effectively identifying and extracting the core entities, action/operation instructions, and key constraints in the question. Each sub-query should correctly correspond to a specific combination of subject, instruction, and its constraints, and the full set of sub-queries should completely cover all the information needs of the original question.

2.Decomposition Independence : Each sub-query in the query plan should have an independent, non-overlapping retrieval goal. There should be no sub-queries with functional overlap or semantic redundancy.

3.Expression Accuracy : Each sub-query should accurately convey the meaning of the original question, be clear and unambiguous, and be understandable without relying on surrounding context. No vague or ambiguous wording should be present.

4.Retrieval Timeliness : For questions that include time constraints, the query plan should correctly parse, preserve, or supplement the time requirements to ensure the retrieved results meet the timeliness needs.

5.Expression Conciseness : Sub-queries should use standard written phrasing, be semantically concise yet complete, and avoid colloquial language, verbosity, or unnecessary embellishments.

6.Expansion Relevance : If the user query is expanded, the query plan should perform reasonable semantic expansion, and the expansion dimensions should be highly relevant to the core needs, with no redundant expansions or low-relevance expansion content. If there is no expansion, this criterion is considered satisfied by default.

'''


# Default Rubrics for R-Agent
DEFAULT_R_AGENT_RUBRICS = '''
  1.Document Relevance : The document should fully match the core needs of the user query and the generated query plans, and should satisfy the user’s primary information need or directly answer the user’s question.
  
  2.Document Timeliness : If the user query and the generated query plans involve a specific point in time or a time range, the document’s content must align with the specified time. For example, for “Can a 610 score in the 2025 Gaokao get into Tsinghua University?”, the document must provide the 2025 cutoff score. Note: if a time range is specified, content covering any part of that range is considered compliant—for example, for “China’s GDP from Jan–Jul 2024,” a document that provides the GDP for any single month within Jan–Jul 2024 is acceptable. If neither the user query nor the query plans include a time requirement, the document is considered timely by default.
  
  3.Document Authority : Authority requirements vary by domain. The document must meet the authority expectations of the domain implied by the user query and the query plans. Determine whether the document satisfies the authority requirement by analyzing the document’s URL domain name and the source indicators in its title and snippet.
  '''


# RM-Agent System Prompt for Q-Agent Scoring
RM_Q_AGENT_SYSTEM_PROMPT = '''You are a professional evaluator for a multi-agent deep search system. Your task is to objectively and rigorously assess the quality of Q-Agent's query planning actions for ALL rounds based on the provided rubrics and the complete trajectory.

  ## System Context

  You are evaluating a multi-agent system with the following components:
  - **Q-Agent (Planner)**: Responsible for query decomposition, keyword extraction, and generating sub-queries
  - **R-Agent (Ranker)**: Responsible for document selection and relevance filtering
  - **A-Agent (Solver)**: Responsible for final answer generation

  The system operates in rounds following the pattern: Q → R → Q → R → ... → Q → A

  ## Input Components

  You will receive the following information:

  1. **User Query**: The user's original question or information need

  2. **Complete Trajectory**: The full reasoning trajectory including all rounds with:
     - All Q-Agent's think processes and generated plans
     - All R-Agent's think processes and selected documents
     - This provides complete context for understanding the entire search process

  3. **Rubrics**: Evaluation criteria for assessing Q-Agent's query planning quality

  ## Evaluation Rules

  ### Rule 1: Binary Assessment with Score
  For each rubric criterion in each Q-Agent round, you must provide:
  - **judgment**: Either "PRESENT" (fully satisfies) or "NOT_PRESENT" (does not satisfy or partially satisfies)
  - **score**: A numerical score from 0-10
    - 0-3: Severe issues, fundamental requirements not met
    - 4-6: Moderate issues, some requirements met but significant gaps
    - 7-8: Good quality, most requirements met with minor issues
    - 9-10: Excellent quality, all requirements fully satisfied
  - **reason**: Clear explanation for the judgment and score

  ### Rule 2: Context-Aware Evaluation
  When evaluating each round's plans, you MUST consider:
  - **Historical trajectory**: What has been searched and discovered in previous rounds
  - **Search progress**: Whether the current plans build upon or complement previous searches
  - **Redundancy**: Whether the current plans unnecessarily repeat previous searches
  - **Coherence**: Whether the current plans logically follow from the historical context
  - **Impact on subsequent searches**: How the current round's planning influences and guides subsequent search rounds, including whether it effectively narrows down the search space, identifies key information gaps, or sets up strategic directions for future exploration

  ### Rule 3: Multi-Criteria Comprehensive Assessment
  If a rubric contains multiple criteria, ALL must be satisfied for a PRESENT judgment. If ANY criterion is not met, the judgment should be NOT_PRESENT.

  ### Rule 4: Incremental Progress Assessment
  Evaluate whether each round's plans demonstrate:
  - **Progressive refinement**: Building on insights from previous rounds
  - **Gap filling**: Addressing information needs not covered by previous searches
  - **Strategic thinking**: Showing awareness of what has been tried and what remains to be explored

  ## Evaluation Process

  ### Step 1: Understand the Complete Context
  - Read the user query to identify the core information need
  - Review the complete trajectory to understand:
    - What has been searched in each round
    - What documents were selected and why
    - How the search progressed over time
  - Identify the search strategy and information gathering pattern

  ### Step 2: Analyze Each Q-Agent Round
  For each Q-Agent round in the trajectory:
  - Examine the Q-Agent's think process:
    - Does it show awareness of historical context?
    - Does it identify remaining information gaps?
    - Is the reasoning logical and well-justified?
  - Evaluate the generated plans:
    - Relevance to the user's core need
    - Complementarity with previous searches
    - Specificity and clarity of expression
    - Likelihood of retrieving useful new information

  ### Step 3: Evaluate Against Each Criterion
  For each Q-Agent round and each rubric criterion:
  - Parse all conditions and requirements
  - Check whether the plans satisfy these conditions IN CONTEXT of the historical trajectory
  - Consider both the plans themselves and the Q-Agent's reasoning
  - Assign an appropriate score reflecting the quality level

  ### Step 4: Provide Judgment, Score, and Reasoning
  For each round and each criterion, provide:
  - Clear binary judgment (PRESENT/NOT_PRESENT)
  - Numerical score (0-10)
  - Concise reasoning (2-4 sentences) explaining:
    - How the plans perform against the criterion
    - Relevant evidence from plans, think process, or historical context
    - Specific strengths or weaknesses
    - Why the particular score was assigned

  ## Output Format

  Your response must strictly follow this JSON format, with evaluations for ALL Q-Agent rounds:

  ```json
  {{
    "round_1": {{
      "1": {{
        "judgment": "PRESENT",
        "score": 8,
        "reason": "The first round's plans accurately decompose the original question into core information needs. The Q-Agent identifies key entities and constraints effectively. Minor issue: one sub-query could be more specific."
      }},
      "2": {{
        "judgment": "PRESENT",
        "score": 9,
        "reason": "Each sub-query has independent retrieval goals with no functional overlap. The decomposition is clean and well-structured."
      }}
    }},
    "round_2": {{
      "1": {{
        "judgment": "PRESENT",
        "score": 7,
        "reason": "The second round's plans build upon Round 1 results and address remaining information gaps. However, one sub-query shows slight overlap with Round 1."
      }},
      "2": {{
        "judgment": "NOT_PRESENT",
        "score": 5,
        "reason": "While most plans are independent, there is partial overlap between two sub-queries in this round, reducing efficiency."
      }}
    }}
  }}
  ```

  ## Important Notes

  1. **Complete Trajectory Evaluation**: You must evaluate ALL Q-Agent rounds in the trajectory
  2. **Historical Context is Critical**: Always evaluate each round's plans in relation to what has been done before
  3. **Objectivity**: Base evaluation on actual content, not speculation
  4. **Completeness**: Provide judgments for ALL rubric criteria for ALL Q-Agent rounds
  5. **Score Calibration**: Use the full 0-10 range appropriately; don't cluster all scores in the middle
  6. **Reasoning Quality**: Provide specific, evidence-based reasoning that clearly justifies both the judgment and score

  ---

  Now, please evaluate the following content:

  **User Query:**
  ```
  {user_query}
  ```

  **Complete Trajectory (All Rounds):**
  ```
  {complete_trajectory}
  ```

  **Rubrics:**
  ```
  {rubrics}
  ```

  Based on the above information, please evaluate EACH Q-Agent round's query planning action against each rubric criterion, considering the complete trajectory context. Your JSON evaluation:
  '''

RM_R_AGENT_SYSTEM_PROMPT = '''You are a professional evaluator for a document retrieval system. Your task is to objectively and rigorously assess the quality of each document returned by a web search system according to the provided rubrics.

  ## Input Components

  You will receive the following information:

  1. **Current Round Think**: The reasoning and analysis output from the thinking process in the current ranking round, explaining the retrieval strategy and information needs.

  2. **Historical Context** (if applicable): Information from previous retrieval rounds, including:
    - **Previous Round Number**: The identifier of the historical round
    - **Historical Think**: The thinking output from that historical round
    - **Historical Documents**: Documents retrieved and ranked in that round, each containing:
      - **plan**: The sub-query that retrieved this document
      - **id**: Unique identifier for the document
      - **url**: The document's web address
      - **title**: Document title
      - **snippet**: Document summary excerpt
      - **content**: The complete body text of the document

  3. **Current Document List**: A collection of documents returned by the web search system in the current round. Each document contains the following fields:
    - **plan**: The sub-query plan that retrieved this document, representing the specific information need this document should address
    - **id**: Unique identifier for the document
    - **url**: The document's web address
    - **title**: Document title
    - **snippet**: Document summary excerpt
    - **content**: The complete body text of the document

  4. **Rubrics**: A set of evaluation criteria for assessing document quality. Each criterion has a unique identifier and specific assessment requirements.


  ## Evaluation Rules

  ### Rule 1: Binary Assessment
  For each rubric criterion, you must make a clear binary judgment:
  - **PRESENT**: The document fully satisfies the requirements of the rubric criterion
  - **NOT_PRESENT**: The document does not satisfy the rubric criterion (including cases of partial satisfaction)

  **Example:**
  - Sub-query plan: "Health benefits of apples"
  - Retrieved Document: Content describes apple cooking recipes
  - Rubric: "The document should fully match the core needs of the sub-query plan"
  - Judgment: NOT_PRESENT (The document clearly deviates from the sub-query, discussing cooking rather than health benefits)

  ### Rule 2: Multi-Criteria Comprehensive Assessment
  If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be NOT_PRESENT. Only return PRESENT if all of the criteria are met.

  ### Rule 3: Lenient Handling of Example-Based Conditions
  One important exception to the above rule is that if a rubric says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.

  ### Rule 4: Context-Dependent Evaluation
  The role of current round think and historical context depends on the specific rubric being evaluated:
  - **For sub-query alignment rubrics**: Focus primarily on whether the document matches its sub-query plan. Current think and historical context provide background understanding.
  - **For context-related rubrics**: If a rubric explicitly evaluates aspects like "consistency with previous information," "complementarity to historical documents," "context coherence," or similar criteria, then historical context and current round think become essential evaluation materials, not just auxiliary information.
  - Always read each rubric carefully to determine how to use the provided context information.


  ## Evaluation Process

  ### Step 1: Understand All Available Information
  - Review the current round think to understand the overall retrieval strategy
  - If provided, review historical context including previous thinks and documents
  - For each document:
    - Carefully read the sub-query plan to understand the specific information need
    - Read the document's title, snippet, and complete content

  ### Step 2: Evaluate Each Criterion
  For each rubric criterion:
  - **First**, parse the criterion: Identify all conditions and requirements
  - **Second**, determine what information is needed for evaluation:
    - Does this criterion focus on sub-query alignment? → Focus on document vs. sub-query plan
    - Does this criterion involve contextual factors (e.g., consistency with history, complementarity, avoiding redundancy)? → You MUST use historical context and current think as primary evaluation materials
    - Does this criterion assess other aspects (e.g., authority, timeliness)? → Use all provided information as appropriate
  - **Third**, check the document: Verify whether the document satisfies the criterion based on the relevant information sources
  - Pay special attention to:
    - Whether multiple conditions exist (all must be satisfied)
    - Whether example-based expressions exist (satisfying one is sufficient)
    - Whether the criterion requires comparison with historical information
    - Whether the criterion requires understanding of the current retrieval strategy

  ### Step 3: Provide Judgment and Reasoning
  For each criterion, you must:
  - Give a clear judgment: PRESENT or NOT_PRESENT
  - Provide concise but sufficient reasoning, explaining:
    - Which conditions the document satisfied (or failed to satisfy)
    - Where in the document (or historical context, if relevant) the specific evidence comes from
    - For context-related rubrics, explicitly reference how the document relates to historical information or current think
    - If NOT_PRESENT, clearly indicate what is missing or inconsistent


  ## Output Format
  Your response must strictly follow this format:
  - Begin with a valid JSON object
  - Output the evaluation results for each document, using the document ID to represent each document
  - JSON must be wrapped in a code block: starting with ```json and ending with ```
  - Each document includes the rubric criterion and the results. The rubric criteria are represented by identifier. The results consist of two parts:
    - "judgment": String, value must be "PRESENT" or "NOT_PRESENT"
    - "reason": String, explaining the judgment reasoning
  - Ensure JSON format validity:
    - Use double quotes
    - Correct commas and brackets
    - No comments
    - No extraneous text

  ## Output Template
  ```json{{
    "doc_1_id":{{
      "1": {{
        "judgment": "PRESENT",
        "reason": "The document fully matches the sub-query plan 'recent advances in transformer models', providing detailed technical information about new transformer architectures from 2024, directly answering the sub-query's information need."
      }},
      "2": {{
        "judgment": "NOT_PRESENT",
        "reason": "The rubric requires the document to provide complementary information to historical documents. However, this document repeats the same transformer architecture details already covered in Round 1, Document 3, without adding new insights."
      }},
      "3": {{
        "judgment": "PRESENT",
        "reason": "The document source is from arxiv.org with peer-reviewed content, satisfying the authority requirement specified in the rubric."
      }}
    }},
    "doc_2_id":{{}}
  }}
  ```

  ## Important Notes

  - **Rubric-Driven Evaluation**: The nature of each rubric determines how you should use the provided information
  - **Context as Evidence**: When evaluating context-related rubrics, historical context and current think are NOT auxiliary materials—they are essential evidence you must reference
  - **Objectivity**: Base your evaluation on the actual content of documents, historical context, and think outputs; do not add subjective speculation
  - **Completeness**: You must provide judgments for all provided rubric criteria
  - **Accuracy**: Reasoning must accurately cite relevant sources (document content, historical documents, or think outputs)
  - **Conciseness**: Reasoning should be concise and clear, typically 1-2 sentences
  - **Consistency**: Apply the same judgment standards to similar types of criteria
  - **Explicit References**: For context-related rubrics, explicitly mention which historical documents or think outputs you are comparing against

  ---

  Now, please evaluate the following content:

  **Current Round Think:**
  ```
  {current_think}
  ```

  **Historical Context:**
  ```
  {historical_context}
  ```

  **Current Document List:**
  ```
  {documents}
  ```

  **Rubrics:**
  ```
  {rubrics}
  ```

  Based on the above information, please evaluate these documents according to the rubrics. Your JSON evaluation:
  '''


class LLMClient: """ Client for calling LLM APIs (supports both open-source and closed-source models). """
    def __init__(
            self,
            api_url: str = None,
            api_key: str = None,
            model: str = None,
            timeout: int = 120,
    ):
        """
        Initialize LLM client.

        Args:
            api_url: API endpoint URL (e.g., OpenAI-compatible endpoint)
            api_key: API key for authentication
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.api_url = api_url or os.getenv("RM_LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        self.api_key = api_key or os.getenv("RM_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.model = model or os.getenv("RM_LLM_MODEL", "gpt-4")
        self.timeout = timeout


    async def call(
            self,
            prompt: str,
            system_prompt: str = None,
            temperature: float = 0.1,
            max_tokens: int = 4096,
    ) -> Optional[str]:
        """
        Call LLM API.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text, or None if failed
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None


class RMAgent(Agent):
    """
    RM-Agent (Rubric-based Critic): Core component for reward shaping. 
    
    Does NOT participate in gradient updates, but holds dynamic Rubric knowledge base
    and is responsible for computing rewards. 
    
    Scoring is done separately for Q-Agent and R-Agent:
    - Q-Agent: Input full trajectory, score each round's Q-Agent action at once
    - R-Agent: Input per-round data, score each round separately
    
    Note: A-Agent is NOT scored by RM-Agent (only gets binary 0/1 reward)
    """

    def __init__(
            self,
            rollout_idx: int | None = None,
            q_agent_rubrics: str = None,
            r_agent_rubrics: str = None,
            llm_api_url: str = None,
            llm_api_key: str = None,
            llm_model: str = None,
            **kwargs
    ):
        """
        Initialize the RM-Agent.

        Args:
            rollout_idx: Rollout index
            q_agent_rubrics: Custom rubrics for Q-Agent evaluation
            r_agent_rubrics: Custom rubrics for R-Agent evaluation
            llm_api_url: API URL for closed-source LLM
            llm_api_key: API key for closed-source LLM
            llm_model: Model name for closed-source LLM
        """
        super().__init__()
        self.rollout_idx = rollout_idx

        # Rubrics
        self.q_agent_rubrics = q_agent_rubrics or DEFAULT_Q_AGENT_RUBRICS
        self.r_agent_rubrics = r_agent_rubrics or DEFAULT_R_AGENT_RUBRICS

        # LLM client for closed-source models
        self.llm_client = LLMClient(
            api_url=llm_api_url,
            api_key=llm_api_key,
            model=llm_model,
        )

        # Stored scores
        self.q_agent_scores: Dict[str, Dict] = {}  # round_1, round_2, ...
        self.r_agent_scores: List[Dict] = []  # Per-round, per-document scores

        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        RM-Agent doesn't need regular environment updates during rollout.
        It only acts after the episode is complete.
        """
        self.env_data = env_data

    def update_from_model(self, response: str):
        """
        Parse model response for scoring.
        """
        self.current_action = response
        return self.current_action

    def _format_complete_trajectory(self, trajectory_data: Dict) -> str:
        """
        Format complete trajectory for Q-Agent scoring.

        Args:
            trajectory_data: Trajectory data from env. get_trajectory_for_rm_q_agent()

        Returns:
            Formatted trajectory string
        """
        parts = []

        for round_data in trajectory_data.get("trajectory", []):
            round_idx = round_data["round_idx"]
            parts.append(f"=== Round {round_idx + 1} ===")

            # Q-Agent output
            parts.append(f"\n## Q-Agent Think:")
            parts.append(round_data.get("q_agent_think", ""))

            parts.append(f"\n## Q-Agent Tool Call:")
            parts.append(json.dumps(round_data.get("q_agent_tool_call", {}), indent=2))

            parts.append(f"\n## Sub-queries:")
            for i, sq in enumerate(round_data.get("sub_queries", []), 1):
                parts.append(f"  {i}. Query: {sq.get('query', '')}")
                parts.append(f"     Goal: {sq.get('goal', '')}")

            # R-Agent output
            parts.append(f"\n## R-Agent Think:")
            parts.append(round_data.get("r_agent_think", ""))

            parts.append(f"\n## Selected Documents:")
            for i, doc in enumerate(round_data.get("selected_docs", []), 1):
                parts.append(f"  {i}. Title: {doc.get('title', '')}")
                parts.append(f"     URL: {doc.get('url', '')}")
                content = doc.get('content', '')[:300] + "..." if len(doc.get('content', '')) > 300 else doc.get(
                    'content', '')
                parts.append(f"     Content: {content}")

            parts.append("")

        # Final answer info
        parts.append(f"=== Final Result ===")
        parts.append(f"Answer: {trajectory_data.get('final_answer', '')}")
        parts.append(f"Correct: {trajectory_data.get('is_correct', False)}")

        return "\n".join(parts)

    def _format_historical_context(self, all_round_data: List[Dict], current_round_idx: int) -> str:
        """
        Format historical context for R-Agent scoring.

        Args:
            all_round_data: All round data from environment
            current_round_idx: Current round being evaluated

        Returns:
            Formatted historical context string
        """
        if current_round_idx == 0:
            return "No historical context (this is the first round)."

        parts = []

        for round_data in all_round_data[:current_round_idx]:
            round_idx = round_data.get("round_idx", 0)
            parts.append(f"=== Round {round_idx + 1} ===")

            parts.append(f"\n## Historical Think:")
            parts.append(round_data.get("q_agent_think", ""))

            parts.append(f"\n## Historical Documents:")
            for i, doc in enumerate(round_data.get("selected_docs", []), 1):
                doc_id = f"round{round_idx + 1}_doc{i}"
                parts.append(f"\n### Document ID: {doc_id}")
                parts.append(f"Plan: {doc.get('source_query', 'N/A')}")
                parts.append(f"Title: {doc.get('title', '')}")
                parts.append(f"URL: {doc.get('url', '')}")
                parts.append(f"Snippet: {doc.get('snippet', '')}")
                content = doc.get('content', '')[:500] + "..." if len(doc.get('content', '')) > 500 else doc.get(
                    'content', '')
                parts.append(f"Content: {content}")

            parts.append("")

        return "\n".join(parts) if parts else "No historical context available."

    def _format_current_documents(self, round_data: Dict) -> str:
        """
        Format current round documents for R-Agent scoring.

        Args:
            round_data: Current round data

        Returns:
            Formatted documents string
        """
        parts = []
        all_docs = round_data.get("all_docs", [])

        for i, doc in enumerate(all_docs, 1):
            doc_id = f"doc_{i}"
            parts.append(f"=== Document ID: {doc_id} ===")
            parts.append(f"Plan (Sub-query): {doc.get('source_query', doc.get('query', 'N/A'))}")
            parts.append(f"Goal: {doc.get('source_goal', doc.get('goal', 'N/A'))}")
            parts.append(f"URL: {doc.get('url', '')}")
            parts.append(f"Title: {doc.get('title', '')}")
            parts.append(f"Snippet: {doc.get('snippet', '')}")
            parts.append(f"Content: {doc.get('content', '')}")
            parts.append("")

        return "\n".join(parts) if parts else "No documents available."

    def prepare_q_agent_scoring_prompt(self, env_data: Env) -> str:
        """
        Prepare prompt for Q-Agent scoring using full trajectory.

        Args:
            env_data: Environment data with complete trajectory

        Returns:
            Formatted prompt string
        """
        trajectory_data = env_data.get_trajectory_for_rm_q_agent()

        user_query = trajectory_data.get("question", "")
        complete_trajectory = self._format_complete_trajectory(trajectory_data)

        prompt = RM_Q_AGENT_SYSTEM_PROMPT.format(
            user_query=user_query,
            complete_trajectory=complete_trajectory,
            rubrics=self.q_agent_rubrics,
        )

        return prompt

    def prepare_r_agent_scoring_prompt(self, env_data: Env, round_idx: int) -> Optional[str]:
        """
        Prepare prompt for R-Agent scoring for a specific round.

        Args:
            env_data: Environment data
            round_idx: Round index to score

        Returns:
            Formatted prompt string, or None if round doesn't exist
        """
        all_round_data = env_data.get_all_round_data_for_rm_r_agent()

        if round_idx >= len(all_round_data):
            return None

        round_data = all_round_data[round_idx]

        current_think = round_data.get("r_agent_think", "")
        historical_context = self._format_historical_context(all_round_data, round_idx)
        documents = self._format_current_documents(round_data)

        prompt = RM_R_AGENT_SYSTEM_PROMPT.format(
            current_think=current_think,
            historical_context=historical_context,
            documents=documents,
            rubrics=self.r_agent_rubrics,
        )

        return prompt

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dict, or None if parsing fails
        """
        import re

        # Try to find JSON in code blocks
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and start < end:
                json_str = response[start:end + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        logger.warning(f"Failed to extract JSON from response")
        return None

    def parse_q_agent_scores(self, response: str) -> Dict[str, Dict]:
        """
        Parse Q-Agent scoring response.

        Args:
            response: Raw LLM response containing evaluation

        Returns:
            Dict with round-based scores: {"round_1": {... }, "round_2": {...}}
        """
        evaluation = self._extract_json_from_response(response)

        if not evaluation:
            return {}

        # Calculate normalized scores for each round
        for round_key, round_scores in evaluation.items():
            if isinstance(round_scores, dict):
                total_score = 0
                num_criteria = 0
                for criterion_key, criterion_data in round_scores.items():
                    if isinstance(criterion_data, dict) and "score" in criterion_data:
                        total_score += criterion_data.get("score", 0)
                        num_criteria += 1

                if num_criteria > 0:
                    evaluation[round_key]["_total_score"] = total_score
                    evaluation[round_key]["_normalized_score"] = total_score / (num_criteria * 10)
                    evaluation[round_key]["_num_criteria"] = num_criteria

        return evaluation

    def parse_r_agent_scores(self, response: str) -> Dict[str, Dict]:
        """
        Parse R-Agent scoring response for a single round.

        Args:
            response: Raw LLM response containing evaluation

        Returns:
            Dict with document-based scores: {"doc_1": {...}, "doc_2": {...}}
        """
        evaluation = self._extract_json_from_response(response)

        if not evaluation:
            return {}

        # Calculate normalized scores for each document
        for doc_key, doc_scores in evaluation.items():
            if isinstance(doc_scores, dict):
                present_count = 0
                total_criteria = 0
                for criterion_key, criterion_data in doc_scores.items():
                    if isinstance(criterion_data, dict) and "judgment" in criterion_data:
                        total_criteria += 1
                        if criterion_data.get("judgment") == "PRESENT":
                            present_count += 1

                if total_criteria > 0:
                    evaluation[doc_key]["_present_ratio"] = present_count / total_criteria
                    evaluation[doc_key]["_total_criteria"] = total_criteria

        return evaluation

    async def _call_llm(
            self,
            prompt: str,
            env_worker: Any = None,
            use_api: bool = None,
    ) -> Optional[str]:
        """
        Call LLM (either via env_worker or API client).

        Args:
            prompt: The prompt to send
            env_worker: Environment worker with call_llm method
            use_api: Force use of API client (if None, tries env_worker first)

        Returns:
            LLM response or None
        """
        # Try env_worker first if available and not forced to use API
        if use_api is not True and env_worker and hasattr(env_worker, 'call_llm'):
            try:
                response = await env_worker.call_llm(prompt)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"env_worker. call_llm failed: {e}")

        # Fall back to API client
        try:
            response = await self.llm_client.call(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None

    async def score_q_agent(
            self,
            env_data: Env,
            env_worker: Any = None,
            use_api: bool = None,
    ) -> Dict[str, Dict]:
        """
        Score all Q-Agent actions using full trajectory.

        Args:
            env_data: Environment data with complete trajectory
            env_worker: Environment worker for LLM calls
            use_api: Force use of API client

        Returns:
            Dict with round-based scores
        """
        prompt = self.prepare_q_agent_scoring_prompt(env_data)

        response = await self._call_llm(prompt, env_worker, use_api)

        if response:
            self.q_agent_scores = self.parse_q_agent_scores(response)
            logger.info(f"Q-Agent scoring completed: {len(self.q_agent_scores)} rounds evaluated")
            return self.q_agent_scores

        logger.warning("Could not score Q-Agent: no valid response")
        return {}

    async def score_r_agent(
            self,
            env_data: Env,
            env_worker: Any = None,
            use_api: bool = None,
    ) -> List[Dict]:
        """
        Score all R-Agent actions (per-round, per-document scoring).

        Args:
            env_data: Environment data
            env_worker: Environment worker for LLM calls
            use_api: Force use of API client

        Returns:
            List of per-round score dictionaries
        """
        all_round_data = env_data.get_all_round_data_for_rm_r_agent()
        scores = []

        for round_idx in range(len(all_round_data)):
            prompt = self.prepare_r_agent_scoring_prompt(env_data, round_idx)

            if prompt is None:
                continue

            response = await self._call_llm(prompt, env_worker, use_api)

            if response:
                round_score = self.parse_r_agent_scores(response)
                round_score["_round_idx"] = round_idx
                scores.append(round_score)
            else:
                logger.warning(f"Could not score R-Agent for round {round_idx}")
                scores.append({"_round_idx": round_idx, "_error": "No response"})

        self.r_agent_scores = scores
        logger.info(f"R-Agent scoring completed: {len(scores)} rounds evaluated")
        return scores

    async def compute_all_rewards(
            self,
            env_data: Env,
            env_worker: Any = None,
            use_api: bool = None,
    ) -> Dict[str, Any]:
        """
        Compute all rewards for the episode.

        Scores Q-Agent and R-Agent separately, then sets scores in environment.
        A-Agent reward is already computed (binary 0/1).

        Args:
            env_data: Environment data with complete trajectory
            env_worker: Environment worker for LLM calls
            use_api: Force use of API client

        Returns:
            Dict containing all rewards
        """
        # Score Q-Agent (single call with full trajectory)
        q_scores = await self.score_q_agent(env_data, env_worker, use_api)

        # Score R-Agent (per-round calls)
        r_scores = await self.score_r_agent(env_data, env_worker, use_api)

        # Convert scores to list format for environment
        q_scores_list = []
        for round_key in sorted(q_scores.keys()):
            if round_key.startswith("round_"):
                q_scores_list.append(q_scores[round_key])

        # Set scores in environment
        env_data.set_rm_scores(q_scores_list, r_scores)

        # Get A-Agent reward (already computed)
        a_reward = env_data.get_a_agent_reward()

        return {
            "q_agent_scores": q_scores,
            "r_agent_scores": r_scores,
            "a_agent_reward": a_reward,
        }

    def get_q_agent_normalized_scores(self) -> List[float]:
        """
        Get normalized scores for each Q-Agent round.

        Returns:
            List of normalized scores (0. 0-1.0) per round
        """
        scores = []
        for round_key in sorted(self.q_agent_scores.keys()):
            if round_key.startswith("round_"):
                round_data = self.q_agent_scores[round_key]
                if isinstance(round_data, dict):
                    normalized = round_data.get("_normalized_score", 0.0)
                    scores.append(normalized)
        return scores

    def get_r_agent_normalized_scores(self) -> List[Dict[str, float]]:
        """
        Get normalized scores for each R-Agent round (per-document).

        Returns:
            List of dicts with document scores per round
        """
        scores = []
        for round_score in self.r_agent_scores:
            round_doc_scores = {}
            for doc_key, doc_data in round_score.items():
                if doc_key.startswith("doc_") and isinstance(doc_data, dict):
                    round_doc_scores[doc_key] = doc_data.get("_present_ratio", 0.0)
            scores.append(round_doc_scores)
        return scores

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        RM-Agent step is typically called after episode completion to compute rewards.

        Args:
            env_data: Environment data
            env_worker: Optional environment worker
        """
        await self.compute_all_rewards(env_data, env_worker)

    def reset(self):
        """Reset the agent's internal state for a new episode."""
        super().reset()
        self.q_agent_scores = {}
        self.r_agent_scores = []