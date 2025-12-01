import logging
import re
import json
import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set, TYPE_CHECKING
import requests

if TYPE_CHECKING:
    from pettingllms.multi_agent_env.deep_search.deep_search_env import (
        RoundTrajectory, SubQueryInfo, SelectedDocument
    )

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
SERPER_API_URL = "https://google. serper.dev/search"
JINA_READER_URL = "https://r.jina.ai"
MAX_CONTENT_LENGTH = 150000


def truncatefn(s: str, length: int = 300) -> str:
    """Truncate text to specified length while preserving context."""
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= length:
        return s
    return s[:length // 2] + ".. .(truncated)..." + s[-length // 2:]


# ============================================================
# JinaReader Prompts for Goal-Directed Extraction
# ============================================================

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs. 
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""

SNIPPET_EXTRACTOR_PROMPT = """The full webpage content is unavailable. Please analyze the search snippet and extract information relevant to the goal. 

## **Search Snippet**
{snippet}

## **User Goal**
{goal}

## **Task**
1. Extract the most relevant information from the snippet that addresses the goal
2. Provide a focused summary based on available information
3. Be honest about information limitations

**Output JSON format with "evidence" and "summary" fields**
"""


# ============================================================
# Web Search Functions (Serper API + Jina Reader)
# ============================================================

async def perform_web_search_with_summary(
        query: str,
        goal: str,
        max_results: int = 5,
        summary_model: str = "gpt-3.5-turbo",
        env_worker: Any = None,
        visited_urls: Set[str] = None,
        url_content_cache: Dict[str, str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform web search and summarize results with goal-directed summarization.

    Returns:
        Tuple of (raw_search_results, summarized_docs)
    """
    if visited_urls is None:
        visited_urls = set()
    if url_content_cache is None:
        url_content_cache = {}

    # Step 1: Perform Google search via Serper API
    raw_results = await _serper_search(query, max_results)

    if not raw_results:
        logger.warning(f"No search results for query: {query}")
        return [], []

    # Step 2: Fetch and summarize each result
    summarized_docs = []
    for result in raw_results:
        url = result.get("url", "")
        if not url:
            continue

        # Check if URL already crawled
        if url in visited_urls and url in url_content_cache:
            logger.info(f"Re-summarizing cached content for URL: {url} with new goal")
            raw_content = url_content_cache.get(url, "")
            if raw_content:
                summary_result = await _generate_summary_with_goal(
                    content=raw_content,
                    goal=goal,
                    model=summary_model,
                    env_worker=env_worker,
                )
            else:
                summary_result = await _generate_summary_from_snippet(
                    snippet=result.get("snippet", ""),
                    goal=goal,
                    model=summary_model,
                    env_worker=env_worker,
                )
        else:
            # First time seeing this URL - crawl and summarize
            raw_content = await _jina_reader_fetch(url)

            if raw_content:
                url_content_cache[url] = raw_content
                visited_urls.add(url)

                summary_result = await _generate_summary_with_goal(
                    content=raw_content,
                    goal=goal,
                    model=summary_model,
                    env_worker=env_worker,
                )
            else:
                logger.warning(f"Jina Reader failed for {url}, using snippet")
                summary_result = await _generate_summary_from_snippet(
                    snippet=result.get("snippet", ""),
                    goal=goal,
                    model=summary_model,
                    env_worker=env_worker,
                )

        # Format content with evidence and summary
        formatted_content = f"""Evidence from page:
{summary_result.get('evidence', '')}

Summary:
{summary_result.get('summary', '')}"""

        summarized_docs.append({
            "title": result.get("title", ""),
            "url": url,
            "snippet": result.get("snippet", ""),
            "content": formatted_content,
            "evidence": summary_result.get("evidence", ""),
            "summary": summary_result.get("summary", ""),
            "rational": summary_result.get("rational", ""),
        })

    return raw_results, summarized_docs


async def _serper_search(query: str, max_results: int = 5) -> List[Dict]:
    """Perform Google search using Serper API."""
    api_key = os.getenv("SERPER_API_KEY", "")

    if not api_key:
        logger.warning("SERPER_API_KEY not set, using mock search results")
        return _generate_mock_search_results(query, max_results)

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
        "num": max_results,
    }

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(SERPER_API_URL, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
        )

        if response.status_code != 200:
            logger.error(f"Serper API error: {response.status_code}")
            return _generate_mock_search_results(query, max_results)

        data = response.json()

        results = []
        for item in data.get("organic", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        logger.info(f"Serper search returned {len(results)} results for: {query}")
        return results

    except Exception as e:
        logger.error(f"Serper search failed: {e}")
        return _generate_mock_search_results(query, max_results)


async def _jina_reader_fetch(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch full page content using Jina Reader."""
    jina_api_key = os.getenv("JINA_API_KEY", "")

    reader_url = f"{JINA_READER_URL}/{url}"
    headers = {}

    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(reader_url, headers=headers, timeout=timeout)
        )

        if response.status_code == 200:
            content = response.text
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH]
            return content
        else:
            logger.warning(f"Jina Reader failed for {url}: {response.status_code}")
            return None

    except Exception as e:
        logger.warning(f"Jina Reader exception for {url}: {e}")
        return None


async def _generate_summary_with_goal(
        content: str,
        goal: str,
        model: str = "gpt-3.5-turbo",
        env_worker: Any = None,
        max_content_length: int = 100000,
) -> Dict[str, str]:
    """Generate summary using LLM with goal-directed extraction."""
    if len(content) > max_content_length:
        content = content[:max_content_length]

    prompt = EXTRACTOR_PROMPT.format(
        webpage_content=content,
        goal=goal
    )

    for attempt in range(MAX_RETRIES):
        try:
            if env_worker and hasattr(env_worker, 'call_llm'):
                raw_response = await env_worker.call_llm(prompt, model=model)
            else:
                return _simple_extract(content, goal)

            if not raw_response:
                truncate_length = int(0.7 * len(content))
                content = content[:truncate_length]
                prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
                continue

            result = _parse_extractor_response(raw_response)
            if result:
                return result

        except Exception as e:
            logger.warning(f"Summary attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                return {
                    "evidence": "Failed to generate summary after multiple attempts.",
                    "summary": "The webpage content could not be processed properly.",
                    "rational": ""
                }

    return {
        "evidence": "Failed to generate summary.",
        "summary": "Summary generation failed.",
        "rational": ""
    }


async def _generate_summary_from_snippet(
        snippet: str,
        goal: str,
        model: str = "gpt-3. 5-turbo",
        env_worker: Any = None,
) -> Dict[str, str]:
    """Generate summary from snippet when full content is unavailable."""
    prompt = SNIPPET_EXTRACTOR_PROMPT.format(snippet=snippet, goal=goal)

    try:
        if env_worker and hasattr(env_worker, 'call_llm'):
            raw_response = await env_worker.call_llm(prompt, model=model)
            if raw_response:
                result = _parse_extractor_response(raw_response)
                if result:
                    return result
    except Exception as e:
        logger.warning(f"Snippet summary failed: {e}")

    return {
        "evidence": snippet,
        "summary": snippet,
        "rational": ""
    }


def _parse_extractor_response(raw_response: str) -> Optional[Dict[str, str]]:
    """Parse JSON response from extractor prompt."""
    try:
        result = json.loads(raw_response)
        return {
            "evidence": result.get("evidence", ""),
            "summary": result.get("summary", ""),
            "rational": result.get("rational", ""),
        }
    except json.JSONDecodeError:
        left = raw_response.find('{')
        right = raw_response.rfind('}')
        if left != -1 and right != -1 and left <= right:
            json_str = raw_response[left:right + 1]
            try:
                result = json.loads(json_str)
                return {
                    "evidence": result.get("evidence", ""),
                    "summary": result.get("summary", ""),
                    "rational": result.get("rational", ""),
                }
            except:
                pass

        return _manual_extract_fields(raw_response)


def _manual_extract_fields(text: str) -> Dict[str, str]:
    """Manually extract fields from malformed JSON."""
    result = {}
    fields = ["evidence", "summary", "rational"]

    for field in fields:
        patterns = [
            rf'"{field}"\s*:\s*"((? :[^"\\]|\\[\\"/bfnrt]|\\u[0-9a-fA-F]{{4}})*)"',
            rf'"{field}"\s*:\s*"([^"]*(? :\\"[^"]*)*)"',
        ]

        for pattern in patterns:
            try:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    value = match.group(1)
                    value = value.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                    result[field] = value
                    break
            except:
                continue

        if field not in result:
            result[field] = ""

    return result


def _simple_extract(content: str, goal: str, max_length: int = 500) -> Dict[str, str]:
    """Simple keyword-based extraction as fallback."""
    keywords = set(goal.lower().split())
    keywords -= {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "for", "on", "with"}

    sentences = re.split(r'[.!?]+', content)
    relevant_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(kw in sentence_lower for kw in keywords):
            relevant_sentences.append(sentence.strip())
            if len(' '.join(relevant_sentences)) > max_length:
                break

    extracted = ' '.join(relevant_sentences) if relevant_sentences else content[:max_length]

    return {
        "evidence": extracted,
        "summary": extracted,
        "rational": ""
    }


def _generate_mock_search_results(query: str, max_results: int = 5) -> List[Dict]:
    """Generate mock search results for testing."""
    mock_results = []
    for i in range(min(max_results, 3)):
        mock_results.append({
            "title": f"Search result {i + 1} for: {query}",
            "url": f"https://example.com/result{i + 1}? q={query.replace(' ', '+')}",
            "snippet": f"This is a mock search result about {query}. Contains relevant information.",
        })
    return mock_results


# ============================================================
# Response Parsing Functions
# ============================================================

def extract_q_agent_response(response: str) -> Dict[str, Any]:
    """
    Extract Q-Agent response from LLM output.

    Expected format:
    <response>
    <think>... </think>
    <tool_call>
    {"name": "search", "arguments": {"queries": [{"query": ".. .", "goal": "... "}]}}
    OR
    {"name": "generate_answer", "arguments": {}}
    </tool_call>
    </response>
    """
    result = {
        "think": "",
        "action_type": "search",
        "queries": [],
        "tool_call": {},
    }

    # Extract thinking
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    # Extract tool_call JSON
    tool_call_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL)

    if tool_call_match:
        tool_call_text = tool_call_match.group(1).strip()

        json_start = tool_call_text.find('{')
        if json_start != -1:
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(tool_call_text)):
                if tool_call_text[i] == '{':
                    brace_count += 1
                elif tool_call_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end != -1:
                try:
                    tool_call_json = tool_call_text[json_start:json_end]
                    tool_call = json.loads(tool_call_json)
                    result["tool_call"] = tool_call

                    tool_name = tool_call.get("name", "")
                    arguments = tool_call.get("arguments", {})

                    if tool_name == "search":
                        result["action_type"] = "search"
                        result["queries"] = arguments.get("queries", [])
                    elif tool_name == "generate_answer":
                        result["action_type"] = "generate_answer"

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse Q-Agent tool_call JSON: {e}")

    # Fallback check
    if "generate_answer" in response.lower() and not result["queries"]:
        result["action_type"] = "generate_answer"

    return result


def extract_r_agent_response(response: str, all_docs: List[Dict] = None) -> Dict[str, Any]:
    """
    Extract R-Agent response from LLM output.

    Expected format:
    <response>
    <think>...</think>
    <tool_call>
    {"name": "select_documents", "arguments": {"selected_urls": ["url1", "url2", ...]}}
    </tool_call>
    </response>
    """
    result = {
        "think": "",
        "selected_urls": [],
        "selected_indices": [],
        "tool_call": {},
    }

    if all_docs is None:
        all_docs = []

    url_to_idx = {doc.get("url", ""): idx for idx, doc in enumerate(all_docs)}

    # Extract thinking
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    # Extract tool_call JSON
    tool_call_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL)

    if tool_call_match:
        tool_call_text = tool_call_match.group(1).strip()

        json_start = tool_call_text.find('{')
        if json_start != -1:
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(tool_call_text)):
                if tool_call_text[i] == '{':
                    brace_count += 1
                elif tool_call_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end != -1:
                try:
                    tool_call_json = tool_call_text[json_start:json_end]
                    tool_call = json.loads(tool_call_json)
                    result["tool_call"] = tool_call

                    arguments = tool_call.get("arguments", {})
                    selected_urls = arguments.get("selected_urls", [])
                    result["selected_urls"] = selected_urls

                    for url in selected_urls:
                        if url in url_to_idx:
                            result["selected_indices"].append(url_to_idx[url])

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse R-Agent tool_call JSON: {e}")

    # Fallback: try to extract document numbers
    if not result["selected_urls"]:
        doc_pattern = r"Document\s*(\d+)"
        matches = re.findall(doc_pattern, response, re.IGNORECASE)
        if matches:
            result["selected_indices"] = [int(m) - 1 for m in matches if 1 <= int(m) <= len(all_docs)]

    return result


def extract_a_agent_response(response: str) -> Dict[str, str]:
    """
    Extract A-Agent response from LLM output.

    Expected format:
    <response>
    <think>... </think>
    <answer>... </answer>
    </response>
    """
    result = {
        "think": "",
        "answer": "",
    }

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    else:
        result["answer"] = response.strip()

    return result


# ============================================================
# Context Formatting Functions
# ============================================================

def format_history_for_q_agent(trajectory: List["RoundTrajectory"]) -> str:
    """
    Format history trajectory for Q-Agent input.
    
    Q-Agent sees: previous rounds' Q-response, R-response, and selected docs. 
    """
    if not trajectory:
        return "# No information gathered yet.  This is the first round."
    
    history = "# Information Gathered So Far"
    
    for round_traj in trajectory:
        round_num = round_traj.round_idx + 1
        
        # Q-Agent action for this round
        history += f"\n\n## Round {round_num}: Search"
        
        # Format queries with goals
        query_strings = []
        for sq in round_traj.sub_queries:
            query_strings.append(f"{sq.query} (Goal: {sq.goal})")
        history += f"\nQueries: {', '.join(query_strings)}"
        
        # Selected documents with content
        if round_traj. selected_docs:
            history += f"\n\n## Selected Documents from Round {round_num}"
            for doc in round_traj.selected_docs:
                history += f"\n\n### [{doc.title}]({doc.url})"
                if doc.content:
                    history += f"\n**Full Content**:\n{doc. content}"
                else:
                    history += f"\n**Snippet**: {doc.snippet}"
    
    return history


def format_context_for_r_agent(
    question: str,
    q_agent_think: str,
    q_agent_tool_call: Dict,
    sub_queries: List["SubQueryInfo"],
    all_docs: List[Dict],
) -> str:
    """
    Format context for R-Agent input.
    
    R-Agent sees: question + current Q-response + all retrieved docs.
    """
    parts = [f"# User Question\n{question}\n"]
    
    # Q-Agent's analysis for this round
    parts.append("\n# Q-Agent's Analysis for This Round")
    if q_agent_think:
        parts. append(f"**Thinking**: {q_agent_think}")
    
    # Queries with goals
    query_strings = []
    for sq in sub_queries:
        query_strings.append(f"{sq.query} (Goal: {sq.goal})")
    parts.append(f"**Search Queries**: {', '.join(query_strings)}")
    
    # All candidate documents
    parts.append("\n# Current Candidate Documents to Review")
    parts.append(format_docs_for_r_agent(all_docs))
    
    parts.append("\n# Your Task")
    parts.append("Select the most relevant and useful documents from the candidates above.")
    
    return "\n". join(parts)


def format_docs_for_r_agent(docs: List[Dict], max_docs: int = 20) -> str:
    """Format documents for R-Agent display."""
    if not docs:
        return "No documents available."
    
    formatted = ""
    
    for i, doc in enumerate(docs[:max_docs], 1):
        formatted += f"\n## Document {i}\n"
        formatted += f"**Title**: {doc.get('title', 'N/A')}\n"
        formatted += f"**URL**: {doc.get('url', 'N/A')}\n"
        formatted += f"**Snippet**: {doc. get('snippet', 'N/A')}\n"
        if doc.get('content'):
            formatted += f"**Full Content**:\n{doc['content']}\n"
    
    if len(docs) > max_docs:
        formatted += f"\n... and {len(docs) - max_docs} more documents\n"
    
    return formatted


def format_context_for_a_agent(
    question: str,
    trajectory: List["RoundTrajectory"],
) -> str:
    """
    Format context for A-Agent input.
    
    A-Agent sees: question + complete history trajectory.
    """
    parts = [f"# User Question\n{question}\n"]
    
    parts.append("\n# Available Information from Selected Documents\n")
    
    doc_idx = 1
    for round_traj in trajectory:
        for doc in round_traj.selected_docs:
            parts.append(f"\n## Document {doc_idx}: {doc.title}")
            parts.append(f"**Source**: {doc.url}")
            if doc.content:
                parts.append(f"**Content**:\n{doc.content}")
            else:
                parts.append(f"**Content**:\n{doc.snippet}")
            doc_idx += 1
    
    parts.append("\n# Your Task")
    parts.append("Generate a comprehensive answer to the user's question based on the documents above.")
    
    return "\n". join(parts)


# ============================================================
# RM-Agent Formatting Functions (Q-Agent and R-Agent only)
# ============================================================

def format_trajectory_for_rm_q_scoring(
    question: str,
    ground_truth: str,
    trajectory: List[Dict],
    final_answer: str,
    is_correct: bool,
) -> str:
    """
    Format complete trajectory for RM-Agent Q-Agent scoring.
    
    RM-Agent scores all Q-Agent actions at once using the full trajectory.
    Input: initial question + full trajectory (Q-response, R-response, selected docs per round)
    
    Note: This only scores Q-Agent actions, not A-Agent. 
    """
    parts = [f"# Original Question\n{question}\n"]
    parts.append(f"# Ground Truth Answer\n{ground_truth}\n")
    
    parts.append("\n# Complete Trajectory\n")
    
    for round_data in trajectory:
        round_num = round_data["round_idx"] + 1
        
        parts.append(f"\n## Round {round_num}: Q-Agent")
        parts.append(f"<think>{round_data['q_agent_think']}</think>")
        parts.append(f"<tool_call>")
        parts.append(json.dumps(round_data['q_agent_tool_call'], indent=2))
        parts.append(f"</tool_call>")
        
        parts.append(f"\n## Round {round_num}: R-Agent")
        parts.append(f"<think>{round_data['r_agent_think']}</think>")
        parts.append(f"<tool_call>")
        parts.append(json.dumps(round_data['r_agent_tool_call'], indent=2))
        parts.append(f"</tool_call>")
        
        if round_data. get("selected_docs"):
            parts.append(f"\n### Selected Documents:")
            for i, doc in enumerate(round_data["selected_docs"], 1):
                parts. append(f"\n**Document {i}**: {doc.get('title', 'N/A')}")
                parts.append(f"URL: {doc.get('url', 'N/A')}")
                content = doc.get('content', '')
                if len(content) > 500:
                    content = content[:500] + "..."
                parts.append(f"Content: {content}")
    
    parts. append(f"\n# Final Answer\n{final_answer}")
    parts.append(f"\n# Answer Correct: {is_correct}")
    
    parts.append("\n# Your Task")
    parts.append("Evaluate each round's Q-Agent action according to the rubrics and provide detailed scores for each round.")
    
    return "\n".join(parts)


def format_round_for_rm_r_scoring(round_data: Dict) -> str:
    """
    Format single round data for RM-Agent R-Agent scoring. 
    
    Each round is scored independently.
    Input: initial question + Q-response + all_docs + R-response + selected_docs
    """
    question = round_data["question"]
    ground_truth = round_data. get("ground_truth", "")
    round_idx = round_data["round_idx"]
    
    parts = [f"# Original Question\n{question}\n"]
    if ground_truth:
        parts.append(f"# Ground Truth Answer\n{ground_truth}\n")
    
    parts.append(f"\n# Round {round_idx + 1} Data\n")
    
    # Q-Agent response
    parts.append(f"## Q-Agent Response")
    parts.append(f"<think>{round_data['q_agent_think']}</think>")
    parts.append(f"<tool_call>")
    parts.append(json.dumps(round_data['q_agent_tool_call'], indent=2))
    parts.append(f"</tool_call>")
    
    # Sub-queries
    parts.append(f"\n### Search Queries:")
    for sq in round_data.get("sub_queries", []):
        parts.append(f"- Query: {sq['query']}")
        parts.append(f"  Goal: {sq['goal']}")
    
    # All retrieved documents (for R-Agent evaluation)
    parts. append(f"\n## All Retrieved Documents ({len(round_data.get('all_docs', []))} total)")
    for i, doc in enumerate(round_data. get("all_docs", []), 1):
        parts.append(f"\n### Document {i}")
        parts.append(f"**Title**: {doc.get('title', 'N/A')}")
        parts.append(f"**URL**: {doc.get('url', 'N/A')}")
        parts.append(f"**Snippet**: {doc. get('snippet', 'N/A')}")
        content = doc.get('content', '')
        if len(content) > 500:
            content = content[:500] + "..."
        parts.append(f"**Full Content**:\n{content}")
    
    # R-Agent response
    parts.append(f"\n## R-Agent Response")
    parts.append(f"<think>{round_data['r_agent_think']}</think>")
    parts.append(f"<tool_call>")
    parts.append(json.dumps(round_data['r_agent_tool_call'], indent=2))
    parts.append(f"</tool_call>")
    
    # Selected documents
    parts. append(f"\n## Selected Documents ({len(round_data.get('selected_docs', []))} selected)")