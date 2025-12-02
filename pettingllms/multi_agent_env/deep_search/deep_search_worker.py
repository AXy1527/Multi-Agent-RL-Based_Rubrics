"""
Worker classes and utilities for Deep Search operations. 

This module contains the worker implementations for web search,
page crawling, and LLM summarization in isolated Ray environments.
"""

import os
import sys
import asyncio
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import ray
except ImportError:
    ray = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import requests
except ImportError:
    requests = None

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
SERPER_API_URL = "https://google.serper.dev/search"
JINA_READER_URL = "https://r.jina.ai"
MAX_CONTENT_LENGTH = 150000


async def _await_ray_object_ref(obj_ref, timeout_seconds: float = 60.0):
    """
    Await a Ray object reference with timeout.

    Args:
        obj_ref: Ray object reference to await
        timeout_seconds: Maximum time to wait

    Returns:
        Result from Ray task

    Raises:
        asyncio. TimeoutError: If task exceeds timeout
    """
    start_time = time.time()
    while True:
        ready, _ = ray.wait([obj_ref], timeout=0.1)
        if ready:
            return ray.get(obj_ref)

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise asyncio.TimeoutError(f"Ray task timed out after {timeout_seconds}s")

        await asyncio.sleep(0.01)


async def _serper_search_async(
        query: str,
        max_results: int = 5,
        timeout: float = 30.0
) -> List[Dict]:
    """
    Perform Google search using Serper API. 

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        timeout: Request timeout in seconds

    Returns:
        List of search result dictionaries with title, url, snippet
    """
    api_key = os.getenv("SERPER_API_KEY", "")

    if not api_key:
        print("Warning: SERPER_API_KEY not set, returning empty results")
        return []

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
        "num": max_results,
    }

    try:
        if aiohttp:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        SERPER_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status != 200:
                        print(f"Serper API error: {response.status}")
                        return []

                    data = await response.json()
        else:
            # Fallback to synchronous requests
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    SERPER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
            )
            if response.status_code != 200:
                print(f"Serper API error: {response.status_code}")
                return []
            data = response.json()

        results = []
        for item in data.get("organic", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        return results

    except Exception as e:
        print(f"Serper search failed: {e}")
        return []


async def _jina_reader_fetch_async(
        url: str,
        timeout: float = 30.0
) -> Optional[str]:
    """
    Fetch full page content using Jina Reader. 

    Args:
        url: URL to fetch content from
        timeout: Request timeout in seconds

    Returns:
        Page content as string, or None if fetch fails
    """
    jina_api_key = os.getenv("JINA_API_KEY", "")

    reader_url = f"{JINA_READER_URL}/{url}"
    headers = {}

    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"

    try:
        if aiohttp:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        reader_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        if len(content) > MAX_CONTENT_LENGTH:
                            content = content[:MAX_CONTENT_LENGTH]
                        return content
                    else:
                        print(f"Jina Reader failed for {url}: {response.status}")
                        return None
        else:
            # Fallback to synchronous requests
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
                print(f"Jina Reader failed for {url}: {response.status_code}")
                return None

    except Exception as e:
        print(f"Jina Reader exception for {url}: {e}")
        return None


async def _call_llm_async(
        prompt: str,
        model: str = "gpt-3.5-turbo",
        timeout: float = 60.0,
        temperature: float = 0.3,
        max_tokens: int = 2000,
) -> Optional[str]:
    """
    Call LLM API for summarization.

    Args:
        prompt: The prompt to send
        model: Model name
        timeout: Request timeout in seconds
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated response text, or None if failed
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")

    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        if aiohttp:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        print(f"LLM API error: {response.status}")
                        return None
        else:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"LLM API error: {response.status_code}")
                return None

    except Exception as e:
        print(f"LLM API call failed: {e}")
        return None


async def _perform_web_search_with_summary(
        query: str,
        goal: str,
        max_results: int = 5,
        timeout: float = 30.0,
        summary_model: str = "gpt-3.5-turbo",
) -> Dict[str, Any]:
    """
    Perform web search and summarize results with goal-directed extraction.

    Args:
        query: Search query
        goal: The goal for information extraction
        max_results: Maximum number of search results
        timeout: Timeout for each operation
        summary_model: Model to use for summarization

    Returns:
        Dictionary with raw_results and summarized_docs
    """
    # Step 1: Perform search
    raw_results = await _serper_search_async(query, max_results, timeout)

    if not raw_results:
        return {
            "status": "no_results",
            "raw_results": [],
            "summarized_docs": [],
        }

    # Step 2: Fetch and summarize each result
    summarized_docs = []

    for result in raw_results:
        url = result.get("url", "")
        if not url:
            continue

        # Fetch page content
        content = await _jina_reader_fetch_async(url, timeout)

        if content:
            # Generate summary with goal
            summary_prompt = f"""Please extract information relevant to the goal from the following webpage content. 

## Webpage Content
{content[:50000]}

## Goal
{goal}

## Task
1. Extract the most relevant information (evidence)
2. Provide a concise summary

Output JSON format with "evidence" and "summary" fields."""

            summary_response = await _call_llm_async(
                summary_prompt,
                model=summary_model,
                timeout=60.0,
            )

            if summary_response:
                try:
                    # Try to parse JSON
                    json_start = summary_response.find('{')
                    json_end = summary_response.rfind('}')
                    if json_start != -1 and json_end != -1:
                        summary_json = json.loads(summary_response[json_start:json_end + 1])
                        evidence = summary_json.get("evidence", "")
                        summary = summary_json.get("summary", "")
                    else:
                        evidence = summary_response
                        summary = summary_response
                except json.JSONDecodeError:
                    evidence = summary_response
                    summary = summary_response
            else:
                evidence = content[:1000]
                summary = result.get("snippet", "")
        else:
            # Use snippet as fallback
            evidence = result.get("snippet", "")
            summary = result.get("snippet", "")

        formatted_content = f"""Evidence from page:
{evidence}

Summary:
{summary}"""

        summarized_docs.append({
            "title": result.get("title", ""),
            "url": url,
            "snippet": result.get("snippet", ""),
            "content": formatted_content,
            "evidence": evidence,
            "summary": summary,
            "source_query": query,
            "source_goal": goal,
        })

    return {
        "status": "success",
        "raw_results": raw_results,
        "summarized_docs": summarized_docs,
    }


def get_ray_deep_search_worker_cls():
    """
    Get or create the Ray Deep Search worker class. 

    Returns a Ray remote actor class that can perform web searches
    and page summarization with proper timeout handling.

    Returns:
        Ray remote actor class for deep search operations, or None if Ray unavailable
    """
    if ray is None:
        print("Ray is not available, cannot create DeepSearchWorker")
        return None

    # Check if we already have a cached class
    if hasattr(get_ray_deep_search_worker_cls, "_cls"):
        return getattr(get_ray_deep_search_worker_cls, "_cls")

    try:
        @ray.remote(num_cpus=0.1, max_concurrency=100)
        class _RayDeepSearchWorker:
            """
            Ray worker for Deep Search operations.
            Handles web search, page crawling, and content summarization.
            """

            def __init__(self, idx):
                """Initialize worker with index."""
                if not isinstance(idx, (int, float)):
                    print(f"Warning: idx parameter is not numeric: {type(idx)}, converting to int")
                    try:
                        self.idx = int(idx) if idx is not None else 0
                    except (ValueError, TypeError):
                        self.idx = 0
                else:
                    self.idx = int(idx)

                # URL cache to avoid re-crawling
                self._url_cache: Dict[str, str] = {}

            def get_idx(self) -> int:
                """Get the worker's index."""
                return self.idx

            async def search_and_summarize(
                    self,
                    query: str,
                    goal: str,
                    max_results: int = 5,
                    timeout: float = 30.0,
                    summary_model: str = "gpt-3.5-turbo",
            ) -> Dict[str, Any]:
                """
                Perform web search and summarize results.

                Args:
                    query: Search query string
                    goal: The information goal for extraction
                    max_results: Maximum number of results
                    timeout: Timeout per operation
                    summary_model: Model for summarization

                Returns:
                    Dictionary with status, raw_results, and summarized_docs
                """
                try:
                    result = await _perform_web_search_with_summary(
                        query=query,
                        goal=goal,
                        max_results=max_results,
                        timeout=timeout,
                        summary_model=summary_model,
                    )
                    return result
                except Exception as e:
                    print(f"search_and_summarize failed: {e}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "raw_results": [],
                        "summarized_docs": [],
                    }

            async def fetch_page(
                    self,
                    url: str,
                    timeout: float = 30.0,
                    use_cache: bool = True,
            ) -> Dict[str, Any]:
                """
                Fetch page content using Jina Reader.

                Args:
                    url: URL to fetch
                    timeout: Request timeout
                    use_cache: Whether to use cached content

                Returns:
                    Dictionary with status and content
                """
                try:
                    # Check cache first
                    if use_cache and url in self._url_cache:
                        return {
                            "status": "success",
                            "content": self._url_cache[url],
                            "from_cache": True,
                        }

                    content = await _jina_reader_fetch_async(url, timeout)

                    if content:
                        self._url_cache[url] = content
                        return {
                            "status": "success",
                            "content": content,
                            "from_cache": False,
                        }
                    else:
                        return {
                            "status": "error",
                            "error": "Failed to fetch content",
                            "content": None,
                        }
                except Exception as e:
                    print(f"fetch_page failed for {url}: {e}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "content": None,
                    }

            async def summarize_content(
                    self,
                    content: str,
                    goal: str,
                    model: str = "gpt-3.5-turbo",
                    timeout: float = 60.0,
            ) -> Dict[str, Any]:
                """
                Summarize content with goal-directed extraction. 

                Args:
                    content: Content to summarize
                    goal: The information extraction goal
                    model: LLM model to use
                    timeout: Request timeout

                Returns:
                    Dictionary with evidence and summary
                """
                try:
                    prompt = f"""Extract information relevant to the goal from the content. 

## Content
{content[:50000]}

## Goal
{goal}

## Output
JSON format with "evidence" and "summary" fields."""

                    response = await _call_llm_async(prompt, model, timeout)

                    if response:
                        try:
                            json_start = response.find('{')
                            json_end = response.rfind('}')
                            if json_start != -1 and json_end != -1:
                                result = json.loads(response[json_start:json_end + 1])
                                return {
                                    "status": "success",
                                    "evidence": result.get("evidence", ""),
                                    "summary": result.get("summary", ""),
                                }
                        except json.JSONDecodeError:
                            pass

                        return {
                            "status": "success",
                            "evidence": response,
                            "summary": response,
                        }
                    else:
                        return {
                            "status": "error",
                            "error": "LLM call failed",
                            "evidence": "",
                            "summary": "",
                        }
                except Exception as e:
                    print(f"summarize_content failed: {e}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "evidence": "",
                        "summary": "",
                    }

            async def call_rm_agent(
                    self,
                    prompt: str,
                    timeout: float = 120.0,
            ) -> Optional[str]:
                """
                Call RM-Agent (typically a closed-source LLM) for scoring.

                Args:
                    prompt: The evaluation prompt
                    timeout: Request timeout

                Returns:
                    RM-Agent response string, or None if failed
                """
                try:
                    api_url = os.getenv("RM_LLM_API_URL", "https://api.openai.com/v1/chat/completions")
                    api_key = os.getenv("RM_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
                    model = os.getenv("RM_LLM_MODEL", "gpt-4")

                    if not api_key:
                        print("Warning: RM_LLM_API_KEY not set")
                        return None

                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }

                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    }

                    if aiohttp:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                    api_url,
                                    headers=headers,
                                    json=payload,
                                    timeout=aiohttp.ClientTimeout(total=timeout)
                            ) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    return data["choices"][0]["message"]["content"]
                                else:
                                    print(f"RM API error: {response.status}")
                                    return None
                    else:
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            None,
                            lambda: requests.post(
                                api_url,
                                headers=headers,
                                json=payload,
                                timeout=timeout
                            )
                        )
                        if response.status_code == 200:
                            data = response.json()
                            return data["choices"][0]["message"]["content"]
                        else:
                            print(f"RM API error: {response.status_code}")
                            return None

                except Exception as e:
                    print(f"call_rm_agent failed: {e}")
                    return None

            def clear_cache(self):
                """Clear the URL content cache."""
                self._url_cache.clear()

        RayDeepSearchWorker = _RayDeepSearchWorker
        setattr(get_ray_deep_search_worker_cls, "_cls", RayDeepSearchWorker)
        return RayDeepSearchWorker

    except Exception as e:
        print(f"Failed to create RayDeepSearchWorker class: {e}")
        return None


async def get_search_results(
        query: str,
        goal: str,
        max_results: int = 5,
        timeout: float = 30.0,
        ray_actor: Any = None,
        summary_model: str = "gpt-3.5-turbo",
) -> Dict[str, Any]:
    """
    Get search results with summarization, using Ray worker if available.

    This is the main entry point for performing web searches from agents.

    Args:
        query: Search query
        goal: The information extraction goal
        max_results: Maximum number of results
        timeout: Timeout per operation
        ray_actor: Ray actor for distributed execution
        summary_model: Model for summarization

    Returns:
        Dictionary with search results and summaries
    """
    try:
        if ray_actor is not None:
            # Use Ray worker
            timeout_buffer = max(timeout * 2.
            0, 30.0)
            total_timeout = timeout + timeout_buffer

            obj_ref = ray_actor.search_and_summarize.remote(
                query=query,
                goal=goal,
                max_results=max_results,
                timeout=timeout,
                summary_model=summary_model,
            )
            result = await _await_ray_object_ref(obj_ref, total_timeout)
            return result
        else:
            # Direct execution
            result = await _perform_web_search_with_summary(
                query=query,
                goal=goal,
                max_results=max_results,
                timeout=timeout,
                summary_model=summary_model,
            )
            return result

    except asyncio.TimeoutError as e:
        error_msg = f"Search timed out after {timeout}s"
        print(f"Error: {error_msg}")
        return {
            "status": "error",
            "error": error_msg,
            "raw_results": [],
            "summarized_docs": [],
        }
    except Exception as e:
        error_msg = f"Search failed: {e}"
        print(f"Error: {error_msg}")
        return {
            "status": "error",
            "error": error_msg,
            "raw_results": [],
            "summarized_docs": [],
        }