"""Cerebras LLM Reranker for movie recommendations."""

import hashlib
import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from cerebras.cloud.sdk import Cerebras


class LLMReranker:
    """Reranks movie candidates using an LLM served by Cerebras."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-oss-120b",
        max_candidates_per_request: int = 25,
        cache_dir: Optional[str] = "outputs/reranker_cache",
        max_retries: int = 4,
    ):
        """
        Initialize the Cerebras reranker.

        Args:
            api_key: Cerebras API key. If None, reads from CEREBRAS_API_KEY env var.
            model: Model name to use
            max_candidates_per_request: Max movies per API call. Tokens are the
                scarce resource on free tiers: the tail of a long candidate list
                is not where reranking earns anything.
            cache_dir: Disk cache for responses keyed by prompt hash — re-running
                the same comparison (temperature 0.1 is near-deterministic) costs
                zero tokens. None disables.
            max_retries: Exponential-backoff attempts on rate limits (429).
        """
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY not set and no api_key provided")

        self.client = Cerebras(api_key=self.api_key)
        self.model = model
        self.max_candidates_per_request = max_candidates_per_request
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        # gpt-oss-120b is a reasoning model: it may reject json_object mode or
        # spend tokens reasoning before the JSON. Downgrade gracefully once.
        self._json_mode = True
        self._reasoning_effort: Optional[str] = "low"
        # Token usage of the most recent rerank() call ({"prompt_tokens",
        # "completion_tokens", "cached"}); None until the first call.
        self.last_usage = None

    def _build_prompt(self, query: str, movies: List[Dict[str, Any]], top_k: int) -> str:
        """Build the reranking prompt. Title + genres only: tags are the token
        hog and the noisiest signal; the LLM reorders fine without them."""
        movie_lines = []
        for i, movie in enumerate(movies):
            title = movie.get("title", "Unknown")
            genres = movie.get("genres", "")
            movie_lines.append(f"{i}: {title} | {genres}")

        movie_block = "\n".join(movie_lines)
        n = min(top_k, len(movies))

        prompt = f"""You are a movie recommendation expert. Rank these candidates by relevance to the user's query.

User Query: "{query}"

Candidates:
{movie_block}

Return a JSON object {{"ranking": [4, 0, 7]}} with exactly the {n} best candidate
indices (0-based), best first. No prose, JSON only."""

        return prompt

    def _parse_response(self, response_text: str, num_candidates: int) -> List[Dict[str, Any]]:
        """Parse LLM response into ranked (index, score) pairs.

        Expects {"ranking": [4, 0, 7]} (JSON mode, best first); salvages a bare
        integer list from truncated or prose-wrapped output as a last resort.
        Scores are synthesized from rank order (best = num_candidates).
        """
        indices = []
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
            indices = [i for i in parsed if isinstance(i, int)]
        except json.JSONDecodeError:
            match = re.search(r"\[\s*\d+(?:\s*,\s*\d+)*", response_text)
            if not match:
                print("Warning: Failed to parse LLM response and nothing salvageable")
                return []
            indices = [int(i) for i in re.findall(r"\d+", match.group())]
            print(f"Warning: LLM response was not valid JSON; salvaged {len(indices)} indices")

        results = []
        seen = set()
        for rank, idx in enumerate(indices):
            if 0 <= idx < num_candidates and idx not in seen:
                seen.add(idx)
                results.append({"index": idx, "score": float(num_candidates - rank)})
        return results

    def _cache_path(self, prompt: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(f"{self.model}|{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _rerank_batch(
        self,
        query: str,
        movies: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Rerank a single batch of movies (disk-cached, 429s retried with backoff)."""
        prompt = self._build_prompt(query, movies, top_k)

        cache_path = self._cache_path(prompt)
        if cache_path is not None and cache_path.exists():
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached": True}
            return json.loads(cache_path.read_text())

        for attempt in range(self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a movie recommendation expert. Return only valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    # Reasoning models spend tokens thinking before the JSON; keep
                    # the ceiling high and the reasoning effort low so the ranking
                    # itself never gets truncated away.
                    max_tokens=4096,
                )
                if self._reasoning_effort:
                    kwargs["reasoning_effort"] = self._reasoning_effort
                if self._json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception as e:
                    msg = str(e)
                    if self._json_mode and "response_format" in msg:
                        print("Note: model rejected json_object mode; retrying without")
                        self._json_mode = False
                        kwargs.pop("response_format")
                        response = self.client.chat.completions.create(**kwargs)
                    elif self._reasoning_effort and "reasoning_effort" in msg:
                        print("Note: model rejected reasoning_effort; retrying without")
                        self._reasoning_effort = None
                        kwargs.pop("reasoning_effort")
                        response = self.client.chat.completions.create(**kwargs)
                    else:
                        raise
                response_text = response.choices[0].message.content or ""
                if response.usage is not None:
                    self.last_usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "cached": False,
                    }
                results = self._parse_response(response_text, len(movies))
                if results and cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(results))
                return results

            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e)
                if is_rate_limit and attempt < self.max_retries:
                    delay = 5 * 2**attempt
                    print(f"Cerebras rate limit, backing off {delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                print(f"Error calling Cerebras API: {e}")
                return []
        return []

    def rerank(
        self,
        query: str,
        movies: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank movies by relevance to query using Groq LLM.

        Args:
            query: User's natural language query
            movies: List of movie dicts with keys: movieId/id, title, genres, tags
            top_k: Number of top results to return

        Returns:
            List of movie dicts with added 'rerank_score' key, sorted by score desc
        """
        if not movies:
            return []

        all_results = []
        ranked_ids = set()

        for i in range(0, len(movies), self.max_candidates_per_request):
            batch = movies[i : i + self.max_candidates_per_request]
            batch_results = self._rerank_batch(query, batch, top_k)

            for result in batch_results:
                idx = result["index"]
                movie = batch[idx].copy()
                movie["rerank_score"] = result["score"]
                all_results.append(movie)
                ranked_ids.add(id(batch[idx]))

        all_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        if not all_results:
            print("Warning: reranker produced no results, falling back to retrieval order")

        # Pad short (or empty) LLM output with the remaining candidates in upstream
        # (retrieval) order — the caller always gets a full top_k list.
        if len(all_results) < top_k:
            for movie in movies:
                if len(all_results) >= top_k:
                    break
                if id(movie) not in ranked_ids:
                    movie = movie.copy()
                    movie["rerank_score"] = 0.0
                    all_results.append(movie)

        return all_results[:top_k]


def create_reranker(config: Optional[Dict[str, Any]] = None) -> LLMReranker:
    """Factory function to create reranker from config."""
    if config is None:
        config = {}
    return LLMReranker(
        api_key=config.get("api_key"),
        model=config.get("model", "gpt-oss-120b"),
        max_candidates_per_request=config.get("max_candidates_per_request", 25),
        cache_dir=config.get("cache_dir", "outputs/reranker_cache"),
        max_retries=config.get("max_retries", 4),
    )