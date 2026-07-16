"""Groq LLM Reranker for movie recommendations."""

import hashlib
import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from groq import Groq


class GroqReranker:
    """Reranks movie candidates using Groq's Llama-3-8B model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        max_candidates_per_request: int = 25,
        cache_dir: Optional[str] = "outputs/reranker_cache",
        max_retries: int = 4,
    ):
        """
        Initialize the Groq reranker.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model: Model name to use
            max_candidates_per_request: Max movies per API call. Tokens are the
                scarce resource (Groq free tier is 500K/day): the tail of a long
                candidate list is not where reranking earns anything.
            cache_dir: Disk cache for responses keyed by prompt hash — re-running
                the same comparison (temperature 0.1 is near-deterministic) costs
                zero tokens. None disables.
            max_retries: Exponential-backoff attempts on rate limits (429).
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set and no api_key provided")

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.max_candidates_per_request = max_candidates_per_request
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries

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
            return json.loads(cache_path.read_text())

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a movie recommendation expert. Return only valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                response_text = response.choices[0].message.content or ""
                results = self._parse_response(response_text, len(movies))
                if results and cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(results))
                return results

            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e)
                if is_rate_limit and attempt < self.max_retries:
                    delay = 5 * 2**attempt
                    print(f"Groq rate limit, backing off {delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                print(f"Error calling Groq API: {e}")
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


def create_reranker(config: Optional[Dict[str, Any]] = None) -> GroqReranker:
    """Factory function to create reranker from config."""
    if config is None:
        config = {}
    return GroqReranker(
        api_key=config.get("api_key"),
        model=config.get("model", "llama-3.1-8b-instant"),
        max_candidates_per_request=config.get("max_candidates_per_request", 25),
        cache_dir=config.get("cache_dir", "outputs/reranker_cache"),
        max_retries=config.get("max_retries", 4),
    )