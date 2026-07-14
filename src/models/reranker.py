"""Groq LLM Reranker for movie recommendations."""

import os
import json
import re
from typing import List, Dict, Any, Optional

from groq import Groq


class GroqReranker:
    """Reranks movie candidates using Groq's Llama-3-8B model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        max_candidates_per_request: int = 50,
    ):
        """
        Initialize the Groq reranker.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model: Model name to use (default: llama3-8b-8192)
            max_candidates_per_request: Max movies per API call (Groq context limit)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set and no api_key provided")

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.max_candidates_per_request = max_candidates_per_request

    def _build_prompt(self, query: str, movies: List[Dict[str, Any]]) -> str:
        """Build the reranking prompt for the LLM."""
        movie_lines = []
        for i, movie in enumerate(movies):
            movie_id = movie.get("movieId", movie.get("id", i))
            title = movie.get("title", "Unknown")
            genres = movie.get("genres", "")
            tags = movie.get("tags", "")
            movie_lines.append(f"{i}: [{movie_id}] {title} | Genres: {genres} | Tags: {tags}")

        movie_block = "\n".join(movie_lines)

        prompt = f"""You are a movie recommendation expert. Rank these candidates by relevance to the user's query.

User Query: "{query}"

Candidates:
{movie_block}

Return a JSON array of objects with "index" (0-based position in candidates) and "score" (1-10 relevance).
Only include movies you would recommend. Sort by score descending.
Example: [{{"index": 0, "score": 9}}, {{"index": 2, "score": 7}}]"""

        return prompt

    def _parse_response(self, response_text: str, num_candidates: int) -> List[Dict[str, Any]]:
        """Parse LLM response into list of (index, score) pairs."""
        try:
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text)

            results = []
            for item in parsed:
                idx = item.get("index")
                score = item.get("score")
                if isinstance(idx, int) and isinstance(score, (int, float)):
                    if 0 <= idx < num_candidates:
                        results.append({"index": idx, "score": float(score)})

            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return []

    def _rerank_batch(
        self,
        query: str,
        movies: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank a single batch of movies."""
        prompt = self._build_prompt(query, movies)

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
                max_tokens=2048,
            )

            response_text = response.choices[0].message.content or ""
            return self._parse_response(response_text, len(movies))

        except Exception as e:
            print(f"Error calling Groq API: {e}")
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

        for i in range(0, len(movies), self.max_candidates_per_request):
            batch = movies[i : i + self.max_candidates_per_request]
            batch_results = self._rerank_batch(query, batch)

            for result in batch_results:
                idx = result["index"]
                movie = batch[idx].copy()
                movie["rerank_score"] = result["score"]
                all_results.append(movie)

        all_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return all_results[:top_k]


def create_reranker(config: Optional[Dict[str, Any]] = None) -> GroqReranker:
    """Factory function to create reranker from config."""
    if config is None:
        config = {}
    return GroqReranker(
        api_key=config.get("api_key"),
        model=config.get("model", "llama-3.1-8b-instant"),
        max_candidates_per_request=config.get("max_candidates_per_request", 50),
    )