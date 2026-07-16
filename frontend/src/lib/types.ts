/**
 * One source of truth for the API schema, written against the TARGET backend
 * contract. Fields the backend does not yet return are optional; components
 * must badge or hide on absence, never invent a value.
 */

export interface RecommendRequest {
  user_id: number;
  query: string;
  top_k: number;
}

export interface ColdRecommendRequest {
  liked_movie_ids: number[];
  query: string;
  top_k: number;
}

export interface MovieRecommendation {
  movieId: number;
  title: string;
  genres: string;
  /** Preference score derived from the LLM's ranking of the 25 candidates
   * (higher = ranked earlier). NOT a calibrated probability — never render
   * as a percentage. */
  rerank_score: number;
  /** 1-based position in the retrieval candidate list. */
  retrieval_rank?: number;
  /** Cosine distance between the user embedding and this movie's embedding
   * in 128-dim space. Lower is closer. */
  distance?: number;
  /** Short per-movie explanation from the reranker. Not yet returned. */
  reason?: string;
}

export interface Candidate {
  movieId: number;
  title: string;
  genres: string;
  retrieval_rank: number;
  distance: number;
}

export interface Timing {
  hnsw_ms: number;
  rerank_ms: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  /** True when the reranker response was served from the on-disk cache. */
  cached: boolean;
}

export interface RecommendResponse {
  recommendations: MovieRecommendation[];
  candidates: Candidate[];
  timing: Timing;
  usage?: Usage | null;
}

/* ---- Offline artifacts (Bucket 3) ---- */

export interface DemoUser {
  user_id: number;
  num_ratings: number;
  num_liked: number;
  genre_distribution: Record<string, number>;
  decade_distribution: Record<string, number>;
  recent_liked_titles: string[];
}

export interface PickerMovie {
  movieId: number;
  title: string;
  genres: string;
  year: number | null;
}

export interface ProjectionPoint {
  movieId: number;
  title: string;
  genre: string;
  x: number;
  y: number;
  /** Third PCA component; present in current artifacts (3D galaxy view). */
  z?: number;
}

export interface EvalMetrics {
  loo: Record<string, number>;
  cold_start: {
    cold: Record<string, number>;
    warm: Record<string, number>;
    cold_frac: number;
  };
  timesplit: Record<string, Record<string, number>>;
  system: {
    movies_indexed: number;
    users_trained: number;
    ratings_trained: string;
    embedding_dim: number;
    hnsw: { M: number; ef_construction: number; ef_search: number };
    reranker_model: string;
    candidate_pool: number;
    rerank_candidates: number;
  };
}

/* ---- Fixture file shapes ---- */

export interface RecordedRecommend {
  request: RecommendRequest;
  response: RecommendResponse;
}

export interface RecordedColdRecommend {
  name: string;
  request: ColdRecommendRequest;
  response: RecommendResponse;
}

export interface FixtureFile {
  recommend: RecordedRecommend[];
  recommend_cold: RecordedColdRecommend[];
}
