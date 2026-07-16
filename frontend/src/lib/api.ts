/**
 * Typed API layer with two build-time modes:
 *
 * - Fixture replay (default, and the only mode in deployed builds): real
 *   recorded responses from fixtures/, captured from the live system by
 *   scripts/capture_fixtures.py.
 * - Live (when NEXT_PUBLIC_API_URL is set at build time, e.g. in .env.local
 *   for local testing): real fetches against the FastAPI backend — any user,
 *   any query, any cold-start picks.
 *
 * The env var is inlined at build time, so a build without it contains only
 * the fixture path and stays a pure static site.
 */

import type {
  ColdRecommendRequest,
  DemoUser,
  EvalMetrics,
  FixtureFile,
  PickerMovie,
  ProjectionPoint,
  RecommendRequest,
  RecommendResponse,
} from "./types";

import fixtureData from "../../fixtures/recommendations.json";
import usersData from "../../fixtures/users.json";
import moviesData from "../../fixtures/movies.json";
import evalData from "../../fixtures/eval.json";

const fixtures = fixtureData as unknown as FixtureFile;

/** True when this build talks to a real backend instead of replaying fixtures. */
export const LIVE_MODE = Boolean(process.env.NEXT_PUBLIC_API_URL);
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export class NoFixtureError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NoFixtureError";
  }
}

async function livePost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const err = await res.json();
      if (err.detail) detail = String(err.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const normalize = (s: string) => s.trim().toLowerCase();

/** Replay pacing: resolve after the recorded stage time, capped so a slow
 * recorded rerank doesn't stall the demo. The displayed numbers are always
 * the real recorded values, not the capped wait. */
const capMs = (ms: number, cap: number) => Math.min(ms, cap);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function recommend(req: RecommendRequest): Promise<RecommendResponse> {
  if (LIVE_MODE) return livePost<RecommendResponse>("/recommend", req);
  const hit = fixtures.recommend.find(
    (f) => f.request.user_id === req.user_id && normalize(f.request.query) === normalize(req.query),
  );
  if (!hit) {
    throw new NoFixtureError(
      `No recorded response for user #${req.user_id} with that query. ` +
        `This demo replays real captured runs — pick a suggested query and demo user.`,
    );
  }
  await sleep(capMs(hit.response.timing.hnsw_ms + hit.response.timing.rerank_ms, 3500));
  return hit.response;
}

export async function recommendCold(req: ColdRecommendRequest): Promise<RecommendResponse> {
  if (LIVE_MODE) return livePost<RecommendResponse>("/recommend_cold", req);
  const wanted = [...req.liked_movie_ids].sort().join(",");
  const hit = fixtures.recommend_cold.find(
    (f) =>
      [...f.request.liked_movie_ids].sort().join(",") === wanted &&
      normalize(f.request.query) === normalize(req.query),
  );
  if (!hit) {
    throw new NoFixtureError(
      "No recorded response for that pick + query combination. " +
        "This demo replays real captured runs — use one of the recorded scenarios.",
    );
  }
  await sleep(capMs(hit.response.timing.hnsw_ms + hit.response.timing.rerank_ms, 3500));
  return hit.response;
}

/* ---- Static artifacts (loaded at build time) ---- */

export const demoUsers = usersData as unknown as DemoUser[];
export const pickerMovies = moviesData as unknown as PickerMovie[];
export const evalMetrics = evalData as unknown as EvalMetrics;

/** The exact query strings that have recorded responses (for suggestion chips). */
export const recordedQueries = Array.from(new Set(fixtures.recommend.map((f) => f.request.query)));

/** The recorded cold-start scenarios (for onboarding presets). */
export const coldScenarios = fixtures.recommend_cold.map((f) => ({
  name: f.name,
  liked_movie_ids: f.request.liked_movie_ids,
  query: f.request.query,
}));

export const demoUserIds = demoUsers.map((u) => u.user_id);
