# movie-recommender frontend

A static portfolio demo for the recommendation engine in the parent repo. It has exactly
one mode: it **replays real recorded responses** captured from the live system — there is
no live backend connection, so it deploys to Vercel (or any static host) permanently with
zero server dependency.

## What's real

Every number on screen traces to a named field in the API contract or an offline artifact:

- `fixtures/recommendations.json` — actual `/recommend` and `/recommend_cold` responses
  (movies, rerank order, retrieval ranks, cosine distances, stage latencies, token usage),
  captured by `../scripts/capture_fixtures.py` through the real FastAPI app.
- `fixtures/eval.json` — leave-one-out metrics over 138,493 users from the repo's
  evaluation artifacts.
- `fixtures/projection.json` — 2D PCA of the served model's item embeddings.
- `fixtures/users.json` / `movies.json` — demo-user taste profiles and the picker catalog,
  computed from real rating history.

Fields the backend doesn't return yet (e.g. per-movie `reason`) render an
`AWAITING BACKEND` badge — nothing is invented.

## Develop

```bash
npm install
npm run dev     # http://localhost:3000
npm run build   # static production build (all routes prerender)
```

To refresh the fixtures after retraining or backend changes:

```bash
cd .. && PYTHONPATH=. python3 scripts/capture_fixtures.py
```

## Stack

Next.js (App Router) · React 19 · TypeScript strict · Tailwind v4 · Framer Motion ·
TanStack Query · Zustand · Lucide.
