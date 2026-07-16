"use client";

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AlertCircle, ArrowLeft } from "lucide-react";
import { recommend, recommendCold, NoFixtureError } from "@/lib/api";
import { useSettings } from "@/lib/store";
import type { MovieRecommendation, RecommendResponse } from "@/lib/types";
import { fmtMs, genreList, titleOnly, yearOf } from "@/lib/format";
import { GenreTag, Tooltip } from "@/components/ui";
import { LoadingSequence } from "./loading-sequence";
import { RankFlow } from "./rank-flow";
import { DevDrawer } from "./dev-drawer";

function PipelineBar({ response }: { response: RecommendResponse }) {
  const { timing } = response;
  const stages = [
    { name: "user embedding", ms: null },
    { name: "hnsw search", ms: timing.hnsw_ms },
    { name: "llm rerank", ms: timing.rerank_ms },
  ];
  return (
    <div className="border-gradient glass mb-6 flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl px-4 py-2.5">
      {stages.map((s, i) => (
        <span key={s.name} className="flex items-center gap-2 text-xs">
          {i > 0 && <span className="text-accent/60">→</span>}
          <span className="text-text-dim">{s.name}</span>
          {s.ms !== null && <span className="font-mono text-text">{fmtMs(s.ms)}</span>}
        </span>
      ))}
      {response.usage?.cached && (
        <span className="ml-auto font-mono text-[11px] text-text-faint" title="Rerank response served from the on-disk cache">
          rerank cached
        </span>
      )}
    </div>
  );
}

/** Consumer card: poster-first design with a deliberate typographic fallback
 * (posters are a Bucket-3 artifact that ships with the TMDB enrichment job). */
function MovieCard({ movie, rank }: { movie: MovieRecommendation; rank: number }) {
  const year = yearOf(movie.title);
  const genres = genreList(movie.genres);
  return (
    <motion.article
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: rank * 0.06, duration: 0.25 }}
      whileHover={{ y: -3 }}
      className="border-gradient glass glow-accent-hover flex flex-col overflow-hidden rounded-2xl transition-all"
    >
      <div className="relative flex aspect-[2/3] items-end bg-[radial-gradient(circle_at_20%_15%,rgba(139,92,246,0.22),transparent_55%),radial-gradient(circle_at_85%_90%,rgba(99,102,241,0.14),transparent_50%)] p-4">
        <span className="glow-accent absolute left-3 top-3 rounded-md bg-bg/80 px-2 py-0.5 font-mono text-xs text-accent">
          #{rank}
        </span>
        <h3 className="text-lg font-semibold leading-snug tracking-tight">{titleOnly(movie.title)}</h3>
      </div>
      <div className="flex flex-1 flex-col gap-2.5 p-4">
        <div className="flex items-center justify-between font-mono text-xs text-text-faint">
          <span>{year ?? ""}</span>
          <Tooltip label="Preference score derived from the LLM's ranking of the 25 retrieval candidates. Higher = ranked earlier. Not a probability.">
            <span>score {movie.rerank_score.toFixed(0)}</span>
          </Tooltip>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {genres.slice(0, 3).map((g) => (
            <GenreTag key={g} genre={g} />
          ))}
        </div>
        {movie.reason && <p className="text-xs leading-relaxed text-text-dim">{movie.reason}</p>}
      </div>
    </motion.article>
  );
}

function Filters({
  genres,
  genre,
  setGenre,
  sort,
  setSort,
}: {
  genres: string[];
  genre: string;
  setGenre: (g: string) => void;
  sort: string;
  setSort: (s: string) => void;
}) {
  const select =
    "rounded-lg border border-border bg-card px-3 py-1.5 text-xs text-text-dim outline-none focus:border-border-strong";
  return (
    <div className="mb-5 flex flex-wrap items-center gap-2">
      <select value={genre} onChange={(e) => setGenre(e.target.value)} aria-label="Filter by genre" className={select}>
        <option value="">all genres</option>
        {genres.map((g) => (
          <option key={g}>{g}</option>
        ))}
      </select>
      <select value={sort} onChange={(e) => setSort(e.target.value)} aria-label="Sort" className={select}>
        <option value="rerank">sort: rerank score</option>
        <option value="year">sort: year</option>
      </select>
    </div>
  );
}

function ResultsInner() {
  const params = useSearchParams();
  const devMode = useSettings((s) => s.devMode);
  const [genre, setGenre] = useState("");
  const [sort, setSort] = useState("rerank");

  const mode = params.get("mode") ?? "warm";
  const query = params.get("q") ?? "";
  const userId = Number(params.get("user") ?? 0);
  const picks = (params.get("picks") ?? "").split(",").filter(Boolean).map(Number);

  const { data, isPending, error } = useQuery({
    queryKey: ["recommend", mode, userId, picks.join(","), query],
    queryFn: () =>
      mode === "cold"
        ? recommendCold({ liked_movie_ids: picks, query, top_k: 10 })
        : recommend({ user_id: userId, query, top_k: 10 }),
  });

  const filtered = useMemo(() => {
    if (!data) return [];
    let list = [...data.recommendations];
    if (genre) list = list.filter((m) => genreList(m.genres).includes(genre));
    if (sort === "year") list.sort((a, b) => (yearOf(b.title) ?? 0) - (yearOf(a.title) ?? 0));
    return list;
  }, [data, genre, sort]);

  const allGenres = useMemo(
    () => Array.from(new Set((data?.recommendations ?? []).flatMap((m) => genreList(m.genres)))).sort(),
    [data],
  );

  if (isPending) return <LoadingSequence query={query} />;

  if (error) {
    const noFixture = error instanceof NoFixtureError;
    return (
      <div className="mx-auto mt-20 max-w-md rounded-2xl border border-border bg-card p-6 text-center">
        <AlertCircle className="mx-auto mb-3 text-error" size={22} />
        <h2 className="mb-2 text-sm font-semibold">{noFixture ? "No recorded response" : "Something went wrong"}</h2>
        <p className="mb-5 text-xs leading-relaxed text-text-dim">{error.message}</p>
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-xs font-medium text-white hover:bg-accent-dim"
        >
          <ArrowLeft size={13} /> Back to search
        </Link>
      </div>
    );
  }

  return (
    <div>
      <header className="mb-6 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">
            {mode === "cold" ? "cold-start · no account" : `personalized · user #${userId}`}
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">“{query}”</h1>
        </div>
        <Link href="/" className="flex items-center gap-1.5 text-xs text-text-dim hover:text-text">
          <ArrowLeft size={13} /> new search
        </Link>
      </header>

      <PipelineBar response={data} />

      {devMode ? (
        <div className="flex flex-col gap-6">
          <RankFlow response={data} />
          <DevDrawer response={data} />
        </div>
      ) : (
        <>
          <Filters genres={allGenres} genre={genre} setGenre={setGenre} sort={sort} setSort={setSort} />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5">
            {filtered.map((m) => (
              <MovieCard key={m.movieId} movie={m} rank={data.recommendations.indexOf(m) + 1} />
            ))}
          </div>
          {filtered.length === 0 && (
            <p className="mt-10 text-center text-sm text-text-dim">No results match that filter.</p>
          )}
        </>
      )}
    </div>
  );
}

export default function ResultsPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <Suspense fallback={null}>
        <ResultsInner />
      </Suspense>
    </div>
  );
}
