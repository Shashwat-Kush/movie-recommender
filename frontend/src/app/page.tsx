"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowRight, Check, Cpu, Layers, Shuffle, Sparkles, UserRound } from "lucide-react";
import Link from "next/link";
import { coldScenarios, demoUsers, pickerMovies, recordedQueries, LIVE_MODE } from "@/lib/api";
import { titleOnly } from "@/lib/format";
import { Tooltip } from "@/components/ui";

/** Faint embedding-space dot field behind the hero (poster wall ships when the
 * TMDB enrichment artifact exists). Deterministic so SSR and client agree. */
function DotField() {
  const dots = useMemo(() => {
    let seed = 42;
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    return Array.from({ length: 90 }, () => ({
      x: rand() * 100,
      y: rand() * 100,
      r: 1 + rand() * 2,
      o: 0.04 + rand() * 0.1,
    }));
  }, []);
  return (
    <svg aria-hidden className="pointer-events-none absolute inset-0 h-full w-full">
      {dots.map((d, i) => (
        <circle key={i} cx={`${d.x}%`} cy={`${d.y}%`} r={d.r} fill="#8b5cf6" opacity={d.o} />
      ))}
    </svg>
  );
}

function WarmSearch() {
  const router = useRouter();
  const [userIdx, setUserIdx] = useState(0);
  const [customUserId, setCustomUserId] = useState<string | null>(null);
  const [query, setQuery] = useState(recordedQueries[0] ?? "");
  const user = demoUsers[userIdx];
  const effectiveUserId = customUserId !== null && customUserId !== "" ? Number(customUserId) : user.user_id;

  const go = () => {
    if (!query.trim()) return;
    router.push(`/results?mode=warm&user=${effectiveUserId}&q=${encodeURIComponent(query)}`);
  };

  return (
    <div className="w-full max-w-2xl">
      <div className="border-gradient glass flex flex-col gap-3 rounded-2xl p-4 shadow-2xl sm:flex-row sm:items-center">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && go()}
          placeholder={LIVE_MODE ? "ask for anything — this is live" : "what are you in the mood for?"}
          aria-label="Query"
          className="flex-1 bg-transparent px-2 py-2 text-base outline-none placeholder:text-text-faint"
        />
        <button
          onClick={go}
          className="glow-accent glow-accent-hover flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent to-indigo-500 px-5 py-2.5 text-sm font-medium text-white transition-all hover:brightness-110"
        >
          Recommend <ArrowRight size={15} />
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          {recordedQueries.map((q) => (
            <button
              key={q}
              onClick={() => setQuery(q)}
              className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                q === query
                  ? "border-accent bg-accent-faint text-accent"
                  : "border-border text-text-dim hover:border-border-strong hover:text-text"
              }`}
            >
              {q}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 text-xs text-text-dim">
          <UserRound size={13} />
          <Tooltip label="Retrieval is personalized: the user's watch history drives the taste embedding that searches the index. The query only enters at the rerank stage.">
            viewing as user
          </Tooltip>
          {LIVE_MODE ? (
            <input
              value={customUserId ?? String(user.user_id)}
              onChange={(e) => setCustomUserId(e.target.value.replace(/\D/g, ""))}
              aria-label="User ID"
              className="w-20 rounded-md border border-border bg-card px-2 py-0.5 font-mono text-text outline-none focus:border-accent"
            />
          ) : (
            <span className="font-mono text-text">#{user.user_id}</span>
          )}
          <button
            onClick={() => {
              setUserIdx((userIdx + 1) % demoUsers.length);
              setCustomUserId(null);
            }}
            aria-label="Shuffle demo user"
            className="rounded-md border border-border p-1 text-text-dim transition-colors hover:border-border-strong hover:text-text"
          >
            <Shuffle size={12} />
          </button>
        </div>
      </div>

      <p className="mt-2 text-right font-mono text-[11px] text-text-faint">
        {customUserId === null || customUserId === String(user.user_id)
          ? `${user.num_ratings.toLocaleString()} ratings · top genres: ${Object.keys(user.genre_distribution).slice(0, 3).join(", ")}`
          : "custom user — any of the 138,493 trained users works live"}
      </p>
    </div>
  );
}

function ColdStart() {
  const router = useRouter();
  const [scenarioIdx, setScenarioIdx] = useState<number | null>(null);
  const [picks, setPicks] = useState<number[]>([]);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");

  const applyScenario = (i: number) => {
    setScenarioIdx(i);
    setPicks(coldScenarios[i].liked_movie_ids);
    setQuery(coldScenarios[i].query);
  };

  const togglePick = (movieId: number) => {
    setScenarioIdx(null);
    setPicks((p) => (p.includes(movieId) ? p.filter((m) => m !== movieId) : p.length < 5 ? [...p, movieId] : p));
  };

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return pickerMovies.filter((m) => m.title.toLowerCase().includes(q)).slice(0, 24);
  }, [search]);

  const titleFor = (id: number) => {
    const m = pickerMovies.find((x) => x.movieId === id);
    return m ? titleOnly(m.title) : `#${id}`;
  };

  const go = () => {
    if (picks.length === 0 || !query.trim()) return;
    router.push(`/results?mode=cold&picks=${picks.join(",")}&q=${encodeURIComponent(query)}`);
  };

  return (
    <div className="w-full max-w-2xl text-left">
      <p className="mb-3 text-sm text-text-dim">
        No account needed — the user tower pools the embeddings of a few movies you pick.{" "}
        {!LIVE_MODE && (
          <span className="text-text-faint">
            The static demo replays the two recorded scenarios; custom picks need the live backend.
          </span>
        )}
        {LIVE_MODE && <span className="text-success">Live mode: pick any movies you like.</span>}
      </p>

      <div className="mb-4 grid gap-3 sm:grid-cols-2">
        {coldScenarios.map((s, i) => (
          <button
            key={s.name}
            onClick={() => applyScenario(i)}
            className={`rounded-xl border p-3 text-left transition-colors ${
              scenarioIdx === i ? "border-accent bg-accent-faint" : "border-border bg-card hover:border-border-strong"
            }`}
          >
            <p className="mb-1 flex items-center gap-2 text-sm font-medium capitalize">
              {scenarioIdx === i && <Check size={14} className="text-accent" />}
              {s.name.replace("-", " ")}
            </p>
            <p className="text-xs leading-relaxed text-text-dim">{s.liked_movie_ids.map(titleFor).join(" · ")}</p>
            <p className="mt-1.5 font-mono text-[11px] text-text-faint">“{s.query}”</p>
          </button>
        ))}
      </div>

      <div className="rounded-xl border border-border bg-card p-3">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="search the catalog…"
          aria-label="Search movies"
          className="mb-3 w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm outline-none placeholder:text-text-faint focus:border-border-strong"
        />
        <div className="grid max-h-44 grid-cols-2 gap-1.5 overflow-y-auto scroll-thin sm:grid-cols-3">
          {filtered.map((m) => (
            <button
              key={m.movieId}
              onClick={() => togglePick(m.movieId)}
              className={`truncate rounded-lg border px-2 py-1.5 text-left text-xs transition-colors ${
                picks.includes(m.movieId)
                  ? "border-accent bg-accent-faint text-accent"
                  : "border-transparent text-text-dim hover:bg-bg"
              }`}
              title={m.title}
            >
              {titleOnly(m.title)}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && go()}
          placeholder="and what are you in the mood for?"
          aria-label="Cold-start query"
          className="flex-1 rounded-xl border border-border bg-card px-4 py-2.5 text-sm outline-none placeholder:text-text-faint focus:border-border-strong"
        />
        <button
          onClick={go}
          disabled={picks.length === 0 || !query.trim()}
          className="flex items-center justify-center gap-2 rounded-xl bg-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-dim disabled:cursor-not-allowed disabled:opacity-40"
        >
          Recommend <ArrowRight size={15} />
        </button>
      </div>
      <p className="mt-2 font-mono text-[11px] text-text-faint">
        {picks.length}/5 picked{picks.length > 0 && `: ${picks.map(titleFor).join(", ")}`}
      </p>
    </div>
  );
}

const FEATURES = [
  {
    icon: Layers,
    title: "Two-Tower Retrieval",
    body: "Your watch history is pooled into a 128-dim taste embedding — no per-user parameters, so brand-new users work too.",
    href: "/lab#pipeline",
  },
  {
    icon: Cpu,
    title: "Custom C++ HNSW",
    body: "A from-scratch HNSW index over 26,744 movie embeddings with SIMD dot products, searched in milliseconds.",
    href: "/lab#system",
  },
  {
    icon: Sparkles,
    title: "LLM Reranking",
    body: "Cerebras gpt-oss-120b reorders the top 25 candidates against your natural-language query.",
    href: "/lab#pipeline",
  },
];

export default function Home() {
  const [mode, setMode] = useState<"warm" | "cold">("warm");

  return (
    <div className="relative">
      <DotField />
      <section className="relative mx-auto flex max-w-6xl flex-col items-center px-6 pb-10 pt-20 text-center">
        <motion.h1
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-gradient max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl"
        >
          What should I watch tonight?
        </motion.h1>
        <p className="mt-4 max-w-2xl text-base text-text-dim">
          A recommendation engine built from scratch — Two-Tower retrieval, a custom C++ HNSW index, and LLM
          reranking.
        </p>

        <div className="mt-8 flex rounded-full border border-border bg-card p-1 text-sm">
          <button
            onClick={() => setMode("warm")}
            aria-pressed={mode === "warm"}
            className={`rounded-full px-4 py-1.5 transition-colors ${
              mode === "warm" ? "bg-accent text-white" : "text-text-dim hover:text-text"
            }`}
          >
            Existing user
          </button>
          <button
            onClick={() => setMode("cold")}
            aria-pressed={mode === "cold"}
            className={`rounded-full px-4 py-1.5 transition-colors ${
              mode === "cold" ? "bg-accent text-white" : "text-text-dim hover:text-text"
            }`}
          >
            New user — pick movies
          </button>
        </div>

        <div className="mt-8 flex w-full justify-center">{mode === "warm" ? <WarmSearch /> : <ColdStart />}</div>
      </section>

      <section className="relative mx-auto grid max-w-5xl gap-4 px-6 pb-16 sm:grid-cols-3">
        {FEATURES.map(({ icon: Icon, title, body, href }) => (
          <motion.div key={title} whileHover={{ y: -3 }} transition={{ duration: 0.2 }}>
            <Link
              href={href}
              className="border-gradient glass glow-accent-hover flex h-full flex-col gap-2 rounded-2xl p-5 transition-all"
            >
              <span className="glow-accent flex h-8 w-8 items-center justify-center rounded-lg bg-accent-faint">
                <Icon size={16} className="text-accent" />
              </span>
              <h3 className="text-sm font-semibold">{title}</h3>
              <p className="text-xs leading-relaxed text-text-dim">{body}</p>
            </Link>
          </motion.div>
        ))}
      </section>
    </div>
  );
}
