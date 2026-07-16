"use client";

import { motion } from "framer-motion";
import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { RecommendResponse } from "@/lib/types";
import { titleOnly, yearOf } from "@/lib/format";
import { Tooltip } from "@/components/ui";

const ROW_H = 46; // px, fixed so connecting-line endpoints are exact
const GAP = 6;

/** The retrieve-then-rerank visualization: HNSW candidates on the left in
 * retrieval order, the LLM's final order on the right, movement lines between.
 * Every number is real recorded output. */
export function RankFlow({ response }: { response: RecommendResponse }) {
  const { candidates, recommendations } = response;
  const y = (i: number) => i * (ROW_H + GAP) + ROW_H / 2;
  const leftIndex = new Map(candidates.map((c, i) => [c.movieId, i]));

  const lines = recommendations
    .filter((r) => leftIndex.has(r.movieId))
    .map((r, rightI) => {
      const leftI = leftIndex.get(r.movieId)!;
      return { key: r.movieId, y1: y(leftI), y2: y(rightI), moved: leftI - rightI };
    });

  const svgH = Math.max(candidates.length, recommendations.length) * (ROW_H + GAP);

  return (
    <div className="grid grid-cols-[1fr_72px_1fr] gap-0">
      {/* Left: retrieval order */}
      <div>
        <h3 className="mb-3 font-mono text-xs uppercase tracking-[0.15em] text-text-faint">
          <Tooltip label="Candidates in the order the HNSW index returned them (after popularity blending and seen-movie filtering). Personalized by the user embedding alone — the query hasn't entered yet.">
            HNSW retrieval
          </Tooltip>{" "}
          · {candidates.length}
        </h3>
        <div className="flex flex-col" style={{ gap: GAP }}>
          {candidates.map((c, i) => {
            const kept = recommendations.some((r) => r.movieId === c.movieId);
            return (
              <motion.div
                key={c.movieId}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: kept ? 1 : 0.45, x: 0 }}
                transition={{ delay: i * 0.015, duration: 0.2 }}
                style={{ height: ROW_H }}
                className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card px-3"
              >
                <div className="flex min-w-0 items-center gap-2.5">
                  <span className="w-6 shrink-0 text-right font-mono text-xs text-text-faint">{c.retrieval_rank}</span>
                  <span className="truncate text-sm">{titleOnly(c.title)}</span>
                </div>
                <span className="shrink-0 font-mono text-[11px] text-text-faint" title="cosine distance (lower = closer)">
                  {c.distance.toFixed(3)}
                </span>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Middle: movement lines */}
      <svg width="72" height={svgH} className="mt-9 shrink-0" aria-hidden>
        {lines.map((l, i) => (
          <motion.path
            key={l.key}
            d={`M 0 ${l.y1} C 36 ${l.y1}, 36 ${l.y2}, 72 ${l.y2}`}
            fill="none"
            stroke={l.moved > 0 ? "#8b5cf6" : l.moved < 0 ? "#62626c" : "#3a3a42"}
            strokeWidth={1.5}
            strokeOpacity={0.8}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ delay: 0.4 + i * 0.05, duration: 0.4 }}
          />
        ))}
      </svg>

      {/* Right: after LLM rerank */}
      <div>
        <h3 className="mb-3 font-mono text-xs uppercase tracking-[0.15em] text-text-faint">
          <Tooltip label="Final order after Groq llama-3.1-8b-instant ranked the 25 candidates against your query.">
            after LLM rerank
          </Tooltip>{" "}
          · {recommendations.length}
        </h3>
        <div className="flex flex-col" style={{ gap: GAP }}>
          {recommendations.map((r, i) => {
            const moved = (r.retrieval_rank ?? i + 1) - (i + 1);
            return (
              <motion.div
                key={r.movieId}
                initial={{ opacity: 0, x: 6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + i * 0.05, duration: 0.2 }}
                style={{ height: ROW_H }}
                className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card px-3 hover:border-border-strong"
              >
                <div className="flex min-w-0 items-center gap-2.5">
                  <span className="w-6 shrink-0 text-right font-mono text-xs text-accent">{i + 1}</span>
                  <span className="truncate text-sm">{titleOnly(r.title)}</span>
                  <span className="shrink-0 font-mono text-[11px] text-text-faint">{yearOf(r.title) ?? ""}</span>
                </div>
                <span
                  className={`flex shrink-0 items-center gap-0.5 font-mono text-[11px] ${
                    moved > 0 ? "text-success" : moved < 0 ? "text-error" : "text-text-faint"
                  }`}
                  title={`Moved ${moved > 0 ? "up" : moved < 0 ? "down" : "0"} from retrieval position ${r.retrieval_rank}`}
                >
                  {moved > 0 ? <ArrowUp size={11} /> : moved < 0 ? <ArrowDown size={11} /> : <Minus size={11} />}
                  {moved !== 0 && Math.abs(moved)}
                </span>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
