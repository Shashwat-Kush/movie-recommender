"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { RecommendResponse } from "@/lib/types";
import { fmtMs, titleOnly } from "@/lib/format";
import { AwaitingBackendBadge, MonoStat, Tooltip } from "@/components/ui";

/** Collapsible technical drawer: per-movie table, stage timings, token usage,
 * raw JSON. Anything the backend doesn't return yet is badged, never invented. */
export function DevDrawer({ response }: { response: RecommendResponse }) {
  const [open, setOpen] = useState(true);
  const { recommendations, timing, usage } = response;

  return (
    <aside className="rounded-2xl border border-border bg-card">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-3 font-mono text-xs uppercase tracking-[0.15em] text-text-dim hover:text-text"
      >
        developer drawer
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <div className="flex flex-col gap-5 border-t border-border p-4">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <MonoStat
              label="hnsw"
              value={fmtMs(timing.hnsw_ms)}
              tooltip="Wall-clock time of the C++ HNSW index search for 500 candidates."
            />
            <MonoStat
              label="rerank"
              value={fmtMs(timing.rerank_ms)}
              tooltip="Wall-clock time of the Groq llama-3.1-8b-instant rerank call, including network."
            />
            <MonoStat
              label="prompt tokens"
              value={usage ? (usage.cached ? "0 (cached)" : usage.prompt_tokens.toLocaleString()) : <AwaitingBackendBadge />}
              tooltip="Groq-reported prompt tokens for the rerank call. 0 when served from the on-disk response cache."
            />
            <MonoStat
              label="completion tokens"
              value={usage ? (usage.cached ? "0 (cached)" : usage.completion_tokens.toLocaleString()) : <AwaitingBackendBadge />}
              tooltip="Groq-reported completion tokens for the rerank call."
            />
          </div>

          <div className="overflow-x-auto scroll-thin">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border font-mono uppercase tracking-wider text-text-faint">
                  <th className="py-2 pr-3 font-normal">#</th>
                  <th className="py-2 pr-3 font-normal">movie</th>
                  <th className="py-2 pr-3 font-normal">
                    <Tooltip label="1-based position in the HNSW retrieval candidate list, before the LLM saw the query.">
                      retrieval_rank
                    </Tooltip>
                  </th>
                  <th className="py-2 pr-3 font-normal">
                    <Tooltip label="Cosine distance between your taste embedding and this movie's embedding in 128-dim space. Lower is closer.">
                      distance
                    </Tooltip>
                  </th>
                  <th className="py-2 pr-3 font-normal">
                    <Tooltip label="Preference score derived from the LLM's ranking of the 25 candidates (higher = ranked earlier). Not a probability.">
                      rerank_score
                    </Tooltip>
                  </th>
                  <th className="py-2 font-normal">reason</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {recommendations.map((r, i) => (
                  <tr key={r.movieId} className="border-b border-border/50">
                    <td className="py-1.5 pr-3 text-text-faint">{i + 1}</td>
                    <td className="max-w-48 truncate py-1.5 pr-3 font-sans text-text">{titleOnly(r.title)}</td>
                    <td className="py-1.5 pr-3">{r.retrieval_rank ?? "—"}</td>
                    <td className="py-1.5 pr-3">{r.distance?.toFixed(4) ?? "—"}</td>
                    <td className="py-1.5 pr-3">{r.rerank_score.toFixed(1)}</td>
                    <td className="py-1.5">{r.reason ?? <AwaitingBackendBadge />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <details>
            <summary className="cursor-pointer font-mono text-xs uppercase tracking-wider text-text-faint hover:text-text-dim">
              raw json response
            </summary>
            <pre className="scroll-thin mt-2 max-h-80 overflow-auto rounded-lg border border-border bg-bg p-3 font-mono text-[11px] leading-relaxed text-text-dim">
              {JSON.stringify(response, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </aside>
  );
}
