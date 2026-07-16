"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";

/** Honest pacing: the first three stages are near-instant in the real system;
 * the rerank stage is the long one. Timings shown after completion are the
 * real recorded values — this component only paces the wait. */
const STEPS = [
  { label: "Loading your taste profile", detail: "history → 128-dim user embedding", ms: 150 },
  { label: "Searching C++ HNSW index", detail: "26,744 movies · M=32 · ef_search=100", ms: 200 },
  { label: "Retrieved 500 candidates", detail: "seen-filtered → top 25", ms: 250 },
  { label: "Reranking against your query", detail: "Cerebras gpt-oss-120b", ms: Infinity },
  { label: "Done", detail: "", ms: Infinity },
] as const;

export function LoadingSequence({ query }: { query: string }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    // Advance through the near-instant stages; stay on "Reranking" until the
    // replayed request resolves and this component unmounts.
    let cancelled = false;
    let acc = 0;
    STEPS.slice(0, 3).forEach((s, i) => {
      acc += s.ms;
      setTimeout(() => !cancelled && setStep(i + 1), acc);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto mt-16 w-full max-w-md rounded-2xl border border-border bg-card p-6">
      <p className="mb-5 truncate font-mono text-xs text-text-faint">“{query}”</p>
      <ol className="flex flex-col gap-4">
        {STEPS.slice(0, 4).map((s, i) => {
          const done = step > i;
          const active = step === i;
          return (
            <motion.li
              key={s.label}
              initial={{ opacity: 0.4 }}
              animate={{ opacity: done || active ? 1 : 0.4 }}
              className="flex items-start gap-3"
            >
              <span className="mt-0.5">
                {done ? (
                  <Check size={15} className="text-success" />
                ) : active ? (
                  <Loader2 size={15} className="animate-spin text-accent" />
                ) : (
                  <span className="block h-[15px] w-[15px] rounded-full border border-border-strong" />
                )}
              </span>
              <span>
                <span className={`block text-sm ${done || active ? "text-text" : "text-text-dim"}`}>{s.label}</span>
                {s.detail && <span className="block font-mono text-[11px] text-text-faint">{s.detail}</span>}
              </span>
            </motion.li>
          );
        })}
      </ol>
    </div>
  );
}
