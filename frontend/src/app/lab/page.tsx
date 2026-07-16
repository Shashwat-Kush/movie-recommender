"use client";

import { motion } from "framer-motion";
import { evalMetrics } from "@/lib/api";
import projectionData from "../../../fixtures/projection.json";
import type { ProjectionPoint } from "@/lib/types";
import { fmtPct } from "@/lib/format";
import { MonoStat, SectionHeading, Tooltip } from "@/components/ui";
import { EmbeddingScatter } from "./scatter";

const projection = projectionData as unknown as ProjectionPoint[];

function PipelineDiagram() {
  const stages = [
    { title: "Watch history", detail: "20 most recent liked movies, recency-weighted" },
    { title: "User embedding", detail: "history tower MLP → 128-dim, L2-normalized" },
    { title: "C++ HNSW search", detail: "26,744 movies · top 500 candidates" },
    { title: "Filter + trim", detail: "drop already-seen → top 25" },
    { title: "LLM rerank", detail: "query + candidates → llama-3.1-8b-instant" },
    { title: "Top-k results", detail: "personalized retrieval, query-guided reranking" },
  ];
  return (
    <div className="flex flex-col gap-0">
      {stages.map((s, i) => (
        <motion.div
          key={s.title}
          initial={{ opacity: 0, x: -10 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.08 }}
          className="flex items-stretch gap-4"
        >
          <div className="flex flex-col items-center">
            <span
              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border font-mono text-xs ${
                i === 4 ? "border-accent bg-accent-faint text-accent" : "border-border bg-card text-text-dim"
              }`}
            >
              {i + 1}
            </span>
            {i < stages.length - 1 && <span className="w-px flex-1 bg-border" />}
          </div>
          <div className="pb-6">
            <p className="text-sm font-medium">{s.title}</p>
            <p className="font-mono text-xs text-text-faint">{s.detail}</p>
          </div>
        </motion.div>
      ))}
      <p className="mt-2 max-w-xl text-xs leading-relaxed text-text-faint">
        Note the query does <span className="text-text-dim">not</span> search the index — retrieval is driven
        entirely by the user embedding. The natural-language query enters at the rerank stage only.
      </p>
    </div>
  );
}

function Evaluation() {
  const { loo, cold_start, timesplit } = evalMetrics;
  const timesplitRows = Object.entries(timesplit);
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <MonoStat
          label="recall@10"
          value={fmtPct(loo["recall@10"])}
          tooltip="How often the user's held-out last-watched movie appears in the top 10 recommendations."
        />
        <MonoStat
          label="ndcg@10"
          value={fmtPct(loo["ndcg@10"])}
          tooltip="Rank-weighted hit quality: hits near the top of the list score higher."
        />
        <MonoStat
          label="popularity baseline"
          value={fmtPct(loo["popularity_recall@10"])}
          tooltip="Recall@10 of simply recommending the most-rated movies to everyone — the floor any model must beat."
        />
        <MonoStat label="users evaluated" value={loo["num_users_evaluated"].toLocaleString()} />
      </div>

      <p className="max-w-3xl text-sm leading-relaxed text-text-dim">
        These are leave-one-out numbers over all {loo["num_users_evaluated"].toLocaleString()} users: for each
        user, the model trains on their whole history except the most recent movie, then has to place that
        held-out movie in its top 10 from a 26,744-movie catalog with everything the user already watched
        masked out. Recall of {fmtPct(loo["recall@10"])} means it succeeds for roughly one user in ten —
        about twice the popularity baseline, which is the honest way to read recommender metrics: the
        absolute numbers look small because the task (one specific movie out of 26,744) is hard.
      </p>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-border bg-card p-4">
          <h4 className="mb-2 text-sm font-medium">
            <Tooltip label="20% of catalog items embedded from metadata alone (no learned ID embedding), simulating brand-new movies. Same users, same targets.">
              Cold-start items
            </Tooltip>
          </h4>
          <div className="flex gap-6 font-mono text-sm">
            <span>
              <span className="text-text-faint">cold </span>
              {fmtPct(cold_start.cold["recall@10"])}
            </span>
            <span>
              <span className="text-text-faint">warm </span>
              {fmtPct(cold_start.warm["recall@10"])}
            </span>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-text-faint">
            A movie with zero ratings gets within 1.5pt of a fully-trained one — the item tower trains with
            ID-embedding dropout so the metadata-only path works.
          </p>
        </div>
        <div className="rounded-xl border border-border bg-card p-4">
          <h4 className="mb-2 text-sm font-medium">
            <Tooltip label="Recall@10 bucketed by when the held-out movie was rated, against the time-split windows used in the data pipeline.">
              By time period
            </Tooltip>
          </h4>
          <div className="flex flex-col gap-1 font-mono text-xs">
            {timesplitRows.map(([name, m]) => (
              <span key={name} className="flex justify-between">
                <span className="text-text-faint">{name.split(" ")[0]}</span>
                <span>{fmtPct(m["recall@10"])}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function SpecSheet() {
  const { system } = evalMetrics;
  const rows: [string, string][] = [
    ["Catalog indexed", `${system.movies_indexed.toLocaleString()} movies`],
    ["Training data", `${system.ratings_trained} · ${system.users_trained.toLocaleString()} users`],
    ["Embedding dim", String(system.embedding_dim)],
    ["Retrieval model", "History-pooling two-tower (PyTorch), in-batch softmax + logQ correction"],
    ["Index", `Custom C++ HNSW · M=${system.hnsw.M} · ef_construction=${system.hnsw.ef_construction} · ef_search=${system.hnsw.ef_search}`],
    ["Candidates", `${system.candidate_pool} retrieved → ${system.rerank_candidates} reranked`],
    ["Reranker", system.reranker_model],
    ["Serving", "FastAPI · MPS (Apple Silicon)"],
  ];
  return (
    <dl className="overflow-hidden rounded-2xl border border-border">
      {rows.map(([k, v], i) => (
        <div key={k} className={`flex flex-col gap-1 px-4 py-3 sm:flex-row sm:items-baseline ${i % 2 ? "bg-card" : "bg-bg"}`}>
          <dt className="w-44 shrink-0 text-xs uppercase tracking-wider text-text-faint">{k}</dt>
          <dd className="font-mono text-sm text-text-dim">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function LabPage() {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-20 px-6 py-14">
      <section id="pipeline" className="scroll-mt-20">
        <SectionHeading
          eyebrow="pipeline"
          title="Personalized retrieval, query-guided reranking"
          sub="Two stages with different jobs: the embedding stage decides what you'd plausibly watch; the LLM stage decides what fits tonight's mood."
        />
        <PipelineDiagram />
      </section>

      <section id="embedding-space" className="scroll-mt-20">
        <SectionHeading
          eyebrow="embedding space"
          title="26,744 movies as the model sees them"
          sub="A 2D PCA projection of the item embeddings the index actually serves. Similar movies cluster; hover a point to light up its nearest neighbors."
        />
        <EmbeddingScatter points={projection} />
      </section>

      <section id="evaluation" className="scroll-mt-20">
        <SectionHeading
          eyebrow="evaluation"
          title="Measured, not vibes"
          sub="Every number below comes from the evaluation artifacts in the repo — nothing on this page is estimated."
        />
        <Evaluation />
      </section>

      <section id="system" className="scroll-mt-20">
        <SectionHeading eyebrow="system" title="Spec sheet" />
        <SpecSheet />
      </section>
    </div>
  );
}
