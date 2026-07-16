"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ProjectionPoint } from "@/lib/types";
import { titleOnly } from "@/lib/format";

const GENRE_COLORS: Record<string, string> = {
  Action: "#8b5cf6",
  Adventure: "#a78bfa",
  Animation: "#f0abfc",
  Comedy: "#fbbf24",
  Crime: "#f87171",
  Documentary: "#94a3b8",
  Drama: "#60a5fa",
  Fantasy: "#c4b5fd",
  Horror: "#fb7185",
  Romance: "#f9a8d4",
  "Sci-Fi": "#34d399",
  Thriller: "#fb923c",
};
const FALLBACK = "#62626c";

/** Canvas scatter of the 2D PCA projection of the served model's item
 * embeddings. Hover highlights the point's nearest neighbors in the
 * projection; the search box jumps to a title. */
export function EmbeddingScatter({ points }: { points: ProjectionPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<ProjectionPoint | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<ProjectionPoint | null>(null);

  const bounds = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }, [points]);

  const W = 860;
  const H = 520;
  const PAD = 24;
  const px = (p: ProjectionPoint) => PAD + ((p.x - bounds.minX) / (bounds.maxX - bounds.minX)) * (W - 2 * PAD);
  const py = (p: ProjectionPoint) => PAD + ((p.y - bounds.minY) / (bounds.maxY - bounds.minY)) * (H - 2 * PAD);

  const focus = hover ?? selected;

  const neighbors = useMemo(() => {
    if (!focus) return new Set<number>();
    const d = points
      .filter((p) => p.movieId !== focus.movieId)
      .map((p) => ({ id: p.movieId, d: (p.x - focus.x) ** 2 + (p.y - focus.y) ** 2 }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 8);
    return new Set(d.map((x) => x.id));
  }, [focus, points]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    for (const p of points) {
      const isFocus = focus?.movieId === p.movieId;
      const isNeighbor = neighbors.has(p.movieId);
      ctx.beginPath();
      ctx.arc(px(p), py(p), isFocus ? 5 : isNeighbor ? 3.5 : 2, 0, Math.PI * 2);
      ctx.fillStyle = GENRE_COLORS[p.genre] ?? FALLBACK;
      ctx.globalAlpha = focus ? (isFocus ? 1 : isNeighbor ? 0.9 : 0.12) : 0.55;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, focus, neighbors]);

  const onMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * W;
    const my = ((e.clientY - rect.top) / rect.height) * H;
    let best: ProjectionPoint | null = null;
    let bestD = 12 ** 2;
    for (const p of points) {
      const d = (px(p) - mx) ** 2 + (py(p) - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    setHover(best);
  };

  const matches = search.length >= 2
    ? points.filter((p) => p.title.toLowerCase().includes(search.toLowerCase())).slice(0, 6)
    : [];

  return (
    <div className="rounded-2xl border border-border bg-card p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="relative">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="find a movie…"
            aria-label="Search embedding space"
            className="w-56 rounded-lg border border-border bg-bg px-3 py-1.5 text-xs outline-none placeholder:text-text-faint focus:border-border-strong"
          />
          {matches.length > 0 && (
            <div className="absolute top-full z-30 mt-1 w-72 rounded-lg border border-border bg-card p-1 shadow-xl">
              {matches.map((m) => (
                <button
                  key={m.movieId}
                  onClick={() => {
                    setSelected(m);
                    setSearch("");
                  }}
                  className="block w-full truncate rounded-md px-2 py-1.5 text-left text-xs text-text-dim hover:bg-bg hover:text-text"
                >
                  {m.title}
                </button>
              ))}
            </div>
          )}
        </div>
        <p className="font-mono text-[11px] text-text-faint">
          {focus ? `${titleOnly(focus.title)} · ${focus.genre}` : `${points.length.toLocaleString()} most-rated movies · PCA of 128-dim item embeddings`}
        </p>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", aspectRatio: `${W}/${H}` }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label="2D projection of movie embeddings, colored by primary genre"
      />
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
        {Object.entries(GENRE_COLORS).map(([g, c]) => (
          <span key={g} className="flex items-center gap-1 text-[10px] text-text-faint">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: c }} />
            {g}
          </span>
        ))}
      </div>
    </div>
  );
}
