"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ProjectionPoint } from "@/lib/types";
import { titleOnly } from "@/lib/format";
import { useSettings } from "@/lib/store";

import { GENRE_COLORS, GENRE_FALLBACK as FALLBACK } from "@/lib/colors";

const W = 860;
const H = 540;
const FOCAL = 2.6; // perspective strength

/** The catalog as a galaxy: a rotating 3D point cloud of the served model's
 * item embeddings (3-component PCA). Drag to rotate, hover for titles and
 * nearest neighbors, search to fly to a movie. Pure canvas — no 3D deps. */
export function EmbeddingScatter({ points }: { points: ProjectionPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<ProjectionPoint | null>(null);
  const [selected, setSelected] = useState<ProjectionPoint | null>(null);
  const [search, setSearch] = useState("");
  const reduceMotion = useSettings((s) => s.reduceMotion);

  // Rotation state lives in refs so the RAF loop never re-renders React.
  const rot = useRef({ yaw: 0.6, pitch: 0.25 });
  const targetRot = useRef<{ yaw: number; pitch: number } | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const projected = useRef<Float32Array>(new Float32Array(0));

  // Normalize coordinates to a unit-ish cube once.
  const cloud = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const zs = points.map((p) => p.z ?? 0);
    const span = (a: number[]) => Math.max(...a) - Math.min(...a) || 1;
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
    const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const cz = (Math.max(...zs) + Math.min(...zs)) / 2;
    const s = 2 / Math.max(span(xs), span(ys), span(zs));
    return points.map((p) => ({
      ...p,
      nx: (p.x - cx) * s,
      ny: (p.y - cy) * s,
      nz: ((p.z ?? 0) - cz) * s,
      color: GENRE_COLORS[p.genre] ?? FALLBACK,
    }));
  }, [points]);

  const focus = hover ?? selected;
  const neighborIds = useMemo(() => {
    if (!focus) return new Set<number>();
    const f = cloud.find((p) => p.movieId === focus.movieId);
    if (!f) return new Set<number>();
    return new Set(
      cloud
        .filter((p) => p.movieId !== f.movieId)
        .map((p) => ({ id: p.movieId, d: (p.nx - f.nx) ** 2 + (p.ny - f.ny) ** 2 + (p.nz - f.nz) ** 2 }))
        .sort((a, b) => a.d - b.d)
        .slice(0, 10)
        .map((x) => x.id),
    );
  }, [focus, cloud]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;

    projected.current = new Float32Array(cloud.length * 3);
    let raf = 0;

    const draw = () => {
      // Ease toward a fly-to target; otherwise idle-spin (unless dragging/reduced).
      if (targetRot.current) {
        const t = targetRot.current;
        rot.current.yaw += (t.yaw - rot.current.yaw) * 0.08;
        rot.current.pitch += (t.pitch - rot.current.pitch) * 0.08;
        if (Math.abs(t.yaw - rot.current.yaw) + Math.abs(t.pitch - rot.current.pitch) < 0.002) {
          targetRot.current = null;
        }
      } else if (!drag.current && !reduceMotion) {
        rot.current.yaw += 0.0016;
      }

      const { yaw, pitch } = rot.current;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // Depth-sort indices each frame (cheap at 3k points)
      const order: number[] = [];
      for (let i = 0; i < cloud.length; i++) {
        const p = cloud[i];
        const x1 = p.nx * cy + p.nz * sy;
        const z1 = -p.nx * sy + p.nz * cy;
        const y2 = p.ny * cp - z1 * sp;
        const z2 = p.ny * sp + z1 * cp;
        const scale = FOCAL / (FOCAL + z2);
        const px = W / 2 + x1 * scale * (H / 2.6);
        const py = H / 2 + y2 * scale * (H / 2.6);
        projected.current[i * 3] = px;
        projected.current[i * 3 + 1] = py;
        projected.current[i * 3 + 2] = z2;
        order.push(i);
      }
      order.sort((a, b) => projected.current[b * 3 + 2] - projected.current[a * 3 + 2]);

      for (const i of order) {
        const p = cloud[i];
        const px = projected.current[i * 3];
        const py = projected.current[i * 3 + 1];
        const z = projected.current[i * 3 + 2];
        const depth = (1 - z / 1.6) / 2 + 0.5; // ~[0.5, 1] front boost
        const isFocus = focus?.movieId === p.movieId;
        const isNeighbor = neighborIds.has(p.movieId);

        ctx.beginPath();
        ctx.arc(px, py, isFocus ? 5.5 : isNeighbor ? 3.5 : 1.6 * depth + 0.6, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = focus ? (isFocus ? 1 : isNeighbor ? 0.95 : 0.07) : 0.28 + 0.4 * depth;
        ctx.fill();

        if (isFocus) {
          ctx.globalAlpha = 0.9;
          ctx.strokeStyle = p.color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(px, py, 9, 0, Math.PI * 2);
          ctx.stroke();
          ctx.font = "11px var(--font-mono, monospace)";
          ctx.fillStyle = "#ededf0";
          ctx.fillText(titleOnly(p.title), Math.min(px + 12, W - 160), py - 8);
        }
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [cloud, focus, neighborIds, reduceMotion]);

  const canvasPos = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: ((e.clientX - rect.left) / rect.width) * W, y: ((e.clientY - rect.top) / rect.height) * H };
  };

  const onMove = (e: React.MouseEvent) => {
    if (drag.current) {
      const dx = e.clientX - drag.current.x;
      const dy = e.clientY - drag.current.y;
      drag.current = { x: e.clientX, y: e.clientY };
      rot.current.yaw += dx * 0.005;
      rot.current.pitch = Math.max(-1.2, Math.min(1.2, rot.current.pitch + dy * 0.005));
      targetRot.current = null;
      return;
    }
    const { x: mx, y: my } = canvasPos(e);
    let best: ProjectionPoint | null = null;
    let bestD = 11 ** 2;
    for (let i = 0; i < cloud.length; i++) {
      const d = (projected.current[i * 3] - mx) ** 2 + (projected.current[i * 3 + 1] - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = cloud[i];
      }
    }
    setHover(best);
  };

  const flyTo = (p: ProjectionPoint) => {
    const c = cloud.find((q) => q.movieId === p.movieId);
    if (!c) return;
    // Rotate so the point faces the camera (z minimal): solve yaw/pitch.
    const yaw = Math.atan2(-c.nx, -c.nz);
    const r = Math.hypot(c.nx, c.nz);
    const pitch = Math.atan2(c.ny, r);
    targetRot.current = { yaw, pitch };
    setSelected(p);
    setSearch("");
  };

  const matches =
    search.length >= 2 ? cloud.filter((p) => p.title.toLowerCase().includes(search.toLowerCase())).slice(0, 6) : [];

  return (
    <div className="border-gradient glass rounded-2xl p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="relative">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="fly to a movie…"
            aria-label="Search embedding space"
            className="w-56 rounded-lg border border-border bg-bg px-3 py-1.5 text-xs outline-none placeholder:text-text-faint focus:border-accent"
          />
          {matches.length > 0 && (
            <div className="glass absolute top-full z-30 mt-1 w-72 rounded-lg border border-border p-1 shadow-xl">
              {matches.map((m) => (
                <button
                  key={m.movieId}
                  onClick={() => flyTo(m)}
                  className="block w-full truncate rounded-md px-2 py-1.5 text-left text-xs text-text-dim hover:bg-bg hover:text-text"
                >
                  {m.title}
                </button>
              ))}
            </div>
          )}
        </div>
        <p className="font-mono text-[11px] text-text-faint">
          {focus
            ? `${titleOnly(focus.title)} · ${focus.genre}`
            : `${points.length.toLocaleString()} movies · 3D PCA of 128-dim item embeddings · drag to rotate`}
        </p>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", aspectRatio: `${W}/${H}`, cursor: drag.current ? "grabbing" : "grab" }}
        onMouseMove={onMove}
        onMouseLeave={() => {
          setHover(null);
          drag.current = null;
        }}
        onMouseDown={(e) => (drag.current = { x: e.clientX, y: e.clientY })}
        onMouseUp={() => (drag.current = null)}
        role="img"
        aria-label="Rotating 3D projection of movie embeddings, colored by primary genre"
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
