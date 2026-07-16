"use client";

import { ReactNode, useId, useState } from "react";

/** Accessible hover/focus tooltip. Every metric label gets one. */
export function Tooltip({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span tabIndex={0} aria-describedby={id} className="cursor-help border-b border-dotted border-border-strong">
        {children}
      </span>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-border bg-card px-3 py-2 text-xs leading-relaxed text-text-dim shadow-xl"
        >
          {label}
        </span>
      )}
    </span>
  );
}

/** Badge for target-schema fields the backend does not return yet. */
export function AwaitingBackendBadge() {
  return (
    <span className="rounded border border-border-strong bg-card px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-text-faint">
      awaiting backend
    </span>
  );
}

export function MonoStat({
  label,
  value,
  tooltip,
}: {
  label: string;
  value: ReactNode;
  tooltip?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs uppercase tracking-wider text-text-faint">
        {tooltip ? <Tooltip label={tooltip}>{label}</Tooltip> : label}
      </span>
      <span className="font-mono text-sm text-text">{value}</span>
    </div>
  );
}

export function GenreTag({ genre }: { genre: string }) {
  return (
    <span className="rounded-md border border-border bg-bg px-2 py-0.5 text-xs text-text-dim">{genre}</span>
  );
}

export function SectionHeading({ eyebrow, title, sub }: { eyebrow?: string; title: string; sub?: string }) {
  return (
    <div className="mb-8">
      {eyebrow && <p className="mb-2 font-mono text-xs uppercase tracking-[0.2em] text-accent">{eyebrow}</p>}
      <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      {sub && <p className="mt-2 max-w-2xl text-sm leading-relaxed text-text-dim">{sub}</p>}
    </div>
  );
}
