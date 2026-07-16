"use client";

import { useSettings } from "@/lib/store";
import { SectionHeading } from "@/components/ui";

function Toggle({
  label,
  detail,
  checked,
  onChange,
}: {
  label: string;
  detail: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-6 rounded-xl border border-border bg-card p-4">
      <span>
        <span className="block text-sm font-medium">{label}</span>
        <span className="block text-xs text-text-dim">{detail}</span>
      </span>
      <button
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${checked ? "bg-accent" : "bg-border-strong"}`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
            checked ? "translate-x-[22px]" : "translate-x-0.5"
          }`}
        />
      </button>
    </label>
  );
}

export default function SettingsPage() {
  const { devMode, reduceMotion, setDevMode, setReduceMotion, reset } = useSettings();
  return (
    <div className="mx-auto max-w-xl px-6 py-14">
      <SectionHeading eyebrow="settings" title="Settings" />
      <div className="flex flex-col gap-3">
        <Toggle
          label="Developer mode"
          detail="Show retrieval ranks, distances, latencies, token usage, and raw JSON on results."
          checked={devMode}
          onChange={setDevMode}
        />
        <Toggle
          label="Reduce motion"
          detail="Disable animations and transitions app-wide."
          checked={reduceMotion}
          onChange={setReduceMotion}
        />
        <button
          onClick={reset}
          className="mt-2 self-start rounded-lg border border-border px-4 py-2 text-xs text-text-dim transition-colors hover:border-border-strong hover:text-text"
        >
          Reset to defaults
        </button>
      </div>
    </div>
  );
}
