"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Braces, FlaskConical, Home, Search, Settings, UserRound } from "lucide-react";
import { demoUsers, recordedQueries } from "@/lib/api";
import { useSettings } from "@/lib/store";

interface Command {
  id: string;
  label: string;
  hint?: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  run: () => void;
}

/** ⌘K / Ctrl-K command palette: pages, demo users, recorded queries, dev mode. */
export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const { devMode, setDevMode } = useSettings();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
        setQuery("");
        setActive(0);
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const commands = useMemo<Command[]>(
    () => [
      { id: "home", label: "Go to Recommend", icon: Home, run: () => router.push("/") },
      { id: "lab", label: "Go to AI Lab", icon: FlaskConical, run: () => router.push("/lab") },
      { id: "settings", label: "Go to Settings", icon: Settings, run: () => router.push("/settings") },
      {
        id: "dev",
        label: devMode ? "Turn developer mode off" : "Turn developer mode on",
        icon: Braces,
        run: () => setDevMode(!devMode),
      },
      ...recordedQueries.flatMap((q) =>
        demoUsers.slice(0, 2).map((u) => ({
          id: `q-${u.user_id}-${q}`,
          label: `“${q}”`,
          hint: `as user #${u.user_id}`,
          icon: Search,
          run: () => router.push(`/results?mode=warm&user=${u.user_id}&q=${encodeURIComponent(q)}`),
        })),
      ),
      ...demoUsers.map((u) => ({
        id: `u-${u.user_id}`,
        label: `Browse as user #${u.user_id}`,
        hint: `${u.num_ratings.toLocaleString()} ratings · ${Object.keys(u.genre_distribution)[0] ?? ""}`,
        icon: UserRound,
        run: () =>
          router.push(`/results?mode=warm&user=${u.user_id}&q=${encodeURIComponent(recordedQueries[0] ?? "")}`),
      })),
    ],
    [router, devMode, setDevMode],
  );

  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    return q ? commands.filter((c) => `${c.label} ${c.hint ?? ""}`.toLowerCase().includes(q)) : commands;
  }, [commands, query]);

  const runCommand = (c: Command) => {
    setOpen(false);
    c.run();
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center bg-black/50 pt-[18vh] backdrop-blur-sm"
      onClick={() => setOpen(false)}
      role="dialog"
      aria-modal
      aria-label="Command palette"
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.97, y: -8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.15 }}
        className="border-gradient glass w-full max-w-lg overflow-hidden rounded-2xl shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") setActive((a) => Math.min(a + 1, filtered.length - 1));
            if (e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
            if (e.key === "Enter" && filtered[active]) runCommand(filtered[active]);
          }}
          placeholder="Type a command, query, or user…"
          className="w-full border-b border-border bg-transparent px-4 py-3 text-sm outline-none placeholder:text-text-faint"
        />
        <div className="max-h-80 overflow-y-auto scroll-thin p-1.5">
          {filtered.map((c, i) => (
            <button
              key={c.id}
              onClick={() => runCommand(c)}
              onMouseEnter={() => setActive(i)}
              className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                i === active ? "bg-accent-faint text-text" : "text-text-dim"
              }`}
            >
              <c.icon size={14} className={i === active ? "text-accent" : "text-text-faint"} />
              <span className="min-w-0 flex-1 truncate">{c.label}</span>
              {c.hint && <span className="shrink-0 font-mono text-[10px] text-text-faint">{c.hint}</span>}
            </button>
          ))}
          {filtered.length === 0 && <p className="px-3 py-6 text-center text-xs text-text-faint">No matches.</p>}
        </div>
        <p className="border-t border-border px-4 py-2 font-mono text-[10px] text-text-faint">
          ↑↓ navigate · ⏎ run · esc close
        </p>
      </motion.div>
    </div>
  );
}
