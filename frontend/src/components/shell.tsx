"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode, useEffect } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { FlaskConical, Settings, Film, Braces } from "lucide-react";

function GithubIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}
import { useSettings } from "@/lib/store";

const REPO_URL = "https://github.com/Shashwat-Kush/movie-recommender";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
});

function DemoBanner() {
  return (
    <div className="border-b border-border bg-card/60 px-4 py-1.5 text-center text-xs text-text-dim">
      Demo — replaying recorded responses from the real system.{" "}
      <a href={REPO_URL} className="text-accent underline-offset-2 hover:underline" target="_blank" rel="noreferrer">
        Code on GitHub
      </a>
    </div>
  );
}

function Nav() {
  const pathname = usePathname();
  const { devMode, setDevMode } = useSettings();
  const links = [
    { href: "/", label: "Recommend", icon: Film },
    { href: "/lab", label: "AI Lab", icon: FlaskConical },
    { href: "/settings", label: "Settings", icon: Settings },
  ];
  return (
    <nav className="sticky top-0 z-40 border-b border-border bg-bg/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-accent-faint font-mono text-xs text-accent">
            ▶
          </span>
          movie-recommender
        </Link>
        <div className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition-colors ${
                pathname === href ? "bg-card text-text" : "text-text-dim hover:bg-card hover:text-text"
              }`}
            >
              <Icon size={14} />
              {label}
            </Link>
          ))}
          <button
            onClick={() => setDevMode(!devMode)}
            aria-pressed={devMode}
            title="Developer mode: show ranks, distances, latencies, and raw JSON"
            className={`ml-2 flex items-center gap-1.5 rounded-lg border px-3 py-1.5 font-mono text-xs transition-colors ${
              devMode
                ? "border-accent bg-accent-faint text-accent"
                : "border-border text-text-dim hover:border-border-strong hover:text-text"
            }`}
          >
            <Braces size={13} />
            dev
          </button>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="ml-1 rounded-lg p-2 text-text-dim transition-colors hover:bg-card hover:text-text"
            aria-label="GitHub repository"
          >
            <GithubIcon size={16} />
          </a>
        </div>
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="mt-24 border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-6 py-8 text-xs text-text-faint sm:flex-row sm:items-center sm:justify-between">
        <div className="flex gap-4">
          <a href={REPO_URL} className="hover:text-text-dim" target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a href={`${REPO_URL}/blob/main/ARCHITECTURE.md`} className="hover:text-text-dim" target="_blank" rel="noreferrer">
            ARCHITECTURE.md
          </a>
        </div>
        <p className="font-mono">
          PyTorch two-tower · custom C++ HNSW · Cerebras gpt-oss-120b · FastAPI · Next.js
        </p>
      </div>
    </footer>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const reduceMotion = useSettings((s) => s.reduceMotion);
  useEffect(() => {
    document.documentElement.dataset.reduceMotion = String(reduceMotion);
  }, [reduceMotion]);
  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex min-h-screen flex-col">
        <DemoBanner />
        <Nav />
        <main className="flex-1">{children}</main>
        <Footer />
      </div>
    </QueryClientProvider>
  );
}
