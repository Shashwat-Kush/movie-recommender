/** Parse the year from a MovieLens title string, e.g. "Heat (1995)" -> 1995. */
export function yearOf(title: string): number | null {
  const m = title.match(/\((\d{4})\)\s*$/);
  return m ? parseInt(m[1], 10) : null;
}

/** Title without the trailing "(1995)". */
export function titleOnly(title: string): string {
  return title.replace(/\s*\(\d{4}\)\s*$/, "");
}

export function genreList(genres: string): string[] {
  if (!genres || genres === "(no genres listed)") return [];
  return genres.split("|");
}

export const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`);

export const fmtPct = (v: number, digits = 2) => `${(v * 100).toFixed(digits)}%`;
