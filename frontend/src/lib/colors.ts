/** One genre palette for every visualization (scatter, taste fingerprint). */
export const GENRE_COLORS: Record<string, string> = {
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

export const GENRE_FALLBACK = "#62626c";

export const genreColor = (g: string) => GENRE_COLORS[g] ?? GENRE_FALLBACK;
