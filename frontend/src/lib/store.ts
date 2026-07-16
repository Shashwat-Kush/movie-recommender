"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
  devMode: boolean;
  reduceMotion: boolean;
  setDevMode: (v: boolean) => void;
  setReduceMotion: (v: boolean) => void;
  reset: () => void;
}

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      devMode: false,
      reduceMotion: false,
      setDevMode: (v) => set({ devMode: v }),
      setReduceMotion: (v) => set({ reduceMotion: v }),
      reset: () => set({ devMode: false, reduceMotion: false }),
    }),
    { name: "movie-recommender-settings" },
  ),
);
