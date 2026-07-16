import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Shell } from "@/components/shell";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const description =
  "A recommendation engine built from scratch — Two-Tower retrieval, a custom C++ HNSW index, and LLM reranking. Personalized retrieval, query-guided reranking.";

export const metadata: Metadata = {
  title: {
    default: "movie-recommender — Two-Tower retrieval + C++ HNSW + LLM reranking",
    template: "%s · movie-recommender",
  },
  description,
  openGraph: {
    title: "movie-recommender",
    description,
    type: "website",
    siteName: "movie-recommender",
  },
  twitter: {
    card: "summary_large_image",
    title: "movie-recommender",
    description,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
