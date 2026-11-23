import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BananaSense — Smart Ripeness Scanner",
  description: "Klasifikasi kematangan pisang menggunakan CNN.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}
