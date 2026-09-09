import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Aitken — London PM2.5 Observatory',
  description:
    'Explore daily London PM2.5 prediction maps and the verified seven-model research benchmark.',
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
