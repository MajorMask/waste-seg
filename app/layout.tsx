
import './globals.css';
import type { Metadata, Viewport } from 'next';

export const metadata: Metadata = {
  title: 'WasteSort AI - Intelligent Waste Classification',
  description: 'AI-powered waste sorting and recycling classification using computer vision',
  keywords: ['AI', 'waste classification', 'recycling', 'computer vision', 'sustainability'],
  metadataBase: new URL('http://localhost:3000'),
  openGraph: {
    title: 'WasteSort AI',
    description: 'AI-powered waste sorting and recycling classification',
    type: 'website',
    images: ['/og-image.jpg'],
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#16a34a',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}