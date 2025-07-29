import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'WasteSort AI - Intelligent Waste Classification',
  description: 'AI-powered waste sorting and recycling classification using computer vision',
  keywords: ['AI', 'waste classification', 'recycling', 'computer vision', 'sustainability'],
  authors: [{ name: 'WasteSort AI Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#16a34a',
  openGraph: {
    title: 'WasteSort AI - Intelligent Waste Classification',
    description: 'AI-powered waste sorting and recycling classification using computer vision',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'WasteSort AI',
    description: 'AI-powered waste sorting and recycling classification',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {children}
      </body>
    </html>
  );
}`,

  "app/globals.css": `@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  * {
    @apply border-gray-200;
  }
  
  body {
    @apply bg-gradient-to-br from-gray-50 to-primary-50 text-gray-900;
  }
}

@layer components {
  .animate-fade-in {
    animation: fadeIn 0.5s ease-in-out;
  }
  
  .animate-slide-up {
    animation: slideUp 0.3s ease-out;
  }
  
  .animate-scale-in {
    animation: scaleIn 0.2s ease-out;
  }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from { 
    transform: translateY(10px);
    opacity: 0;
  }
  to { 
    transform: translateY(0);
    opacity: 1;
  }
}

@keyframes scaleIn {
  from { 
    transform: scale(0.95);
    opacity: 0;
  }
  to { 
    transform: scale(1);
    opacity: 1;
  }
}
