import { type ClassValue, clsx } from 'clsx';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}

export function validateImageFile(file: File): { valid: boolean; error?: string } {
  if (!file.type.startsWith('image/')) {
    return { valid: false, error: 'Please select an image file' };
  }
  
  if (file.size > 10 * 1024 * 1024) { // 10MB
    return { valid: false, error: 'Image size must be less than 10MB' };
  }
  
  return { valid: true };
}

export function getConfidenceLevel(confidence: number): {
  level: 'high' | 'medium' | 'low';
  text: string;
  color: string;
} {
  const percent = confidence * 100;
  
  if (percent >= 80) {
    return { level: 'high', text: 'High Confidence', color: 'text-green-700 bg-green-100' };
  }
  
  if (percent >= 60) {
    return { level: 'medium', text: 'Medium Confidence', color: 'text-yellow-700 bg-yellow-100' };
  }
  
  return { level: 'low', text: 'Low Confidence', color: 'text-red-700 bg-red-100' };
}

export function trackEvent(eventName: string, properties?: Record<string, any>) {
  // Add your analytics tracking here
  if (typeof window !== 'undefined') {
    console.log('Event tracked:', eventName, properties);
    
    // Example: Google Analytics 4
    if (window.gtag) {
      window.gtag('event', eventName, properties);
    }
    
    // Example: Custom analytics
    if (window.analytics) {
      window.analytics.track(eventName, properties);
    }
  }
}`,

  "components/ui/Button.tsx": `import React from 'react';
import { cn } from '@/lib/utils';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading, icon, children, disabled, ...props }, ref) => {
    const baseClasses = 'inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
    
    const variants = {
      primary: 'bg-primary-600 text-white hover:bg-primary-700 focus:ring-primary-500 shadow-md hover:shadow-lg hover:-translate-y-0.5',
      secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-500',
      outline: 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 focus:ring-gray-500',
      ghost: 'text-gray-700 hover:bg-gray-100 focus:ring-gray-500',
    };
    
    const sizes = {
      sm: 'px-3 py-1.5 text-sm gap-1.5',
      md: 'px-4 py-2 text-sm gap-2',
      lg: 'px-6 py-3 text-base gap-3',
    };
    
    return (
      <button
        ref={ref}
        className={cn(
          baseClasses,
          variants[variant],
          sizes[size],
          loading && 'pointer-events-none',
          className
        )}
        disabled={disabled || loading}
        {...props}
      >
        {loading && (
          <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        )}
        {!loading && icon && icon}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;`,

  "components/ui/Card.tsx": `import React from 'react';
import { cn } from '@/lib/utils';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'outlined' | 'elevated';
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', ...props }, ref) => {
    const variants = {
      default: 'bg-white border border-gray-200',
      outlined: 'bg-white border-2 border-gray-300',
      elevated: 'bg-white shadow-lg border border-gray-200',
    };
    
    return (
      <div
        ref={ref}
        className={cn(
          'rounded-2xl p-6',
          variants[variant],
          className
        )}
        {...props}
      />
    );
  }
);

Card.displayName = 'Card';

export default Card;
