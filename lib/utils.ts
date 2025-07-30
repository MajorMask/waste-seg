
import { type ClassValue, clsx } from 'clsx';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function validateImageFile(file: File): { valid: boolean; error?: string } {
  if (!file.type.startsWith('image/')) {
    return { valid: false, error: 'Please select an image file' };
  }
  if (file.size > 10 * 1024 * 1024) {
    return { valid: false, error: 'Image size must be less than 10MB' };
  }
  return { valid: true };
}

export function getConfidenceLevel(confidence: number) {
  const percent = confidence * 100;
  if (percent >= 80) return { level: 'high', text: 'High Confidence', color: 'text-green-700' };
  if (percent >= 60) return { level: 'medium', text: 'Medium Confidence', color: 'text-yellow-700' };
  return { level: 'low', text: 'Low Confidence', color: 'text-red-700' };
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}

export function getCurrentLocation(): Promise<GeolocationPosition> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation is not supported'));
      return;
    }
    
    navigator.geolocation.getCurrentPosition(resolve, reject, {
      enableHighAccuracy: true,
      timeout: 5000,
      maximumAge: 0
    });
  });
}

export function extractImageMetadata(file: File): Promise<any> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      // Basic metadata extraction
      const metadata = {
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: file.lastModified,
        // Add EXIF extraction here if needed
      };
      resolve(metadata);
    };
    reader.readAsArrayBuffer(file);
  });
}