module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}`,

  ".eslintrc.json": {
    "extends": ["next/core-web-vitals", "next/typescript"],
    "rules": {
      "@typescript-eslint/no-unused-vars": "error",
      "@typescript-eslint/no-explicit-any": "warn",
      "prefer-const": "error",
      "no-var": "error"
    }
  },

  "types/index.ts": `export interface WasteClassification {
  class: string;
  confidence: number;
  all_probabilities: Record<string, number>;
}

export interface RecyclingInfo {
  recyclable: boolean;
  bin: string;
  tips: string[];
  environmental_impact: string;
}

export interface ClassificationResult {
  prediction: WasteClassification;
  recycling_info: RecyclingInfo;
  recommendations: {
    action: string;
    confidence_level: string;
    tips: string[];
  };
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface UploadedFile {
  file: File;
  preview: string;
  id: string;
}`,

  "lib/api.ts": `const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(response.status, errorData.message || 'An error occurred');
  }
  return response.json();
}

export const api = {
  async checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
    const response = await fetch(\`\${API_BASE_URL}/health\`);
    return handleResponse(response);
  },

  async classifyImage(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(\`\${API_BASE_URL}/classify\`, {
      method: 'POST',
      body: formData,
    });

    return handleResponse(response);
  },

  async getClasses(): Promise<{ classes: string[]; recycling_info: Record<string, any> }> {
    const response = await fetch(\`\${API_BASE_URL}/classes\`);
    return handleResponse(response);
  },

  async classifyBatch(files: File[]): Promise<any> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await fetch(\`\${API_BASE_URL}/classify/batch\`, {
      method: 'POST',
      body: formData,
    });

    return handleResponse(response);
  }
};
