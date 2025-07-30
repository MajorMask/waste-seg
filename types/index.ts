export interface WasteClassification {
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
}
