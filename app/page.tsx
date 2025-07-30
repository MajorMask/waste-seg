'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CameraIcon, DocumentArrowUpIcon, MapPinIcon, XMarkIcon } from '@heroicons/react/24/outline';

// Since we don't have the lib files yet, let's include the essential functions here
const API_BASE_URL = 'http://localhost:8000';

interface ClassificationResult {
  prediction: {
    class: string;
    confidence: number;
    all_probabilities: Record<string, number>;
  };
  recycling_info: {
    recyclable: boolean;
    bin: string;
    tips: string[];
    environmental_impact: string;
  };
  recommendations: {
    action: string;
    confidence_level: string;
    tips: string[];
  };
}

// API functions
const api = {
  async checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('API not available');
    return response.json();
  },

  async classifyImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/classify`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Classification failed');
    return response.json();
  }
};

// Utility functions
function validateImageFile(file: File): { valid: boolean; error?: string } {
  if (!file.type.startsWith('image/')) {
    return { valid: false, error: 'Please select an image file' };
  }
  if (file.size > 10 * 1024 * 1024) {
    return { valid: false, error: 'Image size must be less than 10MB' };
  }
  return { valid: true };
}

function getConfidenceLevel(confidence: number) {
  const percent = confidence * 100;
  if (percent >= 80) return { level: 'high', text: 'High Confidence', color: 'text-green-700' };
  if (percent >= 60) return { level: 'medium', text: 'Medium Confidence', color: 'text-yellow-700' };
  return { level: 'low', text: 'Low Confidence', color: 'text-red-700' };
}

function getCurrentLocation(): Promise<GeolocationPosition> {
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

export default function HomePage() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [location, setLocation] = useState<GeolocationPosition | null>(null);
  const [locationEnabled, setLocationEnabled] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Check API status
    api.checkHealth()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));

    // Get location permission
    getCurrentLocation()
      .then((pos) => {
        setLocation(pos);
        setLocationEnabled(true);
      })
      .catch(() => setLocationEnabled(false));
  }, []);

  const handleFile = (file: File) => {
    const validation = validateImageFile(file);
    if (!validation.valid) {
      setError(validation.error!);
      return;
    }

    setImage(file);
    setError(null);
    setResult(null);
    
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const classify = async () => {
    if (!image) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await api.classifyImage(image);
      setResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  const openCamera = () => {
    if (fileRef.current) {
      fileRef.current.setAttribute('capture', 'environment');
      fileRef.current.click();
    }
  };

  const removeImage = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileRef.current) {
      fileRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-green-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <motion.header 
          className="text-center mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-green-600 to-green-500 rounded-xl flex items-center justify-center text-white font-bold text-xl shadow-lg">
              WS
            </div>
            <h1 className="text-4xl font-bold text-gray-900">WasteSort AI</h1>
          </div>
          <p className="text-xl text-gray-600 mb-2">Intelligent Waste Classification</p>
          <div className="flex items-center justify-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${
              apiStatus === 'online' ? 'bg-green-500 animate-pulse' : 
              apiStatus === 'offline' ? 'bg-red-500' : 'bg-gray-400'
            }`}></div>
            <span className="text-gray-600">
              {apiStatus === 'online' ? 'System Online' : 
               apiStatus === 'offline' ? 'System Offline' : 'Checking...'}
            </span>
            {locationEnabled && (
              <>
                <span className="mx-2">•</span>
                <MapPinIcon className="w-4 h-4 text-gray-500" />
                <span className="text-gray-500">Location enabled</span>
              </>
            )}
          </div>
        </motion.header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <motion.div 
            className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <h2 className="text-2xl font-semibold mb-6">Classify Waste Item</h2>
            
            <div 
              className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center cursor-pointer hover:border-green-400 hover:bg-green-50 transition-all duration-300"
              onClick={() => fileRef.current?.click()}
            >
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <DocumentArrowUpIcon className="w-6 h-6 text-gray-600" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Upload Image</h3>
              <p className="text-gray-500 mb-6">Click to upload or drag and drop your image</p>
              
              <div className="flex gap-3 justify-center">
                <button 
                  onClick={(e) => { e.stopPropagation(); openCamera(); }} 
                  className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-all duration-200 hover:-translate-y-0.5 shadow-md hover:shadow-lg"
                >
                  <CameraIcon className="w-4 h-4" />
                  Take Photo
                </button>
                <button 
                  onClick={(e) => { e.stopPropagation(); fileRef.current?.click(); }} 
                  className="flex items-center gap-2 px-6 py-3 bg-white text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-all duration-200"
                >
                  <DocumentArrowUpIcon className="w-4 h-4" />
                  Browse Files
                </button>
              </div>
            </div>

            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              className="hidden"
            />

            <AnimatePresence>
              {preview && (
                <motion.div 
                  className="mt-6"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                >
                  <div className="relative rounded-xl overflow-hidden shadow-lg">
                    <img src={preview} alt="Preview" className="w-full h-64 object-cover" />
                    <button
                      onClick={removeImage}
                      className="absolute top-3 right-3 w-8 h-8 bg-white/90 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white transition-all duration-200 hover:scale-110 shadow-md"
                    >
                      <XMarkIcon className="w-4 h-4 text-gray-600" />
                    </button>
                  </div>
                  <button
                    onClick={classify}
                    disabled={loading}
                    className="w-full mt-4 px-6 py-4 bg-green-600 text-white rounded-lg font-semibold disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-green-700 transition-all duration-200 hover:-translate-y-1 shadow-lg hover:shadow-xl"
                  >
                    {loading ? 'Analyzing...' : 'Analyze Image'}
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Results Section */}
          <motion.div 
            className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 min-h-96"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h2 className="text-2xl font-semibold mb-6">Classification Results</h2>
            
            <AnimatePresence mode="wait">
              {error && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg"
                >
                  {error}
                </motion.div>
              )}

              {loading && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="flex flex-col items-center py-12"
                >
                  <div className="w-10 h-10 border-3 border-gray-200 border-t-green-600 rounded-full animate-spin mb-4"></div>
                  <p className="text-gray-600">Analyzing image...</p>
                </motion.div>
              )}

              {result && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  <div className="bg-green-50 border border-green-200 p-6 rounded-lg">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-2xl font-bold capitalize text-gray-900">
                        {result.prediction.class}
                      </h3>
                      <span className={`px-3 py-1 rounded-md text-white text-xs font-semibold uppercase tracking-wide ${
                        getConfidenceLevel(result.prediction.confidence).level === 'high' ? 'bg-green-500' :
                        getConfidenceLevel(result.prediction.confidence).level === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                      }`}>
                        {getConfidenceLevel(result.prediction.confidence).text}
                      </span>
                    </div>
                    
                    <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                      <motion.div
                        className="bg-gradient-to-r from-green-600 to-green-400 h-3 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.prediction.confidence * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                      />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-gray-500 uppercase tracking-wide font-semibold text-xs mb-1">
                          Confidence
                        </div>
                        <div className="font-bold text-gray-900 text-lg">
                          {(result.prediction.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-500 uppercase tracking-wide font-semibold text-xs mb-1">
                          Category
                        </div>
                        <div className="font-bold text-gray-900 text-lg capitalize">
                          {result.prediction.class}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-6 border-l-4 border-green-500">
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <div className="text-gray-500 uppercase tracking-wide font-semibold text-xs mb-1">
                          Disposal Method
                        </div>
                        <div className="font-semibold text-gray-900">
                          {result.recycling_info.bin}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-500 uppercase tracking-wide font-semibold text-xs mb-1">
                          Recyclable
                        </div>
                        <div className={`font-semibold ${
                          result.recycling_info.recyclable ? 'text-green-700' : 'text-red-700'
                        }`}>
                          {result.recycling_info.recyclable ? 'Yes' : 'No'}
                        </div>
                      </div>
                    </div>
                    
                    {result.recycling_info.tips && result.recycling_info.tips.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-gray-900 mb-3">
                          Disposal Instructions
                        </div>
                        <div className="space-y-2">
                          {result.recycling_info.tips.map((tip, index) => (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: 0.1 * index }}
                              className="bg-white p-3 rounded-md border-l-3 border-green-400 text-sm text-gray-700 shadow-sm"
                            >
                              {tip}
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {result.recycling_info.environmental_impact && (
                      <div>
                        <div className="text-sm font-semibold text-gray-900 mb-3">
                          Environmental Impact
                        </div>
                        <div className="bg-white p-4 rounded-md border-l-3 border-green-400 text-sm text-gray-700 shadow-sm">
                          {result.recycling_info.environmental_impact}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {!result && !error && !loading && (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center justify-center h-full py-12 text-center text-gray-400"
                >
                  <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-600 mb-2">Ready to Analyze</h3>
                  <p className="text-sm max-w-sm">
                    Upload an image to see classification results and recycling information
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </div>
  );
}