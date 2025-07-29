'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CameraIcon, 
  DocumentArrowUpIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowUpTrayIcon
} from '@heroicons/react/24/outline';

import Button from '@/components/ui/Button';
import Card from '@/components/ui/Card';
import { api, ApiError } from '@/lib/api';
import { validateImageFile, getConfidenceLevel, trackEvent, generateId } from '@/lib/utils';
import type { ClassificationResult, UploadedFile } from '@/types';

const WasteClassificationApp: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [dragActive, setDragActive] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);
  
  // Check API status on mount
  useEffect(() => {
    checkApiStatus();
  }, []);
  
  const checkApiStatus = async () => {
    try {
      await api.checkHealth();
      setApiStatus('online');
    } catch (error) {
      setApiStatus('offline');
    }
  };
  
  const handleFileSelect = useCallback((file: File) => {
    const validation = validateImageFile(file);
    
    if (!validation.valid) {
      setError(validation.error!);
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const newFile: UploadedFile = {
        file,
        preview: e.target?.result as string,
        id: generateId()
      };
      
      setUploadedFile(newFile);
      setError(null);
      setResults(null);
      
      trackEvent('image_uploaded', {
        fileSize: file.size,
        fileType: file.type,
        fileName: file.name
      });
    };
    
    reader.readAsDataURL(file);
  }, []);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    dragCounter.current = 0;
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);
  
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current++;
    if (dragCounter.current === 1) {
      setDragActive(true);
    }
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setDragActive(false);
    }
  }, []);
  
  const openFileDialog = () => {
    fileInputRef.current?.click();
  };
  
  const openCamera = () => {
    if (fileInputRef.current) {
      fileInputRef.current.setAttribute('capture', 'environment');
      fileInputRef.current.click();
    }
  };
  
  const removeImage = () => {
    setUploadedFile(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    trackEvent('image_removed');
  };
  
  const classifyImage = async () => {
    if (!uploadedFile) {
      setError('Please select an image first');
      return;
    }
    
    const startTime = performance.now();
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await api.classifyImage(uploadedFile.file);
      const endTime = performance.now();
      const processingTime = Math.round(endTime - startTime);
      
      setResults(result);
      setProcessingTime(processingTime);
      
      trackEvent('image_classified', {
        wasteClass: result.prediction.class,
        confidence: result.prediction.confidence,
        processingTime,
        recyclable: result.recycling_info.recyclable
      });
    } catch (error) {
      console.error('Classification error:', error);
      
      if (error instanceof ApiError) {
        setError(`Classification failed: ${error.message}`);
      } else {
        setError('Failed to classify image. Please check your connection and try again.');
      }
      
      trackEvent('classification_error', {
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-primary-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.header 
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card variant="elevated" className="p-6">
            <div className="flex flex-col md:flex-row justify-between items-center gap-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-gradient-to-br from-primary-600 to-primary-500 rounded-xl flex items-center justify-center text-white font-bold text-xl shadow-lg">
                  WS
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">WasteSort AI</h1>
                  <p className="text-gray-600">Intelligent Waste Classification</p>
                </div>
              </div>
              
              <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${
                apiStatus === 'online' 
                  ? 'bg-green-50 border-green-200 text-green-700' 
                  : apiStatus === 'offline'
                  ? 'bg-red-50 border-red-200 text-red-700'
                  : 'bg-gray-50 border-gray-200 text-gray-700'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  apiStatus === 'online' ? 'bg-green-500 animate-pulse' : 
                  apiStatus === 'offline' ? 'bg-red-500' : 'bg-gray-400'
                }`}></div>
                <span className="text-sm font-medium">
                  {apiStatus === 'online' ? 'System Online' : 
                   apiStatus === 'offline' ? 'System Offline' : 'Checking...'}
                </span>
              </div>
            </div>
          </Card>
        </motion.header>
        
        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <motion.section
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card variant="elevated">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Classify Waste Item</h2>
              
              <div
                className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 cursor-pointer ${
                  dragActive 
                    ? 'border-primary-400 bg-primary-50 scale-105' 
                    : 'border-gray-300 hover:border-primary-400 hover:bg-primary-50'
                }`}
                onDrop={handleDrop}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onClick={openFileDialog}
              >
                <motion.div 
                  className={`w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center transition-all duration-300 ${
                    dragActive ? 'bg-primary-200 scale-110' : 'bg-gray-100'
                  }`}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <ArrowUpTrayIcon className="w-6 h-6 text-gray-600" />
                </motion.div>
                
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Image</h3>
                <p className="text-gray-500 text-sm mb-6">
                  Drag and drop your image here, or click to browse
                </p>
                
                <div className="flex flex-col sm:flex-row gap-3 justify-center">
                  <Button
                    onClick={(e) => { e.stopPropagation(); openCamera(); }}
                    icon={<CameraIcon className="w-4 h-4" />}
                    size="md"
                  >
                    Take Photo
                  </Button>
                  <Button
                    onClick={(e) => { e.stopPropagation(); openFileDialog(); }}
                    variant="outline"
                    icon={<DocumentArrowUpIcon className="w-4 h-4" />}
                    size="md"
                  >
                    Browse Files
                  </Button>
                </div>
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                className="hidden"
              />
              
              {/* Image Preview */}
              <AnimatePresence>
                {uploadedFile && (
                  <motion.div
                    className="mt-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="relative rounded-xl overflow-hidden shadow-lg">
                      <img
                        src={uploadedFile.preview}
                        alt="Preview"
                        className="w-full h-64 object-cover"
                      />
                      <button
                        onClick={removeImage}
                        className="absolute top-3 right-3 w-8 h-8 bg-white/90 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white transition-all duration-200 hover:scale-110 shadow-md"
                      >
                        <XMarkIcon className="w-4 h-4 text-gray-600" />
                      </button>
                    </div>
                    
                    <Button
                      onClick={classifyImage}
                      disabled={isLoading || !uploadedFile}
                      loading={isLoading}
                      className="w-full mt-4"
                      size="lg"
                    >
                      {isLoading ? 'Analyzing...' : 'Analyze Image'}
                    </Button>
                  </motion.div>
                )}
              </AnimatePresence>
            </Card>
          </motion.section>
          
          {/* Results Section */}
          <motion.section
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card variant="elevated" className="min-h-96">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Classification Results</h2>
              
              <AnimatePresence mode="wait">
                {/* Error State */}
                {error && (
                  <motion.div
                    key="error"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg flex items-center gap-3"
                  >
                    <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0" />
                    <span>{error}</span>
                  </motion.div>
                )}
                
                {/* Loading State */}
                {isLoading && (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="flex flex-col items-center justify-center h-full py-12"
                  >
                    <div className="w-12 h-12 border-3 border-gray-200 border-t-primary-600 rounded-full animate-spin mb-4"></div>
                    <div className="text-gray-600 font-medium">Analyzing image...</div>
                    <div className="text-sm text-gray-500 mt-2">This may take a few seconds</div>
                  </motion.div>
                )}
                
                {/* Results */}
                {results && (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    {/* Classification Result */}
                    <div className="bg-primary-50 border border-primary-200 rounded-lg p-6">
                      <div className="flex justify-between items-center mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 capitalize flex items-center gap-2">
                          <CheckCircleIcon className="w-6 h-6 text-primary-600" />
                          {results.prediction.class}
                        </h3>
                        <span className={`px-3 py-1 rounded-md text-white text-xs font-semibold uppercase tracking-wide ${
                          getConfidenceLevel(results.prediction.confidence).level === 'high' ? 'bg-green-500' :
                          getConfidenceLevel(results.prediction.confidence).level === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                        }`}>
                          {getConfidenceLevel(results.prediction.confidence).text}
                        </span>
                      </div>
                      
                      <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                        <motion.div
                          className="bg-gradient-to-r from-primary-600 to-primary-400 h-3 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${results.prediction.confidence * 100}%` }}
                          transition={{ duration: 0.8, delay: 0.2 }}
                        ></motion.div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="text-gray-500 uppercase tracking-wide font-semibold text-xs mb-1">
                            Confidence
                          </div>
                          <div className="font-bold text-gray-900 text-lg">
                            {(results.prediction.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500 uppercase tracking-wide font-semibold text-xs mb-1">
                            Processing Time
                          </div>
                          <div className="font-bold text-gray-900 text-lg">
                            {processingTime}ms
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Recycling Information */}
                    <div className="bg-gray-50 rounded-lg p-6 border-l-4 border-primary-500">
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <div className="text-gray-500 uppercase tracking-wide font-semibold text-xs mb-1">
                            Disposal Method
                          </div>
                          <div className="font-semibold text-gray-900">
                            {results.recycling_info.bin}
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500 uppercase tracking-wide font-semibold text-xs mb-1">
                            Recyclable
                          </div>
                          <div className={`font-semibold ${
                            results.recycling_info.recyclable ? 'text-green-700' : 'text-red-700'
                          }`}>
                            {results.recycling_info.recyclable ? 'Yes' : 'No'}
                          </div>
                        </div>
                      </div>
                      
                      {results.recycling_info.tips && results.recycling_info.tips.length > 0 && (
                        <div className="mb-4">
                          <div className="text-sm font-semibold text-gray-900 mb-3">
                            Disposal Instructions
                          </div>
