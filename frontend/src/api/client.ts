import axios from 'axios';
import { STORAGE_KEY_API_KEY } from './constants';

// API base URL - uses environment variable or defaults to localhost
const API_BASE_URL = (process.env.REACT_APP_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

// Check if we're in development mode
const isDevelopment = process.env.NODE_ENV === 'development';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // Prevent the UI from spinning forever if the backend call hangs
  timeout: 30000, // 30s
});

// Add API key to requests if available
apiClient.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem(STORAGE_KEY_API_KEY);
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }
  return config;
});

// Handle errors consistently
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Only log errors in development mode
    if (isDevelopment) {
      if (error.response?.status === 401) {
        console.error('Authentication required');
      } else if (error.response?.status === 429) {
        console.error('Rate limit exceeded');
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;
