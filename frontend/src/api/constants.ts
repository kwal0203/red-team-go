/**
 * API and UI constants for the RedTeamGO frontend.
 * Centralizes magic numbers and configuration values.
 */

// ============================================================================
// Toast Notification Durations (milliseconds)
// ============================================================================

/** Duration for success toast notifications */
export const TOAST_DURATION_SUCCESS = 3000;

/** Duration for error toast notifications */
export const TOAST_DURATION_ERROR = 5000;

/** Duration for warning toast notifications */
export const TOAST_DURATION_WARNING = 3000;

// ============================================================================
// API Polling & Refresh Intervals (milliseconds)
// ============================================================================

/** Health check polling interval on Dashboard */
export const HEALTH_CHECK_INTERVAL = 30000;

// ============================================================================
// Thresholds
// ============================================================================

/** Toxicity score threshold (0.0-1.0) - scores above this are considered toxic */
export const TOXICITY_THRESHOLD = 0.5;

/** Confidence score thresholds for risk levels (percentage) */
export const CONFIDENCE_THRESHOLDS = {
  LOW_RISK: 70,    // >= 70% confidence = low risk
  MEDIUM_RISK: 50, // >= 50% confidence = medium risk
  HIGH_RISK: 30,   // >= 30% confidence = high risk
  // < 30% = critical risk
} as const;

// ============================================================================
// Local Storage Keys
// ============================================================================

/** Key for storing the API key in localStorage */
export const STORAGE_KEY_API_KEY = 'redteam_api_key';

// ============================================================================
// Default Values
// ============================================================================

/** Default number of perturbations for adversarial testing */
export const DEFAULT_NUM_PERTURBATIONS = 5;

/** Default number of prompts for generation */
export const DEFAULT_NUM_PROMPTS = 5;

/** Default number of samples for benchmarks */
export const DEFAULT_NUM_SAMPLES = 10;
