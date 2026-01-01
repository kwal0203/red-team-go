/**
 * Axios error structure for API calls.
 * Use this type instead of 'any' in mutation error handlers.
 */
export interface ApiError {
  response?: {
    data?: {
      detail?: string;
    };
    status?: number;
  };
  message?: string;
}

/**
 * Helper function to extract error message from ApiError.
 * Returns a user-friendly error message.
 */
export function getErrorMessage(error: ApiError, fallback: string): string {
  return error.response?.data?.detail || error.message || fallback;
}

/**
 * Model configuration for API requests.
 * Used across all endpoints that require a target model.
 */
export interface Model {
  /** Model identifier in format "provider:model-name" (e.g., "openai:gpt-4") */
  name: string;
  /** Human-readable description of the model's purpose */
  description: string;
  /** Optional base URL for custom/self-hosted models */
  base_url?: string;
  /** Optional override for the model name sent to the API */
  model_name?: string;
}

/**
 * Generic API response wrapper.
 * Most batch endpoints wrap their results in this structure.
 */
export interface ApiResponse<T> {
  result: T;
}

/**
 * Standard error response from the API.
 */
export interface ErrorResponse {
  detail: string;
}

// ============================================================================
// Detection Endpoints
// ============================================================================

/** Request body for batch toxicity/bias detection endpoints */
export interface DetectionBatchRequest {
  model: Model;
  user_prompts: string[];
}

/** Request body for realtime toxicity/bias detection endpoints */
export interface DetectionRealtimeRequest {
  model: Model;
  prompt: string;
}

/**
 * Toxicity evaluation scores from ensemble models.
 * Scores range from 0.0 (safe) to 1.0 (toxic).
 */
export interface ToxicityScore {
  /** Score from OpenAI Moderation API */
  openai_score: number;
  /** Score from Paradetox model */
  paradetox_score: number;
  /** Average of both scores */
  average_score: number;
  /** True if average_score >= 0.5 */
  is_toxic: boolean;
}

/** Bias detection result using DBias methodology */
export interface BiasScore {
  bias_detected: boolean;
  explanation: string;
}

/** Response from /toxicity-detection-realtime endpoint */
export interface ToxicityRealtimeResult {
  prompt: string;
  model_response: string;
  toxicity: ToxicityScore;
}

/** Response from /bias-detection-realtime endpoint */
export interface BiasRealtimeResult {
  prompt: string;
  model_response: string;
  bias: BiasScore;
}

// ============================================================================
// Guardrails Endpoints
// ============================================================================

/** Individual guardrail violation details */
export interface GuardrailViolation {
  guardrail: string;
  category: string;
  /** Confidence score from 0.0 to 1.0 */
  confidence: number;
  explanation: string;
}

/** Request body for /evaluate/guardrails (red-team mode) */
export interface GuardrailEvaluateRequest {
  model: Model;
  prompt: string;
  /** Optional list of guardrails to check (defaults to all) */
  guardrails?: string[];
}

/** Request body for /protect/guardrails (middleware mode) */
export interface GuardrailProtectRequest {
  input_text?: string;
  output_text?: string;
  guardrails?: string[];
  remediation_action?: 'block' | 'flag' | 'redact';
}

/** Response from guardrail evaluation endpoints */
export interface GuardrailResult {
  passed: boolean;
  /** Risk level: 'low' | 'medium' | 'high' | 'critical' */
  risk_level: string;
  violations: GuardrailViolation[];
  bypassed_guardrails?: string[];
}

// ============================================================================
// Adversarial Testing Endpoints
// ============================================================================

/** Request body for /adversarial-robustness endpoint */
export interface AdversarialRequest {
  model: Model;
  prompt: string;
  /** Perturbation types: 'character' | 'word' | 'semantic' */
  perturbation_types?: string[];
  num_perturbations?: number;
}

/** Request body for /generate-adversarial-prompts endpoint */
export interface PromptGenerationRequest {
  model: Model;
  /** Target category: 'jailbreak' | 'harmful' | 'bias' | 'toxicity' */
  target_category: string;
  generation_method: 'llm' | 'genetic' | 'pair';
  num_prompts?: number;
  seed_prompts?: string[];
}

/** Generated adversarial prompt with metadata */
export interface GeneratedPrompt {
  prompt: string;
  category: string;
  method: string;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Benchmark Endpoints
// ============================================================================

/** Request body for /stereotype-benchmark endpoint */
export interface StereotypeBenchmarkRequest {
  model: Model;
  benchmark: 'stereoset' | 'crows_pairs' | 'bbq';
  num_samples?: number;
}

// ============================================================================
// Reliability Testing Endpoints
// ============================================================================

/** Request body for /consistency-reliability endpoint */
export interface ConsistencyRequest {
  model: Model;
  prompt: string;
  /** Test types: 'sycophancy' | 'stability' | 'self_consistency' | 'instruction_following' */
  test_types?: string[];
  num_samples?: number;
}

/** Request body for /misinformation-factuality endpoint */
export interface MisinformationRequest {
  model: Model;
  prompt: string;
  /** Test types: 'knowledge_cutoff' | 'temporal_reasoning' | 'confidence_calibration' | 'citation_verification' */
  test_types?: string[];
}

/** Request body for /refusal-consistency endpoint */
export interface RefusalRequest {
  model: Model;
  prompt: string;
  /** Harm category: 'weapons' | 'illegal_activities' | 'self_harm' | 'violence' | 'cbrn' | 'dual_use' */
  harm_category?: string;
  /** Test types: 'paraphrase' | 'pressure' | 'multi_turn' | 'context_switching' */
  test_types?: string[];
}

// ============================================================================
// Privacy Testing Endpoints
// ============================================================================

/** Request body for /privacy-redteam endpoint */
export interface PrivacyRequest {
  model: Model;
  /** Test types: 'training_extraction' | 'membership_inference' | 'prompt_leakage' */
  test_types?: string[];
  /** Specific information to probe for */
  target_info?: string;
  /** System prompt to test for leakage */
  system_prompt?: string;
}

// ============================================================================
// Hallucination Detection Endpoints
// ============================================================================

/** Request body for /hallucination-confidence endpoint */
export interface HallucinationRequest {
  model: Model;
  prompt: string;
  /** Confidence calculation method */
  method?: 'geometric' | 'average' | 'minimum' | 'entropy' | 'variance';
}

/** Response from /hallucination-confidence endpoint */
export interface HallucinationResult {
  prompt: string;
  response: string;
  /** Confidence score as percentage (0-100) */
  confidence_score: number;
  /** Risk level: 'low' | 'medium' | 'high' | 'critical' */
  risk_level: string;
  interpretation: string;
  method: string;
}

// ============================================================================
// Common Test Result Types
// ============================================================================

/** Individual test result with grade */
export interface TestResult {
  test_type: string;
  /** Score from 0.0 to 1.0 */
  score: number;
  /** Letter grade: 'A' | 'B' | 'C' | 'D' | 'F' */
  grade: string;
  passed: boolean;
  details: Record<string, unknown>;
}

/** Aggregated evaluation result across multiple tests */
export interface EvaluationResult {
  /** Overall score from 0.0 to 1.0 */
  overall_score: number;
  /** Overall letter grade */
  overall_grade: string;
  passed: boolean;
  test_results: TestResult[];
}
