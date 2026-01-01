// Shared model configuration
export interface Model {
  name: string;
  description: string;
  base_url?: string | null;
  model_name?: string | null;
}

// Common response wrapper
export interface ApiResponse<T> {
  result: T;
}

export interface ErrorResponse {
  detail: string;
}

// Detection endpoints
export interface DetectionBatchRequest {
  model: Model;
  user_prompts: string[];
}

export interface DetectionRealtimeRequest {
  model: Model;
  prompt: string;
}

export interface ToxicityScore {
  openai_score: number;
  paradetox_score: number;
  average_score: number;
  is_toxic: boolean;
}

export interface BiasScore {
  bias_detected: boolean;
  explanation: string;
}

export interface ToxicityRealtimeResult {
  prompt: string;
  model_response: string;
  toxicity: ToxicityScore;
}

export interface BiasRealtimeResult {
  prompt: string;
  model_response: string;
  bias: BiasScore;
}

// Guardrails endpoints
export interface GuardrailViolation {
  guardrail: string;
  category: string;
  confidence: number;
  explanation: string;
}

export interface GuardrailEvaluateRequest {
  model: Model;
  prompt: string;
  guardrails?: string[];
}

export interface GuardrailProtectRequest {
  input_text?: string;
  output_text?: string;
  guardrails?: string[];
  remediation_action?: 'block' | 'flag' | 'redact';
}

export interface GuardrailResult {
  passed: boolean;
  risk_level: string;
  violations: GuardrailViolation[];
  bypassed_guardrails?: string[];
}

// Adversarial robustness
export interface AdversarialRequest {
  model: Model;
  prompt: string;
  perturbation_types?: string[];
  num_perturbations?: number;
}

// Prompt generation
export interface PromptGenerationRequest {
  model: Model;
  target_category: string;
  generation_method: 'llm' | 'genetic' | 'pair';
  num_prompts?: number;
  seed_prompts?: string[];
}

export interface GeneratedPrompt {
  prompt: string;
  category: string;
  method: string;
  metadata?: Record<string, unknown>;
}

// Stereotype benchmarks
export interface StereotypeBenchmarkRequest {
  model: Model;
  benchmark: 'stereoset' | 'crows_pairs' | 'bbq';
  num_samples?: number;
}

// Consistency & Reliability
export interface ConsistencyRequest {
  model: Model;
  prompt: string;
  test_types?: string[];
  num_samples?: number;
}

// Misinformation & Factuality
export interface MisinformationRequest {
  model: Model;
  prompt: string;
  test_types?: string[];
}

// Refusal Consistency
export interface RefusalRequest {
  model: Model;
  prompt: string;
  harm_category?: string;
  test_types?: string[];
}

// Privacy Red Team
export interface PrivacyRequest {
  model: Model;
  test_types?: string[];
  target_info?: string;
  system_prompt?: string;
}

// Hallucination Confidence
export interface HallucinationRequest {
  model: Model;
  prompt: string;
  method?: 'geometric' | 'average' | 'minimum' | 'entropy' | 'variance';
}

export interface HallucinationResult {
  prompt: string;
  response: string;
  confidence_score: number;
  risk_level: string;
  interpretation: string;
  method: string;
}

// Test result with grade
export interface TestResult {
  test_type: string;
  score: number;
  grade: string;
  passed: boolean;
  details: Record<string, unknown>;
}

export interface EvaluationResult {
  overall_score: number;
  overall_grade: string;
  passed: boolean;
  test_results: TestResult[];
}
