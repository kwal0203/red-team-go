"""Guardrails service for input/output safety evaluation.

Provides two modes:
- Evaluate Mode: Red-team testing to detect if target model has proper guardrails
- Protect Mode: Production middleware to block/filter harmful content
"""

import logging

from services.guardrails.src.detectors.base import RiskLevel
from services.guardrails.src.pipeline import GuardrailPipeline
from services.guardrails.src.remediation import (
    ContentRemediator,
    RemediationAction,
    RemediationResult,
)
from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from utils.models import Model

logger = logging.getLogger(__name__)


def _create_target_model(model: Model):
    """Create the appropriate model wrapper based on model configuration."""
    if "openai" in model["name"]:
        logger.info(f"Creating OpenAI model wrapper for {model['name']}")
        return APIModelOpenai(name=model["name"], description=model["description"])
    elif "huggingface" in model["name"]:
        logger.info(f"Creating HuggingFace model wrapper for {model['name']}")
        return APIModelHuggingFace(
            base_url=model["base_url"],
            name=model["name"],
            description=model["description"],
        )
    else:
        raise ValueError(
            f"Invalid model name '{model['name']}': must contain 'openai' or 'huggingface'"
        )


def guardrails_evaluate_service(
    model: Model,
    prompt: str,
    guardrails: list[str] | None = None,
) -> dict:
    """Evaluate guardrails for red-team testing.

    Sends a prompt to the target model and evaluates both the input prompt
    and the model's response for various safety violations.

    Args:
        model: Target LLM configuration.
        prompt: The prompt to send to the model.
        guardrails: List of guardrails to apply. If None, uses all.

    Returns:
        Dictionary containing:
        - prompt: Original prompt
        - model_response: Model's response
        - input_analysis: Guardrail results for the input
        - output_analysis: Guardrail results for the output
        - overall_risk: Aggregated risk level
        - guardrails_bypassed: List of guardrails that detected violations
    """
    logger.info(f"Starting guardrail evaluation for model: {model['name']}")

    # Create target model
    target_model = _create_target_model(model)

    # Create pipeline
    pipeline = GuardrailPipeline(guardrails=guardrails)

    # Analyze input prompt
    logger.debug("Analyzing input prompt")
    input_result = pipeline.check(prompt)

    # Get model response
    logger.info("Getting model response")
    response = target_model.model_predict(data=[prompt])
    if isinstance(response, list):
        response = response[0]

    # Analyze output
    logger.debug("Analyzing model response")
    output_result = pipeline.check(response)

    # Determine which guardrails were bypassed
    # (violations in output mean the model's guardrails failed)
    guardrails_bypassed = output_result.violations

    # Compute combined risk (take the higher risk level)
    risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    input_risk_idx = risk_order.index(input_result.overall_risk)
    output_risk_idx = risk_order.index(output_result.overall_risk)
    overall_risk = risk_order[max(input_risk_idx, output_risk_idx)]

    logger.info(
        f"Evaluation complete: input_risk={input_result.overall_risk.value}, "
        f"output_risk={output_result.overall_risk.value}, "
        f"bypassed={guardrails_bypassed}"
    )

    return {
        "prompt": prompt,
        "model_response": response,
        "input_analysis": {
            name: result.to_dict() for name, result in input_result.results.items()
        },
        "output_analysis": {
            name: result.to_dict() for name, result in output_result.results.items()
        },
        "overall_risk": overall_risk.value,
        "guardrails_bypassed": guardrails_bypassed,
    }


def guardrails_protect_service(
    input_text: str | None = None,
    output_text: str | None = None,
    action: str = "redact",
    guardrails: list[str] | None = None,
) -> dict:
    """Protect mode for production middleware.

    Checks input and/or output text for safety violations and applies
    the specified remediation action.

    Args:
        input_text: Optional input text to check.
        output_text: Optional output text to check.
        action: Remediation action ("block", "flag", "redact").
        guardrails: List of guardrails to apply. If None, uses all.

    Returns:
        Dictionary containing:
        - allowed: Whether content should be allowed
        - input_safe: Whether input passed checks (if provided)
        - output_safe: Whether output passed checks (if provided)
        - violations: List of all violations found
        - remediated_output: Modified output (if redaction applied)
        - remediated_input: Modified input (if redaction applied)
        - input_result: Detailed input check result (if input provided)
        - output_result: Detailed output check result (if output provided)
    """
    logger.info(f"Starting protect check with action={action}")

    # Validate action
    try:
        remediation_action = RemediationAction(action)
    except ValueError:
        remediation_action = RemediationAction.REDACT
        logger.warning(f"Invalid action '{action}', defaulting to 'redact'")

    # Create pipeline and remediator
    pipeline = GuardrailPipeline(guardrails=guardrails)
    remediator = ContentRemediator(default_action=remediation_action)

    all_violations: list[str] = []
    input_safe = True
    output_safe = True
    input_result_dict: dict | None = None
    output_result_dict: dict | None = None
    input_remediation_result: RemediationResult | None = None
    output_remediation_result: RemediationResult | None = None

    # Check input if provided
    if input_text:
        logger.debug("Checking input text")
        input_check = pipeline.check(input_text)
        input_safe = len(input_check.violations) == 0
        all_violations.extend(input_check.violations)
        input_result_dict = input_check.to_dict()

        # Apply remediation to input only if violations detected
        if not input_safe:
            input_remediation_result = remediator.remediate(
                input_text, input_check, remediation_action
            )

    # Check output if provided
    if output_text:
        logger.debug("Checking output text")
        output_check = pipeline.check(output_text)
        output_safe = len(output_check.violations) == 0
        all_violations.extend(output_check.violations)
        output_result_dict = output_check.to_dict()

        # Apply remediation to output only if violations detected
        if not output_safe:
            output_remediation_result = remediator.remediate(
                output_text, output_check, remediation_action
            )

    # Determine if content should be allowed based on remediation results
    # Both input and output must be allowed for overall allowed=True
    input_allowed = (
        input_remediation_result.allowed if input_remediation_result else True
    )
    output_allowed = (
        output_remediation_result.allowed if output_remediation_result else True
    )
    allowed = input_allowed and output_allowed

    # Get remediated content
    remediated_input = (
        input_remediation_result.remediated_content
        if input_remediation_result
        else None
    )
    remediated_output = (
        output_remediation_result.remediated_content
        if output_remediation_result
        else None
    )

    logger.info(
        f"Protect check complete: allowed={allowed}, "
        f"input_safe={input_safe}, output_safe={output_safe}, "
        f"violations={all_violations}"
    )

    result = {
        "allowed": allowed,
        "input_safe": input_safe,
        "output_safe": output_safe,
        "violations": list(set(all_violations)),  # Deduplicate
    }

    # Add remediated content if available
    if remediated_input is not None:
        result["remediated_input"] = remediated_input
    if remediated_output is not None:
        result["remediated_output"] = remediated_output

    # Add detailed results if available
    if input_result_dict:
        result["input_result"] = input_result_dict
    if output_result_dict:
        result["output_result"] = output_result_dict

    # Add separate remediation info for input and output
    if input_remediation_result:
        result["input_remediation"] = {
            "action_taken": input_remediation_result.action_taken.value,
            "explanation": input_remediation_result.explanation,
        }
    if output_remediation_result:
        result["output_remediation"] = {
            "action_taken": output_remediation_result.action_taken.value,
            "explanation": output_remediation_result.explanation,
        }

    return result
