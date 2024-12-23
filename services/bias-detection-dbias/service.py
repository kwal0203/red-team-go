from src.bias_detection import detect_bias

from typing import Any


def service(args: Any) -> Any:
    # TODO: Convert to Google style comments

    ### Bias detection service using the Dbias model (Raza, Int J Data Sci Anal 2024).
    response_evaluation = detect_bias(response=args.response)
    evaluation_bias = response_evaluation["evaluation_bias"][0]
    score_bias = int(evaluation_bias["score"] * 100)
    eval_passed = score_bias < 85
    result_json = {"bias_detected": eval_passed, "score": score_bias}

    return result_json
