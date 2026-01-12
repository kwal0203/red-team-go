import csv
import json
from typing import Any


def hijack_log(results_hijack: Any):
    """
    Uses the batch redteam functionality drawing attacks from a database.
    The results of the batch process are stored in a CSV file.
    """
    responses_adversarial = 0
    for result in results_hijack:
        print(result)
        response = result["response"]
        evaluation = result["response_evaluation"]
        eval_passed = evaluation["evaluation_toxicity"][0]["eval_passed_toxicity"]
        reason = evaluation["evaluation_toxicity"][0]["reason_toxicity"]
        if not eval_passed:
            responses_adversarial += 1

        response.strip("'\"")
        with open("tester.csv", mode="a") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow([eval_passed, response, reason])

    return responses_adversarial


# import fitz
# import re
# def extract_text_from_pdf(file):
#     num_lines = 0
#     pdf_document = fitz.open(stream=file.read(), filetype="pdf")
#     text = ""
#     for page_num in range(len(pdf_document)):
#         page = pdf_document.load_page(page_num)
#         page_text = page.get_text()

#         lines = page_text.splitlines()
#         filtered_lines = [line for line in lines if not line.strip().isdigit()]
#         num_lines += len(filtered_lines)
#         text += " ".join(filtered_lines)
#         if num_lines > 20:
#             break

#     text = re.sub(r"-\s+", "", text)
#     return text


def generate_report(
    results: dict, generated_summary: str, logprobs: list[dict], entropy: Any
) -> None:
    """
    Generate JSON object containing results of hallucination detection and write
    to disk for the front end to read and display.
    """
    # TODO: Report feature currently broken as refactoring services into IVCAP

    # Weighted average of seq_logprobs and FActScore
    # overall_score = (0.33 * seq_logprob) + (0.33 * factscore) + (0.33 * entropy)
    # overall_score = int(overall_score)

    # Dumping results to .json file which can be read by the React frontend
    # TODO: need to connect back and front end by websockets or something
    result_json: dict[str, Any] = {}
    # result_json["summary"] = generated_summary
    # result_json["overall_score"] = overall_score
    # result_json["factscore"] = factscore
    # result_json["seq_logprob"] = seq_logprob
    # result_json["semantic_entropy"] = entropy
    with open("/scratch3/wal740/redteam_react/public/data.json", "w") as f:
        json.dump(result_json, f, indent=4)
