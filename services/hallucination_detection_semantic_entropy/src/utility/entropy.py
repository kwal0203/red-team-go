from typing import Any

from utils.utils import response_generator


def get_generations(
    session_state: Any,
    prompt: str,
    num_generations: int = 10,
    temperature: float = 0.5,
):
    # We sample one low temperature answer on which we will compute the
    # accuracy and args.num_generation high temperature answers which will
    # be used to estimate the entropy variants.
    #
    # Note: I'm not calculating accuracies so I don't need the initial
    #       low temperature generation
    responses = []
    for _ in range(num_generations):
        response = response_generator(session_state=session_state, prompt=prompt)
        response_dict = response.to_dict()
        predicted_answer = response_dict["choices"][0]["message"]["content"]
        log_likelihoods = response_dict["choices"][0]["logprobs"]["content"]
        responses.append((predicted_answer, log_likelihoods))

    return responses


def semantic_entropy(
    entailment_model: str, model: Any, prompt: str, session_state: Any
) -> dict:
    from services.semantic_entropy import get_semantic_ids
    from utils.models import EntailmentDeberta

    # from semantic_entropy import cluster_assignment_entropy
    # import numpy as np
    # from semantic_entropy import predictive_entropy
    # from semantic_entropy import logsumexp_by_id
    # from semantic_entropy import predictive_entropy_rao

    # Load entailment model
    if entailment_model == "deberta":
        print("Entailment deberta")
        entailment_model = EntailmentDeberta()
    # else:
    #     entailment_model = EntailmentGPT35()

    semantic_ids = []
    print("Get generations")
    generations = get_generations(
        session_state=session_state,
        prompt=prompt,
        num_generations=2,
        temperature=0.9,
    )

    print("Get semantic ids")
    responses = [response[0] for response in generations]

    # We also show that simply using the number of semantically distinct answers as an
    # uncertainty measure on its own performs reasonably well:
    # Kuhn, 2023 (https://arxiv.org/pdf/2302.09664)
    #
    # NOTE: Maybe just use the number of semantic ids as the semantic score
    # semantic_score = number_of_distinct_clusters / num_generations
    semantic_ids = get_semantic_ids(strings_list=responses, model=entailment_model)
    semantic_id_score = -(len(set(semantic_ids)) / len(generations)) + 1
    semantic_id_score = int(semantic_id_score * 100)
    print(f"Semantic Ids:      {semantic_ids}")
    print(f"Semantic ID score: {semantic_id_score}")

    # NOTE: For demo, removing the semantic entropy score because not sure how to use it
    #       in the context of the weighted average hallucination score (i.e., how to
    #       make the entropy be between 0 and 1).
    #       Semantic entropy calculation is below.
    #
    # entropies = collections.defaultdict(list)
    # entropies["cluster_assignment_entropy"].append(
    #     cluster_assignment_entropy(semantic_ids)
    # )
    # print(f"cluster_assignment_entropy: {entropies['cluster_assignment_entropy']}")

    # response_logprobs = [response[1] for response in generations]
    # generation_logprobs = []
    # for generation in response_logprobs:
    #     token_logprobs = []
    #     for token in generation:
    #         token_logprobs.append(token["logprob"])
    #     generation_logprobs.append(token_logprobs)

    # log_likelihoods = [
    #     np.mean(log_likelihoods) for log_likelihoods in generation_logprobs
    # ]
    # print(f"mean_log_likelihoods: {log_likelihoods}")

    # log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_likelihoods)
    # print(f"log_likelihood_per_semantic_id: {log_likelihood_per_semantic_id}")

    # semantic_entropy_value = predictive_entropy_rao(log_likelihood_per_semantic_id)
    # entropies["semantic_entropy"].append(
    #     predictive_entropy_rao(log_likelihood_per_semantic_id)
    # )
    # print(f"semantic_entropy: {semantic_entropy_value}")

    return semantic_id_score
