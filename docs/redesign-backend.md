# Backend Redesign: Functional Grouping

## Goal
Restructure the `services/` directory to group services by function (Detection, Safety, Evaluation) rather than by paper/implementation. This aligns the backend architecture with the frontend structure and improves discoverability.

## Proposed Structure

```text
services/
├── detection/               # Passive analysis of content (Input/Output)
│   ├── bias/                # (was bias_detection_dbias)
│   ├── factuality/          # (was misinformation_factuality)
│   ├── hallucination/       # Grouping all hallucination strategies
│   │   ├── factscore/       # (was hallucination_detection_factscore)
│   │   ├── confidence/      # (was hallucination_detection_model_confidence)
│   │   └── entropy/         # (was hallucination_detection_semantic_entropy)
│   └── toxicity/            # (was toxicity_detection)
│
├── safety/                  # Active defenses and guardrails
│   └── guardrails/          # (was guardrails)
│
├── evaluation/              # Active testing loops, benchmarking, and red teaming
│   ├── attacks/             # (or 'generators') - The offensive engines
│   │   ├── aart/
│   │   ├── autodan/
│   │   ├── advprompter/
│   │   ├── blackbox_pair/
│   │   ├── cold_attack/
│   │   ├── crt/
│   │   ├── dsn/
│   │   ├── gptfuzzer/
│   │   ├── jailbreakhub/
│   │   └── sap/
│   │
│   ├── benchmarks/          # Scenarios and metrics runners
│   │   ├── consistency_reliability/
│   │   ├── refusal_consistency/
│   │   ├── privacy/         # (was privacy_redteam)
│   │   ├── robustness/      # (was adversarial_robustness)
│   │   └── stereotypes/     # (was stereotype_benchmarks)
│   │
│   └── prompt_generation/   # Service wrapper for generation/attacks
│
└── shared/                  # Common infrastructure
    ├── datasets/
    └── model_wrappers/
```

## Migration Plan

1.  Create new directory structure.
2.  Move existing service directories to their new locations.
3.  Update all import statements in the codebase.
4.  Verify functionality with tests.
