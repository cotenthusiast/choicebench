# src/mcq_eval/config/experiment.py

# MMLU dataset constants --------------------------------
MCQ_ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
MMLU_QUESTIONS_PER_SUBJECT = 50

# Shared method names -----------------------------------
BASELINE_METHOD = "baseline"
TWOPROMPT_METHOD = "two_prompt"
CYCLIC_METHOD = "cyclic"
TWOPROMPT_CYCLIC_METHOD = "two_prompt_cyclic"
PRIDE_METHOD = "pride"

ALL_METHODS = [
    BASELINE_METHOD,
    TWOPROMPT_METHOD,
    CYCLIC_METHOD,
    PRIDE_METHOD,
]

# Track A: robustness / accuracy ------------------------
ROBUSTNESS_TRACK_NAME = "robustness"

ROBUSTNESS_SUBJECTS = [
    "high_school_physics",
    "college_mathematics",
    "anatomy",
    "college_chemistry",
    "computer_security",
    "medical_genetics",
    "college_biology",
    "clinical_knowledge",
    "high_school_psychology",
    "econometrics",
    "sociology",
    "philosophy",
    "high_school_world_history",
    "jurisprudence",
    "professional_law",
    "professional_medicine",
    "professional_accounting",
    "moral_scenarios",
    "nutrition",
    "global_facts",
    "abstract_algebra",
    "astronomy",
    "business_ethics",
    "college_computer_science",
    "college_medicine",
    "college_physics",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_statistics",
    "high_school_us_history",
    "international_law",
    "logical_fallacies",
    "machine_learning",
    "prehistory",
    "professional_psychology",
    "security_studies",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

ROBUSTNESS_NO_OF_SUBJECTS = len(ROBUSTNESS_SUBJECTS)
ROBUSTNESS_QUESTIONS_PER_SUBJECT = 20
ROBUSTNESS_TOTAL_QUESTIONS = (
    ROBUSTNESS_NO_OF_SUBJECTS * ROBUSTNESS_QUESTIONS_PER_SUBJECT
)

ROBUSTNESS_METHODS = [
    BASELINE_METHOD,
    TWOPROMPT_METHOD,
    CYCLIC_METHOD,
]

ROBUSTNESS_SPLIT_SEED = 42

# Shared review split -----------------------------------
# This single split is reused for:
# 1) stronger-model evaluation
# 2) faithfulness / human-review analysis
REVIEW_TRACK_NAME = "review"

REVIEW_SUBJECTS = [
    "high_school_physics",
    "college_mathematics",
    "anatomy",
    "college_chemistry",
    "computer_security",
    "medical_genetics",
    "college_biology",
    "clinical_knowledge",
    "high_school_psychology",
    "econometrics",
    "sociology",
    "philosophy",
    "high_school_world_history",
    "jurisprudence",
    "professional_law",
    "professional_medicine",
    "professional_accounting",
    "moral_scenarios",
    "nutrition",
    "global_facts",
]

REVIEW_NO_OF_SUBJECTS = len(REVIEW_SUBJECTS)
REVIEW_QUESTIONS_PER_SUBJECT = 3
REVIEW_TOTAL_QUESTIONS = REVIEW_NO_OF_SUBJECTS * REVIEW_QUESTIONS_PER_SUBJECT
REVIEW_SPLIT_SEED = 42

REVIEW_METHODS = [
    BASELINE_METHOD,
    TWOPROMPT_METHOD,
    CYCLIC_METHOD,
]

# Track B: stronger models ------------------------------
STRONG_MODELS_TRACK_NAME = "strong_models"
STRONG_MODELS_USE_REVIEW_SPLIT = True
STRONG_MODELS_METHODS = REVIEW_METHODS

# Track C: faithfulness ---------------------------------
FAITHFULNESS_TRACK_NAME = "faithfulness"
FAITHFULNESS_USE_REVIEW_SPLIT = True
FAITHFULNESS_METHODS = REVIEW_METHODS

# Review / scoring protocols ----------------------------
ANSWER_MATCHING_PROTOCOL = "answer_matching"
HUMAN_REVIEW_PROTOCOL = "human_review"

REVIEW_PROTOCOLS = [
    ANSWER_MATCHING_PROTOCOL,
    HUMAN_REVIEW_PROTOCOL,
]

# Backwards-compatible aliases --------------------------
BENCHMARK_TOTAL_QUESTIONS = ROBUSTNESS_TOTAL_QUESTIONS
BENCHMARK_SUBJECTS = ROBUSTNESS_SUBJECTS

FAITHFULNESS_SUBJECTS = REVIEW_SUBJECTS
FAITHFULNESS_QUESTIONS_PER_SUBJECT = REVIEW_QUESTIONS_PER_SUBJECT
FAITHFULNESS_TOTAL_QUESTIONS = REVIEW_TOTAL_QUESTIONS

# Human review labels -----------------------------------
HUMAN_LABEL_CORRECT = "correct"
HUMAN_LABEL_INCORRECT = "incorrect"
HUMAN_LABEL_AMBIGUOUS = "ambiguous"

HUMAN_LABELS = [
    HUMAN_LABEL_CORRECT,
    HUMAN_LABEL_INCORRECT,
    HUMAN_LABEL_AMBIGUOUS,
]