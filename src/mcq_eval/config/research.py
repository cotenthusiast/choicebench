# src/mcq_eval/config/research.py

from mcq_eval.constants import MCQ_OPTIONS

# MMLU dataset constants --------------------------------
MCQ_ANSWER_MAP = {i: L for i, L in enumerate(MCQ_OPTIONS)}
MMLU_QUESTIONS_PER_SUBJECT = 50

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

ROBUSTNESS_SPLIT_SEED = 42

# Shared review split -----------------------------------
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
