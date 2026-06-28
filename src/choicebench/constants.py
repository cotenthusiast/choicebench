# src/choicebench/constants.py

MCQ_OPTIONS: list[str] = ["A", "B", "C", "D"]

MCQ_ANSWER_MAP: dict[int, str] = {i: letter for i, letter in enumerate(MCQ_OPTIONS)}
