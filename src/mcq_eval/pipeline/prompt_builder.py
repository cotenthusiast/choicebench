# src/mcq_eval/pipeline/prompt_builder.py

from pathlib import Path


_TEMPLATE_NAMES = ("direct_mcq", "free_text", "option_matching")


def load_prompt_templates(version: str, prompts_dir: Path) -> dict[str, str]:
    """Load all prompt templates for a given version from disk.

    Templates are plain text files with Python str.format-style placeholders.
    Each version lives in its own subdirectory under prompts_dir:

        prompts_dir / v1 / direct_mcq.txt
        prompts_dir / v1 / free_text.txt
        prompts_dir / v1 / option_matching.txt

    Args:
        version: Version string matching a subdirectory name (e.g. "v1").
        prompts_dir: Root directory containing versioned prompt folders.

    Returns:
        Dict mapping template name to raw template string.

    Raises:
        FileNotFoundError: If the version directory or any template file is missing.
    """
    version_dir = prompts_dir / version
    if not version_dir.is_dir():
        raise FileNotFoundError(
            f"Prompt version directory not found: {version_dir}. "
            f"Create {version_dir}/ with direct_mcq.txt, free_text.txt, "
            f"and option_matching.txt to use prompt version {version!r}."
        )

    templates = {}
    for name in _TEMPLATE_NAMES:
        path = version_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prompt template: {path}. "
                f"Expected one of: {[f'{n}.txt' for n in _TEMPLATE_NAMES]}"
            )
        templates[name] = path.read_text(encoding="utf-8")

    return templates


def build_direct_mcq_prompt(
    template: str,
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
) -> str:
    """Format the direct MCQ template with question and option text.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Question stem to present to the model.
        option_a: Text of answer option A.
        option_b: Text of answer option B.
        option_c: Text of answer option C.
        option_d: Text of answer option D.

    Returns:
        Fully formatted prompt string.
    """
    return template.format(
        question=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )


def build_free_text_prompt(template: str, question: str) -> str:
    """Format the free-text template with a question stem.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Question stem to present without answer options.

    Returns:
        Fully formatted prompt string.
    """
    return template.format(question=question)


def build_option_matching_prompt(
    template: str,
    question: str,
    free_text: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
) -> str:
    """Format the option-matching template for stage two of two-stage methods.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Original question stem.
        free_text: Free-text answer produced in stage one.
        option_a: Text of answer option A.
        option_b: Text of answer option B.
        option_c: Text of answer option C.
        option_d: Text of answer option D.

    Returns:
        Fully formatted prompt string.
    """
    return template.format(
        question=question,
        free_text=free_text,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )
