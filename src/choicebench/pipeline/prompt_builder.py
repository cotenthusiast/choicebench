# src/choicebench/pipeline/prompt_builder.py

from pathlib import Path


_TEMPLATE_NAMES = ("direct_mcq", "free_text", "option_matching")


def _build_options_block(options: dict[str, str]) -> str:
    """Render real options in label order without inventing missing choices."""
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


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
    options: dict[str, str],
    subject: str | None = None,
) -> str:
    """Format the direct MCQ template with question, option text, and subject.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Question stem to present to the model.
        options: Mapping from answer label to option text.
        subject: Optional subject/category label (e.g. MMLU's per-question
            subject). Underscores are replaced with spaces before
            substitution, matching the paper's reference implementation
            convention (e.g. "abstract_algebra" -> "abstract algebra"). Only
            templates with a {subject} placeholder need this (v1's does not);
            omitted entirely from .format()'s kwargs when None, so a template
            that does reference {subject} without one being supplied raises
            KeyError rather than silently rendering the literal word "None".

    Returns:
        Fully formatted prompt string.
    """
    format_kwargs = {"question": question, "options": _build_options_block(options)}
    if subject is not None:
        format_kwargs["subject"] = subject.replace("_", " ")
    return template.format(**format_kwargs)


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
    options: dict[str, str],
) -> str:
    """Format the option-matching template for stage two of two-stage methods.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Original question stem.
        free_text: Free-text answer produced in stage one.
        options: Mapping from answer label to option text.

    Returns:
        Fully formatted prompt string.
    """
    return template.format(
        question=question,
        free_text=free_text,
        options=_build_options_block(options),
    )
