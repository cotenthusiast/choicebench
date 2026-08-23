# src/choicebench/pipeline/prompt_builder.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from choicebench.config.paths import PROMPTS_DIR
from choicebench.identity import integrity_digest, short_id


_TEMPLATE_NAMES = ("direct_mcq", "free_text", "option_matching")


def _build_options_block(options: dict[str, str]) -> str:
    """Render real options in label order without inventing missing choices."""
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def load_prompt_templates(version: str, prompts_dir: Path | None = None) -> dict[str, str]:
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
    if not version or version in {".", ".."} or "/" in version or "\\" in version:
        raise ValueError(f"Invalid logical prompt version: {version!r}")
    root = prompts_dir or PROMPTS_DIR
    version_dir = root / version
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


def prompt_bundle_identity(version: str, prompts_dir: Path | None = None) -> dict[str, object]:
    templates = load_prompt_templates(version, prompts_dir)
    files = {name: {"sha256": integrity_digest(text), "content": text} for name, text in templates.items()}
    payload = {"version": version, "files": files}
    return {"prompt_id": f"prompt_{integrity_digest(payload)[:16]}", **payload}


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


@dataclass(frozen=True)
class Rotation:
    """One cyclic rotation plus its explicit positional bijection.

    ``mapping`` renders display letter -> option text (what the model sees).
    ``slot_to_canonical[j]`` is the canonical index displayed at display
    position ``j`` under this rotation — the single source of the inverse
    mapping. Invariant: displayed option position -> exactly one canonical
    option position, independent of option text; the inverse of a parsed
    display letter is ``canonical_letters[slot_to_canonical[display_index]]``.
    """

    mapping: dict[str, str]
    slot_to_canonical: tuple[int, ...]


def build_rotations(options: dict[str, str]) -> list[Rotation]:
    """Generate every cyclic rotation of the canonical ordering.

    Each rotation carries both the rendered letter->text mapping and its
    positional bijection back to canonical slots, produced by the same
    operation so rendering and inverse logic can never drift apart (F1 fix:
    the inverse is positional and never consults option text). The number of
    rotations equals the number of options.
    """
    keys = list(options.keys())
    values = list(options.values())
    n = len(keys)
    return [
        Rotation(
            mapping=dict(zip(keys, values[i:] + values[:i])),
            slot_to_canonical=tuple((i + j) % n for j in range(n)),
        )
        for i in range(n)
    ]


def build_permuted_prompt(
    question_row: Any,
    permuted_options: dict[str, str],
    template: str,
) -> str:
    """Build a direct MCQ prompt using a permuted option ordering.

    Args:
        question_row: Normalized question record.
        permuted_options: Permuted letter-to-text mapping.
        template: Raw direct_mcq template string.

    Returns:
        Fully formatted prompt string with permuted options.
    """
    return build_direct_mcq_prompt(
        template=template,
        question=question_row["question_text"],
        options=permuted_options,
        subject=question_row["subject"],
    )
