# methods/

Researcher-facing home for MCQ experimental methods. Every method here talks
only to a `BaseBackend` (`src/twoprompt/backends/base.py`) — never to a
provider client, never to checkpoint/config plumbing directly. That's the
framework's core guarantee: implementing a new method means editing exactly
one file and touching zero framework internals.

All five methods now live here. `src/twoprompt/runners/` still exists with
the same five files, but each carries a `# DEPRECATED` notice pointing back
to its counterpart here — it is not deleted yet, only superseded.

## Included methods

- **Direct MCQ (`direct_mcq.py`)** — Single prompt, all four lettered options
  shown, parse the chosen letter. The unmitigated baseline every other
  method is measured against. Backend requirements: generate only.

- **Cyclic Permutation (`permutation.py`)** — N cyclic rotations of option
  order, one backend call per rotation, majority vote across the un-permuted
  results. Mitigates positional bias by averaging a question's answer over
  every position the correct option could occupy. Reference: Zheng et al.,
  ICLR 2024 (arXiv:2309.03882). Backend requirements: generate only.

- **Two-Stage Prompting (`two_stage.py`)** — Stage one elicits a free-text
  answer with no options shown (removing positional anchoring at the
  reasoning step); stage two asks the model to match that free-text answer
  to a lettered option. This project's primary research method. Backend
  requirements: generate only.

- **Two-Stage + Cyclic Permutation (`two_stage_permutation.py`)** — Combines
  the two above: one free-text elicitation, then N permuted option-matching
  calls with majority vote. Tests whether the two mitigations compose.
  Backend requirements: generate only.

- **PriDe (`pride.py`)** — Estimates a model's positional bias prior from a
  calibration set (cyclic permutation rollouts + Eq. 7), then debiases each
  evaluation question's observed answer distribution against that prior
  (Eq. 8) and picks the argmax via `backend.score_options()`. Ported from
  `model-generalization-paper`'s backend-based PriDe runner (read-only
  reference; that repo was not modified). Reference: Zheng et al., ICLR 2024
  (arXiv:2309.03882). **Backend requirements: generate + score_options.**

## How to add your own method

1. Copy `templates/base_method.py` to `methods/<your_method_name>.py`.
2. Rename the class and fill in the docstring block at the top (Description,
   Reference, Backend requirements, Logprob support required) — a researcher
   should be able to read just this block and know what your method does.
3. Implement your method using only `self.backend.generate(...)` and, if
   needed, `self.backend.score_options(...)`. Do not import from `clients/`,
   `infra/` (checkpointing), or read `config/` directly — if your method
   needs configuration, accept it as constructor arguments.
4. If your method needs logprobs, check `self.backend.supports_logprobs`
   before calling `score_options()` — most backends (every API backend in
   this project) do not implement it and will raise `NotImplementedError`.

## score_options() / logprob requirement

Only backends with direct model access can implement `score_options()`.
Today that means **`HuggingFaceBackend` only** — API backends
(`APIBackend`) are intentionally closed/opaque and raise
`NotImplementedError` on `score_options()` by design, since API providers do
not expose clean full-vocabulary logits the way local inference does.

**Methods that require `score_options()` (currently only PriDe) only work
with backends where `supports_logprobs` is `True` (currently:
`HuggingFaceBackend`).** Constructing such a method against any other
backend raises a `ValueError` immediately, before any backend call is made.
