These three files are byte-for-byte copies of `two-stage-prompting/prompts/v1/*.txt`
(verified via `diff`, commit `b8e784f3eb5d2a727a97eb675140b383a34584fa`).

`choicebench.methods.base.ExperimentRunner.__init__` calls
`load_prompt_templates(prompt_version, prompts_dir)`, which requires all of
`direct_mcq.txt`, `free_text.txt`, and `option_matching.txt` to exist in the
same versioned directory — that is a structural contract of the shared base
class, not something `VisibleLlmMatcherRunner` can opt out of without
overriding `__init__` entirely.

Only `option_matching.txt` is actually used by `VisibleLlmMatcherRunner`
(Stage 2, the LLM matcher — see `../../historical_protocol.py`). This
runner makes no Stage-1 call and no fallback call, so `free_text.txt` and
`direct_mcq.txt` are present only to satisfy the loader and are otherwise
dead weight here. They are kept identical to TSP's originals (rather than,
say, stub files) so that if a later change ever does need them, there is no
question of provenance.
