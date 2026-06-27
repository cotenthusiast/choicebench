# src/mcq_eval/config/providers.py

from __future__ import annotations

import os

from dotenv import load_dotenv

from mcq_eval.config.schema import DEFAULT_MAX_NEW_TOKENS

# API keys --------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Default request settings ----------------------------------------------
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = DEFAULT_MAX_NEW_TOKENS
TIMEOUT = 30
MAX_RETRIES = 3
