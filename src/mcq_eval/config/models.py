# src/mcq_eval/config/models.py

from __future__ import annotations

import os

from dotenv import load_dotenv

# API keys --------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def validate_api_keys():
    if GEMINI_API_KEY is None:
        print("Gemini API key missing")
    if OPENAI_API_KEY is None:
        print("OpenAI API key missing")
    if GROQ_API_KEY is None:
        print("Groq API key missing")
    if TOGETHER_API_KEY is None:
        print("Together AI API key missing")

# Default request settings ----------------------------------------------
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 500
TIMEOUT = 30
MAX_RETRIES = 3

# Providers -------------------------------------------------------------
OPENAI_PROVIDER = "openai"
GEMINI_PROVIDER = "gemini"
GROQ_PROVIDER = "groq"
TOGETHER_PROVIDER = "together"

SUPPORTED_PROVIDERS = {
    OPENAI_PROVIDER,
    GEMINI_PROVIDER,
    GROQ_PROVIDER,
    TOGETHER_PROVIDER,
}

# Core / cheaper models -------------------------------------------------
OPENAI_CORE_MODEL = "gpt-4.1-mini"
GEMINI_CORE_MODEL = "gemini-2.5-flash"
GROQ_CORE_MODEL = "llama-3.1-8b-instant"

CORE_MODELS = [
    OPENAI_CORE_MODEL,
    GEMINI_CORE_MODEL,
    GROQ_CORE_MODEL,
]

# Stronger models -------------------------------------------------------
OPENAI_STRONG_MODEL = "gpt-4.1"
GEMINI_STRONG_MODEL = "gemini-2.5-pro"
GROQ_STRONG_MODEL = "llama-3.3-70b-versatile"

STRONG_MODELS = [
    OPENAI_STRONG_MODEL,
    GEMINI_STRONG_MODEL,
    GROQ_STRONG_MODEL,
]

# Together / open-weights models ----------------------------------------
TOGETHER_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"

TOGETHER_MODELS = [
    TOGETHER_QWEN_MODEL,
]

# All supported models --------------------------------------------------
SUPPORTED_MODELS_BY_PROVIDER = {
    OPENAI_PROVIDER: {
        OPENAI_CORE_MODEL,
        OPENAI_STRONG_MODEL,
    },
    GEMINI_PROVIDER: {
        GEMINI_CORE_MODEL,
        GEMINI_STRONG_MODEL,
    },
    GROQ_PROVIDER: {
        GROQ_CORE_MODEL,
        GROQ_STRONG_MODEL,
    },
    TOGETHER_PROVIDER: {
        TOGETHER_QWEN_MODEL,
    },
}

ALL_SUPPORTED_MODELS = sorted(
    {
        model_name
        for provider_models in SUPPORTED_MODELS_BY_PROVIDER.values()
        for model_name in provider_models
    }
)