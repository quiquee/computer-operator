"""
Loads prompt templates from the prompts/ directory.

Templates use simple {key} placeholders. Use prompts.fmt() to substitute
values without conflicting with literal curly braces that may appear in the
prompt text (e.g. JSON examples).
"""
import os

_dir = os.path.dirname(__file__)


def _load(name: str) -> str:
    with open(os.path.join(_dir, "prompts", name), encoding="utf-8") as f:
        return f.read()


GEMINI_SYSTEM_INSTRUCTION = _load("gemini.txt")
OLLAMA_SYSTEM_PROMPT      = _load("ollama.txt")


def fmt(template: str, **kwargs) -> str:
    """Replace {key} placeholders without calling str.format().

    This avoids KeyError / ValueError when the template contains literal
    curly braces (e.g. JSON schema examples such as {"action": "..."}).
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result
