"""
Central configuration.

All constants are loaded here. SCREEN_WIDTH / SCREEN_HEIGHT start as safe
defaults and are overwritten by vision.init_capture_card() at runtime once a
real frame is grabbed from the capture card.
"""


def _load_secrets(path="secrets.txt"):
    secrets = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                secrets[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return secrets


_secrets = _load_secrets()

# --- Gemini ---
API_KEY            = _secrets.get("API_KEY", "")
COMPUTER_USE_MODEL = "gemini-2.5-computer-use-preview-10-2025"

# --- Pi Zero ---
PI_IP_ADDRESS        = _secrets.get("PI_IP_ADDRESS", "192.168.1.1")

# --- Capture card ---
CAPTURE_DEVICE_INDEX = int(_secrets.get("CAPTURE_DEVICE_INDEX", "1"))

# Defaults — overwritten dynamically by vision.init_capture_card()
SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080

# --- Ollama ---
OLLAMA_HOST  = _secrets.get("OLLAMA_HOST",  "localhost")
OLLAMA_PORT  = _secrets.get("OLLAMA_PORT",  "11434")
OLLAMA_MODEL = _secrets.get("OLLAMA_MODEL", "qwen2.5vl:7b")
