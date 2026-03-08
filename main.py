"""
Entry point — backend and mode selection only.
"""
from gemini_agent import start_agent
from ollama_agent import start_ollama_agent
from calibration import run_calibration_test


def _pick(prompt: str, choices: tuple) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value in choices:
                return value
            print(f"Please enter one of: {', '.join(str(c) for c in choices)}")
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":
    print("=== AI Computer Operator ===")
    print("Backend:")
    print("  1. Google Gemini Computer Use API")
    print("  2. Ollama (local model)")
    backend = _pick("Select backend (1 or 2): ", (1, 2))

    if backend == 2:
        start_ollama_agent()
    else:
        print("\nMode:")
        print("  1. Normal mode (Computer Use agent)")
        print("  2. Calibration test mode")
        mode = _pick("Select mode (1 or 2): ", (1, 2))

        if mode == 1:
            start_agent()
        else:
            run_calibration_test()
