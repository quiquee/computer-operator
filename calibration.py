"""
Calibration test mode.

Moves the mouse to the four screen corners and asks a chosen Gemini model to
verify the cursor position in each screenshot.
"""
import time
import base64

from google import genai
from google.genai import types

import config
import hardware
import vision

client = genai.Client(api_key=config.API_KEY)


def run_calibration_test() -> None:
    print("Fetching available models...")
    available_models = [
        m.name for m in client.models.list() if hasattr(m, "name")
    ]

    print("\nAvailable Models:")
    for i, name in enumerate(available_models, 1):
        print(f"  {i}. {name}")

    while True:
        try:
            choice = int(input("Select a model (number): ")) - 1
            if 0 <= choice < len(available_models):
                selected_model = available_models[choice]
                break
            print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"Using model: {selected_model}")

    cap = vision.init_capture_card()

    OFFSET = 40
    corners = [
        ("Top-Left",     OFFSET,                          OFFSET),
        ("Top-Right",    config.SCREEN_WIDTH  - OFFSET,   OFFSET),
        ("Bottom-Right", config.SCREEN_WIDTH  - OFFSET,   config.SCREEN_HEIGHT - OFFSET),
        ("Bottom-Left",  OFFSET,                          config.SCREEN_HEIGHT - OFFSET),
    ]

    print("\n--- Calibration Test Mode ---")
    print("The mouse will be moved to each corner. Gemini will verify the cursor position.\n")

    try:
        for label, x, y in corners:
            print(f"Moving mouse to {label} corner ({x}, {y})...")
            hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.0)

            screenshot_bytes = vision.get_screen_bytes(cap)
            b64 = base64.b64encode(screenshot_bytes).decode()

            prompt = (
                f"The screen resolution is {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}. "
                f"The mouse cursor was just moved to the {label} corner at approximately ({x}, {y}). "
                "Look at the screenshot and locate the mouse cursor. "
                "Report: (1) whether you can see the cursor, (2) your best estimate of its X,Y position, "
                "and (3) whether this matches the expected corner position. "
                "Be concise."
            )

            print(f"Asking Gemini to verify cursor at {label}...")
            response = client.models.generate_content(
                model=selected_model,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=b64)),
                    ])
                ],
            )
            print(f"  Gemini says: {response.text.strip()}\n")

    except KeyboardInterrupt:
        print("\nCalibration test stopped by user.")
    finally:
        cap.release()

    print("--- Calibration Test Complete ---")
