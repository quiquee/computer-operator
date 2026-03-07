import cv2
import time
import base64
import requests
from google import genai
from google.genai import types

# --- CONFIGURATION ---
API_KEY = "AIzaSyBhq5PJGzKj0QnTbBG2BImV_AGeejDwGFw"
PI_IP_ADDRESS = "192.168.1.116"  # Update this to your Pi Zero's IP
CAPTURE_DEVICE_INDEX = 1         # Changed to 1 since /dev/video0 does not exist

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
COMPUTER_USE_MODEL = "gemini-2.5-computer-use-preview-10-2025"

client = genai.Client(api_key=API_KEY)


# --- HARDWARE COMMUNICATION ---
def send_to_pi(payload):
    """Sends the command to the Pi Zero HTTP Server"""
    url = f"http://{PI_IP_ADDRESS}:8080"
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"  Success: {payload['action']}")
        else:
            print(f"  Error from Pi: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"  Connection error to Pi: {e}")


# --- VISION ---
def _capture_frame(cap, flush_frames=3, fmt='.jpg'):
    """Captures a frame from the capture card and returns encoded image bytes.

    Grabs and discards `flush_frames` frames first to drain the capture buffer,
    ensuring the returned image is taken *after* any hardware input has settled.
    Use fmt='.png' when PNG is required (e.g. Computer Use function responses).
    """
    for _ in range(flush_frames):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to read from Capture Card. Check your CAPTURE_DEVICE_INDEX.")
    _, buffer = cv2.imencode(fmt, frame)
    return buffer.tobytes()


def get_screen_bytes(cap, flush_frames=3, fmt='.jpg'):
    return _capture_frame(cap, flush_frames, fmt)


# --- COORDINATE CONVERSION ---
def norm_to_pixel(nx, ny):
    """Convert Gemini's normalized 0-1000 coordinates to pixel coordinates."""
    x = max(0, min(SCREEN_WIDTH,  int(nx / 1000 * SCREEN_WIDTH)))
    y = max(0, min(SCREEN_HEIGHT, int(ny / 1000 * SCREEN_HEIGHT)))
    return x, y


# --- ACTION EXECUTOR ---
def execute_computer_use_action(fn_name, args, state):
    """Route a Gemini Computer Use function call to the Pi Zero HID server.

    `state` is a dict with at least a 'current_url' key that is mutated here
    whenever a navigation action is performed.
    """

    if fn_name == "click_at":
        x, y = norm_to_pixel(args.get("x", 0), args.get("y", 0))
        print(f"  -> mouse_move({x}, {y})")
        send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        print(f"  -> left_click")
        send_to_pi({"action": "left_click"})
        time.sleep(0.5)

    elif fn_name == "double_click_at":
        x, y = norm_to_pixel(args.get("x", 0), args.get("y", 0))
        print(f"  -> mouse_move({x}, {y}) + double_click")
        send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        send_to_pi({"action": "double_click"})
        time.sleep(0.5)

    elif fn_name == "right_click_at":
        x, y = norm_to_pixel(args.get("x", 0), args.get("y", 0))
        print(f"  -> mouse_move({x}, {y}) + right_click")
        send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        send_to_pi({"action": "right_click"})
        time.sleep(0.5)

    elif fn_name == "type_text_at":
        nx = args.get("x")
        ny = args.get("y")
        text = args.get("text", "")
        press_enter = args.get("pressEnter", args.get("press_enter", False))
        # Move to and click the target element first
        if nx is not None and ny is not None:
            x, y = norm_to_pixel(nx, ny)
            print(f"  -> mouse_move({x}, {y}) + left_click")
            send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.5)
            send_to_pi({"action": "left_click"})
            time.sleep(0.5)
        if press_enter:
            text = text + "[enter]"
        print(f"  -> type_text({text!r})")
        send_to_pi({"action": "type_text", "text": text})
        time.sleep(0.5)

    elif fn_name == "scroll_at":
        nx = args.get("x", 500)
        ny = args.get("y", 500)
        x, y = norm_to_pixel(nx, ny)
        direction = args.get("direction", "down")
        amount = int(args.get("amount", 3))
        print(f"  -> mouse_move({x}, {y}) + scroll_{direction} x{amount}")
        send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(0.5)
        for _ in range(amount):
            send_to_pi({"action": f"scroll_{direction}"})
            time.sleep(0.1)
        time.sleep(0.3)

    elif fn_name == "key_press":
        keys = args.get("keys", args.get("key", ""))
        print(f"  -> key_press({keys!r})")
        send_to_pi({"action": "type_text", "text": keys})
        time.sleep(0.5)

    elif fn_name == "open_web_browser":
        # Browser is already open; if a URL is provided navigate to it via kbd
        url = args.get("url", "")
        if url:
            print(f"  -> navigate to {url!r} via Ctrl+L")
            send_to_pi({"action": "type_text", "text": "[ctrl l]"})
            time.sleep(0.5)
            send_to_pi({"action": "type_text", "text": url + "[enter]"})
            time.sleep(1.0)
            state["current_url"] = url
        else:
            print(f"  -> open_web_browser (no URL provided, no action)")

    elif fn_name == "screenshot":
        # Model explicitly requested a screenshot — nothing to do, we always capture
        print(f"  -> screenshot (captured automatically)")

    else:
        print(f"  -> Unrecognized action: {fn_name}({args})")


# --- COMPUTER USE AGENT ---
def start_agent():
    # Tracks the current browser URL so every function response can include it
    # (the Computer Use API requires 'url' or 'current_url' in each response).
    browser_state = {"current_url": "about:blank"}
    print(f"Using model: {COMPUTER_USE_MODEL}")

    print("Initializing Capture Card...")
    cap = cv2.VideoCapture(CAPTURE_DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    computer_use_tool = types.Tool(
        computer_use=types.ComputerUse(
            environment=types.Environment.ENVIRONMENT_BROWSER
        )
    )

    system_instruction = (
        f"You are controlling a desktop computer with a {SCREEN_WIDTH}x{SCREEN_HEIGHT} display. "
        "You receive screenshots and interact with the computer using the provided tools. "
        "Coordinates are normalized: 0 is left/top and 1000 is right/bottom. "
        "Observe each screenshot carefully before acting. "
        "When the goal is fully accomplished, say so clearly."
    )

    user_goal = input("Enter your goal: ")

    print("Taking initial screenshot...")
    screenshot_bytes = get_screen_bytes(cap)
    b64 = base64.b64encode(screenshot_bytes).decode()

    contents = [
        types.Content(role="user", parts=[
            types.Part(text=user_goal),
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=b64)),
        ])
    ]

    print("\n--- AI Computer Use Agent Active ---")
    try:
        while True:
            print("\nSending to Gemini Computer Use API...")
            response = client.models.generate_content(
                model=COMPUTER_USE_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[computer_use_tool],
                    system_instruction=system_instruction,
                ),
            )

            candidate = response.candidates[0]
            # Append the model's response to the conversation history
            contents.append(candidate.content)

            function_response_parts = []
            any_action = False

            for part in candidate.content.parts:
                if part.text:
                    print(f"\nAI: {part.text.strip()}")

                if part.function_call:
                    fn = part.function_call
                    fn_name = fn.name
                    args = dict(fn.args) if fn.args else {}
                    any_action = True

                    print(f"\nAction -> {fn_name}({args})")
                    execute_computer_use_action(fn_name, args, browser_state)

                    # Let the screen settle, then capture the result
                    # Computer Use API requires image/png in function responses
                    time.sleep(2.0)
                    png_bytes = get_screen_bytes(cap, fmt='.png')
                    png_b64 = base64.b64encode(png_bytes).decode()

                    # Build a function response that includes the screenshot.
                    # The API always requires 'url' or 'current_url' in response.
                    function_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=fn.id,
                                name=fn_name,
                                response={
                                    "output": "success",
                                    "current_url": browser_state["current_url"],
                                },
                                parts=[
                                    types.FunctionResponsePart(
                                        inline_data=types.FunctionResponseBlob(
                                            mime_type="image/png",
                                            data=png_b64,
                                        )
                                    )
                                ],
                            )
                        )
                    )

            if function_response_parts:
                # Return all action results in a single user turn
                contents.append(
                    types.Content(role="user", parts=function_response_parts)
                )
            elif not any_action:
                # Model only produced text — task is done or it needs guidance
                user_input = input("\nEnter feedback, or press Enter to continue: ")
                if not user_input.strip():
                    user_input = "Continue with the task."
                screenshot_bytes = get_screen_bytes(cap)
                b64 = base64.b64encode(screenshot_bytes).decode()
                contents.append(
                    types.Content(role="user", parts=[
                        types.Part(text=user_input),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=b64)),
                    ])
                )

            time.sleep(1.0)  # Brief throttle between API calls

    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
    finally:
        cap.release()


# --- CALIBRATION TEST ---
def run_calibration_test():
    """Test mode: moves the mouse to the 4 corners and asks Gemini to confirm position."""
    print("Fetching available models...")
    available_models = []
    for m in client.models.list():
        if hasattr(m, 'name'):
            available_models.append(m.name)

    print("\nAvailable Models:")
    for i, m_name in enumerate(available_models):
        print(f"{i+1}. {m_name}")

    while True:
        try:
            choice = int(input("Select a model (number): ")) - 1
            if 0 <= choice < len(available_models):
                selected_model = available_models[choice]
                break
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"Using model: {selected_model}")

    print("Initializing Capture Card...")
    cap = cv2.VideoCapture(CAPTURE_DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    OFFSET = 40
    corners = [
        ("Top-Left",     OFFSET,                    OFFSET),
        ("Top-Right",    SCREEN_WIDTH  - OFFSET,    OFFSET),
        ("Bottom-Right", SCREEN_WIDTH  - OFFSET,    SCREEN_HEIGHT - OFFSET),
        ("Bottom-Left",  OFFSET,                    SCREEN_HEIGHT - OFFSET),
    ]

    print("\n--- Calibration Test Mode ---")
    print("The mouse will be moved to each corner. Gemini will verify the cursor position.\n")

    try:
        for label, x, y in corners:
            print(f"Moving mouse to {label} corner ({x}, {y})...")
            send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.0)

            screenshot_bytes = get_screen_bytes(cap)
            b64 = base64.b64encode(screenshot_bytes).decode()

            prompt = (
                f"The screen resolution is {SCREEN_WIDTH}x{SCREEN_HEIGHT}. "
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


if __name__ == "__main__":
    print("=== AI Computer Operator ===")
    print("1. Normal mode (Gemini Computer Use API)")
    print("2. Calibration test mode")
    while True:
        try:
            mode = int(input("Select mode (1 or 2): "))
            if mode in (1, 2):
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

    if mode == 1:
        start_agent()
    else:
        run_calibration_test()
