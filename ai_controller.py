import cv2
import time
import base64
import json
import os
import re
import requests
import numpy as np
from google import genai
from google.genai import types

# --- CONFIGURATION ---
def _load_secrets(path="secrets.txt"):
    secrets = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            secrets[key.strip()] = value.strip()
    return secrets

_secrets = _load_secrets()
API_KEY       = _secrets["API_KEY"]
PI_IP_ADDRESS = _secrets["PI_IP_ADDRESS"]
CAPTURE_DEVICE_INDEX = 1         # Changed to 1 since /dev/video0 does not exist

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
COMPUTER_USE_MODEL = "gemini-2.5-computer-use-preview-10-2025"

OLLAMA_SYSTEM_PROMPT = """\
You are an AI agent controlling a desktop computer with a {width}x{height} screen.
You receive a screenshot after each action and must issue ONE action at a time.

Respond ONLY with a JSON object — no extra text, no markdown fences. Schema:
Observe each screenshot carefully before acting. 
When the goal is fully accomplished, say so clearly.
        
Respond ONLY with a JSON object — no extra text, no markdown fences. Schema:
{{\"action\": \"action\",     
  \"x\": <int>, \"y\": <int>,
        "   \"thought\": \"<why>\"}} 

Actions are one of:  mouse_move, left_click, double_click, type_text, scroll_down, scroll_up, page_down, page_up
Explain what you intend to do in the field \"thought\"
Coordinates are pixel values: x 0-{width}, y 0-{height}.
Do only use the provided actions, not your own made-up ones like "search" or "navigate". Instead, use the tools you have: mouse clicks, typing, and key presses to accomplish those higher-level tasks.
Special key use bracket notation in text/keys fields: [enter] [tab] [escape]
Control - letter combinations use bracket notation as  [ctrl letter] 
"""

client = genai.Client(api_key=API_KEY)

_pi_responses: list = []  # accumulates Pi Zero command/response records for the current action step


# --- HARDWARE COMMUNICATION ---
def send_to_pi(payload):
    """Sends the command to the Pi Zero HTTP Server and records the result."""
    url = f"http://{PI_IP_ADDRESS}:8080"
    entry = {"cmd": payload}
    try:
        response = requests.post(url, json=payload, timeout=5)
        entry["http_status"] = response.status_code
        entry["body"] = response.text.strip()
        if response.status_code == 200:
            print(f"  OK [{response.status_code}]: {payload['action']}")
        else:
            print(f"  Error [{response.status_code}] from Pi: {response.text}")
    except requests.exceptions.RequestException as e:
        entry["http_status"] = None
        entry["body"] = None
        entry["error"] = str(e)
        print(f"  Connection error to Pi: {e}")
    _pi_responses.append(entry)


# --- INTERACTION LOGGER ---
class InteractionLogger:
    """Saves a PNG screenshot and a structured text file for each action step."""

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        existing = [f for f in os.listdir(log_dir) if re.match(r"capture-\d+\.png", f)]
        nums = [int(re.search(r"(\d+)", f).group(1)) for f in existing]
        self.counter = max(nums) + 1 if nums else 1

    def save(self, goal, model_thought, model_action, model_args, pi_commands, image_bytes):
        """Save screenshot + structured interaction log.

        pi_commands is a list of dicts: {cmd, http_status, body, error?}
        as recorded by send_to_pi().
        """
        n = self.counter
        self.counter += 1

        img_path = os.path.join(self.log_dir, f"capture-{n:03d}.png")
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imwrite(img_path, frame)

        txt_path = os.path.join(self.log_dir, f"interaction-{n:03d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            # --- Goal ---
            f.write(f"=== GOAL ===\n{goal}\n\n")

            # --- What the model decided ---
            model_section = ""
            if model_thought:
                model_section += f"Thought:\n{model_thought}\n\n"
            model_section += f"Action: {model_action}\n"
            if model_args:
                model_section += f"Args:\n{json.dumps(model_args, indent=2)}\n"
            f.write(f"=== MODEL RESPONSE ===\n{model_section}\n")

            # --- Each HTTP command sent to Pi Zero ---
            if pi_commands:
                cmds = ""
                for i, entry in enumerate(pi_commands, 1):
                    cmds += f"[{i}] {json.dumps(entry['cmd'])}\n"
                f.write(f"=== COMMANDS SENT TO Pi Zero ===\n{cmds}\n")
            else:
                f.write("=== COMMANDS SENT TO Pi Zero ===\n(none)\n\n")

            # --- Response / status from Pi Zero for each command ---
            if pi_commands:
                resps = ""
                for i, entry in enumerate(pi_commands, 1):
                    if entry.get("error"):
                        resps += f"[{i}] ERROR: {entry['error']}\n"
                    else:
                        status = entry.get("http_status", "?")
                        body   = entry.get("body", "")
                        resps += f"[{i}] HTTP {status}: {body}\n"
                f.write(f"=== Pi Zero RESPONSES ===\n{resps}\n")
            else:
                f.write("=== Pi Zero RESPONSES ===\n(none)\n\n")

        self._update_manifest(n)
        print(f"  [Log] capture-{n:03d}.png + interaction-{n:03d}.txt")

    def _update_manifest(self, new_n):
        """Keep logs/manifest.json in sync so the JS viewer needs no server logic."""
        manifest_path = os.path.join(self.log_dir, "manifest.json")
        try:
            with open(manifest_path) as f:
                nums = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            nums = []
        if new_n not in nums:
            nums.append(new_n)
            nums.sort()
        with open(manifest_path, "w") as f:
            json.dump(nums, f)


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
        
        "Respond ONLY with a JSON object — no extra text, no markdown fences. Schema:"
        "{{\"action\": \"action\",     " 
        "   \"x\": <int>, \"y\": <int>,"
        "   \"thought\": \"<why>\"}} "

        "Actions are one of:  mouse_move, left_click, double_click, type_text, scroll_down, scroll_up, page_down, page_up"
        "Explain what you intend to do in the field \"thought\""
        "Coordinates are pixel values: x 0-{width}, y 0-{height}."

        "Special key use bracket notation in text/keys fields: [enter] [tab] [escape]"
        "Control - letter combinations use bracket notation as  [ctrl letter] "
    )

    user_goal = input("Enter your goal: ")
    logger = InteractionLogger()

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
            try:
                response = client.models.generate_content(
                    model=COMPUTER_USE_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=[computer_use_tool],
                        system_instruction=system_instruction,
                    ),
                )
            except Exception as api_err:
                print(f"\n[API ERROR] {api_err}")
                # Roll back the conversation to the last clean user turn so the
                # loop can retry.  The bad exchange looks like:
                #   [..., user(screenshot/goal), model(bad fn call), user(fn response)]
                # We pop until we're back to a user turn that has plain text
                # (i.e. not a function-response turn), then retry.
                while contents and not any(
                    getattr(p, "text", None)
                    for p in getattr(contents[-1], "parts", [])
                ):
                    contents.pop()
                print(f"Rolled back to {len(contents)} turn(s). Retrying after 3 s...")
                time.sleep(3)
                continue

            candidate = response.candidates[0]
            # Append the model's response to the conversation history
            contents.append(candidate.content)

            function_response_parts = []
            any_action = False
            model_text_parts = []

            for part in candidate.content.parts:
                if part.text:
                    print(f"\nAI: {part.text.strip()}")
                    model_text_parts.append(part.text.strip())

                if part.function_call:
                    fn = part.function_call
                    fn_name = fn.name
                    args = dict(fn.args) if fn.args else {}
                    any_action = True

                    print(f"\nAction -> {fn_name}({args})")
                    _pi_responses.clear()
                    execute_computer_use_action(fn_name, args, browser_state)

                    # Let the screen settle, then capture the result
                    # Computer Use API requires image/png in function responses
                    time.sleep(2.0)
                    png_bytes = get_screen_bytes(cap, fmt='.png')
                    png_b64 = base64.b64encode(png_bytes).decode()

                    logger.save(
                        goal=user_goal,
                        model_thought="\n".join(model_text_parts),
                        model_action=fn_name,
                        model_args=args,
                        pi_commands=list(_pi_responses),
                        image_bytes=png_bytes,
                    )

                    # Build a function response that includes the screenshot.
                    # The API requires 'current_url' and 'safety_decision_acknowledged'
                    # in every function response — the latter confirms the client has
                    # seen and accepted the safety decision attached to the function call.
                    function_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=fn.id,
                                name=fn_name,
                                response={
                                    "output": "success",
                                    "current_url": browser_state["current_url"],
                                    "safety_decision_acknowledged": True,
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


# --- OLLAMA BACKEND ---

def _ollama_chat(ollama_url, model, messages):
    """POST to Ollama /api/chat and return the assistant message content."""
    resp = requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            # Do NOT use format:"json" with vision models — the grammar sampler
            # conflicts with multimodal token processing and causes the model to
            # emit repeated <|im_start|> tokens instead of valid output.
            # JSON output is enforced through the system prompt instead.
            "options": {
                "temperature": 0.1,
                "repeat_penalty": 1.15,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_action(raw):
    """Extract a JSON action dict from the model's response string."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    return json.loads(cleaned)


def execute_ollama_action(action_dict):
    """Execute an action dict produced by the Ollama model. Returns True when done."""
    action = action_dict.get("action", "")
    thought = action_dict.get("thought", "")
    if thought:
        print(f"  Thought: {thought}")

    if action == "done":
        print(f"\nTask complete: {action_dict.get('message', '')}")
        return True

    elif action == "click_at":
        x, y = int(action_dict["x"]), int(action_dict["y"])
        print(f"  -> mouse_move({x}, {y}) + left_click")
        send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        send_to_pi({"action": "left_click"})
        time.sleep(0.5)

    elif action == "double_click_at":
        x, y = int(action_dict["x"]), int(action_dict["y"])
        print(f"  -> mouse_move({x}, {y}) + double_click")
        send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        send_to_pi({"action": "double_click"})
        time.sleep(0.5)

    elif action == "right_click_at":
        x, y = int(action_dict["x"]), int(action_dict["y"])
        print(f"  -> mouse_move({x}, {y}) + right_click")
        send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        send_to_pi({"action": "right_click"})
        time.sleep(0.5)

    elif action == "type_text":
        nx, ny = action_dict.get("x"), action_dict.get("y")
        text = action_dict.get("text", "")
        if nx is not None and ny is not None:
            x, y = int(nx), int(ny)
            print(f"  -> mouse_move({x}, {y}) + left_click")
            send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.5)
            send_to_pi({"action": "left_click"})
            time.sleep(0.5)
        print(f"  -> type_text({text!r})")
        send_to_pi({"action": "type_text", "text": text})
        time.sleep(0.5)

    elif action == "scroll":
        x = int(action_dict.get("x", SCREEN_WIDTH // 2))
        y = int(action_dict.get("y", SCREEN_HEIGHT // 2))
        direction = action_dict.get("direction", "down")
        amount = int(action_dict.get("amount", 3))
        print(f"  -> mouse_move({x}, {y}) + scroll_{direction} x{amount}")
        send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(0.5)
        for _ in range(amount):
            send_to_pi({"action": f"scroll_{direction}"})
            time.sleep(0.1)
        time.sleep(0.3)

    elif action == "key_press":
        keys = action_dict.get("keys", "")
        print(f"  -> key_press({keys!r})")
        send_to_pi({"action": "type_text", "text": keys})
        time.sleep(0.5)

    else:
        print(f"  -> Unknown action: {action}")

    return False


def start_ollama_agent():
    ollama_host  = _secrets.get("OLLAMA_HOST",  "localhost")
    ollama_port  = _secrets.get("OLLAMA_PORT",  "11434")
    ollama_model = _secrets.get("OLLAMA_MODEL", "qwen2.5vl:7b")
    ollama_url   = f"http://{ollama_host}:{ollama_port}"

    print(f"Using Ollama model: {ollama_model} at {ollama_url}")

    print("Initializing Capture Card...")
    cap = cv2.VideoCapture(CAPTURE_DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    system_prompt = OLLAMA_SYSTEM_PROMPT.format(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    user_goal = input("Enter your goal: ")
    logger = InteractionLogger()

    print("Taking initial screenshot...")
    jpg_bytes = get_screen_bytes(cap)
    b64 = base64.b64encode(jpg_bytes).decode()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_goal, "images": [b64]},
    ]

    print("\n--- Ollama Agent Active ---")
    try:
        while True:
            print("\nSending to Ollama...")
            raw = _ollama_chat(ollama_url, ollama_model, messages)
            print(f"Response: {raw[:300]}")

            try:
                action_dict = _parse_action(raw)
            except Exception as e:
                print(f"Failed to parse action JSON: {e}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Your response was not valid JSON. Reply with ONLY a JSON action object, no extra text."})
                continue

            messages.append({"role": "assistant", "content": raw})

            _pi_responses.clear()
            done = execute_ollama_action(action_dict)
            if done:
                break

            # Capture result screen and continue the loop
            time.sleep(2.0)
            jpg_bytes = get_screen_bytes(cap)
            b64 = base64.b64encode(jpg_bytes).decode()

            action_name = action_dict.get("action", "")
            action_args = {k: v for k, v in action_dict.items() if k not in ("action", "thought")}
            logger.save(
                goal=user_goal,
                model_thought=action_dict.get("thought", ""),
                model_action=action_name,
                model_args=action_args,
                pi_commands=list(_pi_responses),
                image_bytes=jpg_bytes,
            )

            messages.append({
                "role": "user",
                "content": "Action executed. Here is the updated screen. Continue towards the goal.",
                "images": [b64],
            })
            time.sleep(1.0)

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
    print("Backend:")
    print("  1. Google Gemini Computer Use API")
    print("  2. Ollama (local model)")
    while True:
        try:
            backend = int(input("Select backend (1 or 2): "))
            if backend in (1, 2):
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

    if backend == 2:
        start_ollama_agent()
    else:
        print("\nMode:")
        print("  1. Normal mode (Computer Use agent)")
        print("  2. Calibration test mode")
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
