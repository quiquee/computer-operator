import cv2
import time
import base64
import requests
import google.generativeai as genai
from google.generativeai.types import content_types

# --- CONFIGURATION ---
API_KEY = "AIzaSyBhq5PJGzKj0QnTbBG2BImV_AGeejDwGFw"
PI_IP_ADDRESS = "192.168.1.116"  # Update this to your Pi Zero's IP
CAPTURE_DEVICE_INDEX = 1         # Changed to 1 since /dev/video0 does not exist

# Configure Gemini
genai.configure(api_key=API_KEY)

# --- HARDWARE COMMUNICATION ---
def send_to_pi(payload):
    """Sends the command to the Pi Zero HTTP Server"""
    url = f"http://{PI_IP_ADDRESS}:8080"
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"Success: {payload['action']}")
        else:
            print(f"Error from Pi: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Connection error to Pi: {e}")

# --- GEMINI TOOLS SCHEMA ---
tools_schema = [
    {
        "function_declarations": [
            {
                "name": "mouse_move",
                "description": "Moves the mouse cursor to a specific X,Y coordinate on the 1920x1080 screen.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "x": {"type": "NUMBER", "description": "X coordinate (0-1920) representing horizontal position from left to right"},
                        "y": {"type": "NUMBER", "description": "Y coordinate (0-1080) representing vertical position from top to bottom"}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "left_click",
                "description": "Performs a left mouse click at the current position.",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "double_click",
                "description": "Performs a double left mouse click at the current position.",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "type_text",
                "description": "Types text on the keyboard.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "text": {"type": "STRING", "description": "The text to type"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "task_completed",
                "description": "Call this tool when the overall goal has been successfully accomplished.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "message": {"type": "STRING", "description": "A brief summary of what was accomplished."}
                    }
                }
            }
        ]
    }
]

# --- VISION AND LOOP ---
def get_screen_base64(cap, flush_frames=3):
    """Captures a frame from the capture card and encodes it.

    Grabs and discards `flush_frames` frames first to drain the capture buffer,
    ensuring the returned image is taken *after* any mouse move has been sent.
    """
    for _ in range(flush_frames):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to read from Capture Card. Check your CAPTURE_DEVICE_INDEX.")
    
    # Optional: Resize to save tokens/bandwidth if your API calls are too large
    # Make sure to implement coordinate scaling if you resize!
    # frame = cv2.resize(frame, (1024, 576))
    
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def start_agent():
    print("Fetching available models...")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    
    print("\nAvailable Models:")
    for i, m_name in enumerate(available_models):
        print(f"{i+1}. {m_name}")
    
    while True:
        try:
            choice = int(input("Select a model (number): ")) - 1
            if 0 <= choice < len(available_models):
                selected_model = available_models[choice]
                if selected_model.startswith('models/'):
                    selected_model = selected_model[7:]
                break
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")
            
    print(f"Using model: {selected_model}")

    # 1. Initialize Capture Card
    print("Initializing Capture Card...")
    cap = cv2.VideoCapture(CAPTURE_DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # Ensure buffer is 1 for lowest latency on Linux
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    # 2. Initialize Gemini Model
    model = genai.GenerativeModel(selected_model, tools=tools_schema)
    chat = model.start_chat(enable_automatic_function_calling=False)

    print("\n--- AI Agent Active ---")
    
    # Context note to explicitly inform the model of the screen dimensions
    system_setup_instruction = (
        "IMPORTANT: The screen is in landscape orientation, width is 1920 and height is 1080. "
        "For your FIRST response, you must ONLY propose a step-by-step course of action based on the screenshot to achieve the user's goal. "
        "DO NOT make any tool calls in this first response. Wait for the user to approve the plan. "
        "Always provide a brief explanation of what you are doing and why before making tool calls. "
        "CRITICAL: To edit a browser's navigation bar/URL, ALWAYS use keyboard shortcuts like typing 'ctrl l' instead of clicking the address bar. "
        "Enter should be expressed as [enter] and tab as [tab] when typing text."
    )
    
    # The initial command to start the loop
    user_goal = input("Enter your initial goal for the AI (e.g., 'Open the web browser and go to Wikipedia'): ")
    instruction = system_setup_instruction + user_goal

    try:
        while True:
            # 1. Look at the screen
            b64_image = get_screen_base64(cap)
            
            # 2. Send image and instruction to Gemini
            print("Sending vision to Gemini...")
            response = chat.send_message([
                instruction,
                {"mime_type": "image/jpeg", "data": b64_image}
            ])

            # 3. Parse Gemini's response for tool calls
            action_taken = False
            task_is_completed = False
            for part in response.parts:
                try:
                    if part.text:
                        print(f"\nAI Thought -> {part.text.strip()}")
                except Exception:
                    pass

                if fn := part.function_call:
                    action_taken = True
                    action_name = fn.name
                    args = dict(fn.args)
                    
                    if action_name == "task_completed":
                        task_is_completed = True
                        print(f"\nAI Decision -> task_completed: {args.get('message', 'Task done.')}")
                        break

                    # --- ADD THESE TWO LINES ---
                    if 'x' in args: args['x'] = int(args['x'])
                    if 'y' in args: args['y'] = int(args['y'])
                    print(f"\nAI Decision -> {action_name}({args})")
                    
                    # 4. Route the command to the Pi Zero
                    payload = {"action": action_name}
                    payload.update(args)
                    send_to_pi(payload)

                    # Allow the OS to process the hardware input.
                    # mouse_move needs extra settle time before a screenshot is useful.
                    if action_name == "mouse_move":
                        time.sleep(1.5)
                    else:
                        time.sleep(0.5)
            
            if task_is_completed:
                print("\nTask successfully completed. Exiting AI control loop.")
                break

            # 5. Determine the next step
            if action_taken:
                # Tell Gemini the action was completed so it can look at the screen again
                instruction = "Action completed. Look at the screen and execute the next step. If the full goal has been achieved, call the 'task_completed' tool."
            else:
                # If Gemini just responded with text (no tool calls), print it out
                print(f"\nGemini proposes/says:\n{response.text}")
                instruction = input("\nEnter feedback, or press Enter to approve and proceed: ")
                if not instruction.strip():
                    instruction = "The plan is approved. Proceed."
                    
            time.sleep(2.5)  # Throttle API calls and ensure the screen has settled before the next capture

    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
    finally:
        cap.release()

def run_calibration_test():
    """Test mode: moves the mouse to the 4 corners and asks Gemini to confirm position."""
    print("Fetching available models...")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)

    print("\nAvailable Models:")
    for i, m_name in enumerate(available_models):
        print(f"{i+1}. {m_name}")

    while True:
        try:
            choice = int(input("Select a model (number): ")) - 1
            if 0 <= choice < len(available_models):
                selected_model = available_models[choice]
                if selected_model.startswith('models/'):
                    selected_model = selected_model[7:]
                break
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"Using model: {selected_model}")

    print("Initializing Capture Card...")
    cap = cv2.VideoCapture(CAPTURE_DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    model = genai.GenerativeModel(selected_model)

    # Inset from the true corners so the cursor is clearly visible
    OFFSET = 40
    corners = [
        ("Top-Left",     OFFSET,        OFFSET),
        ("Top-Right",    1920 - OFFSET, OFFSET),
        ("Bottom-Right", 1920 - OFFSET, 1080 - OFFSET),
        ("Bottom-Left",  OFFSET,        1080 - OFFSET),
    ]

    print("\n--- Calibration Test Mode ---")
    print("The mouse will be moved to each corner. Gemini will verify the cursor position.\n")

    try:
        for label, x, y in corners:
            print(f"Moving mouse to {label} corner ({x}, {y})...")
            send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.0)  # Allow the OS to process the move

            b64_image = get_screen_base64(cap)

            prompt = (
                f"The screen resolution is 1920x1080. "
                f"The mouse cursor was just moved to the {label} corner at approximately ({x}, {y}). "
                "Look at the screenshot and locate the mouse cursor. "
                "Report: (1) whether you can see the cursor, (2) your best estimate of its X,Y position, "
                "and (3) whether this matches the expected corner position. "
                "Be concise."
            )

            print(f"Asking Gemini to verify cursor at {label}...")
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": b64_image}
            ])
            print(f"  Gemini says: {response.text.strip()}\n")

    except KeyboardInterrupt:
        print("\nCalibration test stopped by user.")
    finally:
        cap.release()

    print("--- Calibration Test Complete ---")


if __name__ == "__main__":
    print("=== AI Computer Operator ===")
    print("1. Normal mode")
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
