"""
Ollama local-model agent.

Sends screenshots to a locally running Ollama vision model, parses the JSON
action it returns, and dispatches the action to the Pi Zero HID server.
"""
import json
import re
import time
import base64

import requests

import config
import prompts
import hardware
import vision
from logger import InteractionLogger


# ---------------------------------------------------------------------------
# Ollama API helpers
# ---------------------------------------------------------------------------

def _ollama_chat(ollama_url: str, model: str, messages: list) -> str:
    """POST to Ollama /api/chat and return the assistant message content string."""
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


def _parse_action(raw: str) -> dict:
    """Extract a JSON action dict from the model's raw response string."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def execute_ollama_action(action_dict: dict) -> bool:
    """Execute one action dict. Returns True when the model signals 'done'."""
    action = action_dict.get("action", "")
    thought = action_dict.get("thought", "")
    if thought:
        print(f"  Thought: {thought}")

    if action == "done":
        print(f"\nTask complete: {action_dict.get('message', '')}")
        return True

    elif action == "mouse_move":
        x, y = int(action_dict["x"]), int(action_dict["y"])
        print(f"  -> mouse_move({x}, {y})")
        hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(0.5)

    elif action == "left_click":
        nx, ny = action_dict.get("x"), action_dict.get("y")
        if nx is not None and ny is not None:
            x, y = int(nx), int(ny)
            print(f"  -> mouse_move({x}, {y}) + left_click")
            hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.5)
        hardware.send_to_pi({"action": "left_click"})
        time.sleep(0.5)

    elif action == "double_click":
        nx, ny = action_dict.get("x"), action_dict.get("y")
        if nx is not None and ny is not None:
            x, y = int(nx), int(ny)
            print(f"  -> mouse_move({x}, {y}) + double_click")
            hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.5)
        hardware.send_to_pi({"action": "double_click"})
        time.sleep(0.5)

    elif action == "type_text":
        nx, ny = action_dict.get("x"), action_dict.get("y")
        text = action_dict.get("text", "")
        if nx is not None and ny is not None:
            x, y = int(nx), int(ny)
            print(f"  -> mouse_move({x}, {y}) + left_click")
            hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.5)
            hardware.send_to_pi({"action": "left_click"})
            time.sleep(0.5)
        print(f"  -> type_text({text!r})")
        hardware.send_to_pi({"action": "type_text", "text": text})
        time.sleep(0.5)

    elif action == "scroll_down":
        clicks = int(action_dict.get("clicks", 3))
        print(f"  -> scroll_down x{clicks}")
        hardware.send_to_pi({"action": "scroll_down", "clicks": clicks})
        time.sleep(0.3)

    elif action == "scroll_up":
        clicks = int(action_dict.get("clicks", 3))
        print(f"  -> scroll_up x{clicks}")
        hardware.send_to_pi({"action": "scroll_up", "clicks": clicks})
        time.sleep(0.3)

    elif action == "page_down":
        print("  -> page_down")
        hardware.send_to_pi({"action": "page_down"})
        time.sleep(0.3)

    elif action == "page_up":
        print("  -> page_up")
        hardware.send_to_pi({"action": "page_up"})
        time.sleep(0.3)

    else:
        print(f"  -> Unknown action: {action}")

    return False


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def start_ollama_agent() -> None:
    ollama_url = f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}"
    print(f"Using Ollama model: {config.OLLAMA_MODEL} at {ollama_url}")

    cap = vision.init_capture_card()

    system_prompt = prompts.fmt(
        prompts.OLLAMA_SYSTEM_PROMPT,
        width=config.SCREEN_WIDTH,
        height=config.SCREEN_HEIGHT,
    )

    user_goal = input("Enter your goal: ")
    logger = InteractionLogger()

    print("Taking initial screenshot...")
    jpg_bytes = vision.get_screen_bytes(cap)
    b64 = base64.b64encode(jpg_bytes).decode()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_goal, "images": [b64]},
    ]

    print("\n--- Ollama Agent Active ---")
    try:
        while True:
            print("\nSending to Ollama...")
            raw = _ollama_chat(ollama_url, config.OLLAMA_MODEL, messages)
            print(f"Response: {raw[:300]}")

            try:
                action_dict = _parse_action(raw)
            except Exception as e:
                print(f"Failed to parse action JSON: {e}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "Your response was not valid JSON. "
                               "Reply with ONLY a JSON action object, no extra text.",
                })
                continue

            messages.append({"role": "assistant", "content": raw})

            hardware._pi_responses.clear()
            done = execute_ollama_action(action_dict)
            if done:
                break

            time.sleep(2.0)
            jpg_bytes = vision.get_screen_bytes(cap)
            b64 = base64.b64encode(jpg_bytes).decode()

            action_name = action_dict.get("action", "")
            action_args = {k: v for k, v in action_dict.items() if k not in ("action", "thought")}
            logger.save(
                goal=user_goal,
                model_thought=action_dict.get("thought", ""),
                model_action=action_name,
                model_args=action_args,
                pi_commands=list(hardware._pi_responses),
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
