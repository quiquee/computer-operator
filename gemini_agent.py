"""
Gemini Computer Use agent.

Translates Gemini's built-in computer-use function calls into Pi Zero HID
commands and drives the conversation loop.
"""
import time
import base64
import json

from google import genai
from google.genai import types

import config
import prompts
import hardware
import vision
from logger import InteractionLogger

client = genai.Client(api_key=config.API_KEY)


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def execute_computer_use_action(fn_name: str, args: dict, state: dict) -> None:
    """Map a Gemini Computer Use function call to one or more Pi Zero HID commands.

    `state` carries at least {"current_url": <str>} and is mutated by
    open_web_browser so subsequent function responses contain the right URL.
    """
    if fn_name == "click_at":
        x, y = vision.norm_to_pixel(args.get("x", 0), args.get("y", 0))
        print(f"  -> mouse_move({x}, {y})")
        hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        print("  -> left_click")
        hardware.send_to_pi({"action": "left_click"})
        time.sleep(0.5)

    elif fn_name == "double_click_at":
        x, y = vision.norm_to_pixel(args.get("x", 0), args.get("y", 0))
        print(f"  -> mouse_move({x}, {y}) + double_click")
        hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        hardware.send_to_pi({"action": "double_click"})
        time.sleep(0.5)

    elif fn_name == "right_click_at":
        x, y = vision.norm_to_pixel(args.get("x", 0), args.get("y", 0))
        print(f"  -> mouse_move({x}, {y}) + right_click")
        hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(1.5)
        hardware.send_to_pi({"action": "right_click"})
        time.sleep(0.5)

    elif fn_name == "type_text_at":
        nx = args.get("x")
        ny = args.get("y")
        text = args.get("text", "")
        press_enter = args.get("pressEnter", args.get("press_enter", False))
        if nx is not None and ny is not None:
            x, y = vision.norm_to_pixel(nx, ny)
            print(f"  -> mouse_move({x}, {y}) + left_click")
            hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
            time.sleep(1.5)
            hardware.send_to_pi({"action": "left_click"})
            time.sleep(0.5)
        if press_enter:
            text = text + "[enter]"
        print(f"  -> type_text({text!r})")
        hardware.send_to_pi({"action": "type_text", "text": text})
        time.sleep(0.5)

    elif fn_name == "scroll_at":
        nx = args.get("x", 500)
        ny = args.get("y", 500)
        x, y = vision.norm_to_pixel(nx, ny)
        direction = args.get("direction", "down")
        amount = int(args.get("amount", 3))
        print(f"  -> mouse_move({x}, {y}) + scroll_{direction} x{amount}")
        hardware.send_to_pi({"action": "mouse_move", "x": x, "y": y})
        time.sleep(0.5)
        for _ in range(amount):
            hardware.send_to_pi({"action": f"scroll_{direction}"})
            time.sleep(0.1)
        time.sleep(0.3)

    elif fn_name == "key_press":
        keys = args.get("keys", args.get("key", ""))
        print(f"  -> key_press({keys!r})")
        hardware.send_to_pi({"action": "type_text", "text": keys})
        time.sleep(0.5)

    elif fn_name == "open_web_browser":
        url = args.get("url", "")
        if url:
            print(f"  -> navigate to {url!r} via Ctrl+L")
            hardware.send_to_pi({"action": "type_text", "text": "[ctrl l]"})
            time.sleep(0.5)
            hardware.send_to_pi({"action": "type_text", "text": url + "[enter]"})
            time.sleep(1.0)
            state["current_url"] = url
        else:
            print("  -> open_web_browser (no URL provided, no action)")

    elif fn_name == "screenshot":
        print("  -> screenshot (captured automatically)")

    else:
        print(f"  -> Unrecognized action: {fn_name}({args})")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def start_agent() -> None:
    browser_state = {"current_url": "about:blank"}
    print(f"Using model: {config.COMPUTER_USE_MODEL}")

    cap = vision.init_capture_card()

    computer_use_tool = types.Tool(
        computer_use=types.ComputerUse(
            environment=types.Environment.ENVIRONMENT_BROWSER
        )
    )

    system_instruction = prompts.fmt(
        prompts.GEMINI_SYSTEM_INSTRUCTION,
        width=config.SCREEN_WIDTH,
        height=config.SCREEN_HEIGHT,
    )

    user_goal = input("Enter your goal: ")
    logger = InteractionLogger()

    print("Taking initial screenshot...")
    screenshot_bytes = vision.get_screen_bytes(cap)
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
                    model=config.COMPUTER_USE_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=[computer_use_tool],
                        system_instruction=system_instruction,
                    ),
                )
            except Exception as api_err:
                print(f"\n[API ERROR] {api_err}")
                # Roll back to the last turn that contains plain user text so
                # the loop can retry cleanly without the bad model/fn-response turns.
                while contents and not any(
                    getattr(p, "text", None)
                    for p in getattr(contents[-1], "parts", [])
                ):
                    contents.pop()
                print(f"Rolled back to {len(contents)} turn(s). Retrying after 3 s...")
                time.sleep(3)
                continue

            candidate = response.candidates[0]
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
                    hardware._pi_responses.clear()
                    execute_computer_use_action(fn_name, args, browser_state)

                    # Settle, then capture result (Computer Use API requires PNG)
                    time.sleep(2.0)
                    png_bytes = vision.get_screen_bytes(cap, fmt='.png')
                    png_b64 = base64.b64encode(png_bytes).decode()

                    logger.save(
                        goal=user_goal,
                        model_thought="\n".join(model_text_parts),
                        model_action=fn_name,
                        model_args=args,
                        pi_commands=list(hardware._pi_responses),
                        image_bytes=png_bytes,
                    )

                    # The API requires current_url and safety_decision_acknowledged
                    # in every function response.
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
                contents.append(
                    types.Content(role="user", parts=function_response_parts)
                )
            elif not any_action:
                # Model produced only text — task may be done or needs guidance
                user_input = input("\nEnter feedback, or press Enter to continue: ")
                if not user_input.strip():
                    user_input = "Continue with the task."
                screenshot_bytes = vision.get_screen_bytes(cap)
                b64 = base64.b64encode(screenshot_bytes).decode()
                contents.append(
                    types.Content(role="user", parts=[
                        types.Part(text=user_input),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=b64)),
                    ])
                )

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
    finally:
        cap.release()
