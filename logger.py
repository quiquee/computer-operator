"""
Interaction logger.

Saves a PNG screenshot and a structured text file for every action step,
then keeps logs/manifest.json in sync for the JS viewer.
"""
import os
import re
import json
import cv2
import numpy as np


class InteractionLogger:

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        existing = [f for f in os.listdir(log_dir) if re.match(r"capture-\d+\.png", f)]
        nums = [int(re.search(r"(\d+)", f).group(1)) for f in existing]
        self.counter = max(nums) + 1 if nums else 1

    def save(
        self,
        goal: str,
        model_thought: str,
        model_action: str,
        model_args: dict,
        pi_commands: list,
        image_bytes: bytes,
    ) -> None:
        """Save screenshot + structured text log for one interaction step.

        pi_commands is a list of dicts produced by hardware.send_to_pi():
        each dict has keys: cmd, http_status, body, (optional) error.
        """
        n = self.counter
        self.counter += 1

        # --- Screenshot ---
        img_path = os.path.join(self.log_dir, f"capture-{n:03d}.png")
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imwrite(img_path, frame)

        # --- Text log ---
        txt_path = os.path.join(self.log_dir, f"interaction-{n:03d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== GOAL ===\n{goal}\n\n")

            model_section = ""
            if model_thought:
                model_section += f"Thought:\n{model_thought}\n\n"
            model_section += f"Action: {model_action}\n"
            if model_args:
                model_section += f"Args:\n{json.dumps(model_args, indent=2)}\n"
            f.write(f"=== MODEL RESPONSE ===\n{model_section}\n")

            if pi_commands:
                cmds = "".join(
                    f"[{i}] {json.dumps(e['cmd'])}\n"
                    for i, e in enumerate(pi_commands, 1)
                )
                f.write(f"=== COMMANDS SENT TO Pi Zero ===\n{cmds}\n")
            else:
                f.write("=== COMMANDS SENT TO Pi Zero ===\n(none)\n\n")

            if pi_commands:
                resps = ""
                for i, e in enumerate(pi_commands, 1):
                    if e.get("error"):
                        resps += f"[{i}] ERROR: {e['error']}\n"
                    else:
                        resps += f"[{i}] HTTP {e.get('http_status', '?')}: {e.get('body', '')}\n"
                f.write(f"=== Pi Zero RESPONSES ===\n{resps}\n")
            else:
                f.write("=== Pi Zero RESPONSES ===\n(none)\n\n")

        self._update_manifest(n)
        print(f"  [Log] capture-{n:03d}.png + interaction-{n:03d}.txt")

    def _update_manifest(self, new_n: int) -> None:
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
