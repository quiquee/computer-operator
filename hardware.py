"""
Pi Zero HID communication.

send_to_pi() posts a JSON command to the Pi Zero HTTP server and appends
a structured result record to _pi_responses so the caller can log it.
Clear _pi_responses before each new agent action with _pi_responses.clear().
"""
import requests
import config

_pi_responses: list = []


def send_to_pi(payload: dict) -> None:
    """POST a command to the Pi Zero and record the result in _pi_responses."""
    url = f"http://{config.PI_IP_ADDRESS}:8080"
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
