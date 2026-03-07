# AI Computer Operator

An AI agent that sees and controls any computer using Gemini's Computer Use API, a Raspberry Pi Zero acting as a USB HID device, and an HDMI capture card for vision.

```
┌──────────────────────────────────┐   HTTP / LAN   ┌─────────────────────────┐   USB HID   ┌──────────────────┐
│        Operator machine          │───────────────►│       Pi Zero           │────────────►│ Target computer  │
│                                  │                │                         │             │                  │
│  ai_controller.py                │                │  pi_hid_server.py       │             │                  │
│  • Gemini Computer Use API       │                │  • /dev/hidg0 keyboard  │             │                  │
│  • Uscreen HDMI capture          │                │  • /dev/hidg1 mouse     │             │                  │
│    /dev/video* (USB-C)           │◄───────────────────────────────────────────── HDMI out ┘
└──────────────────────────────────┘
```

---

## Hardware requirements

| Item | Role |
|---|---|
| **Uscreen HDMI capture card** | Captures the target computer's HDMI output; connected to the operator machine via USB-C and visible as `/dev/video*` |
| **Raspberry Pi Zero W** (or Zero 2 W) | Emulates a USB keyboard + absolute mouse; plugged into a USB data port on the **target computer** |
| Network (LAN / Wi-Fi) | The operator machine sends HTTP requests to the Pi Zero's server over the local network |

No software agent, no OS access, and no direct network connection to the target machine is required — the Pi Zero appears to it as a generic USB keyboard and mouse, and the screen is captured purely through the HDMI capture card.

---

## Part 1 — Raspberry Pi Zero setup (`pi_hid_server.py`)

This is the server side. It runs on the Pi Zero, which is connected to the **target computer** via USB. The Pi Zero uses the Linux USB gadget framework to register itself as a composite HID device (keyboard on `/dev/hidg0`, absolute mouse on `/dev/hidg1`).

### 1.1 Enable USB gadget mode

Follow a USB HID gadget guide for your Pi Zero to create a composite gadget with:
- **HID keyboard** — 8-byte boot-protocol report descriptor
- **HID absolute mouse** — 5-byte report (buttons, X-low, X-high, Y-low, Y-high) mapped to the 0–32767 USB absolute coordinate range

A typical gadget setup script writes the descriptors to `/sys/kernel/config/usb_gadget/` and binds the gadget to a UDC.

### 1.2 Install dependencies

```bash
# Python 3 is included in Raspberry Pi OS; no extra packages needed
```

### 1.3 Run the server

```bash
sudo python3 pi_hid_server.py
```

The server listens on port **8080** and accepts JSON POST requests:

| Action | Payload |
|---|---|
| `mouse_move` | `{"action": "mouse_move", "x": 960, "y": 540}` |
| `left_click` | `{"action": "left_click"}` |
| `double_click` | `{"action": "double_click"}` |
| `type_text` | `{"action": "type_text", "text": "hello[enter]"}` |

**`type_text` special tokens** — wrap special keys in square brackets inside the text string:

`[enter]` `[tab]` `[backspace]` `[delete]` `[escape]`
`[up]` `[down]` `[left]` `[right]` `[home]` `[end]` `[pageup]` `[pagedown]`
`[f1]`–`[f12]`

**Ctrl shortcuts** — prefix the key with `^`, e.g. `^l` for Ctrl+L, `^c` for Ctrl+C.

> **Note:** The key map in `pi_hid_server.py` is configured for a **Spanish ISO keyboard layout**. If your target computer uses a different layout, update the `KEY_MAP` dictionary accordingly.

### 1.4 Auto-start on boot (optional)

Create `/etc/systemd/system/pi-hid.service`:

```ini
[Unit]
Description=Pi HID Server
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/pi_hid_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable pi-hid
sudo systemctl start pi-hid
```

---

## Part 2 — Operator machine setup (`ai_controller.py`)

This side runs on **your machine** (the one with the HDMI capture card attached). It grabs frames from the capture card, sends them to the Gemini Computer Use API together with your goal, and forwards the resulting HID actions to the Pi Zero over HTTP.

### 2.1 Prerequisites

- Python 3.10+
- A Google AI Studio account with access to the **Gemini Computer Use API** (the model `gemini-2.5-computer-use-preview-10-2025` requires "UI actions" to be enabled for your project — request access via [Google AI Studio](https://aistudio.google.com))

### 2.2 Connect the capture card

Plug the **Uscreen HDMI capture card** into a USB-C port. Connect the HDMI output of the target computer to the capture card's HDMI input. Verify the device appears:

```bash
ls /dev/video*
# e.g. /dev/video0, /dev/video1
```

### 2.3 Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python
requests
google-genai
```

### 2.4 Configuration

Edit the constants at the top of `ai_controller.py`:

```python
API_KEY            = "YOUR_GOOGLE_AI_API_KEY"
PI_IP_ADDRESS      = "192.168.1.116"   # LAN IP of the Raspberry Pi Zero
CAPTURE_DEVICE_INDEX = 1               # Index of the /dev/video* capture device
SCREEN_WIDTH       = 1920
SCREEN_HEIGHT      = 1080
```

### 2.5 Run

```bash
python ai_controller.py
```

You will be offered two modes:

**1 — Normal mode (Gemini Computer Use API)**
Enter a natural-language goal (e.g. *"Open Chrome with the Pili profile"*). The agent will loop: capture screen → send to Gemini → execute action on Pi → capture result → repeat, until the goal is achieved or you press Ctrl+C.

**2 — Calibration test mode**
Moves the mouse to each of the four screen corners and asks a Gemini vision model to confirm the cursor position. Use this to verify that the coordinate mapping between the capture card and the HID mouse is correct.

---

## How it works

1. A screenshot is captured from the capture card (JPEG for the initial prompt, PNG for subsequent function responses — the Computer Use API requires PNG).
2. The screenshot and goal are sent to `gemini-2.5-computer-use-preview-10-2025` with the `computer_use` tool enabled.
3. Gemini replies with one or more function calls (`click_at`, `type_text_at`, `scroll_at`, `key_press`, `open_web_browser`, etc.) using **normalized 0–1000 coordinates**.
4. `ai_controller.py` converts those coordinates to pixels and forwards them as JSON to the Pi Zero's HTTP server.
5. The Pi Zero writes raw HID reports to `/dev/hidg0` and `/dev/hidg1`, which the target computer receives as genuine keyboard/mouse input.
6. After each action, a fresh screenshot is taken and returned to the model as a function response, closing the perception–action loop.

---

## Project structure

```
computer-operator/
├── ai_controller.py          # Operator-side agent (runs on your machine)
├── requirements.txt
├── README.md
└── remote_pi_folder/
    └── pi_hid_server.py      # HID server (copy this to the Raspberry Pi Zero)
```
