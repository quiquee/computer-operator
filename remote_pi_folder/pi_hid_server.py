import os
import time
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- HID REPORT HELPERS ---
def write_report(device, report):
    """Writes raw bytes to the /dev/hidgX device"""
    try:
        with open(device, 'rb+') as fd:
            fd.write(report)
    except BlockingIOError:
        print(f"Device {device} busy, retrying...")
        time.sleep(0.01)
        write_report(device, report)

# --- MOUSE CONTROLS (/dev/hidg1) ---
# --- MOUSE CONTROLS (/dev/hidg1) ---
# We must store the current position so when we click, we don't snap back to 0,0
current_x = 0
current_y = 0

def send_absolute_mouse(buttons, x, y):
    global current_x, current_y
    current_x = x
    current_y = y
    
    # Constrain to screen bounds
    x = max(0, min(1920, int(x)))
    y = max(0, min(1080, int(y)))
    
    # Map 1920x1080 resolution to the 0-32767 USB absolute scale
    abs_x = int((x / 1920.0) * 32767)
    abs_y = int((y / 1080.0) * 32767)
    
    # Pack into 5 bytes: [Buttons, X-Low, X-High, Y-Low, Y-High]
    report = bytes([
        buttons,
        abs_x & 0xFF, (abs_x >> 8) & 0xFF,
        abs_y & 0xFF, (abs_y >> 8) & 0xFF
    ])
    write_report('/dev/hidg1', report)

def move_mouse_absolute(target_x, target_y):
    send_absolute_mouse(0, target_x, target_y)

def send_mouse_scroll(clicks):
    """Send a scroll-wheel event without moving the pointer.

    `clicks` is a signed integer: positive = scroll up, negative = scroll down.
    Requires the HID gadget descriptor for /dev/hidg1 to include a wheel axis
    (6-byte report: buttons, X-low, X-high, Y-low, Y-high, wheel).
    """
    # Keep the current pointer position; only the wheel byte changes.
    abs_x = int((max(0, min(1920, int(current_x))) / 1920.0) * 32767)
    abs_y = int((max(0, min(1080, int(current_y))) / 1080.0) * 32767)
    # Clamp wheel to signed byte range and convert to unsigned for bytes()
    wheel = max(-127, min(127, int(clicks)))
    report = bytes([
        0,
        abs_x & 0xFF, (abs_x >> 8) & 0xFF,
        abs_y & 0xFF, (abs_y >> 8) & 0xFF,
        wheel & 0xFF,
    ])
    write_report('/dev/hidg1', report)

def left_click():
    send_absolute_mouse(1, current_x, current_y) # Press
    time.sleep(0.05)
    send_absolute_mouse(0, current_x, current_y) # Release

def double_click():
    left_click()
    time.sleep(0.05)
    left_click()

# --- KEYBOARD CONTROLS (/dev/hidg0) ---

# --- KEYBOARD CONTROLS (/dev/hidg0) ---
def send_keyboard(modifier, keycode):
    """Sends an 8-byte keyboard report."""
    report = bytes([modifier, 0, keycode, 0, 0, 0, 0, 0])
    write_report('/dev/hidg0', report)

# Map every character to (keycode, modifier). Modifier 0x02 is Left Shift.
KEY_MAP = {
    # Lowercase Letters
    'a': (0x04, 0), 'b': (0x05, 0), 'c': (0x06, 0), 'd': (0x07, 0), 'e': (0x08, 0),
    'f': (0x09, 0), 'g': (0x0A, 0), 'h': (0x0B, 0), 'i': (0x0C, 0), 'j': (0x0D, 0),
    'k': (0x0E, 0), 'l': (0x0F, 0), 'm': (0x10, 0), 'n': (0x11, 0), 'o': (0x12, 0),
    'p': (0x13, 0), 'q': (0x14, 0), 'r': (0x15, 0), 's': (0x16, 0), 't': (0x17, 0),
    'u': (0x18, 0), 'v': (0x19, 0), 'w': (0x1A, 0), 'x': (0x1B, 0), 'y': (0x1C, 0),
    'z': (0x1D, 0),

    # Uppercase Letters (Uses Shift Modifier 0x02)
    'A': (0x04, 2), 'B': (0x05, 2), 'C': (0x06, 2), 'D': (0x07, 2), 'E': (0x08, 2),
    'F': (0x09, 2), 'G': (0x0A, 2), 'H': (0x0B, 2), 'I': (0x0C, 2), 'J': (0x0D, 2),
    'K': (0x0E, 2), 'L': (0x0F, 2), 'M': (0x10, 2), 'N': (0x11, 2), 'O': (0x12, 2),
    'P': (0x13, 2), 'Q': (0x14, 2), 'R': (0x15, 2), 'S': (0x16, 2), 'T': (0x17, 2),
    'U': (0x18, 2), 'V': (0x19, 2), 'W': (0x1A, 2), 'X': (0x1B, 2), 'Y': (0x1C, 2),
    'Z': (0x1D, 2),

    # Numbers and Shifted Symbols (SPANISH ISO LAYOUT)
    '1': (0x1E, 0), '!': (0x1E, 2),
    '2': (0x1F, 0), '"': (0x1F, 2), # Shift+2 is "
    '3': (0x20, 0), '·': (0x20, 2), # Shift+3 is ·
    '4': (0x21, 0), '$': (0x21, 2),
    '5': (0x22, 0), '%': (0x22, 2),
    '6': (0x23, 0), '&': (0x23, 2), # Shift+6 is &
    '7': (0x24, 0), '/': (0x24, 2), # Shift+7 is /
    '8': (0x25, 0), '(': (0x25, 2), # Shift+8 is (
    '9': (0x26, 0), ')': (0x26, 2), # Shift+9 is )
    '0': (0x27, 0), '=': (0x27, 2), # Shift+0 is =

    # Spanish Specific Punctuation
    "'": (0x2D, 0), '?': (0x2D, 2), 
    '¡': (0x2E, 0), '¿': (0x2E, 2),
    '+': (0x30, 0), '*': (0x30, 2),
    'ç': (0x31, 0), 'Ç': (0x31, 2),
    'ñ': (0x33, 0), 'Ñ': (0x33, 2),
    ',': (0x36, 0), ';': (0x36, 2), # Shift+, is ;
    '.': (0x37, 0), ':': (0x37, 2), # Shift+. is :
    '-': (0x38, 0), '_': (0x38, 2), # Shift+- is _
    
    # Special Whitespace
    ' ': (0x2C, 0),       # Space

    # Bracket sequences — used by type_text to unambiguously represent special keys
    '[enter]':     (0x28, 0),    # Enter
    '[tab]':       (0x2B, 0),    # Tab
    '[backspace]': (0x2A, 0),    # Backspace
    '[delete]':    (0x4C, 0),    # Delete
    '[escape]':    (0x29, 0),    # Escape
    '[up]':        (0x52, 0),    # Arrow Up
    '[down]':      (0x51, 0),    # Arrow Down
    '[left]':      (0x50, 0),    # Arrow Left
    '[right]':     (0x4F, 0),    # Arrow Right
    '[home]':      (0x4A, 0),    # Home
    '[end]':       (0x4D, 0),    # End
    '[pageup]':    (0x4B, 0),    # Page Up
    '[pagedown]':  (0x4E, 0),    # Page Down
    '[f1]':        (0x3A, 0),    # F1
    '[f2]':        (0x3B, 0),    # F2
    '[f3]':        (0x3C, 0),    # F3
    '[f4]':        (0x3D, 0),    # F4
    '[f5]':        (0x3E, 0),    # F5
    '[f6]':        (0x3F, 0),    # F6
    '[f7]':        (0x40, 0),    # F7
    '[f8]':        (0x41, 0),    # F8
    '[f9]':        (0x42, 0),    # F9
    '[f10]':       (0x43, 0),    # F10
    '[f11]':       (0x44, 0),    # F11
    '[f12]':       (0x45, 0),    # F12
    
    # AltGr Characters (Right Alt is 0x40)
    '@': (0x1F, 0x40), # AltGr + 2
    '€': (0x08, 0x40), # AltGr + E
    '\\': (0x35, 0x40), # AltGr + º (keycode 0x35 is º/ª in ES layout)
}


def _tokenize(text):
    """Split text into a list of tokens.

    Tokens are either:
    - A bracket sequence such as '[enter]', '[tab]', '[f5]' (case-insensitive, lowercased)
    - A single character
    """
    tokens = []
    i = 0
    while i < len(text):
        if text[i] == '[':
            end = text.find(']', i)
            if end != -1:
                seq = text[i:end + 1].lower()  # e.g. '[Enter]' -> '[enter]'
                tokens.append(seq)
                i = end + 1
                continue
        tokens.append(text[i])
        i += 1
    return tokens


def type_text(text):
    # Pre-process the text: Convert "Ctrl l", "ctrl-l", "CTRL L" into "^l"
    # (?i) makes it case-insensitive.
    # It catches "ctrl" followed by a space or hyphen, then a letter or number.
    text = re.sub(r'(?i)ctrl[- ]([a-z0-9])', r'^\1', text)

    print(f"Attempting to type: {text}")

    tokens = _tokenize(text)
    i = 0
    while i < len(tokens):
        char = tokens[i]
        
        # --- Handle Control Sequences (^l, ^c, ^v) ---
        if char == '^' and i + 1 < len(tokens):
            next_char = tokens[i+1]
            
            # If it's a double caret (^^), type a literal ^
            if next_char == '^':
                if '^' in KEY_MAP:
                    keycode, modifier = KEY_MAP['^']
                    send_keyboard(modifier, keycode)
                    send_keyboard(0, 0)
                i += 2
                continue
                
            # Otherwise, treat it as a Ctrl modifier
            lower_char = next_char.lower()
            if lower_char in KEY_MAP:
                keycode, _ = KEY_MAP[lower_char] # We only need the base keycode
                modifier = 0x01 # 0x01 is the Left Ctrl modifier
                
                print(f"Typing Ctrl+{lower_char} -> Keycode: {hex(keycode)}, Mod: {hex(modifier)}")
                send_keyboard(modifier, keycode)
                send_keyboard(0, 0)
                time.sleep(0.05)
            else:
                print(f"!!! Skipping unknown Ctrl sequence: ^{next_char} !!!")
                
            i += 2
            continue

        # --- Handle Normal Characters ---
        if char in KEY_MAP:
            keycode, modifier = KEY_MAP[char]
            print(f"Typing '{char}' -> Keycode: {hex(keycode)}, Mod: {hex(modifier)}")
            send_keyboard(modifier, keycode) 
            send_keyboard(0, 0)              
            time.sleep(0.02)                 
        else:
            print(f"!!! Skipping unknown character: '{char}' !!!")
            
        i += 1

# --- HTTP SERVER TO RECEIVE COMMANDS ---
class HIDServer(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        command = json.loads(post_data.decode('utf-8'))
        
        action = command.get('action')
        print(f"Received action: {action}")
        
        try:
            if action == 'mouse_move':
                print(f"Destination {command['x']},{command['y']}")
                move_mouse_absolute(command['x'], command['y'])
            elif action == 'left_click':
                left_click()
            elif action == 'double_click':
                double_click()
            elif action == 'type_text':
                print(f"Typing: {command['text']}")
                type_text(command['text'])
            elif action == 'scroll_down':
                clicks = int(command.get('clicks', 3))
                print(f"Scrolling down {clicks} click(s)")
                for _ in range(clicks):
                    send_mouse_scroll(-1)
                    time.sleep(0.05)
                send_mouse_scroll(0)  # release wheel
            elif action == 'scroll_up':
                clicks = int(command.get('clicks', 3))
                print(f"Scrolling up {clicks} click(s)")
                for _ in range(clicks):
                    send_mouse_scroll(1)
                    time.sleep(0.05)
                send_mouse_scroll(0)  # release wheel
            elif action == 'page_down':
                print("Page Down")
                type_text('[pagedown]')
            elif action == 'page_up':
                print("Page Up")
                type_text('[pageup]')

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"success"}')
            
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'{{"status":"error", "message":"{str(e)}"}}'.encode())

if __name__ == '__main__':
    # Ensure devices exist before starting
    if not os.path.exists('/dev/hidg0') or not os.path.exists('/dev/hidg1'):
        print("ERROR: /dev/hidg0 or /dev/hidg1 not found. Did you run the gadget script?")
        exit(1)

    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, HIDServer)
    print("Pi HID Server running on port 8080...")
    httpd.serve_forever()
