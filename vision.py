"""
Capture card initialisation, frame capture, and coordinate conversion.

Call init_capture_card() once at startup — it detects the actual output
resolution and writes it back into config.SCREEN_WIDTH / SCREEN_HEIGHT so
every other module sees the correct values.
"""
import cv2
import config


def init_capture_card():
    """Open the capture card, auto-detect resolution, return the cv2.VideoCapture."""
    print("Initializing Capture Card...")
    cap = cv2.VideoCapture(config.CAPTURE_DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Read one frame to detect the native output resolution
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        config.SCREEN_WIDTH  = w
        config.SCREEN_HEIGHT = h
        print(f"  Detected resolution: {w}x{h}")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.SCREEN_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.SCREEN_HEIGHT)
        print(f"  Could not detect resolution, using {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")

    return cap


def _capture_frame(cap, flush_frames: int = 3, fmt: str = '.jpg') -> bytes:
    """Flush stale buffered frames, grab a fresh one, and return encoded bytes."""
    for _ in range(flush_frames):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(
            "Failed to read from capture card. Check CAPTURE_DEVICE_INDEX in secrets.txt."
        )
    _, buf = cv2.imencode(fmt, frame)
    return buf.tobytes()


def get_screen_bytes(cap, flush_frames: int = 3, fmt: str = '.jpg') -> bytes:
    """Return the current screen as encoded image bytes (JPEG by default, PNG if fmt='.png')."""
    return _capture_frame(cap, flush_frames, fmt)


def norm_to_pixel(nx: float, ny: float) -> tuple[int, int]:
    """Convert Gemini's normalised 0–1000 coordinates to pixel coordinates."""
    x = max(0, min(config.SCREEN_WIDTH,  int(nx / 1000 * config.SCREEN_WIDTH)))
    y = max(0, min(config.SCREEN_HEIGHT, int(ny / 1000 * config.SCREEN_HEIGHT)))
    return x, y
