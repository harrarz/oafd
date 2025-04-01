import threading
from datetime import datetime

# Thread-safety utilities
sync_lock = threading.Lock()


def secure_print(*args, **kwargs):
    """
    Thread-safe console output with timestamp.
    Usage: secure_print("Message", key=value, ...)
    """
    with sync_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}]", *args, **kwargs)