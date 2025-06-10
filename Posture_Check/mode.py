import requests
from pynput import keyboard

# Server settings
url = "http://0.0.0.0:8000/set-mode/" # Change to your server IP if needed
enabled = False
message = "System temporarily offline. Try again later."

print("Press 't' to toggle forced mode ON/OFF. Press 'q' to quit.")

def on_press(key: keyboard.Key) -> None:
    """Handle key press events."""
    global enabled

    try:
        if key.char == 't':
            enabled = not enabled
            data = {
                "enabled": enabled,
                "message": message if enabled else ""
            }
            response = requests.post(url, data=data)
            print(f"\nToggled forced mode to {'ON' if enabled else 'OFF'}")
            print("Response:", response.json())

        elif key.char == 'q':
            print("Exiting.")
            return False  # Stop listener

    except AttributeError:
        pass  # Handle special keys like shift, ctrl, etc.

# Start listening
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
