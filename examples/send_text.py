"""Send stdin text to the external-input URL displayed in the listening browser.

Usage: python examples/send_text.py http://SERVER:8000/api/sessions/ID/events
The producer should flush stdout. Line input is an example, not an API restriction.
"""

import sys
import time

import httpx


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/send_text.py <external-input URL>")
    with httpx.Client(timeout=20, trust_env=False, follow_redirects=False) as client:
        def send(event):
            deadline = time.monotonic() + 180
            while True:
                response = client.post(sys.argv[1], json=event)
                if response.status_code != 429:
                    response.raise_for_status()
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError("入力キューが満杯のままです。再生端末を確認してください。")
                time.sleep(1)

        for line in sys.stdin:
            for offset in range(0, len(line), 8192):
                send({"type": "text", "text": line[offset:offset + 8192]})
        send({"type": "end"})


if __name__ == "__main__":
    main()
