"""Read-only configuration / text-only WebSocket compatibility probe.

Creates short empty sessions, never invokes an LLM or VOICEVOX synthesis.
"""

import argparse
import asyncio
import json

from websockets.asyncio.client import connect


async def main(url):
    base = {"type": "start", "source": "text", "audio": False,
            "mode": "balanced", "speaker": 3, "speed": 1.2,
            "min_chars": 24, "max_chars": 160, "lookahead": 12, "prompt": ""}
    for label, payload in [
        ("baseline", base),
        ("current_lab", {**base, "adjust_pauses": False, "sentence_pause_extra_ms": 200, "comma_pause_scale": 0.8}),
        ("syntax", {**base, "mode": "syntax"}),
        ("raw", {**base, "mode": "raw"}),
        ("invalid_speed_example", {**base, "speed": 0}),
    ]:
        async with asyncio.timeout(10), connect(url, proxy=None) as ws:
            await ws.send(json.dumps(payload))
            first = json.loads(await ws.recv())
            events = [first]
            if first["type"] == "ready":
                await ws.send(json.dumps({"type": "end"}))
                while events[-1]["type"] not in {"done", "error"}:
                    events.append(json.loads(await ws.recv()))
            print(json.dumps({"probe": label, "events": events}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    asyncio.run(main(parser.parse_args().url))
