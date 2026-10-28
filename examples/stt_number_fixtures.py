"""Generate public numeric minimal-pair fixtures with a running VOICEVOX Engine."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx

from restream.providers import Settings, synthesize

SAMPLES = [
    ("walk", "今日は近所の公園を散歩します。"),
    ("real_numbers", "ペンを3本買って、30分後の午後3時に駅で会いましょう。"),
    ("numbers_date", "会議は2026年10月22日、料金は1,500円です。"),
]


async def main(args):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.get(args.voicevox.rstrip("/") + "/version")
        response.raise_for_status()
        engine_version = response.json()
        cases = []
        for name, text in SAMPLES:
            audio = await synthesize(client, Settings(voicevox_url=args.voicevox), text, 3, 1.0)
            path = directory / f"{name}.wav"
            path.write_bytes(audio)
            cases.append({"name": name, "audio": str(path), "reference": text,
                          "sha256": hashlib.sha256(audio).hexdigest()})
    (directory / "CREDIT.txt").write_text("VOICEVOX:ずんだもん\n", encoding="utf-8")
    manifest = {"engine_version": engine_version, "speaker_id": 3, "speed": 1.0, "cases": cases}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voicevox", default="http://127.0.0.1:50021")
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
