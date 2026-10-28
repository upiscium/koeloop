"""Real selected STT and real browser capture; file-backed fake mic, no LLM calls."""

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import httpx
from playwright.async_api import async_playwright, expect

from examples.browser_smoke import serve
from examples.stt_live import to_16k
from restream.app import create_app
from restream.stt import BACKENDS, create_stt, validate_recording


async def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    recording = validate_recording(to_16k(Path(args.audio).read_bytes()))
    fixture = output / "fake-microphone.wav"
    fixture.write_bytes(recording.wav)
    service = create_stt(args.backend)
    assert service.status()["available"], service.status()
    sockets, remote, errors = [], [], []

    def forbidden(request):
        remote.append(str(request.url))
        raise AssertionError("No upstream request is allowed in a dictation-only test")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client, serve(
        create_app(client=client)
    ) as base, async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=shutil.which("chromium"), headless=True, chromium_sandbox=True,
            args=["--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={fixture.resolve()}%noloop"],
        )
        try:
            context = await browser.new_context(permissions=["microphone"], viewport={"width": 1280, "height": 920})
            page = await context.new_page()
            page.on("websocket", lambda ws: sockets.append(ws.url))
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(base)
            await expect(page.locator("#sttBackend")).to_be_enabled()
            await page.locator("#sttBackend").select_option(service.status()["backend"])
            await expect(page.locator("#mic")).to_be_enabled()
            await expect(page.locator("#sttStatus")).to_contain_text(service.status()["engine_label"])
            await page.locator("#prompt").fill("下書き（まだ送信しません）")
            await page.locator("#mic").click()
            await expect(page.locator("#sttStatus")).to_contain_text("録音中")
            await page.wait_for_timeout(min(recording.seconds * 1000 + 350, 29000))
            await page.locator("#mic").click()
            await expect(page.locator("#sttStatus")).to_contain_text("入力欄に追加しました", timeout=90000)
            text = await page.locator("#prompt").input_value()
            assert text.startswith("下書き（まだ送信しません）\n")
            assert text.split("\n", 1)[1].strip()
            await expect(page.locator(".message")).to_have_count(0)
            await expect(page.locator("#send")).to_be_enabled()
            assert not sockets and not remote and not errors
            await page.screenshot(path=str(output / "draft-only.png"))
            report = {"status": "passed", "backend": service.status()["backend"],
                      "engine_label": service.status()["engine_label"], "input_audio": args.audio,
                      "draft": text, "physical_microphone": False, "llm_websockets": len(sockets),
                      "upstream_requests": len(remote), "page_errors": errors}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False))
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--backend", choices=list(BACKENDS))
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
