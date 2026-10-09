"""Captures pixel-perfect screenshots across viewports using Chrome DevTools Protocol."""
import asyncio
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9222
TARGET_URL = "http://127.0.0.1:5000/?demo=1"
SCREENSHOTS_DIR = Path("tests/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

VIEWPORTS = [
    ("screenshot_1920x1080.png", 1920, 1080, False),
    ("screenshot_1366x768.png", 1366, 768, False),
    ("screenshot_390x844.png", 390, 844, True),
]


async def send_cmd(ws, msg_id, method, params=None):
    payload = {"id": msg_id, "method": method}
    if params:
        payload["params"] = params
    await ws.send(json.dumps(payload))
    while True:
        raw = await ws.recv()
        data = json.loads(raw)
        if data.get("id") == msg_id:
            return data


async def capture_all():
    # 1. Start Chrome headless with remote debugging
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        TARGET_URL,
    ])

    try:
        await asyncio.sleep(2.5)
        # 2. Get pages list from CDP HTTP endpoint
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=10) as resp:
            pages = json.loads(resp.read().decode())
        
        page = next((p for p in pages if p.get("type") == "page"), pages[0])
        ws_url = page["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=30 * 1024 * 1024) as ws:
            await send_cmd(ws, 1, "Page.enable")
            await asyncio.sleep(1.2)

            for idx, (filename, width, height, is_mobile) in enumerate(VIEWPORTS, start=10):
                # Set viewport
                await send_cmd(ws, idx * 2, "Emulation.setDeviceMetricsOverride", {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 1,
                    "mobile": is_mobile,
                })
                await asyncio.sleep(0.8)

                # Capture screenshot
                res = await send_cmd(ws, idx * 2 + 1, "Page.captureScreenshot", {"format": "png"})
                img_data = base64.b64decode(res["result"]["data"])
                out_path = SCREENSHOTS_DIR / filename
                out_path.write_bytes(img_data)
                print(f"Captured {filename} ({width}x{height}) -> {len(img_data)} bytes")

            # Also capture scrolled view of detail card at 1920x1080
            await send_cmd(ws, 90, "Emulation.setDeviceMetricsOverride", {
                "width": 1920,
                "height": 1080,
                "deviceScaleFactor": 1,
                "mobile": False,
            })
            await send_cmd(ws, 91, "Runtime.evaluate", {
                "expression": "window.scrollTo(0, 520);"
            })
            await asyncio.sleep(0.8)
            res = await send_cmd(ws, 92, "Page.captureScreenshot", {"format": "png"})
            scrolled_path = SCREENSHOTS_DIR / "screenshot_detail_scrolled.png"
            scrolled_path.write_bytes(base64.b64decode(res["result"]["data"]))
            print("Captured screenshot_detail_scrolled.png ->", len(res["result"]["data"]))

            # Capture ELA & visual forensics at scroll 1100
            await send_cmd(ws, 93, "Runtime.evaluate", {
                "expression": "window.scrollTo(0, 1100);"
            })
            await asyncio.sleep(0.8)
            res_ela = await send_cmd(ws, 94, "Page.captureScreenshot", {"format": "png"})
            ela_path = SCREENSHOTS_DIR / "screenshot_detail_ela.png"
            ela_path.write_bytes(base64.b64decode(res_ela["result"]["data"]))
            print("Captured screenshot_detail_ela.png ->", len(res_ela["result"]["data"]))

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    asyncio.run(capture_all())
