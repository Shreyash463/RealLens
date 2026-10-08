"""Captures pixel-perfect mobile screenshots at exact device viewport using Chrome DevTools Protocol."""
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9222
TARGET_URL = "http://127.0.0.1:5000/?demo=1"
OUTPUT_FILE = Path("tests/screenshots/screenshot_390x844.png")


def capture_mobile_screenshot():
    # 1. Start Chrome with remote debugging
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        "--disable-gpu",
        "--hide-scrollbars",
        TARGET_URL,
    ])

    try:
        time.sleep(2)
        # 2. Get pages list from CDP HTTP endpoint
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=10) as resp:
            pages = json.loads(resp.read().decode())
        
        page = next((p for p in pages if p.get("type") == "page"), pages[0])
        ws_url = page["webSocketDebuggerUrl"]

        # 3. Use python's built-in or urllib to send CDP commands via simple websocket or send HTTP Target
        # In modern CDP, we can send commands using websockets or python-native ws
        try:
            import websockets
            import asyncio

            async def cdp_session():
                async with websockets.connect(ws_url) as ws:
                    # Emulate mobile 390x844
                    await ws.send(json.dumps({
                        "id": 1,
                        "method": "Emulation.setDeviceMetricsOverride",
                        "params": {
                            "width": 390,
                            "height": 844,
                            "deviceScaleFactor": 1,
                            "mobile": True,
                        }
                    }))
                    _ = await ws.recv()
                    await asyncio.sleep(0.5)

                    # Capture screenshot
                    await ws.send(json.dumps({
                        "id": 2,
                        "method": "Page.captureScreenshot",
                        "params": {
                            "format": "png",
                        }
                    }))
                    res = json.loads(await ws.recv())
                    img_data = base64.b64decode(res["result"]["data"])
                    OUTPUT_FILE.write_bytes(img_data)
                    print(f"Captured exact CDP mobile screenshot to {OUTPUT_FILE} ({len(img_data)} bytes)")

            asyncio.run(cdp_session())

        except ImportError:
            print("websockets library not installed, fallback to command line.")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    capture_mobile_screenshot()
