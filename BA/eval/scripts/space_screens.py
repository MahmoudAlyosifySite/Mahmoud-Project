"""Screenshots of the running Space app (venv /tmp/ba/venvs/space, Playwright + local Chrome).

Usage: python space_screens.py <url> <out_dir>
Desktop (1280 px) and phone (390 px) full-page screenshots, plus the desktop view after pressing
Generate with the consent box unticked (the refusal must be visible).
"""
import sys
from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/google/chrome/chrome", args=["--no-sandbox"])
    for name, vp in (("desktop", dict(width=1280, height=900)), ("phone", dict(width=390, height=844))):
        pg = b.new_page(viewport=vp, device_scale_factor=2 if name == "phone" else 1)
        pg.goto(url, wait_until="load", timeout=60000)      # Gradio keeps a connection open: no "networkidle"
        pg.get_by_role("button", name="Generate").wait_for(timeout=60000)
        pg.wait_for_timeout(3000)
        pg.screenshot(path=f"{out}/space_{name}.png", full_page=True)
        if name == "desktop":
            pg.get_by_role("textbox").nth(1).fill("أنا رايح الشغل دلوقتي")
            pg.get_by_role("button", name="Generate").click()
            pg.wait_for_timeout(2500)
            pg.screenshot(path=f"{out}/space_desktop_consent_error.png", full_page=False)
        pg.close()
    b.close()
print("ok")
