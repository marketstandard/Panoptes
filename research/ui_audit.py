"""Screenshot audit of the Panoptes UI — captures every section for review."""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "figures" / "ui-audit"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8000"


def main() -> None:
    fixture = sys.argv[1] if len(sys.argv) > 1 else "ai-prose"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5)
        page.goto(BASE, wait_until="networkidle")
        page.screenshot(path=OUT / "00-landing.png")

        # run a fixture analysis
        page.get_by_role("button", name="AI prose", exact=True).click()
        page.wait_for_selector(".observatory-answer", timeout=15000)
        time.sleep(1.2)

        # full page
        page.screenshot(path=OUT / "01-full.png", full_page=True)

        # per-section captures
        sections = {
            "02-answer": ".observatory-answer",
            "03-evidence": ".evidence-grid",
            "04-matrices": ".matrix-grid-section",
            "05-figures": ".figures-grid",
            "06-corpus": ".corpus-section",
        }
        for name, sel in sections.items():
            el = page.query_selector(sel)
            if el:
                el.scroll_into_view_if_needed()
                time.sleep(0.4)
                el.screenshot(path=OUT / f"{name}.png")
            else:
                print(f"MISSING: {sel}")

        # hover the prior-sensitivity chart to verify the crosshair readout
        chart = page.query_selector(".figures-grid svg.chart-interactive")
        if chart:
            box = chart.bounding_box()
            if box:
                page.mouse.move(box["x"] + box["width"] * 0.62, box["y"] + box["height"] * 0.5)
                time.sleep(0.4)
                page.query_selector_all(".figures-grid .figure-card")[0].screenshot(path=OUT / "card-00-hover.png")

        # technical drilldown is expanded by default; capture it directly
        lab = page.query_selector(".technical-lab")
        if lab:
            lab.scroll_into_view_if_needed()
            time.sleep(0.4)
            lab.screenshot(path=OUT / "07-drilldown.png")
        page.screenshot(path=OUT / "08-full-expanded.png", full_page=True)

        # individual figure cards for detail inspection
        cards = page.query_selector_all(".figures-grid .figure-card, .corpus-section .figure-card")
        for i, card in enumerate(cards):
            card.screenshot(path=OUT / f"card-{i:02d}.png")

        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
