"""Screenshot every figure in paper.html for visual review."""

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "figures" / "paper-audit"
OUT.mkdir(parents=True, exist_ok=True)
PAPER = (Path(__file__).parent.parent / "frontend" / "public" / "paper.html").as_uri()


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 700}, device_scale_factor=2.0)
        page.goto(PAPER, wait_until="networkidle")
        figures = page.query_selector_all("figure")
        print(f"{len(figures)} figures found")
        for i, fig in enumerate(figures):
            fig.scroll_into_view_if_needed()
            page.wait_for_timeout(250)
            fig.screenshot(path=OUT / f"fig-{i:02d}.png")
        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
