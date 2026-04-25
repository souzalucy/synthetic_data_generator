"""Convert HTML visualization reports to PNG images using Playwright.

Requires:
    pip install playwright
    playwright install chromium

Usage:
    # Convert a single file
    python -m ecosystem_sim.visualization.html_to_png path/to/report.html

    # Convert all reports in a directory
    python -m ecosystem_sim.visualization.html_to_png path/to/outputs/viz/reports

    # Convert with custom viewport
    python -m ecosystem_sim.visualization.html_to_png report.html --width 1920 --height 1080

    # Convert all reports after a simulation run
    python -m ecosystem_sim.visualization.html_to_png ecosystem_sim/outputs/viz --recursive
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def html_to_png(
    html_path: str,
    png_path: str,
    width: int = 1400,
    height: int = 900,
    full_page: bool = True,
    timeout_ms: int = 30000,
) -> None:
    """Render an HTML file to a PNG image using headless Chromium.

    Args:
        html_path: Path to the input HTML file.
        png_path: Path for the output PNG image.
        width: Viewport width in pixels.
        height: Viewport height in pixels.
        full_page: If True, capture the full scrollable page.
        timeout_ms: Navigation timeout in milliseconds.

    Raises:
        FileNotFoundError: If the HTML file does not exist.
        RuntimeError: If Playwright is not installed or Chromium is not found.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright is not installed. Run: pip install playwright && playwright install chromium"
        )

    html_file = Path(html_path)
    if not html_file.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file}")

    # Convert to file:// URI (handles Windows backslashes automatically)
    file_uri = html_file.resolve().as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(file_uri, wait_until="networkidle", timeout=timeout_ms)

        # Wait a bit for any late-rendering Plotly charts
        page.wait_for_timeout(2000)

        page.screenshot(path=png_path, full_page=full_page)
        browser.close()


def find_html_files(path: Path, recursive: bool = False) -> List[Path]:
    """Find all .html files under the given path.

    Args:
        path: File or directory path.
        recursive: If True, search subdirectories.

    Returns:
        List of HTML file paths.
    """
    if path.is_file():
        if path.suffix.lower() == ".html":
            return [path]
        return []

    pattern = "**/*.html" if recursive else "*.html"
    return sorted(path.glob(pattern))


def convert_all(
    source: Path,
    output_dir: Optional[Path] = None,
    width: int = 1400,
    height: int = 900,
    full_page: bool = True,
    recursive: bool = False,
) -> List[Path]:
    """Convert all HTML files under source to PNG images.

    Args:
        source: Path to an HTML file or directory containing HTML files.
        output_dir: Directory for output PNGs. If None, PNGs are placed
                    alongside the source HTML files.
        width: Viewport width.
        height: Viewport height.
        full_page: Capture full scrollable page.
        recursive: Search subdirectories for HTML files.

    Returns:
        List of generated PNG file paths.
    """
    html_files = find_html_files(source, recursive=recursive)

    if not html_files:
        print(f"No HTML files found in: {source}")
        return []

    generated: List[Path] = []
    for html_file in html_files:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            png_path = output_dir / f"{html_file.stem}.png"
        else:
            png_path = html_file.with_suffix(".png")

        print(f"  Converting: {html_file.name} -> {png_path.name}")
        html_to_png(
            html_path=str(html_file),
            png_path=str(png_path),
            width=width,
            height=height,
            full_page=full_page,
        )
        generated.append(png_path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HTML visualization reports to PNG images using Playwright.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help="Path to an HTML file or a directory containing HTML files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for PNG images (default: same location as HTML files).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1400,
        help="Viewport width in pixels (default: 1400).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Viewport height in pixels (default: 900).",
    )
    parser.add_argument(
        "--no-full-page",
        action="store_true",
        help="Capture only the visible viewport, not the full scrollable page.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search subdirectories recursively for HTML files.",
    )

    args = parser.parse_args()
    source = Path(args.source)
    output_dir = Path(args.output) if args.output else None

    if not source.exists():
        print(f"Error: path not found: {source}", file=sys.stderr)
        sys.exit(1)

    print(f"[HTML -> PNG] Converting HTML files from: {source}")
    generated = convert_all(
        source=source,
        output_dir=output_dir,
        width=args.width,
        height=args.height,
        full_page=not args.no_full_page,
        recursive=args.recursive,
    )

    if generated:
        print(f"\nDone. Generated {len(generated)} PNG image(s):")
        for png in generated:
            print(f"  {png}")
    else:
        print("No conversions performed.")


if __name__ == "__main__":
    main()
