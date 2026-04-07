from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from config import CRAWL_CONCURRENCY, CRAWL_DELAY, RAW_PAGES_DIR


def url_to_filename(url: str) -> str:
    """Convert a URL to a safe filename."""
    path = url.replace("https://docs.dynatrace.com/", "").strip("/")
    safe = re.sub(r"[^\w\-]", "_", path)
    return f"{safe}.json"


def extract_page_content(html: str, url: str) -> dict | None:
    """Extract structured content from rendered HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Find main content area - Dynatrace docs use 'search-content content' class
    main = (
        soup.select_one("div.search-content.content")
        or soup.select_one("main")
        or soup.select_one('[role="main"]')
        or soup.select_one("article")
    )
    if not main:
        return None

    # Remove nav, sidebar, footer, cookie banners, table of contents
    for tag in main.select("nav, aside, footer, .cookie-banner, .feedback-section, .sidebar, .on-this-page"):
        tag.decompose()

    # Extract title
    title_tag = soup.select_one("h1") or soup.select_one("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract breadcrumbs
    breadcrumb_elem = soup.select_one('[aria-label="breadcrumb"]') or soup.select_one(".breadcrumb")
    breadcrumbs = []
    if breadcrumb_elem:
        breadcrumbs = [a.get_text(strip=True) for a in breadcrumb_elem.select("a, span")]
        breadcrumbs = [b for b in breadcrumbs if b]

    # Extract headings with hierarchy
    headings = []
    for h in main.select("h1, h2, h3, h4"):
        headings.append({
            "level": int(h.name[1]),
            "text": h.get_text(strip=True),
        })

    # Extract body text preserving structure
    # Replace code blocks with markers to preserve them
    code_blocks = []
    for i, code in enumerate(main.select("pre")):
        marker = f"__CODE_BLOCK_{i}__"
        code_blocks.append(code.get_text())
        code.replace_with(marker)

    # Get text with some structure
    text = main.get_text(separator="\n")

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        marker = f"__CODE_BLOCK_{i}__"
        text = text.replace(marker, f"\n```\n{block}\n```\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    if not text.strip():
        return None

    return {
        "url": url,
        "title": title,
        "breadcrumbs": breadcrumbs,
        "headings": headings,
        "text": text,
    }


async def crawl_page(page, url: str, output_dir: Path) -> bool:
    """Crawl a single page and save its content."""
    filename = url_to_filename(url)
    output_path = output_dir / filename

    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        # Wait for the content div to appear (SPA rendering)
        try:
            await page.wait_for_selector("div.search-content.content", timeout=5000)
        except Exception:
            pass  # Fall through and try to extract what we can
        await page.wait_for_timeout(1000)

        html = await page.content()
        content = extract_page_content(html, url)

        if content:
            content["scraped_at"] = datetime.now(timezone.utc).isoformat()
            output_path.write_text(json.dumps(content, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"  No content extracted: {url}")
            return False

    except Exception as e:
        print(f"  Error crawling {url}: {e}")
        return False


async def crawl_urls(
    urls: list[dict],
    output_dir: Path | None = None,
    force: bool = False,
    max_pages: int | None = None,
) -> dict:
    """Crawl a list of URLs using Playwright."""
    output_dir = output_dir or RAW_PAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter already-scraped unless force
    to_crawl = []
    for entry in urls:
        filename = url_to_filename(entry["url"])
        if not force and (output_dir / filename).exists():
            continue
        to_crawl.append(entry)

    if max_pages:
        to_crawl = to_crawl[:max_pages]

    print(f"Crawling {len(to_crawl)} pages ({len(urls) - len(to_crawl)} already scraped)")

    stats = {"success": 0, "failed": 0, "skipped": len(urls) - len(to_crawl)}

    if not to_crawl:
        return stats

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="DynatraceDocsBot/1.0 (educational RAG agent)"
        )

        # Create a pool of pages for concurrent crawling
        semaphore = asyncio.Semaphore(CRAWL_CONCURRENCY)

        async def crawl_with_limit(entry: dict):
            async with semaphore:
                page = await context.new_page()
                try:
                    print(f"  Crawling: {entry['url']}")
                    success = await crawl_page(page, entry["url"], output_dir)
                    if success:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                    await asyncio.sleep(CRAWL_DELAY)
                finally:
                    await page.close()

        tasks = [crawl_with_limit(entry) for entry in to_crawl]
        await asyncio.gather(*tasks)

        await browser.close()

    return stats
