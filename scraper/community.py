from __future__ import annotations

import asyncio
import gzip
import json
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from lxml import etree

from config import CRAWL_DELAY, RAW_PAGES_DIR

COMMUNITY_SITEMAP_URL = "https://community.dynatrace.com/sitemap.xml"
USER_AGENT = "DynatraceDocsBot/1.0 (educational RAG agent)"


async def _parse_sitemap(client, url: str, ns: dict) -> tuple[list[str], list[dict]]:
    """Parse a sitemap XML, returning (sub_sitemap_urls, page_urls)."""
    resp = await client.get(url)
    resp.raise_for_status()

    content = resp.content
    if url.endswith(".gz"):
        try:
            content = gzip.decompress(content)
        except Exception:
            pass

    root = etree.fromstring(content)

    sub_urls = [s.findtext("s:loc", namespaces=ns) for s in root.findall("s:sitemap", ns)]
    sub_urls = [u for u in sub_urls if u]

    page_urls = []
    for url_elem in root.findall("s:url", ns):
        loc = url_elem.findtext("s:loc", namespaces=ns)
        lastmod = url_elem.findtext("s:lastmod", namespaces=ns)
        if loc:
            page_urls.append({"url": loc, "lastmod": lastmod})

    return sub_urls, page_urls


async def fetch_community_urls(max_urls: int | None = None) -> list[dict]:
    """Fetch thread URLs from community sitemap (handles nested sitemap indexes)."""
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    all_urls = []

    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": USER_AGENT}) as client:
        # BFS through sitemap indexes (up to 2 levels deep)
        to_visit = [COMMUNITY_SITEMAP_URL]
        visited = set()

        for depth in range(3):  # Max 3 levels of nesting
            next_level = []
            for sitemap_url in to_visit:
                if sitemap_url in visited:
                    continue
                visited.add(sitemap_url)
                try:
                    sub_urls, page_urls = await _parse_sitemap(client, sitemap_url, ns)
                    all_urls.extend(page_urls)
                    next_level.extend(sub_urls)
                except Exception as e:
                    print(f"  Error fetching sitemap {sitemap_url}: {e}")
            to_visit = next_level
            if not to_visit:
                break

    # Filter to thread/discussion pages (skip profile pages, etc.)
    thread_urls = [u for u in all_urls if "/t5/" in u["url"]]

    if max_urls:
        thread_urls = thread_urls[:max_urls]

    return thread_urls


def url_to_filename(url: str) -> str:
    """Convert community URL to safe filename."""
    path = url.replace("https://community.dynatrace.com/", "").strip("/")
    safe = re.sub(r"[^\w\-]", "_", path)
    return f"community_{safe}.json"


def extract_community_content(html: str, url: str) -> dict | None:
    """Extract content from a community thread page."""
    soup = BeautifulSoup(html, "lxml")

    # Find thread title
    title_tag = (
        soup.select_one(".lia-message-subject")
        or soup.select_one("h1")
        or soup.select_one(".page-header h1")
    )
    title = title_tag.get_text(strip=True) if title_tag else ""

    if not title:
        return None

    # Extract all message bodies in the thread
    messages = []
    for msg in soup.select(".lia-message-body, .message-body, .lia-quilt-forum-message"):
        text = msg.get_text(separator="\n", strip=True)
        if text:
            messages.append(text)

    # Fallback: get main content area
    if not messages:
        main = soup.select_one(".lia-body-content") or soup.select_one('[role="main"]') or soup.select_one("main")
        if main:
            # Remove sidebar, nav
            for tag in main.select("nav, aside, .sidebar, .lia-component-common-widget-kudos"):
                tag.decompose()
            text = main.get_text(separator="\n", strip=True)
            if text:
                messages = [text]

    if not messages:
        return None

    full_text = f"{title}\n\n" + "\n\n---\n\n".join(messages)

    # Clean up whitespace
    lines = [line.strip() for line in full_text.splitlines()]
    full_text = "\n".join(line for line in lines if line)

    return {
        "url": url,
        "title": title,
        "breadcrumbs": [],
        "headings": [{"level": 1, "text": title}],
        "text": full_text,
        "source": "community",
    }


async def crawl_community(
    urls: list[dict],
    output_dir: Path | None = None,
    force: bool = False,
    max_pages: int | None = None,
) -> dict:
    """Crawl community pages using simple HTTP (server-rendered)."""
    output_dir = output_dir or RAW_PAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    to_crawl = []
    for entry in urls:
        filename = url_to_filename(entry["url"])
        if not force and (output_dir / filename).exists():
            continue
        to_crawl.append(entry)

    if max_pages:
        to_crawl = to_crawl[:max_pages]

    print(f"Crawling {len(to_crawl)} community pages ({len(urls) - len(to_crawl)} already scraped)")

    stats = {"success": 0, "failed": 0, "skipped": len(urls) - len(to_crawl)}

    if not to_crawl:
        return stats

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        for entry in to_crawl:
            url = entry["url"]
            filename = url_to_filename(url)
            output_path = output_dir / filename

            try:
                print(f"  Crawling: {url}")
                resp = await client.get(url)
                resp.raise_for_status()

                content = extract_community_content(resp.text, url)
                if content:
                    content["scraped_at"] = datetime.now(timezone.utc).isoformat()
                    output_path.write_text(json.dumps(content, indent=2, ensure_ascii=False))
                    stats["success"] += 1
                else:
                    print(f"    No content extracted")
                    stats["failed"] += 1

                await asyncio.sleep(CRAWL_DELAY)

            except Exception as e:
                print(f"    Error: {e}")
                stats["failed"] += 1

    return stats
