from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from lxml import etree

from config import CRAWL_DELAY, RAW_PAGES_DIR

BLOG_SITEMAP_INDEX_URL = "https://www.dynatrace.com/news/blog/sitemap.xml"
USER_AGENT = "DynatraceDocsBot/1.0 (educational RAG agent)"


async def fetch_blog_urls(max_urls: int | None = None) -> list[dict]:
    """Fetch blog post URLs from WordPress sitemap index."""
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        resp = await client.get(BLOG_SITEMAP_INDEX_URL)
        resp.raise_for_status()

    root = etree.fromstring(resp.content)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Get sub-sitemap URLs (focus on post sitemaps)
    sub_urls = []
    for sitemap in root.findall("s:sitemap", ns):
        loc = sitemap.findtext("s:loc", namespaces=ns)
        if loc and "post-sitemap" in loc:
            sub_urls.append(loc)

    all_urls = []
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        for sub_url in sub_urls:
            try:
                resp = await client.get(sub_url)
                resp.raise_for_status()
                sub_root = etree.fromstring(resp.content)
                for url_elem in sub_root.findall("s:url", ns):
                    loc = url_elem.findtext("s:loc", namespaces=ns)
                    lastmod = url_elem.findtext("s:lastmod", namespaces=ns)
                    if loc:
                        all_urls.append({"url": loc, "lastmod": lastmod})
            except Exception as e:
                print(f"  Error fetching blog sitemap {sub_url}: {e}")

    if max_urls:
        all_urls = all_urls[:max_urls]

    return all_urls


def url_to_filename(url: str) -> str:
    """Convert blog URL to safe filename."""
    path = url.replace("https://www.dynatrace.com/news/blog/", "").strip("/")
    safe = re.sub(r"[^\w\-]", "_", path)
    return f"blog_{safe}.json"


def extract_blog_content(html: str, url: str) -> dict | None:
    """Extract article content from a Dynatrace blog post."""
    soup = BeautifulSoup(html, "lxml")

    # Find article title
    title_tag = soup.select_one("h1.entry-title") or soup.select_one("h1") or soup.select_one("article h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    if not title:
        return None

    # Find article body
    article = (
        soup.select_one("article .entry-content")
        or soup.select_one("article")
        or soup.select_one(".post-content")
        or soup.select_one('[class*="article-body"]')
    )

    if not article:
        return None

    # Remove nav, sidebar, share buttons, related posts
    for tag in article.select("nav, aside, .share-buttons, .related-posts, .author-bio, .comments, script, style"):
        tag.decompose()

    # Extract headings
    headings = []
    for h in article.select("h1, h2, h3, h4"):
        headings.append({"level": int(h.name[1]), "text": h.get_text(strip=True)})

    # Extract text
    text = article.get_text(separator="\n", strip=True)

    # Clean up
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    if len(text) < 100:
        return None

    return {
        "url": url,
        "title": title,
        "breadcrumbs": [],
        "headings": headings,
        "text": text,
        "source": "blog",
    }


async def crawl_blog(
    urls: list[dict],
    output_dir: Path | None = None,
    force: bool = False,
    max_pages: int | None = None,
) -> dict:
    """Crawl blog posts using simple HTTP (WordPress, server-rendered)."""
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

    print(f"Crawling {len(to_crawl)} blog posts ({len(urls) - len(to_crawl)} already scraped)")

    stats = {"success": 0, "failed": 0, "skipped": len(urls) - len(to_crawl)}

    if not to_crawl:
        return stats

    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        for entry in to_crawl:
            url = entry["url"]
            filename = url_to_filename(url)
            output_path = output_dir / filename

            try:
                print(f"  Crawling: {url}")
                resp = await client.get(url)
                resp.raise_for_status()

                content = extract_blog_content(resp.text, url)
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
