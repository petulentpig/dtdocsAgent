import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SITEMAP_URL, WHATS_NEW_PREFIX
from scraper.sitemap import fetch_sitemap_urls
from scraper.crawler import crawl_urls
from scraper.community import fetch_community_urls, crawl_community
from scraper.blog import fetch_blog_urls, crawl_blog
from scraper.github import fetch_github_docs, crawl_github

SOURCES = ["docs", "whats-new", "community", "blog", "github"]


def print_stats(name, stats):
    print(f"  [{name}] Success: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")


async def scrape_docs(args):
    print(f"\n=== Dynatrace Docs ===")
    print(f"Fetching sitemap from {SITEMAP_URL}...")
    urls = await fetch_sitemap_urls(SITEMAP_URL, prefix=args.prefix)
    print(f"Found {len(urls)} URLs in sitemap")
    stats = await crawl_urls(urls, force=args.force, max_pages=args.max)
    print_stats("docs", stats)
    return stats


async def scrape_whats_new(args):
    """Scrape the what's-new / release-notes section.

    Always re-crawls (force=True): release-note pages are updated in place
    each sprint, so skip-if-exists would serve stale content forever.
    """
    print(f"\n=== Dynatrace What's New ===")
    print(f"Fetching sitemap from {SITEMAP_URL} (prefix {WHATS_NEW_PREFIX})...")
    urls = await fetch_sitemap_urls(SITEMAP_URL, prefix=WHATS_NEW_PREFIX)
    print(f"Found {len(urls)} what's-new URLs in sitemap")
    stats = await crawl_urls(urls, force=True, max_pages=args.max)
    print_stats("whats-new", stats)
    return stats


async def scrape_community(args):
    print(f"\n=== Dynatrace Community ===")
    urls = await fetch_community_urls(max_urls=args.max)
    print(f"Found {len(urls)} community thread URLs")
    stats = await crawl_community(urls, force=args.force, max_pages=args.max)
    print_stats("community", stats)
    return stats


async def scrape_blog(args):
    print(f"\n=== Dynatrace Blog ===")
    urls = await fetch_blog_urls(max_urls=args.max)
    print(f"Found {len(urls)} blog post URLs")
    stats = await crawl_blog(urls, force=args.force, max_pages=args.max)
    print_stats("blog", stats)
    return stats


async def scrape_github(args):
    print(f"\n=== Dynatrace GitHub ===")
    files = await fetch_github_docs(max_files=args.max)
    stats = await crawl_github(files, force=args.force, max_pages=args.max)
    print_stats("github", stats)
    return stats


async def main():
    parser = argparse.ArgumentParser(description="Scrape Dynatrace documentation sources")
    parser.add_argument(
        "--source",
        choices=SOURCES + ["all"],
        default="all",
        help="Which source to scrape (default: all)",
    )
    parser.add_argument("--prefix", help="Only scrape URLs starting with this prefix (docs only)")
    parser.add_argument("--max", type=int, help="Maximum number of pages per source")
    parser.add_argument("--force", action="store_true", help="Re-scrape already scraped pages")
    args = parser.parse_args()

    sources = SOURCES if args.source == "all" else [args.source]

    for source in sources:
        if source == "docs":
            await scrape_docs(args)
        elif source == "whats-new":
            await scrape_whats_new(args)
        elif source == "community":
            await scrape_community(args)
        elif source == "blog":
            await scrape_blog(args)
        elif source == "github":
            await scrape_github(args)

    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())
