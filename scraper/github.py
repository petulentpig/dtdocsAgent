from __future__ import annotations

import asyncio
import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config import CRAWL_DELAY, RAW_PAGES_DIR

USER_AGENT = "DynatraceDocsBot/1.0 (educational RAG agent)"

# Key Dynatrace repos with significant documentation
DYNATRACE_REPOS = [
    "dynatrace/dynatrace-operator",
    "dynatrace/dynatrace-configuration-as-code",
    "dynatrace/dynatrace-api",
    "dynatrace/OneAgent-SDK-for-Java",
    "dynatrace-oss/api-client-python",
    "dynatrace/backstage-plugin",
    "dynatrace/dynatrace-otel-collector",
]

# File patterns to look for in repos
DOC_PATTERNS = [
    "README.md",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "docs/**/*.md",
    "doc/**/*.md",
    "documentation/**/*.md",
]


async def fetch_github_docs(
    repos: list[str] | None = None,
    max_files: int | None = None,
) -> list[dict]:
    """Fetch markdown documentation files from Dynatrace GitHub repos."""
    repos = repos or DYNATRACE_REPOS
    all_files = []

    async with httpx.AsyncClient(
        timeout=30,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github.v3+json",
        },
        follow_redirects=True,
    ) as client:
        for repo in repos:
            print(f"  Scanning repo: {repo}")
            try:
                # Get repo tree recursively
                resp = await client.get(
                    f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
                )
                if resp.status_code == 404:
                    # Try 'master' branch
                    resp = await client.get(
                        f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
                    )
                if resp.status_code != 200:
                    print(f"    Skipping {repo}: HTTP {resp.status_code}")
                    continue

                tree = resp.json().get("tree", [])

                # Filter for markdown files in doc-relevant paths
                for item in tree:
                    path = item["path"]
                    if not path.endswith(".md"):
                        continue
                    # Include READMEs and files in doc directories
                    is_doc = (
                        path == "README.md"
                        or path.endswith("/README.md")
                        or path.startswith("doc/")
                        or path.startswith("docs/")
                        or path.startswith("documentation/")
                        or path in ("ARCHITECTURE.md", "CONTRIBUTING.md", "HACKING.md")
                    )
                    if is_doc:
                        all_files.append({
                            "repo": repo,
                            "path": path,
                            "url": f"https://github.com/{repo}/blob/main/{path}",
                            "raw_url": f"https://raw.githubusercontent.com/{repo}/main/{path}",
                        })

                await asyncio.sleep(0.5)  # Respect GitHub rate limits

            except Exception as e:
                print(f"    Error scanning {repo}: {e}")

    if max_files:
        all_files = all_files[:max_files]

    print(f"  Found {len(all_files)} doc files across {len(repos)} repos")
    return all_files


def file_to_filename(repo: str, path: str) -> str:
    """Convert repo/path to safe filename."""
    safe = re.sub(r"[^\w\-]", "_", f"{repo}/{path}")
    return f"github_{safe}.json"


async def crawl_github(
    files: list[dict],
    output_dir: Path | None = None,
    force: bool = False,
    max_pages: int | None = None,
) -> dict:
    """Fetch raw markdown content from GitHub."""
    output_dir = output_dir or RAW_PAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    to_crawl = []
    for entry in files:
        filename = file_to_filename(entry["repo"], entry["path"])
        if not force and (output_dir / filename).exists():
            continue
        to_crawl.append(entry)

    if max_pages:
        to_crawl = to_crawl[:max_pages]

    print(f"Fetching {len(to_crawl)} GitHub docs ({len(files) - len(to_crawl)} already fetched)")

    stats = {"success": 0, "failed": 0, "skipped": len(files) - len(to_crawl)}

    if not to_crawl:
        return stats

    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        for entry in to_crawl:
            filename = file_to_filename(entry["repo"], entry["path"])
            output_path = output_dir / filename

            try:
                print(f"  Fetching: {entry['repo']}/{entry['path']}")
                resp = await client.get(entry["raw_url"])
                if resp.status_code != 200:
                    # Try master branch
                    alt_url = entry["raw_url"].replace("/main/", "/master/")
                    resp = await client.get(alt_url)

                resp.raise_for_status()
                md_text = resp.text

                if len(md_text.strip()) < 50:
                    stats["failed"] += 1
                    continue

                title = entry["path"].replace(".md", "").split("/")[-1]
                # Try to extract title from first heading
                for line in md_text.splitlines():
                    if line.startswith("# "):
                        title = line.lstrip("# ").strip()
                        break

                # Extract headings
                headings = []
                for line in md_text.splitlines():
                    match = re.match(r"^(#{1,4})\s+(.+)", line)
                    if match:
                        headings.append({"level": len(match.group(1)), "text": match.group(2).strip()})

                content = {
                    "url": entry["url"],
                    "title": f"{entry['repo'].split('/')[-1]}: {title}",
                    "breadcrumbs": [entry["repo"], entry["path"]],
                    "headings": headings,
                    "text": md_text,
                    "source": "github",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                }

                output_path.write_text(json.dumps(content, indent=2, ensure_ascii=False))
                stats["success"] += 1

                await asyncio.sleep(CRAWL_DELAY)

            except Exception as e:
                print(f"    Error: {e}")
                stats["failed"] += 1

    return stats
