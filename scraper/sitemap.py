from __future__ import annotations

import httpx
from lxml import etree


async def fetch_sitemap_urls(sitemap_url: str, prefix: str | None = None) -> list[dict]:
    """Fetch and parse a sitemap XML, returning list of {url, lastmod} dicts."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(sitemap_url)
        resp.raise_for_status()

    root = etree.fromstring(resp.content)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls = []
    for url_elem in root.findall("s:url", ns):
        loc = url_elem.findtext("s:loc", namespaces=ns)
        lastmod = url_elem.findtext("s:lastmod", namespaces=ns)
        if loc:
            if prefix and not loc.startswith(prefix):
                continue
            urls.append({"url": loc, "lastmod": lastmod})

    return urls
