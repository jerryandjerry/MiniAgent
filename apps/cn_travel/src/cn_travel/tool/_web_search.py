#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web-search adapter returning sourced title, URL, and snippet records."""
import itertools
import os
import time
from typing import Dict, List


def provider() -> str:
    """Return the backend identifier recorded in review sources."""
    return "ddgs"


RETRIES = int(os.getenv("SEARCH_RETRIES", "3"))


def search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Search the web with retries and return normalized result records."""
    from ddgs import DDGS

    limit = max(1, min(limit, 10))
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            rows = itertools.islice(
                DDGS().text(query, region="cn-zh", max_results=limit), limit)
            hits = [{"title": x.get("title", ""),
                     "url": x.get("href", ""),
                     "snippet": x.get("body", "") or ""} for x in rows]
            if hits:
                return hits
            last = RuntimeError("no results")
        except Exception as exc:                      # Transient timeout or empty result
            last = exc
        if attempt < RETRIES - 1:
            time.sleep(1.5 * (attempt + 1))           # Linear backoff is enough here
    raise RuntimeError(f"搜索失败（重试 {RETRIES} 次）: {last}")
