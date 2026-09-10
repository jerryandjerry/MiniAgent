#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional browser-backed hotel-price lookup.

Unverified prices return ``None`` so callers preserve a null price field.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import requests

from cn_travel.paths import CN_TRAVEL

os.environ.setdefault(
    "PLAYWRIGHT_BROWSERS_PATH",
    str(CN_TRAVEL.runtime / "ms-playwright"),
)

BACKEND = os.getenv("PRICE_BACKEND", "none").lower()
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Parse currency prefixes or suffixes, thousands separators, and decimals.
_PRICE = re.compile(r'(?:¥|￥|CNY\s*)\s?([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*元')


def _parse(blob: str) -> Optional[int]:
    """Extract the lowest plausible price from text."""
    found = []
    for m in _PRICE.finditer(blob or ""):
        raw = (m.group(1) or m.group(2) or "").replace(",", "")
        try:
            n = int(float(raw))
        except ValueError:
            continue
        if 30 <= n <= 99999:          # Exclude years, street numbers, and review counts
            found.append(n)
    return min(found) if found else None


def _playwright(hotel: str, city: str) -> Optional[int]:
    """Read the closest matching Booking property with headless Playwright."""
    import asyncio
    import datetime as dt

    async def run() -> Optional[int]:
        from playwright.async_api import async_playwright

        ci = (dt.date.today() + dt.timedelta(days=14)).isoformat()
        co = (dt.date.today() + dt.timedelta(days=15)).isoformat()
        url = (f"https://www.booking.com/searchresults.zh-cn.html"
               f"?ss={requests.utils.quote(city + hotel)}"
               f"&checkin={ci}&checkout={co}&group_adults=1&no_rooms=1"
               f"&selected_currency=CNY")     # Default to USD when unspecified
        async with async_playwright() as pw:
            b = await pw.chromium.launch(headless=True)
            try:
                ctx = await b.new_context(locale="zh-CN", user_agent=_UA)
                pg = await ctx.new_page()
                await pg.goto(url, timeout=60000, wait_until="domcontentloaded")
                try:
                    await pg.wait_for_selector('[data-testid="property-card"]', timeout=25000)
                except Exception:
                    return None
                # Cards render before prices arrive. Waiting only three seconds can
                # expose a currency label without digits and look like a missing price.
                await pg.wait_for_timeout(6000)
                cards = await pg.query_selector_all('[data-testid="property-card"]')
                for card in cards[:1]:                       # First card is the closest match
                    got = _parse(await card.inner_text())
                    if got:
                        return got
                return None
            finally:
                await b.close()

    try:
        return asyncio.run(run())
    except RuntimeError:                      # Already in an event loop, e.g. async caller
        return None



def lowest_price(hotel_name: str, city: str = "") -> Optional[int]:
    """Return the lowest verified nightly CNY price, or ``None``."""
    if BACKEND != "playwright" or not hotel_name:
        return None                      # Disabled means no price, not an error
    try:
        return _playwright(hotel_name, city)
    except Exception:
        return None                      # Scraping is flaky and must not break the tool


def available() -> bool:
    """Return whether the browser price backend is enabled."""
    return BACKEND == "playwright"
