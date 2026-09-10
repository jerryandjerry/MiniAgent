#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Travel-guide retrieval tool backed by the packaged knowledge base."""
import os

from cn_travel.business_logic.contracts import EMPTY, OK, err
from cn_travel.tool.guide.normalization import stem

SOURCE = "rag-api/travel_guides"
RAG_URL = os.getenv("RAG_URL", "").rstrip("/")


def _search(query: str, search_type: str) -> list[dict]:
    """Run retrieval in process, with an optional HTTP adapter."""
    if RAG_URL:
        import requests

        response = requests.post(
            f"{RAG_URL}/search",
            timeout=30,
            json={"query": query, "search_type": search_type, "limit": 3},
        )
        response.raise_for_status()
        return list(response.json().get("results") or [])

    from cn_travel.tool.guide.retrieval import get_rag_service

    return list(
        get_rag_service().search(
            query,
            limit=3,
            search_type=search_type,
            vector_weight=1.0,
            keyword_weight=1.5,
        )
        or []
    )


def _matches(requested: str, city: str, province: str) -> bool:
    """Accept only guides whose city or province matches the request."""
    if not requested:
        return False
    want = stem(requested)
    for field in (city or "", province or ""):
        got = stem(field)
        if got and (got in want or want in got):
            return True
    return False


def search_travel_guide(location: str, search_mode: str = "hybrid") -> dict:
    """Retrieve travel guides using vector, keyword, or hybrid search."""
    if not location:
        return err("search_travel_guide", "location is required", location="")

    try:
        results = _search(location, search_mode or "hybrid")
    except Exception as exc:
        return err("search_travel_guide", f"rag service unavailable: {exc}",
                   location=location)

    guides = []
    for item in results:
        city = item.get("city_name") or ""
        province = item.get("province_name") or ""
        content = item.get("content") or ""
        if not content.strip() or not _matches(location, city, province):
            continue                      # fail closed
        guides.append({"city": city, "province": province, "content": content,
                       "score": float(item.get("score") or 0.0)})

    if not guides:
        return {"status": EMPTY, "source": SOURCE, "location": location, "guides": []}
    return {"status": OK, "source": SOURCE, "location": location, "guides": guides}
