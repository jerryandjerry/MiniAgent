"""Application service facade for travel-guide retrieval."""
from __future__ import annotations

from threading import Lock
from typing import Any


class GuideService:
    """Expose guide-search use cases without leaking tool internals to the API."""

    def __init__(self, retrieval: Any | None = None):
        if retrieval is None:
            from cn_travel.tool.guide.retrieval import get_rag_service

            retrieval = get_rag_service()
        self._retrieval = retrieval

    @property
    def collection_size(self) -> int:
        collection = self._retrieval.collection
        return int(collection.num_entities) if collection else 0

    def search(
        self,
        query: str,
        limit: int,
        search_type: str,
        vector_weight: float,
        keyword_weight: float,
    ) -> list[dict[str, Any]]:
        return list(
            self._retrieval.search(
                query,
                limit,
                search_type,
                vector_weight,
                keyword_weight,
            )
            or []
        )

    def search_by_location(
        self,
        province: str | None,
        city: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        return list(
            self._retrieval.search_by_location(province, city, limit) or []
        )

    def statistics(self) -> dict[str, Any]:
        collection = self._retrieval.collection
        rows = collection.query(
            expr="",
            output_fields=["province_name"],
            limit=100,
        )
        provinces: dict[str, int] = {}
        for row in rows:
            province = row.get("province_name")
            if province:
                provinces[province] = provinces.get(province, 0) + 1
        return {
            "total_travel_guides": int(collection.num_entities),
            "sample_province_distribution": provinces,
            "collection_name": "travel_guides",
        }


_guide_service: GuideService | None = None
_guide_service_lock = Lock()


def get_guide_service() -> GuideService:
    """Return the process-wide guide service."""
    global _guide_service
    if _guide_service is None:
        with _guide_service_lock:
            if _guide_service is None:
                _guide_service = GuideService()
    return _guide_service


__all__ = ["GuideService", "get_guide_service"]
