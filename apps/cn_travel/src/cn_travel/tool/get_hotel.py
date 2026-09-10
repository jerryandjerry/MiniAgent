# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Any

def _generate_fallback_hotels(user_requirements: str) -> List[Dict[str, Any]]:
    """Generate fallback hotel recommendations."""
    fallback_hotels = [
        {
            "hotel_name": "假日酒店",
            "location": "市中心商业区",
            "price_range": "400-600元/晚",
            "room_types": ["标准间", "豪华间", "套房"],
            "rating": 4.2,
            "amenities": ["免费WiFi", "健身房", "早餐", "停车场"],
            "distance_to_transport": "地铁站500米"
        },
        {
            "hotel_name": "商务精选酒店",
            "location": "金融中心",
            "price_range": "600-900元/晚",
            "room_types": ["商务间", "行政套房"],
            "rating": 4.5,
            "amenities": ["商务中心", "会议室", "免费WiFi", "接送服务"],
            "distance_to_transport": "地铁站300米"
        },
        {
            "hotel_name": "经济型连锁酒店",
            "location": "交通枢纽附近",
            "price_range": "200-350元/晚",
            "room_types": ["标准间", "大床房"],
            "rating": 3.8,
            "amenities": ["免费WiFi", "24小时前台"],
            "distance_to_transport": "火车站200米"
        }
    ]
    return fallback_hotels


def _generate_fallback_reviews(hotel_name: str) -> List[Dict[str, Any]]:
    """Generate fallback hotel reviews."""
    fallback_reviews = [
        {
            "reviewer_name": "旅行达人小王",
            "rating": 4,
            "review_date": "2024-12-15",
            "review_content": "酒店位置很好，房间干净整洁，服务人员态度友好。早餐种类丰富，性价比不错。",
        },
        {
            "reviewer_name": "商务出差人",
            "rating": 5,
            "review_date": "2024-12-10",
            "review_content": "商务设施齐全，会议室很专业，网络稳定。房间安静，适合工作和休息。",
        },
    ]
    return fallback_reviews


# ---------------------------------------------------------------------------
# Tool entry points. Contract: tools/contracts.py
#
# recommend_hotels  -> AMap POI (real listings; AMap withholds price on this
#                      tier, so no price is ever reported)
# get_hotel_reviews -> web-search-grounded qwen under a strict no-fabrication
#                      prompt (AMap exposes a rating but no review text)
# ---------------------------------------------------------------------------
_TIER_KEYWORDS = (
    ("五星", "五星级酒店"), ("豪华", "五星级酒店"), ("高档", "五星级酒店"),
    ("经济", "经济型酒店"), ("便宜", "经济型酒店"), ("青年", "青年旅舍"),
    ("民宿", "民宿"), ("公寓", "公寓式酒店"),
)


def get_hotel_recommendations(location: str, requirements: str = "",
                              limit: int = 3) -> dict:
    """Query city-scoped AMap hotel POIs and available prices."""
    from . import _amap
    from . import _hotel_price as _price
    from cn_travel.business_logic.contracts import EMPTY, OK, err

    source = "amap-poi/100000"
    limit = max(1, min(int(limit or 3), 25))
    place = _amap.resolve_city(location)
    if not place:
        # An unresolved place is an error; an empty result means a resolved place
        # has no hotels. Conflating them makes the model misstate the failure.
        return err("recommend_hotels", f"could not resolve location {location!r}",
                   location=location)

    keywords = next((kw for needle, kw in _TIER_KEYWORDS if needle in (requirements or "")),
                    "酒店")
    try:
        pois = _amap.search_poi(keywords, adcode=place["adcode"], types="100000", limit=limit)
    except Exception as exc:
        return err("recommend_hotels", f"poi search failed: {exc}", location=location)

    hotels = []
    for p in pois[:limit]:
        pos = _amap.poi_lat_lng(p) or {}
        rating = (p.get("biz_ext") or {}).get("rating")
        tier = p.get("keytag")
        hotels.append({
            "name": p.get("name") or "",
            "address": p.get("address") or "",
            "district": p.get("adname") or "",
            "rating": float(rating) if rating not in ("", [], None) else None,
            "tier": tier if isinstance(tier, str) else "",
            "tel": (p.get("tel") or "").split(";")[0],
            "lat": pos.get("lat"), "lng": pos.get("lng"),
            "price_cny": _price.lowest_price(p.get("name") or "", location),
        })

    # Filter by budget only when prices exist. Keep hotels without prices so a
    # lookup failure never silently hides a real hotel.
    budget = _budget_ceiling(requirements)
    if budget and any(h["price_cny"] for h in hotels):
        hotels = [h for h in hotels if not h["price_cny"] or h["price_cny"] <= budget]

    if not hotels:
        return {"status": EMPTY, "source": source, "location": location, "hotels": []}
    return {"status": OK, "source": source, "location": location, "hotels": hotels}




def _budget_ceiling(requirements: str):
    """Extract the upper bound from a Chinese-language budget phrase."""
    import re
    nums = [int(n) for n in re.findall(r"(\d{2,5})\s*(?:-|~|到|至|元)", requirements or "")
            if 30 <= int(n) <= 99999]
    return max(nums) if nums else None


_REVIEW_SYSTEM = """你是酒店信息检索工具，只输出检索到的事实，严禁编造。
若检索不到某一项，该字段必须写 null，绝不可猜测或估算。
只输出 JSON，不要任何解释文字，格式：
{"rating": <0-5的数字或null>,
 "price_hint": "<带单位的价格区间原文，或null>",
 "reviews": ["<真实点评原文，最多3条>"],
 "summary": "<两句以内的中文总结；若无任何数据则写“未检索到该酒店的公开评价”>",
 "source": "<信息来源，如平台名称与点评数量>"}"""


def _rating_from_search_text(text: str):
    """Extract the first explicit rating and normalize it to a five-point scale."""
    import re

    patterns = (
        (r"(?:评分|评分为|rating)\s*[:：]?\s*(\d(?:\.\d+)?)\s*(?:分|/\s*5)?", 5),
        (r"(\d(?:\.\d+)?)\s*/\s*5", 5),
        (r"(\d(?:\.\d+)?)\s*/\s*10", 10),
        (r"(\d(?:\.\d+)?)\s*分", 5),
    )
    for pattern, scale in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        value = float(match.group(1))
        if 0 <= value <= scale:
            return round(value / 2, 1) if scale == 10 else value
    return None


def _price_from_search_text(text: str):
    """Extract a quoted nightly price while preserving its source wording."""
    import re

    patterns = (
        r"(?:¥|￥)\s*\d{2,5}(?:\.\d{1,2})?\s*(?:元)?(?:起|/晚|每晚)?",
        r"\d{2,5}\s*元\s*(?:起|/晚|每晚)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None


def _search_excerpts(hits: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Select up to three distinct non-empty excerpts in stable search order."""
    import re

    reviews: List[Dict[str, str]] = []
    seen = set()
    for hit in hits:
        text = re.sub(r"\s+", " ", str(hit.get("snippet") or "")).strip()
        fingerprint = text.casefold()
        if len(text) < 12 or fingerprint in seen:
            continue
        seen.add(fingerprint)
        reviews.append({"text": text})
        if len(reviews) == 3:
            break
    return reviews


def _reviews_via_search(hotel_name: str, location: str) -> dict:
    """Search public pages and deterministically map snippets to the contract."""
    from cn_travel.tool import _web_search as _websearch
    from cn_travel.business_logic.contracts import EMPTY, OK, err

    # Keep the query focused on review terms to preserve relevant source recall.
    query = f"{location} {hotel_name}".strip() + " 点评 评价"
    try:
        hits = _websearch.search(query, limit=6)
    except Exception as exc:
        return err("get_hotel_reviews", f"web search failed: {exc}",
                   hotel_name=hotel_name)
    # Rank review sites first so extraction sees substantive snippets early
    review_sites = ("ctrip", "trip.com", "tripadvisor", "dianping", "meituan", "qunar")
    hits.sort(key=lambda h: 0 if any(d in h.get("url", "") for d in review_sites) else 1)
    if not hits:
        return {"status": EMPTY, "source": f"websearch/{_websearch.provider()} (无结果)",
                "hotel_name": hotel_name, "rating": None, "price_hint": None,
                "reviews": [], "summary": "未检索到该酒店的公开评价"}

    searchable_text = "\n".join(
        f"{hit.get('title', '')}\n{hit.get('snippet', '')}" for hit in hits
    )
    rating = _rating_from_search_text(searchable_text)
    price_hint = _price_from_search_text(searchable_text)
    reviews = _search_excerpts(hits)

    status = OK if (reviews or rating is not None or price_hint) else EMPTY
    cited_urls = [str(hit.get("url") or "") for hit in hits[:3] if hit.get("url")]
    return {
        "status": status,
        "source": f"websearch/{_websearch.provider()} | {' | '.join(cited_urls)}",
        "hotel_name": hotel_name,
        "rating": rating,
        "price_hint": price_hint,
        "reviews": reviews,
        "summary": (
            f"检索到 {len(reviews)} 条公开网页摘要，详情请核对来源链接。"
            if status == OK
            else "未检索到该酒店的公开评价"
        ),
    }


def get_hotel_reviews(hotel_name: str, location: str = "") -> dict:
    """Rating, price hint and review excerpts for a named hotel.

    ``REVIEWS_BACKEND=search`` uses deterministic extraction from public search
    snippets. ``REVIEWS_BACKEND=dashscope`` uses Qwen with web search enabled.
    """
    import json as _json
    import os
    import re

    from cn_travel.business_logic.contracts import EMPTY, OK, err

    if not hotel_name:
        return err("get_hotel_reviews", "hotel_name is required", hotel_name="")

    backend = os.getenv("REVIEWS_BACKEND", "search").strip().lower() or "search"
    if backend == "search":
        return _reviews_via_search(hotel_name, location)
    if backend != "dashscope":
        return err("get_hotel_reviews", f"unsupported reviews backend: {backend}",
                   hotel_name=hotel_name)

    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        try:
            from dotenv import load_dotenv
            from cn_travel.paths import CN_TRAVEL

            load_dotenv(CN_TRAVEL.env_file)
            key = os.getenv("DASHSCOPE_API_KEY", "")
        except Exception:
            pass
    if not key:
        return err("get_hotel_reviews", "DASHSCOPE_API_KEY not configured",
                   hotel_name=hotel_name)

    try:
        from openai import OpenAI
        resp = OpenAI(api_key=key,
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                      ).chat.completions.create(
            model=os.getenv("REVIEW_MODEL", "qwen-plus"),
            messages=[{"role": "system", "content": _REVIEW_SYSTEM},
                      {"role": "user",
                       "content": f"{location} {hotel_name}".strip() + " 的房价与用户评价"}],
            max_tokens=600,
            extra_body={"enable_search": True,
                        "search_options": {"forced_search": True, "enable_source": True}},
        )
        text = resp.choices[0].message.content or ""
    except Exception as exc:
        return err("get_hotel_reviews", f"search request failed: {exc}", hotel_name=hotel_name)

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = fenced.group(1) if fenced else None
    if blob is None:
        s, e = text.find("{"), text.rfind("}")
        blob = text[s:e + 1] if s != -1 and e > s else ""
    try:
        parsed = _json.loads(blob)
    except Exception:
        return err("get_hotel_reviews", "backend returned unparseable output",
                   hotel_name=hotel_name)

    try:
        rating = float(parsed["rating"]) if parsed.get("rating") is not None else None
    except (TypeError, ValueError):
        rating = None
    reviews = [{"text": str(t)} for t in (parsed.get("reviews") or []) if str(t).strip()]
    price_hint = parsed.get("price_hint") or None
    summary = str(parsed.get("summary") or "").strip()

    status = OK if (reviews or rating is not None or price_hint) else EMPTY
    return {
        "status": status,
        "source": f"dashscope-qwen+websearch ({parsed.get('source') or 'unspecified'})",
        "hotel_name": hotel_name,
        "rating": rating,
        "price_hint": str(price_hint) if price_hint else None,
        "reviews": reviews,
        "summary": summary or ("未检索到该酒店的公开评价" if status == EMPTY else ""),
    }


# Example usage
if __name__ == "__main__":
    print("=== 酒店推荐测试 ===")
    result = recommend_hotels("北京", "商务 五星级", limit=3)
    print(f"status={result['status']}  source={result['source']}")
    for i, hotel in enumerate(result["hotels"], 1):
        print(f"\n酒店 {i}: {hotel['name']}")
        print(f"地址: {hotel['address']} ({hotel['district']})")
        print(f"评分: {hotel['rating']}  档次: {hotel['tier']}  电话: {hotel['tel']}")

    print("\n" + "=" * 50)

    print("=== 酒店评价测试 ===")
    reviews = get_hotel_reviews("北京国贸大酒店")
    print(f"status={reviews['status']}  source={reviews['source']}")
    print(f"评分: {reviews['rating']}  价格: {reviews['price_hint']}")
    print(f"总结: {reviews['summary']}")
    for i, review in enumerate(reviews["reviews"], 1):
        print(f"\n评论 {i}: {review['text']}")
