#!/usr/bin/env python3
"""In-process hybrid retrieval for the travel-guide tool."""

import os
import time
from typing import List, Dict, Any
from pymilvus import connections, Collection
import logging
from collections import defaultdict
from threading import Lock
import jieba

from . import embeddings
from dotenv import load_dotenv
from cn_travel.paths import CN_TRAVEL

# Load configuration from the application environment file.
load_dotenv(dotenv_path=CN_TRAVEL.env_file)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration

# Milvus configuration
MILVUS_URI = os.getenv("MILVUS_URI", str(CN_TRAVEL.milvus_db))
COLLECTION_NAME = "travel_guides"

# Each embedding model has its own threshold; score distributions are model-specific.
_THRESHOLD_BY_MODEL = {
    "dashscope:text-embedding-v4:1024": 0.35,
    "local:models/embedding/embeddinggemma-300m:768": 0.33,
    "local:model/embedding/embeddinggemma-300m:768": 0.33,
    "local:tool/guide/knowledge_base/embedding/embeddinggemma-300m:768": 0.33,
}


from .normalization import stem as _stem


def _name_matches(name: str, query: str) -> bool:
    """Match either the full administrative name or its normalized stem."""
    if not name or not query:
        return False
    return name in query or (_stem(name) and _stem(name) in query)


class RAGService:
    # Keyword candidates to retrieve before scoring. The corpus has one document per city, 361 total,
    # so this cap is large enough for scoring to see every match.
    KEYWORD_CANDIDATE_CAP = 2000
    def __init__(self):
        # Indexing and queries share this embedding implementation.
        self.collection = None
        self.setup_milvus()
        # Vector thresholds are model-specific and require recalibration after a model change.
        self.VECTOR_SIMILARITY_THRESHOLD = float(
            os.getenv("VECTOR_SIMILARITY_THRESHOLD",
                      _THRESHOLD_BY_MODEL.get(embeddings.describe(), 0.35)))
        self.KEYWORD_SCORE_THRESHOLD = 1
        self.RRF_SCORE_THRESHOLD = 0.02
        # Initialize Chinese tokenization
        jieba.initialize()
    
    def setup_milvus(self):
        """Connect to Milvus with bounded retries."""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Disconnect any existing connection first
                try:
                    connections.disconnect("default")
                except:
                    pass
                
                # Wait before reconnecting
                if attempt > 0:
                    logger.info(f"重试连接Milvus (第{attempt+1}次)...")
                    time.sleep(retry_delay)
                
                # Establish the connection
                connections.connect(uri=MILVUS_URI)
                self.collection = Collection(COLLECTION_NAME)
                self.collection.load()
                
                logger.info(f"成功连接Milvus集合: {COLLECTION_NAME}")
                logger.info(f"集合中的数据量: {self.collection.num_entities}")
                return
                
            except Exception as e:
                logger.warning(f"连接Milvus失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"连接Milvus失败，已重试{max_retries}次: {e}")
                    raise
                retry_delay *= 2  # Exponential backoff
    
    def ensure_connection(self):
        """Restore the Milvus connection when needed."""
        try:
            if self.collection is None:
                self.setup_milvus()
            else:
                # Test the connection
                _ = self.collection.num_entities
        except Exception as e:
            logger.warning(f"检测到连接问题，重新连接: {e}")
            self.setup_milvus()
    
    def get_embedding(self, text: str) -> List[float]:
        """Embed a query with the same configuration used for indexing."""
        return embeddings.embed_one(text)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract searchable keywords."""
        # Tokenize with jieba
        words = jieba.cut_for_search(text)
        # Filter stop words and short tokens
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '里', '啊', '哦', '哈', '吧', '呢', '吗',"酒店"}
        keywords = [word.strip() for word in words if len(word.strip()) > 1 and word.strip() not in stop_words]
        return keywords
    
    def extract_locations(self, query: str) -> Dict[str, List[str]]:
        """Extract known province and city names from a query."""
        try:
            self.ensure_connection()
            
            # Retrieve all province and city data
            all_locations = self.collection.query(
                expr="",
                output_fields=["province_name", "city_name"],
                limit=1000  # Fetch enough data for matching
            )
            
            # Build province and city lists
            provinces = set()
            cities = set()
            for location in all_locations:
                if location.get("province_name"):
                    provinces.add(location.get("province_name"))
                if location.get("city_name"):
                    cities.add(location.get("city_name"))
            
            # Find matching province and city names in the query
            found_provinces = []
            found_cities = []
            
            for province in provinces:
                if _name_matches(province, query):
                    found_provinces.append(province)
            
            for city in cities:
                if _name_matches(city, query):
                    found_cities.append(city)
            
            return {
                "provinces": found_provinces,
                "cities": found_cities
            }
            
        except Exception as e:
            logger.error(f"位置提取失败: {e}")
            return {"provinces": [], "cities": []}
    
    def keyword_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search the guide corpus by keyword frequency."""
        try:
            self.ensure_connection()
            
            # Extract query keywords
            keywords = self.extract_keywords(query)
            if not keywords:
                return []
            
            # Build the keyword query expression
            # Match against the content field
            keyword_exprs = []
            for keyword in keywords:
                keyword_exprs.append(f'content like "%{keyword}%"')
            
            # Join keywords with OR to require at least one match
            expr = " or ".join(keyword_exprs)
            
            # Execute the query
            # Fetch candidates before TF scoring because Milvus truncates by insertion order.
            results = self.collection.query(
                expr=expr,
                output_fields=["city_code", "city_name", "province_name", "content"],
                limit=self.KEYWORD_CANDIDATE_CAP
            )
            
            # Compute keyword-match scores
            scored_results = []
            for result in results:
                content = result.get("content", "").lower()
                score = 0
                matched_keywords = []
                
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in content:
                        # Simple TF score: keyword occurrence count
                        count = content.count(keyword_lower)
                        score += count
                        matched_keywords.append(keyword)
                
                scored_results.append({
                    "city_code": result.get("city_code"),
                    "city_name": result.get("city_name"),
                    "province_name": result.get("province_name"),
                    "content": result.get("content"),
                    "keyword_score": score,
                    "matched_keywords": matched_keywords,
                    "location": f"{result.get('province_name')}-{result.get('city_name')}"
                })
            
            # Sort by keyword score
            scored_results.sort(key=lambda x: x["keyword_score"], reverse=True)
            return scored_results
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []
    
    def reciprocal_rank_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], 
                             vector_weight: float = 1.0, keyword_weight: float = 1.0, 
                             k: int = 60) -> List[Dict[str, Any]]:
        """Fuse vector and keyword rankings with weighted reciprocal rank."""
        # Map document IDs to results
        doc_scores = defaultdict(float)
        doc_info = {}
        
        # Process vector results and apply the vector weight
        for rank, result in enumerate(vector_results):
            doc_id = f"{result['city_code']}_{result['city_name']}"
            rrf_score = vector_weight * (1.0 / (k + rank + 1))
            doc_scores[doc_id] += rrf_score
            doc_info[doc_id] = result
            doc_info[doc_id]['vector_rank'] = rank + 1
            doc_info[doc_id]['vector_score'] = result.get('score', 0)
            doc_info[doc_id]['vector_rrf_score'] = rrf_score
        
        # Process keyword results and apply the keyword weight
        for rank, result in enumerate(keyword_results):
            doc_id = f"{result['city_code']}_{result['city_name']}"
            rrf_score = keyword_weight * (1.0 / (k + rank + 1))
            doc_scores[doc_id] += rrf_score
            if doc_id not in doc_info:
                doc_info[doc_id] = result
            doc_info[doc_id]['keyword_rank'] = rank + 1
            doc_info[doc_id]['keyword_score'] = result.get('keyword_score', 0)
            doc_info[doc_id]['keyword_rrf_score'] = rrf_score
            doc_info[doc_id]['matched_keywords'] = result.get('matched_keywords', [])
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for doc_id, total_rrf_score in sorted_docs:
            result = doc_info[doc_id].copy()
            result['rrf_score'] = total_rrf_score
            result['search_type'] = 'hybrid'
            result['fusion_info'] = {
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight,
                'vector_rrf_contribution': result.get('vector_rrf_score', 0),
                'keyword_rrf_contribution': result.get('keyword_rrf_score', 0)
            }
            fused_results.append(result)
        
        return fused_results
    
    def location_priority_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Prefer exact province or city retrieval before semantic search."""
        try:
            # 1. Extract locations
            locations = self.extract_locations(query)
            found_provinces = locations["provinces"]
            found_cities = locations["cities"]
            
            # 2. Prefer location search when a location is found
            if found_provinces or found_cities:
                location_results = []
                
                # Search by province
                for province in found_provinces:
                    results = self.search_by_location(province=province, limit=limit)
                    for result in results:
                        result["match_type"] = "province_match"
                        result["matched_location"] = province
                        result["location_score"] = 1.0
                    location_results.extend(results)
                
                # Search by city
                for city in found_cities:
                    results = self.search_by_location(city=city, limit=limit)
                    for result in results:
                        result["match_type"] = "city_match"
                        result["matched_location"] = city
                        result["location_score"] = 1.0
                    location_results.extend(results)
                
                # Deduplicate by city_code and city_name
                seen = set()
                unique_results = []
                for result in location_results:
                    key = f"{result.get('city_code')}_{result.get('city_name')}"
                    if key not in seen:
                        seen.add(key)
                        unique_results.append(result)
                
                if unique_results:
                    return {
                        "search_strategy": "location_priority",
                        "matched_locations": {
                            "provinces": found_provinces,
                            "cities": found_cities
                        },
                        "results": unique_results[:limit]
                    }
            
            # 3. Return no results if no location is found or location search finds nothing
            return {
                "search_strategy": "hybrid_fallback",
                "matched_locations": {
                    "provinces": found_provinces,
                    "cities": found_cities
                },
                "results": []
            }
            
        except Exception as e:
            logger.error(f"位置优先搜索失败: {e}")
            return {
                "search_strategy": "error_fallback",
                "matched_locations": {"provinces": [], "cities": []},
                "results": []
            }
    
    @staticmethod
    def drop_results_about_another_place(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop city documents whose city and province are absent from the query."""
        kept = []
        for r in results:
            city = (r.get("city_name") or "").strip()
            province = (r.get("province_name") or "").strip()
            names = [n for n in (city, province) if n]
            # Strip administrative suffixes before comparing the full name with its stem
            for n in list(names):
                names.append(n.rstrip("市省县区").replace("自治区", ""))
            if any(n and n in query for n in names):
                kept.append(r)
        return kept

    def filter_by_threshold(self, results: List[Dict[str, Any]], search_type: str) -> List[Dict[str, Any]]:
        """Filter results with the threshold for their retrieval method."""
        if not results:
            return results
            
        filtered_results = []
        
        for result in results:
            should_keep = False
            
            if search_type == "vector":
                # Vector-search threshold
                vector_score = result.get("score", 0)
                if vector_score >= self.VECTOR_SIMILARITY_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"vector_score({vector_score:.3f}) >= threshold({self.VECTOR_SIMILARITY_THRESHOLD})"
                else:
                    result["filter_reason"] = f"vector_score({vector_score:.3f}) < threshold({self.VECTOR_SIMILARITY_THRESHOLD})"
                    
            elif search_type == "keyword":
                # Keyword-search threshold
                keyword_score = result.get("keyword_score", 0)
                if keyword_score >= self.KEYWORD_SCORE_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"keyword_score({keyword_score}) >= threshold({self.KEYWORD_SCORE_THRESHOLD})"
                else:
                    result["filter_reason"] = f"keyword_score({keyword_score}) < threshold({self.KEYWORD_SCORE_THRESHOLD})"
                    
            else:  # hybrid
                # Hybrid search uses the RRF threshold
                rrf_score = result.get("rrf_score", 0)
                if rrf_score >= self.RRF_SCORE_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"rrf_score({rrf_score:.4f}) >= threshold({self.RRF_SCORE_THRESHOLD})"
                else:
                    result["filter_reason"] = f"rrf_score({rrf_score:.4f}) < threshold({self.RRF_SCORE_THRESHOLD})"
            
            if should_keep:
                filtered_results.append(result)
        
        return filtered_results
    
    def search(self, query: str, limit: int = 1, search_type: str = "hybrid", 
               vector_weight: float = 0.7, keyword_weight: float = 1.5,
               use_threshold: bool = True) -> List[Dict[str, Any]]:
        """Search by location first, then use thresholded hybrid retrieval."""
        try:
            if search_type == "vector":
                results = self.vector_search(query, limit * 2)  # Fetch extra candidates for filtering
            elif search_type == "keyword":
                results = self.keyword_search(query, limit * 2)
            elif search_type == "location":
                location_result = self.location_priority_search(query, limit)
                return location_result["results"]
            else:
                # Smart hybrid search
                location_result = self.location_priority_search(query, limit)
                
                if location_result["results"]:
                    # Return a successful location search directly; it needs no threshold filtering
                    for result in location_result["results"]:
                        result["search_strategy"] = location_result["search_strategy"]
                        result["matched_locations"] = location_result["matched_locations"]
                    return location_result["results"]
                
                # Fall back to hybrid retrieval when location search fails
                logger.info(f"位置搜索无结果，降级到混合检索: {query}")
                
                vector_results = self.vector_search(query, limit * 3)  # Fetch extra candidates for filtering
                keyword_results = self.keyword_search(query, limit * 3)
                
                if not vector_results and not keyword_results:
                    return []
                elif not vector_results:
                    results = keyword_results
                    search_type = "keyword"
                elif not keyword_results:
                    results = vector_results
                    search_type = "vector"
                else:
                    results = self.reciprocal_rank_fusion(
                        vector_results, keyword_results, 
                        vector_weight, keyword_weight
                    )
                
                # Fallback retrieval must return no match rather than substitute another city.
                # The corpus has one document per city, so a result whose city or province is absent
                # from the query does not answer it. A Sanya query returning Zhuanghe, Haicheng, or
                # Haiyang would cause downstream code to fabricate a Sanya itinerary.
                results = self.drop_results_about_another_place(query, results)

                # Add strategy metadata to hybrid results
                for result in results:
                    result["search_strategy"] = "hybrid_fallback"
                    result["matched_locations"] = location_result["matched_locations"]
            
            # Apply threshold filtering
            if use_threshold:
                results = self.filter_by_threshold(results, search_type)
                logger.info(f"阈值过滤后剩余结果数: {len(results)}")
            
            return results[:limit]
                    
        except Exception as e:
            logger.error(f"智能搜索失败: {e}")
            return []
    
    def vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search the guide corpus by vector similarity."""
        try:
            # Ensure the connection is healthy
            self.ensure_connection()
            
            # Get the query vector
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Search parameters
            search_params = {"metric_type": "IP", "params": {}}
            
            # Execute the search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["city_code", "city_name", "province_name", "content"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "city_code": hit.entity.get("city_code"),
                        "city_name": hit.entity.get("city_name"), 
                        "province_name": hit.entity.get("province_name"),
                        "content": hit.entity.get("content"),
                        "score": float(hit.score),
                        "search_type": "vector",
                        "location": f"{hit.entity.get('province_name')}-{hit.entity.get('city_name')}"
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def search_by_location(self, province: str = None, city: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve guides for an exact province or city filter."""
        try:
            # Ensure the connection is healthy
            self.ensure_connection()
            
            # Build filter conditions
            filter_expr = []
            if province:
                filter_expr.append(f'province_name == "{province}"')
            if city:
                filter_expr.append(f'city_name like "%{city}%"')
            
            expr = " and ".join(filter_expr) if filter_expr else ""
            
            # Execute the query
            # Fetch candidates before local truncation because Milvus orders by insertion.
            results = self.collection.query(
                expr=expr,
                output_fields=["city_code", "city_name", "province_name", "content"],
                limit=self.KEYWORD_CANDIDATE_CAP
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "city_code": result.get("city_code"),
                    "city_name": result.get("city_name"),
                    "province_name": result.get("province_name"), 
                    "content": result.get("content"),
                    "location": f"{result.get('province_name')}-{result.get('city_name')}"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"按位置搜索失败: {e}")
            return []

# Global service instance
rag_service = None
_rag_service_lock = Lock()

def get_rag_service():
    """Return the lazily initialized retrieval service."""
    global rag_service
    if rag_service is None:
        with _rag_service_lock:
            if rag_service is None:
                rag_service = RAGService()
    return rag_service
