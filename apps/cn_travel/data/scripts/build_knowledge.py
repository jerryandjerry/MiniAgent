#!/usr/bin/env python3
"""Build the Milvus travel-guide index concurrently."""

import os

from cn_travel.service import guide_build as embeddings
import json
import glob
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from tqdm import tqdm

from data.paths import PATHS, load_data_env

# Load configuration from the application environment file.
load_data_env()

# The embeddings module owns the model and dimensions from config.yaml or the environment.
EMBEDDING_DIMENSIONS = embeddings.embedding_dimensions()

MILVUS_URI = os.getenv("MILVUS_URI", str(PATHS.milvus_db))
COLLECTION_NAME = "travel_guides"

MAX_WORKERS = 10


@dataclass(frozen=True)
class KnowledgeBuild:
    name: str
    num_entities: int


class TravelGuideImporter:
    def __init__(self):
        # Route indexing and query vectors through the same embedding implementation.
        self.city_mapping = {}
        self.load_city_mapping()
        self._lock = threading.Lock()  # Protect shared counters.
        
    def load_city_mapping(self):
        """Load the city-code mapping."""
        with open(PATHS.knowledge_base / 'city_code_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.city_mapping = data['城市编码映射']
        print(f"已加载 {len(self.city_mapping)} 个城市映射")
    

    def get_embedding(self, text: str) -> List[float]:
        """Create a vector with the same implementation used at query time."""
        return embeddings.embed_one(text)


    def setup_milvus(self, uri: str = MILVUS_URI):
        """Create an empty indexed Milvus collection."""
        connections.connect(uri=uri)

        if utility.has_collection(COLLECTION_NAME):
            Collection(COLLECTION_NAME).drop()
            print(f"删除已存在的集合: {COLLECTION_NAME}")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="city_code", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="city_name", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="province_name", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSIONS),
        ]
        
        schema = CollectionSchema(fields, description="旅游攻略向量数据库")
        collection = Collection(COLLECTION_NAME, schema, consistency_level="Bounded")
        
        index_params = {
            "index_type": "AUTOINDEX",
            "metric_type": "IP",
            "params": {}
        }
        collection.create_index("embedding", index_params)
        print(f"创建集合和索引: {COLLECTION_NAME}")
        
        return collection
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Extract city metadata from a guide filename."""
        basename = os.path.basename(filename)
        parts = basename.replace('_travel_guide.txt', '').split('_')
        if len(parts) >= 2:
            city_code = parts[0]
            city_name = '_'.join(parts[1:])
            
            if city_code in self.city_mapping:
                province_name = self.city_mapping[city_code]['province']
                return {
                    'city_code': city_code,
                    'city_name': city_name,
                    'province_name': province_name
                }
        
        return None
    
    def process_single_file(self, file_path: str) -> Tuple[bool, Any]:
        """Transform one guide into an indexed record."""
        
        try:
            file_info = self.parse_filename(file_path)
            if not file_info:
                return False, f"无法解析文件名: {file_path}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                return False, f"文件内容为空: {file_path}"
            
            embedding = self.get_embedding(content)
            if not embedding:
                return False, f"无法获取embedding: {file_path}"
            
            data_record = {
                'city_code': file_info['city_code'],
                'city_name': file_info['city_name'], 
                'province_name': file_info['province_name'],
                'content': content,
                'embedding': embedding
            }
            
            return True, data_record
            
        except Exception as e:
            return False, f"处理文件失败 {file_path}: {e}"
    
    def insert_batch_data(self, collection: Collection, batch_records: List[Dict]):
        """Insert a batch using Milvus's field-oriented input layout."""
        if not batch_records:
            return
            
        city_codes = []
        city_names = []
        province_names = []
        contents = []
        embeddings = []
        
        for record in batch_records:
            city_codes.append(record['city_code'])
            city_names.append(record['city_name'])
            province_names.append(record['province_name'])
            contents.append(record['content'])
            embeddings.append(record['embedding'])
        
        data_to_insert = [
            city_codes,
            city_names,
            province_names,
            contents,
            embeddings
        ]
        
        collection.insert(data_to_insert)

    @staticmethod
    def _promote_index_pair(staged_db: Path, target_db: Path) -> None:
        staged_fingerprint = staged_db.with_name(embeddings.FINGERPRINT_NAME)
        target_fingerprint = target_db.with_name(embeddings.FINGERPRINT_NAME)
        if not staged_db.is_file() or not staged_fingerprint.is_file():
            raise RuntimeError("verified knowledge build did not produce both artifacts")
        target_db.parent.mkdir(parents=True, exist_ok=True)
        for target in (target_db, target_fingerprint):
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise RuntimeError(f"knowledge artifact is not a regular file: {target}")

        with tempfile.TemporaryDirectory(
            prefix=".knowledge-backup.", dir=target_db.parent
        ) as backup_dir:
            backup_root = Path(backup_dir)
            previous_db = backup_root / "milvus.db"
            previous_fingerprint = backup_root / embeddings.FINGERPRINT_NAME
            had_db = target_db.is_file()
            had_fingerprint = target_fingerprint.is_file()
            if had_db:
                shutil.copy2(target_db, previous_db)
            if had_fingerprint:
                shutil.copy2(target_fingerprint, previous_fingerprint)
            try:
                staged_db.replace(target_db)
                staged_fingerprint.replace(target_fingerprint)
            except Exception:
                if had_db:
                    shutil.copy2(previous_db, target_db)
                else:
                    target_db.unlink(missing_ok=True)
                if had_fingerprint:
                    shutil.copy2(previous_fingerprint, target_fingerprint)
                else:
                    target_fingerprint.unlink(missing_ok=True)
                raise

    def _build_local_index(self, records: List[Dict]) -> KnowledgeBuild:
        uri = str(MILVUS_URI)
        if "://" in uri:
            raise RuntimeError("knowledge build output must be a local Milvus database")
        target_db = Path(uri).expanduser().resolve()
        target_db.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=".knowledge-build.", dir=target_db.parent
        ) as build_dir:
            staged_db = Path(build_dir) / "milvus.db"
            try:
                collection = self.setup_milvus(str(staged_db))
                batch_size = 50
                for start in range(0, len(records), batch_size):
                    self.insert_batch_data(
                        collection, records[start:start + batch_size]
                    )
                collection.flush()
                collection.load()
                count = int(collection.num_entities)
                if count != len(records):
                    raise RuntimeError(
                        f"knowledge collection contains {count} entities; "
                        f"expected {len(records)}"
                    )
                name = str(collection.name)
                embeddings.write_fingerprint(staged_db)
            finally:
                connections.disconnect("default")

            self._promote_index_pair(staged_db, target_db)
        return KnowledgeBuild(name=name, num_entities=count)
    
    def process_travel_guides(self, data_dir: str):
        """Build and atomically promote an index for every guide."""
        travel_guide_pattern = str(PATHS.guides / "*_travel_guide.txt")
        files = sorted(glob.glob(travel_guide_pattern))
        print(f"找到 {len(files)} 个旅游攻略文件")
        if not files:
            raise RuntimeError(f"no travel-guide files found at {PATHS.guides}")

        # Complete every source transformation before replacing the collection.
        embeddings.warm_up()
        
        records = []
        failures = []
        
        print(f"开始使用 {MAX_WORKERS} 个线程并发处理...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(self.process_single_file, file_path): file_path 
                             for file_path in files}

            with tqdm(total=len(files), desc="处理旅游攻略") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success, result = future.result()
                        
                        if success:
                            records.append(result)
                        else:
                            failures.append(str(result))
                            
                    except Exception as e:
                        failures.append(f"{file_path}: {e}")
                    
                    pbar.update(1)
        
        if failures or len(records) != len(files):
            preview = "\n".join(failures[:10])
            raise RuntimeError(
                f"knowledge build rejected {len(failures)} of {len(files)} files"
                + (f"\n{preview}" if preview else "")
            )

        records.sort(key=lambda record: (record["city_code"], record["city_name"]))
        collection = self._build_local_index(records)
        print(f"\n=== 处理完成 ===")
        print(f"成功处理: {len(records)} 条记录")
        print("失败: 0 个文件")
        print(f"集合中实际记录数: {collection.num_entities}")
        
        return collection

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    importer = TravelGuideImporter()
    
    try:
        collection = importer.process_travel_guides(current_dir)
        
        print("\n=== 数据验证 ===")
        print(f"集合名称: {collection.name}")
        print(f"数据总数: {collection.num_entities}")
        
        search_params = {"metric_type": "IP", "params": {}}
        results = collection.search(
            data=[[0.1] * EMBEDDING_DIMENSIONS],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["city_name", "province_name"]
        )
        
        print("查询测试结果:")
        for hits in results:
            for hit in hits:
                print(f"- {hit.entity.get('province_name')}-{hit.entity.get('city_name')} (分数: {hit.score:.4f})")
                
    except Exception as e:
        print(f"导入过程出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
