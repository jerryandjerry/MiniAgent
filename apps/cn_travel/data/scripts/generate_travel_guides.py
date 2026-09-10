#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one- to five-day city guides with the configured model."""

import json
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import threading

from data.paths import PATHS, load_data_env

load_data_env()

from data.clients.claude_cli import run as _claude_run
from data.clients.openai_compat import call_llm as _openai_call_llm


def call_llm(model_name: str, prompt: str, system_prompt: str = None, **kw) -> str:
    """Route Claude aliases to the CLI and other aliases to the API client."""
    if str(model_name).startswith("claude"):
        joined = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return _claude_run(joined, model=model_name,
                           effort=os.getenv("CLAUDE_CLI_EFFORT", "medium"))
    return _openai_call_llm(model_name, prompt, system_prompt, **kw)

# Global lock for thread-safe file operations and logging
file_lock = threading.Lock()
console_lock = threading.Lock()

class TravelGuideGenerator:
    def __init__(self, max_workers: int = 5, model_name: str = ""):
        """Initialize the generator with its concurrency limit."""
        self.max_workers = max_workers
        # GUIDE_MODEL selects the guide-generation model.
        self.model_name = model_name or os.getenv("GUIDE_MODEL", "claude-sonnet-5")
        # Corpus and progress paths follow the application path contract.
        self.city_mapping_file = str(PATHS.knowledge_base / "city_code_mapping.json")
        self.guides_dir = str(PATHS.guides)
        self.progress_file = str(PATHS.root / "generation_progress.json")
        
        os.makedirs(self.guides_dir, exist_ok=True)

        self.cities = self._load_cities()

        self.progress = self._load_progress()
        
        with console_lock:
            print(f"✓ 初始化完成，共{len(self.cities)}个城市")
            print(f"✓ 已完成{len(self.progress.get('completed', []))}个城市的攻略生成")
    
    def _load_cities(self) -> Dict[str, Dict]:
        """Load city metadata from the city-code mapping."""
        try:
            with open(self.city_mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("城市编码映射", {})
        except Exception as e:
            with console_lock:
                print(f"❌ 加载城市数据失败: {e}")
            return {}
    
    def _load_progress(self) -> Dict:
        """Load the resumable generation state."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                with console_lock:
                    print(f"⚠️ 加载进度文件失败: {e}")
        return {"completed": [], "failed": [], "last_update": ""}
    
    def _save_progress(self, city_code: str, city_name: str, success: bool):
        """Persist one city's generation status."""
        with file_lock:
            try:
                if success:
                    if city_code not in self.progress.get("completed", []):
                        self.progress.setdefault("completed", []).append(city_code)
                    if city_code in self.progress.get("failed", []):
                        self.progress["failed"].remove(city_code)
                else:
                    if city_code not in self.progress.get("failed", []):
                        self.progress.setdefault("failed", []).append(city_code)
                
                self.progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(self.progress, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                with console_lock:
                    print(f"❌ 保存进度失败 ({city_name}): {e}")
    
    def _generate_prompt(self, city_name: str, city_info: Dict) -> Tuple[str, str]:
        """Build the system and user prompts for one city guide."""

        days = random.randint(1, 5)
        
        system_prompt = """你是一位专业的旅游攻略规划师，擅长为游客制定详细实用的旅行计划。
请根据用户提供的城市信息，生成一份专业、详细、实用的旅游攻略。

攻略要求：
1. 内容要准确真实，基于该城市的实际情况
2. 包含景点推荐、美食推荐、住宿建议、交通指南等
3. 按天数合理安排行程，避免过于紧凑
4. 提供实用的旅行小贴士
5. 语言生动有趣，富有吸引力"""

        user_prompt = f"""请为{city_info.get('province', '')}的{city_name}制定一份{days}天的详细旅游攻略。

城市信息：
- 城市名称：{city_name}
- 行政级别：{city_info.get('level', '未知')}
- 所属省份：{city_info.get('province', '未知')}
- 攻略天数：{days}天

请按以下格式输出攻略：

# {city_name}{days}日游攻略

## 🏙️ 城市简介
[城市的基本介绍、历史文化背景、最佳旅行时间等]

## 🚗 交通指南
[如何到达该城市，以及市内交通方式]

## 🏨 住宿推荐
[推荐3-5个不同价位的住宿区域或酒店]

## 📅 详细行程安排

### 第1天：[主题]
- **上午**：[具体安排]
- **下午**：[具体安排] 
- **晚上**：[具体安排]
- **推荐美食**：[当地特色菜品]

[如果是多天行程，继续第2天、第3天等...]

## 🍽️ 必吃美食
[详细介绍当地特色美食，包括推荐餐厅]

## 🎁 购物推荐
[特产、纪念品购买建议]

## 💡 实用小贴士
[气候、着装、注意事项等实用信息]

## 💰 预算参考
[大致的花费预算，包含交通、住宿、餐饮、门票等]

请确保内容丰富详实，具有很强的实用性。"""

        return system_prompt, user_prompt
    
    def _generate_single_guide(self, city_code: str, city_info: Dict) -> bool:
        """Generate and persist one city guide."""
        city_name = city_info.get("name", "未知城市")
        
        if city_code in self.progress.get("completed", []):
            with console_lock:
                print(f"⏭️ {city_name} 攻略已存在，跳过")
            return True
        
        filename = f"{city_code}_{city_name.replace('市', '')}_travel_guide.txt"
        filepath = os.path.join(self.guides_dir, filename)
        
        if os.path.exists(filepath):
            with console_lock:
                print(f"⏭️ {city_name} 攻略文件已存在，跳过")
            self._save_progress(city_code, city_name, True)
            return True
        
        try:
            with console_lock:
                print(f"🚀 开始生成 {city_name} 的旅游攻略...")
            
            system_prompt, user_prompt = self._generate_prompt(city_name, city_info)

            start_time = time.time()
            guide_content = call_llm(
                model_name=self.model_name,
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                show_thinking=False
            )
            end_time = time.time()
            
            if not guide_content or len(guide_content.strip()) < 500:
                raise ValueError("生成的攻略内容过短或为空")
            
            with file_lock:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# {city_name}旅游攻略\n")
                    f.write(f"城市编码: {city_code}\n")
                    f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"生成用时: {end_time - start_time:.2f}秒\n")
                    f.write(f"生成模型: {self.model_name}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(guide_content)
            
            self._save_progress(city_code, city_name, True)
            
            with console_lock:
                print(f"✅ {city_name} 攻略生成完成 ({end_time - start_time:.2f}秒)")
            
            # Add jitter to avoid excessive API calls
            time.sleep(random.uniform(1, 3))
            
            return True
            
        except Exception as e:
            with console_lock:
                print(f"❌ {city_name} 攻略生成失败: {e}")
            
            self._save_progress(city_code, city_name, False)
            return False
    
    def generate_all_guides(self):
        """Generate guides for every incomplete city concurrently."""
        if not self.cities:
            print("❌ 没有找到城市数据")
            return
        
        pending_cities = {
            code: info for code, info in self.cities.items() 
            if code not in self.progress.get("completed", [])
        }
        
        if not pending_cities:
            print("🎉 所有城市的攻略都已生成完成！")
            return
        
        print(f"📋 准备为{len(pending_cities)}个城市生成旅游攻略...")
        print(f"🔧 使用{self.max_workers}个并行线程")
        print(f"🤖 使用模型: deepseek-v3")
        print("-" * 50)
        
        success_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_city = {
                executor.submit(self._generate_single_guide, city_code, city_info): (city_code, city_info)
                for city_code, city_info in pending_cities.items()
            }
            
            for future in as_completed(future_to_city):
                city_code, city_info = future_to_city[future]
                city_name = city_info.get("name", "未知城市")
                
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    with console_lock:
                        print(f"❌ {city_name} 处理异常: {e}")
                    failed_count += 1
                
                total_processed = success_count + failed_count
                total_cities = len(pending_cities)
                progress_percent = (total_processed / total_cities) * 100
                
                with console_lock:
                    print(f"📊 总进度: {total_processed}/{total_cities} ({progress_percent:.1f}%) | "
                          f"成功: {success_count} | 失败: {failed_count}")
        
        print("\n" + "=" * 50)
        print("🎯 攻略生成完成!")
        print(f"✅ 成功生成: {success_count}个城市")
        print(f"❌ 生成失败: {failed_count}个城市")
        print(f"📁 攻略文件保存在: {self.guides_dir}")
        
        if failed_count > 0:
            print(f"\n⚠️ 失败的城市可以重新运行脚本进行重试")

    def retry_failed_cities(self):
        """Retry every city recorded as failed."""
        failed_cities = self.progress.get("failed", [])
        if not failed_cities:
            print("😊 没有需要重试的城市")
            return
        
        print(f"🔄 开始重试{len(failed_cities)}个失败的城市...")
        
        self.progress["failed"] = []
        self._save_progress("", "", True)

        for city_code in failed_cities:
            if city_code in self.cities:
                city_info = self.cities[city_code]
                self._generate_single_guide(city_code, city_info)

def main():
    print("🎯 中国城市旅游攻略生成器")
    print("=" * 50)
    
    generator = TravelGuideGenerator(max_workers=3)

    while True:
        print("\n请选择操作:")
        print("1. 生成所有城市攻略")
        print("2. 重试失败的城市")
        print("3. 查看生成进度")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            generator.generate_all_guides()
        elif choice == "2":
            generator.retry_failed_cities()
        elif choice == "3":
            completed = len(generator.progress.get("completed", []))
            failed = len(generator.progress.get("failed", []))
            total = len(generator.cities)
            print(f"\n📊 生成进度统计:")
            print(f"✅ 已完成: {completed}/{total} ({completed/total*100:.1f}%)")
            print(f"❌ 失败: {failed}")
            print(f"📅 最后更新: {generator.progress.get('last_update', '未知')}")
        elif choice == "4":
            print("👋 退出程序")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()
