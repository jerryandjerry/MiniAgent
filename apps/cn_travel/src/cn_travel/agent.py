#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble CN Travel policy, user context, and service-backed tool dispatch."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict

from cn_travel.paths import CN_TRAVEL
from cn_travel.service.agent import Agent, AgentError, Dependency
from cn_travel.service import tools as tool_service

# Conditional workflow dependencies. Workflow 1 skips weather if the guide is empty.
# Workflow 3 skips reviews without a hotel, whose name must come from recommend_hotels.
DEPENDS_ON = {
    "get_weather_info": Dependency("search_travel_guide", "攻略为空"),
    "get_hotel_reviews": Dependency("recommend_hotels", "无酒店结果", args_from_prereq=True),
}


class TravelAssistantFuncCall(Agent):
    """Travel assistant driven by model-issued function calls."""

    def __init__(
        self,
        model_name: str | None = None,
        user_name: str = "用户",
        user_city_id: str = "101010100",
        travel_date_range: str = "",
        start_coordinates: str = "116.481028,39.989643",
        user_city_name: str = "北京",
        **kwargs,
    ):
        # Compute the default departure window relative to the current date.
        travel_date_range = travel_date_range or self._default_date_range()
        self.user_info = {
            "name": user_name,
            "city_id": user_city_id,
            # Tools locate by city name; Amap cannot resolve the weather service city_id
            "city_name": user_city_name,
            "travel_date_range": travel_date_range,
            "start_coordinates": start_coordinates,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
        }
        self._parse_travel_dates()

        super().__init__(
            system_prompt=self._build_system_prompt(),
            tools=json.loads(CN_TRAVEL.tool_schemas.read_text(encoding="utf-8")),
            dispatch=self._build_dispatch(),
            depends_on=DEPENDS_ON,
            model_name=model_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------
    @staticmethod
    def _default_date_range(days: int = 5) -> str:
        """Return a rolling ``days``-day window starting tomorrow."""
        start = datetime.now() + timedelta(days=1)
        return f"{start:%Y-%m-%d}~{start + timedelta(days=days - 1):%Y-%m-%d}"

    def _parse_travel_dates(self) -> None:
        rng = self.user_info["travel_date_range"]
        if "~" in rng:
            start_str, end_str = rng.split("~", 1)
            self.user_info["travel_start_date"] = start_str.strip()
            self.user_info["travel_end_date"] = end_str.strip()
        else:
            self.user_info["travel_start_date"] = rng
            self.user_info["travel_end_date"] = rng

    # ------------------------------------------------------------------
    # Business wiring
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        user_info = self._fill(
            CN_TRAVEL.user_info.read_text(encoding="utf-8"))
        return CN_TRAVEL.system_prompt.read_text(encoding="utf-8").replace("{user_info}", user_info)

    def _fill(self, text: str) -> str:
        """Render the user-info template with the current profile."""
        u = self.user_info
        return (text
                .replace("{self.user_info['name']}", u["name"])
                .replace("{self.user_info['current_date']}", u["current_date"])
                .replace("{self.user_info['city_name']}", u["city_name"])
                .replace("{self.user_info['city_id']}", u["city_id"])
                .replace("{self.user_info['travel_date_range']}", u["travel_date_range"])
                .replace("{self.user_info['start_coordinates']}", u["start_coordinates"]))

    def _build_dispatch(self) -> Dict[str, Any]:
        """Route model arguments through the application's tool-method seams."""
        return {
            "search_travel_guide": lambda a: self.search_travel_guide(
                a.get("location"), a.get("search_mode", "hybrid")),
            "get_weather_info": lambda a: self.get_weather_info(
                a.get("location"), a.get("start_date"), a.get("num_days", 1)),
            "query_route": lambda a: self.query_route(
                a.get("start_location"), a.get("end_location"), a.get("city", "")),
            "recommend_hotels": lambda a: self.recommend_hotels(
                a.get("location", ""), a.get("requirements", "")),
            "get_hotel_reviews": lambda a: self.get_hotel_reviews_func(
                a.get("hotel_name"), a.get("location", "")),
        }


    # ------------------------------------------------------------------
    # Tool methods fill user defaults, call services, and serialize contracts.
    # Dispatch passes through them, making these the application's only seams.
    # ------------------------------------------------------------------
    def search_travel_guide(self, location: str, search_mode: str = "hybrid") -> str:
        return self._serialise(tool_service.search_travel_guide(location, search_mode))

    def get_weather_info(self, location: str, start_date: str, num_days: int = 1) -> str:
        return self._serialise(tool_service.get_weather_info(location, start_date, num_days))

    def query_route(self, start_location: str, end_location: str, city: str = "") -> str:
        return self._serialise(
            tool_service.query_route(
                start_location,
                end_location,
                city or self.user_info["city_name"],
            )
        )

    def recommend_hotels(self, location: str = "", requirements: str = "") -> str:
        return self._serialise(
            tool_service.recommend_hotels(
                location or self.user_info["city_name"],
                requirements,
            )
        )

    def get_hotel_reviews_func(self, hotel_name: str, location: str = "") -> str:
        return self._serialise(tool_service.get_hotel_reviews(hotel_name, location))


def main() -> None:
    """Run the interactive terminal chat."""
    print("🌍 欢迎使用智能旅行助手！")
    print("1. 制定旅行计划（会自动查询天气和攻略）")
    print("2. 查询路线（问路导航）")
    print("3. 推荐酒店和查看评价")
    print("4. 回答旅行相关问题")

    user_name = input("\n请输入您的姓名（默认：旅行者）：").strip() or "旅行者"
    user_city_input = input("请输入您所在的城市（默认：北京）：").strip() or "北京"
    city_id_map = {
        "北京": "101010100", "上海": "101020100", "广州": "101280101",
        "深圳": "101280601", "杭州": "101210101", "南京": "101190101",
    }
    default_range = TravelAssistantFuncCall._default_date_range()
    travel_range = input(f"请输入出发日期范围（默认：{default_range}）：").strip() or default_range
    start_coords = input("请输入起点坐标（默认：116.481028,39.989643）：").strip() or "116.481028,39.989643"

    print("✅ 用户信息设置完成，输入 'quit' 退出\n")
    assistant = TravelAssistantFuncCall(
        user_name=user_name,
        user_city_id=city_id_map.get(user_city_input, "101010100"),
        user_city_name=user_city_input,
        travel_date_range=travel_range,
        start_coordinates=start_coords,
    )

    while True:
        try:
            user_input = input("您: ").strip()
            if user_input.lower() in ("quit", "exit", "退出", "再见"):
                print("🌍 感谢使用旅行助手，祝您旅途愉快！")
                break
            if not user_input:
                continue
            print("\n正在处理您的请求...")
            try:
                print(f"\n助手: {assistant.process_user_input(user_input)}\n")
            except AgentError as exc:
                # A transient API failure should not end a conversation
                print(f"\n❌ {exc}\n请重试，或换个说法。\n")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\n\n🌍 感谢使用旅行助手，祝您旅途愉快！")
            break


if __name__ == "__main__":
    main()
