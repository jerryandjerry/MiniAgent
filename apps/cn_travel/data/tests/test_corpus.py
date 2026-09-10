"""Coverage checks for the source travel-guide corpus."""

from project_paths import DATA_ROOT


IN_CORPUS = {"北京", "上海", "嘉兴", "哈尔滨", "南京", "青岛"}
OUT_OF_CORPUS = {"平壤", "东京", "首尔", "曼谷", "纽约", "新加坡"}


def _corpus_cities() -> set[str]:
    guides = DATA_ROOT / "knowledge_base" / "travel_guides"
    return {path.name.split("_")[1] for path in guides.glob("*_travel_guide.txt")}


def test_source_corpus_contains_expected_cities_and_excludes_foreign_fixtures():
    cities = _corpus_cities()
    assert cities
    assert IN_CORPUS <= cities
    assert OUT_OF_CORPUS.isdisjoint(cities)
