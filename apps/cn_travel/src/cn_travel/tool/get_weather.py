import requests
from datetime import datetime

def get_weather_by_date_range(location: str, start_date: str, num_days: int, query_range: str = "30d",
                              raise_on_error: bool = False):
    """Return weather rows for a coordinate or city ID and date range."""
    from datetime import datetime, timedelta
    
    # Convert city ID to coordinates if needed
    coordinates = _get_coordinates_from_location(location)
    if not coordinates:
        print("❌ 未获取到天气数据: 无法解析地点信息")
        if raise_on_error:
            raise ValueError(f"无法解析地点: {location}")
        return []
    
    # Past dates require the archive endpoint; forecast returns 400 for them.
    today = datetime.now().date()
    url = ("https://archive-api.open-meteo.com/v1/archive"
           if datetime.strptime(start_date, "%Y-%m-%d").date() < today
           else "https://api.open-meteo.com/v1/forecast")
    
    # Calculate end date
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=num_days - 1)
    
    params = {
        "latitude": coordinates[1],  # latitude
        "longitude": coordinates[0],  # longitude
        "start_date": start_date,
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset",
        "timezone": "Asia/Shanghai"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "daily" not in data:
            print("❌ 未获取到天气数据:", data)
            return []
        
        # Convert Open-Meteo data to our format
        weather_list = []
        for i in range(len(data["daily"]["time"])):
            date = data["daily"]["time"][i]
            weather_code = data["daily"]["weather_code"][i]
            
            weather_info = {
                "日期": date,
                "日出": data["daily"]["sunrise"][i].split("T")[1][:5],  # Extract time part
                "日落": data["daily"]["sunset"][i].split("T")[1][:5],   # Extract time part
                "白天天气": _convert_weather_code(weather_code, "day"),
                "夜间天气": _convert_weather_code(weather_code, "night"),
                "最高温": f"{data['daily']['temperature_2m_max'][i]:.0f}℃",
                "最低温": f"{data['daily']['temperature_2m_min'][i]:.0f}℃",
            }
            weather_list.append(weather_info)
        
        return weather_list
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 未获取到天气数据: 网络请求失败 - {e}")
        if raise_on_error:
            raise
        return []
    except Exception as e:
        print(f"❌ 未获取到天气数据: {e}")
        if raise_on_error:
            raise
        return []


def _get_coordinates_from_location(location: str):
    """
    Convert location (city ID or coordinates) to (longitude, latitude)
    """
    # If it's already coordinates (longitude,latitude format)
    if "," in location and not location.isdigit():
        try:
            lon, lat = location.split(",")
            return (float(lon.strip()), float(lat.strip()))
        except:
            pass
    
    # City ID to coordinates mapping (main Chinese cities)
    city_coordinates = {
        "101010100": (116.4074, 39.9042),  # Beijing
        "101020100": (121.4737, 31.2304),  # Shanghai
        "101280101": (113.2644, 23.1291),  # Guangzhou
        "101280601": (114.0579, 22.5431),  # Shenzhen
        "101210101": (120.1551, 30.2741),  # Hangzhou
        "101190101": (118.7674, 32.0415),  # Nanjing
        "101030100": (117.2008, 39.0842),  # Tianjin
        "101040100": (106.5516, 29.5630),  # Chongqing
        "101050100": (126.6425, 45.7564),  # Harbin
        "101060100": (125.3245, 43.8868),  # Changchun
        "101070100": (123.4315, 41.8057),  # Shenyang
        "101080100": (111.7519, 40.8414),  # Hohhot
        "101090100": (114.5149, 38.0428),  # Shijiazhuang
        "101100100": (112.5489, 37.8706),  # Taiyuan
        "101110100": (108.9402, 34.3416),  # Xi'an
        "101120100": (117.0009, 36.6758),  # Jinan
        "101130100": (87.6168, 43.8256),   # Urumqi
        "101140100": (101.7782, 36.6232),  # Xining
        "101150100": (103.8236, 36.0581),  # Lanzhou
        "101160100": (103.8236, 36.0581),  # Yinchuan
        "101170100": (106.2782, 38.4872),  # Yinchuan
        "101180100": (113.6254, 34.7466),  # Zhengzhou
        "101190100": (118.7674, 32.0415),  # Nanjing
        "101200100": (114.2986, 30.5844),  # Wuhan
        "101210100": (120.1551, 30.2741),  # Hangzhou
        "101220100": (117.3272, 31.8612),  # Hefei
        "101230100": (119.3062, 26.0745),  # Fuzhou
        "101240100": (115.8922, 28.6765),  # Nanchang
        "101250100": (112.9823, 28.1949),  # Changsha
        "101260100": (106.7135, 26.5783),  # Guiyang
        "101270100": (104.0665, 30.6595),  # Chengdu
        "101280100": (113.2644, 23.1291),  # Guangzhou
        "101290100": (102.7123, 25.0406),  # Kunming
        "101300100": (108.3661, 22.8172),  # Nanning
        "101310100": (110.3312, 20.0311),  # Haikou
        "500000": (106.5516, 29.5630),     # Chongqing (alternative ID)
    }
    
    if location in city_coordinates:
        return city_coordinates[location]

    # Resolve unknown city IDs instead of silently substituting a default city.
    from . import _amap
    place = _amap.resolve_city(location)
    if place:
        return (place["lng"], place["lat"])

    return None


def _convert_weather_code(code: int, time_of_day: str = "day"):
    """Convert an Open-Meteo code to a Chinese weather description."""
    weather_map = {
        0: {"day": "晴", "night": "晴"},
        1: {"day": "晴", "night": "晴"},
        2: {"day": "多云", "night": "多云"},
        3: {"day": "阴", "night": "阴"},
        45: {"day": "雾", "night": "雾"},
        48: {"day": "雾", "night": "雾"},
        51: {"day": "小雨", "night": "小雨"},
        53: {"day": "小雨", "night": "小雨"},
        55: {"day": "中雨", "night": "中雨"},
        61: {"day": "小雨", "night": "小雨"},
        63: {"day": "中雨", "night": "中雨"},
        65: {"day": "大雨", "night": "大雨"},
        71: {"day": "小雪", "night": "小雪"},
        73: {"day": "中雪", "night": "中雪"},
        75: {"day": "大雪", "night": "大雪"},
        77: {"day": "雪", "night": "雪"},
        80: {"day": "小雨", "night": "小雨"},
        81: {"day": "中雨", "night": "中雨"},
        82: {"day": "大雨", "night": "大雨"},
        85: {"day": "小雪", "night": "小雪"},
        86: {"day": "大雪", "night": "大雪"},
        95: {"day": "雷阵雨", "night": "雷阵雨"},
        96: {"day": "雷阵雨", "night": "雷阵雨"},
        99: {"day": "雷阵雨", "night": "雷阵雨"},
    }
    
    return weather_map.get(code, {"day": "未知", "night": "未知"}).get(time_of_day, "未知")


def get_weather_by_date(location: str, start_date: str, num_days: int = 1) -> dict:
    """Daily forecast for `location` starting at `start_date` (YYYY-MM-DD)."""
    from . import _amap
    from cn_travel.business_logic.contracts import EMPTY, OK, err

    source = "amap-district + open-meteo"
    num_days = max(1, min(int(num_days or 1), 16))

    place = _amap.resolve_city(location)
    if not place:
        return err("get_weather_info", f"could not resolve location {location!r}",
                   location=location)

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return err("get_weather_info", f"bad start_date {start_date!r}", location=location)

    try:
        rows = get_weather_by_date_range(f"{place['lng']},{place['lat']}", start_date,
                                         num_days, raise_on_error=True)
    except Exception as exc:
        # Preserve request failures as errors rather than empty query results.
        return err("get_weather_info", f"weather request failed: {exc}", location=location)
    if not rows:
        return {"status": EMPTY, "source": source, "location": location,
                "resolved": place, "days": []}

    days = [{
        "date": r["日期"],
        "day_weather": r["白天天气"],
        "night_weather": r["夜间天气"],
        "temp_min_c": int(str(r["最低温"]).rstrip("℃")),
        "temp_max_c": int(str(r["最高温"]).rstrip("℃")),
    } for r in rows]

    return {"status": OK, "source": source, "location": location,
            "resolved": place, "days": days}


# Example invocation
if __name__ == "__main__":
    result = get_weather_by_date("101010100", "2025-09-19")
    if result:
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print("未找到对应日期的天气数据")
