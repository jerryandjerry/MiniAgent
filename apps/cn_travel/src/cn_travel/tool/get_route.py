import requests

from ._amap import KEY as AMAP_KEY  # Single source, read only from the environment

def geocode(address, city=None):
    """Resolve a place to coordinates within an optional city or adcode."""
    url = f"https://restapi.amap.com/v3/geocode/geo?address={address}&key={AMAP_KEY}"
    if city:
        url += f"&city={city}"
    from . import _amap
    resp = requests.get(url).json()
    if resp["status"] == "1" and resp["geocodes"]:
        hit = resp["geocodes"][0]
        # Geocoding also fuzzy-matches nonsense to unrelated coordinates.
        # Accept a result only when its address actually matches the query.
        if _amap.relevant(address, hit.get("formatted_address") or ""):
            return hit["location"]  # "lng,lat"

    # Geocoding resolves addresses; generic facilities require POI search

    poi = _amap.search_place(address, adcode=city or "")
    pos = _amap.poi_lat_lng(poi) if poi else None
    if pos:
        return f"{pos['lng']},{pos['lat']}"

    raise ValueError(f"无法找到地址: {address}")

def get_walking(origin, destination):
    url = f"https://restapi.amap.com/v3/direction/walking?origin={origin}&destination={destination}&key={AMAP_KEY}"
    resp = requests.get(url).json()
    if resp["status"] == "1" and resp["route"]["paths"]:
        path = resp["route"]["paths"][0]
        steps = [s["instruction"] for s in path["steps"]]
        return {
            "总时间(分钟)": int(path["duration"]) // 60,
            "总距离(米)": int(path["distance"]),
            # "Detailed route": steps
        }
    return None

def get_transit(origin, destination, city_code):
    url = f"https://restapi.amap.com/v3/direction/transit/integrated?origin={origin}&destination={destination}&city={city_code}&key={AMAP_KEY}"
    resp = requests.get(url).json()
    if resp["status"] == "1" and resp["route"]["transits"]:
        transit = resp["route"]["transits"][0]
        segments = []
        for seg in transit["segments"]:
            # Transit segment
            if "bus" in seg and seg["bus"].get("buslines"):
                buslines = seg["bus"]["buslines"]
                if buslines:  # Ensure it is nonempty
                    busline = buslines[0]
                    segments.append(
                        f"乘坐 {busline['name']} "
                        f"({busline['departure_stop']['name']} 上车 → {busline['arrival_stop']['name']} 下车)"
                    )
            # Walking segment, accepting dict or list shapes
            if "walking" in seg and isinstance(seg["walking"], dict):
                if "distance" in seg["walking"]:
                    segments.append(f"步行 {seg['walking']['distance']} 米")

        return {
            "总时间(分钟)": int(transit["duration"]) // 60,
            "票价(元)": transit.get("cost", "未知"),
            "详细路线": segments
        }
    return None



def get_driving(origin, destination):
    url = f"https://restapi.amap.com/v3/direction/driving?origin={origin}&destination={destination}&extensions=all&key={AMAP_KEY}"
    resp = requests.get(url).json()
    if resp["status"] == "1" and resp["route"]["paths"]:
        path = resp["route"]["paths"][0]
        steps = [s["instruction"] for s in path["steps"]]
        return {
            "总时间(分钟)": int(path["duration"]) // 60,
            "总距离(米)": int(path["distance"]),
            "过路费(元)": path.get("tolls", "0"),
            # "Detailed route": steps
        }
    return None

def query_routes(start_location, end_location, city=""):
    """Route from `start_location` (coords or place) to `end_location`.

    `city` scopes both lookups; a bare landmark is ambiguous nationwide.
    """
    from . import _amap
    from cn_travel.business_logic.contracts import EMPTY, OK, err

    source = "amap-poi + amap-directions"
    scope = _amap.resolve_city(city) if city else None
    adcode = scope["adcode"] if scope else ""

    def _coords(value):
        parts = (value or "").split(",")
        if len(parts) == 2:
            try:
                return f"{float(parts[0])},{float(parts[1])}"
            except ValueError:
                return None
        return None

    origin_coords = _coords(start_location)
    if origin_coords:
        lng, lat = origin_coords.split(",")
        origin = {"query": start_location, "lat": float(lat), "lng": float(lng)}
    else:
        # Resolve administrative areas before city-scoped POI fallback.
        c = _amap.resolve_city(start_location)
        if c:
            origin = {"query": start_location, "name": c["name"],
                      "lat": c["lat"], "lng": c["lng"]}
        else:
            hit = _amap.search_place(start_location, adcode=adcode)
            pos = _amap.poi_lat_lng(hit) if hit else None
            if not pos:
                return err("query_route", f"could not resolve start {start_location!r}")
            origin = {"query": start_location, **pos}
        origin_coords = f"{origin['lng']},{origin['lat']}"

    # Apply the same rule to the destination: administrative names use centroids.
    c = _amap.resolve_city(end_location)
    if c:
        destination = {"query": end_location, "name": c["name"],
                       "lat": c["lat"], "lng": c["lng"]}
    else:
        # Search nearest first to disambiguate generic facilities, then fall back by name.
        hit = _amap.search_nearest(end_location, origin_coords, adcode=adcode)
        if not hit:
            hit = _amap.search_place(end_location, adcode=adcode)
        pos = _amap.poi_lat_lng(hit) if hit else None
        if not pos:
            return err("query_route", f"could not resolve destination {end_location!r}")
        destination = {"query": end_location,
                       "name": hit.get("name") or end_location, **pos}
    dest_coords = f"{destination['lng']},{destination['lat']}"

    routes = {"walking": None, "transit": None, "driving": None}
    for mode, fn in (("walking", lambda: get_walking(origin_coords, dest_coords)),
                     ("driving", lambda: get_driving(origin_coords, dest_coords)),
                     ("transit", lambda: get_transit(origin_coords, dest_coords, adcode))):
        try:
            raw = fn()
        except Exception:
            raw = None
        if not raw:
            continue
        if mode == "walking":
            routes[mode] = {"duration_min": raw["总时间(分钟)"], "distance_m": raw["总距离(米)"]}
        elif mode == "driving":
            routes[mode] = {"duration_min": raw["总时间(分钟)"], "distance_m": raw["总距离(米)"],
                            "tolls_cny": float(raw.get("过路费(元)") or 0)}
        else:
            fare = raw.get("票价(元)")
            routes[mode] = {"duration_min": raw["总时间(分钟)"],
                            "fare_cny": float(fare) if str(fare).replace(".", "").isdigit() else 0.0,
                            "segments": raw.get("详细路线") or []}

    status = OK if any(routes.values()) else EMPTY
    return {"status": status, "source": source, "origin": origin,
            "destination": destination, "routes": routes}


# Example usage
if __name__ == "__main__":
    start = "116.481028,39.989643"   # Start coordinates at Wangjing SOHO
    end = "天坛"
    routes = query_routes(start, end, city="110000")

    for mode, info in routes.items():
        print(f"\n【{mode}】")
        if info:
            for k, v in info.items():
                print(f"{k}: {v}")
        else:
            print("未查询到结果")
