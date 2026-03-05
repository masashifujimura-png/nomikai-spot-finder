import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os
import string
import random
import pickle
import heapq
import json
import urllib.request
import urllib.parse

# ---------------------------------------------------------------------------
# Supabase (REST API direct — no SDK for faster import)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
HOTPEPPER_API_KEY = os.environ.get("HOTPEPPER_API_KEY", "")


def _sb_request(method, path, body=None, params=None):
    """Supabase REST API を直接呼び出す。"""
    base = SUPABASE_URL.rstrip("/") + "/rest/v1/" + path
    if params:
        base += "?" + urllib.parse.urlencode(params, doseq=True)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(base, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _generate_code(length=6):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
st.set_page_config(page_title="飲み会スポットファインダー", layout="wide")

TRAIN_SPEED_KMH = 30
AVG_TRAIN_SPEED_KMH = 35

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 駅データ読み込み (pickle キャッシュ)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="駅データを読み込んでいます...")
def _load_ekidata():
    pickle_file = os.path.join(_DATA_DIR, "ekidata_cache.pkl")

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        # 7-tuple: (station_db, graph, sorted_names, coords, edge_lines, name_to_gcd, gcd_to_name)
        if isinstance(data, tuple) and len(data) >= 7:
            return data[:7]
        if isinstance(data, tuple) and len(data) >= 5:
            return data[:5] + ({}, {})
        if isinstance(data, tuple) and len(data) >= 4:
            return data[:4] + ({}, {}, {})
    raise RuntimeError("ekidata_cache.pkl not found or outdated. Run precompute.py first.")


def _get_ekidata():
    return _load_ekidata()


def _station_db():
    return _get_ekidata()[0]


def _graph():
    return _get_ekidata()[1]


def _station_names_arr():
    return _get_ekidata()[2]


def _station_coords():
    return _get_ekidata()[3]


def _edge_lines():
    return _get_ekidata()[4]


def _name_to_gcd():
    return _get_ekidata()[5]


def _gcd_to_name():
    return _get_ekidata()[6]


@st.cache_data(show_spinner=False)
def _sorted_station_names():
    return list(_get_ekidata()[2])


# ---------------------------------------------------------------------------
# オンデマンド Dijkstra（検索時のみ実行）
# ---------------------------------------------------------------------------
def _dijkstra(graph, start, targets=None):
    """Single-source Dijkstra. If targets is given, stop early when all found."""
    dist = {start: 0.0}
    heap = [(0.0, start)]
    remaining = set(targets) if targets else None
    if remaining:
        remaining.discard(start)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if remaining is not None and u in remaining:
            remaining.discard(u)
            if not remaining:
                break
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _dijkstra_with_path(graph, start, target):
    """Single-source Dijkstra returning shortest path to target."""
    dist = {start: 0.0}
    prev = {start: None}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u == target:
            break
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if target not in dist:
        return None, None
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path, round(dist[target], 1)


@st.cache_data(show_spinner=False)
def _find_route(source_gcd, target_gcd):
    """経路探索（キャッシュ付き）。g_cd で指定。(path_gcd_list, time_min) を返す。"""
    if not source_gcd or not target_gcd or source_gcd == target_gcd:
        return None, 0.0
    graph = _graph()
    if source_gcd not in graph:
        return None, None
    return _dijkstra_with_path(graph, source_gcd, target_gcd)


def _format_route(path_gcds):
    """経路(g_cdリスト)を路線名・乗換駅付きでフォーマット。"""
    if not path_gcds or len(path_gcds) < 2:
        gtn = _gcd_to_name()
        return gtn.get(path_gcds[0], "?") if path_gcds else ""
    el = _edge_lines()
    gtn = _gcd_to_name()
    # 連続する同一路線をまとめてセグメント化
    segments = []  # [(start_gcd, end_gcd, line_name)]
    current_line = el.get((path_gcds[0], path_gcds[1]), "")
    seg_start = path_gcds[0]
    for j in range(1, len(path_gcds) - 1):
        next_line = el.get((path_gcds[j], path_gcds[j + 1]), "")
        if next_line != current_line:
            segments.append((seg_start, path_gcds[j], current_line))
            seg_start = path_gcds[j]
            current_line = next_line
    segments.append((seg_start, path_gcds[-1], current_line))
    # フォーマット: 新宿 →[中央線]→ 四ツ谷 →[丸ノ内線]→ 池袋
    parts = [gtn.get(segments[0][0], "?")]
    for _start, end, line in segments:
        name = gtn.get(end, "?")
        if line:
            parts.append(f" →[{line}]→ {name}")
        else:
            parts.append(f" → {name}")
    return "".join(parts)


@st.cache_data(show_spinner=False)
def _dijkstra_cached(source, _targets_key):
    """Single-source Dijkstra with caching. _targets_key is a sorted tuple for hashing."""
    graph = _graph()
    if source not in graph:
        return {}
    dist = _dijkstra(graph, source, set(_targets_key))
    return {t: round(dist[t], 1) for t in _targets_key if t in dist}


def _batch_dijkstra(sources, targets):
    """Run Dijkstra from each source, returning {source: {target: time}}."""
    targets_key = tuple(sorted(set(targets)))
    graph = _graph()
    result = {}
    for src in sources:
        if src not in graph:
            result[src] = {}
            continue
        result[src] = _dijkstra_cached(src, targets_key)
    return result


# ---------------------------------------------------------------------------
# 距離計算（Haversine）
# ---------------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# ジオコーディング（駅データベースのみ — 外部API不使用）
# ---------------------------------------------------------------------------
def _geocode_station(station_name: str) -> tuple[float | None, float | None, str, int | None]:
    name = station_name.rstrip("駅").strip()
    if not name:
        return None, None, "", None
    sdb = _station_db()
    if name in sdb:
        lat, lon = sdb[name]
        gcd = _name_to_gcd().get(name)
        return lat, lon, f"{name}駅", gcd
    return None, None, "", None


# ---------------------------------------------------------------------------
# 候補駅検索 & スコアリング
# ---------------------------------------------------------------------------
def compute_bounding_circle(participants: list[dict]) -> tuple[float, float, float]:
    all_lats, all_lons = [], []
    for p in participants:
        if p.get("work_lat") is not None:
            all_lats.append(p["work_lat"])
            all_lons.append(p["work_lon"])
        if p.get("home_lat") is not None:
            all_lats.append(p["home_lat"])
            all_lons.append(p["home_lon"])

    center_lat = (min(all_lats) + max(all_lats)) / 2
    center_lon = (min(all_lons) + max(all_lons)) / 2
    max_radius = max(
        haversine(center_lat, center_lon, lat, lon)
        for lat, lon in zip(all_lats, all_lons)
    )
    return center_lat, center_lon, max_radius


def find_candidate_stations(participants: list[dict], margin: float = 1.2) -> list[dict]:
    center_lat, center_lon, radius = compute_bounding_circle(participants)
    search_radius = max(radius * margin, 3.0)

    sdb = _station_db()
    coords = _station_coords()
    names = _station_names_arr()

    # Vectorized haversine distance
    dlat = np.radians(coords[:, 0] - center_lat)
    dlon = np.radians(coords[:, 1] - center_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(center_lat)) * np.cos(np.radians(coords[:, 0]))
         * np.sin(dlon / 2) ** 2)
    distances = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    mask = distances <= search_radius
    stations = []
    ntg = _name_to_gcd()
    for i in np.where(mask)[0]:
        name = names[int(i)]
        lat, lon = sdb[name]
        gcd = ntg.get(name)
        if gcd is not None:
            stations.append({"name": name, "lat": lat, "lon": lon, "gcd": gcd})

    return stations


def _prefilter_stations(stations, participants, work_weight, home_weight, top_n=30):
    """直線距離で粗くスコアリングし、上位 top_n 駅に絞る（Dijkstra 前の高速フィルタ）。"""
    if len(stations) <= top_n:
        return stations
    s_coords = np.array([(s["lat"], s["lon"]) for s in stations])
    costs = np.zeros(len(stations))
    for p in participants:
        for loc_key, weight in [("work", work_weight), ("home", home_weight)]:
            plat = p.get(f"{loc_key}_lat")
            plon = p.get(f"{loc_key}_lon")
            if plat is None or weight <= 0:
                continue
            dlat = np.radians(s_coords[:, 0] - plat)
            dlon = np.radians(s_coords[:, 1] - plon)
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(np.radians(plat)) * np.cos(np.radians(s_coords[:, 0]))
                 * np.sin(dlon / 2) ** 2)
            d = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            costs += (d / TRAIN_SPEED_KMH * 60) * weight
    top_indices = np.argsort(costs)[:top_n]
    return [stations[i] for i in top_indices]


def score_stations(stations, participants, work_weight, home_weight,
                   fairness_weight=0.0) -> list[dict]:
    # 参加者の拠点のみからDijkstra実行（無向グラフなので候補駅からの実行は不要）
    person_sources = set()
    for p in participants:
        wg = p.get("work_gcd")
        if wg:
            person_sources.add(wg)
        hg = p.get("home_gcd")
        if hg:
            person_sources.add(hg)

    candidate_gcds = [s["gcd"] for s in stations if s.get("gcd")]
    # ターゲットは候補駅のみ（参加者拠点→候補駅の距離を求める）
    dist_table = _batch_dijkstra(person_sources, candidate_gcds)

    scored = []
    for st_info in stations:
        total_cost = 0
        max_val = 0
        details = []
        sg = st_info.get("gcd")
        for p in participants:
            # Work -> candidate station
            if p.get("work_lat") is not None:
                wg = p.get("work_gcd")
                if wg and sg and wg == sg:
                    work_val = 0.0
                elif wg and sg:
                    work_val = dist_table.get(wg, {}).get(sg)
                    if work_val is None:
                        dist = haversine(p["work_lat"], p["work_lon"],
                                         st_info["lat"], st_info["lon"])
                        work_val = round(dist / TRAIN_SPEED_KMH * 60, 1)
                else:
                    dist = haversine(p["work_lat"], p["work_lon"],
                                     st_info["lat"], st_info["lon"])
                    work_val = round(dist / TRAIN_SPEED_KMH * 60, 1)
            else:
                work_val = 0

            # Candidate station -> home (無向グラフ: dist(home→candidate) = dist(candidate→home))
            if p.get("home_lat") is not None:
                hg = p.get("home_gcd")
                if hg and sg and hg == sg:
                    home_val = 0.0
                elif hg and sg:
                    home_val = dist_table.get(hg, {}).get(sg)
                    if home_val is None:
                        dist = haversine(st_info["lat"], st_info["lon"],
                                         p["home_lat"], p["home_lon"])
                        home_val = round(dist / TRAIN_SPEED_KMH * 60, 1)
                else:
                    dist = haversine(st_info["lat"], st_info["lon"],
                                     p["home_lat"], p["home_lon"])
                    home_val = round(dist / TRAIN_SPEED_KMH * 60, 1)
            else:
                home_val = 0

            # 自宅→飲み会→自宅の場合、往復同一経路なので一方に揃える
            wg = p.get("work_gcd")
            hg = p.get("home_gcd")
            if wg and hg and wg == hg:
                home_val = work_val

            cost = work_val * work_weight + home_val * home_weight
            total_cost += cost
            total_val = work_val + home_val
            max_val = max(max_val, total_val)
            details.append({
                "name": p["name"],
                "work_val": work_val,
                "home_val": home_val,
                "total_val": round(total_val, 1),
            })

        all_vals = [d["total_val"] for d in details]
        std_dev = float(np.std(all_vals)) if len(all_vals) > 1 else 0.0
        avg_val = sum(all_vals) / len(all_vals)
        final_cost = total_cost + fairness_weight * std_dev * len(details)

        scored.append({
            **st_info,
            "total_cost": round(final_cost, 2),
            "max_person_val": round(max_val, 1),
            "avg_total_val": round(avg_val, 1),
            "std_dev": round(std_dev, 1),
            "details": details,
        })

    scored.sort(key=lambda x: x["total_cost"])
    return scored


# ---------------------------------------------------------------------------
# DB操作
# ---------------------------------------------------------------------------
TRIP_PATTERNS = ["職場→飲み会→自宅", "自宅→飲み会→自宅"]


def create_event(title: str) -> dict:
    code = _generate_code()
    rows = _sb_request("POST", "events", body={"event_code": code, "title": title})
    return rows[0]


def get_event(code: str) -> dict | None:
    rows = _sb_request("GET", "events", params={
        "select": "*", "event_code": f"eq.{code}", "limit": "1",
    })
    return rows[0] if rows else None


def get_participants(event_id: str) -> list[dict]:
    return _sb_request("GET", "participants", params={
        "select": "*", "event_id": f"eq.{event_id}", "order": "created_at",
    })


def add_participant(event_id: str, name: str, pattern: str, work: str, home: str) -> None:
    _sb_request("POST", "participants", body={
        "event_id": event_id,
        "name": name,
        "pattern": pattern,
        "work_location": work,
        "home_location": home,
    })


def update_participant(participant_id: str, name: str, pattern: str, work: str, home: str) -> None:
    _sb_request("PATCH", f"participants?id=eq.{participant_id}", body={
        "name": name,
        "pattern": pattern,
        "work_location": work,
        "home_location": home,
    })


def delete_participant(participant_id: str) -> None:
    _sb_request("DELETE", f"participants?id=eq.{participant_id}")


# ---------------------------------------------------------------------------
# ホットペッパーグルメ検索
# ---------------------------------------------------------------------------
_HP_API_URL = "https://webservice.recruit.co.jp/hotpepper/gourmet/v1/"


@st.cache_data(ttl=600, show_spinner=False)
def _search_hotpepper(lat, lon, keyword="", genre="", count=10,
                      free_drink=0, private_room=0, party_capacity=0, budget=""):
    """ホットペッパーAPIで周辺の飲食店を検索。結果を10分キャッシュ。"""
    if not HOTPEPPER_API_KEY:
        return []
    params = {
        "key": HOTPEPPER_API_KEY,
        "lat": lat,
        "lng": lon,
        "range": 3,  # 1km圏内
        "order": 4,  # おすすめ順
        "count": count,
        "format": "json",
    }
    if genre:
        params["genre"] = genre
    if keyword:
        params["keyword"] = keyword
    if free_drink:
        params["free_drink"] = 1
    if private_room:
        params["private_room"] = 1
    if party_capacity:
        params["party_capacity"] = party_capacity
    if budget:
        params["budget"] = budget
    url = _HP_API_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        shops = data.get("results", {}).get("shop", [])
        return [{
            "name": s["name"],
            "address": s.get("address", ""),
            "access": s.get("mobile_access") or s.get("access", ""),
            "budget": s.get("budget", {}).get("average", ""),
            "capacity": s.get("party_capacity", ""),
            "private_room": s.get("private_room", ""),
            "free_drink": s.get("free_drink", ""),
            "course": s.get("course", ""),
            "genre": s.get("genre", {}).get("name", ""),
            "catch": s.get("catch", ""),
            "photo": s.get("photo", {}).get("pc", {}).get("l", ""),
            "url": s.get("urls", {}).get("pc", ""),
            "open": s.get("open", ""),
        } for s in shops]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# ジオコード参加者
# ---------------------------------------------------------------------------
def geocode_participant(p: dict) -> dict:
    entry = {
        "name": p["name"],
        "pattern": p.get("pattern", TRIP_PATTERNS[0]),
        "work_station": None, "work_lat": None, "work_lon": None, "work_label": None, "work_gcd": None,
        "home_station": None, "home_lat": None, "home_lon": None, "home_label": None, "home_gcd": None,
    }
    is_home_round = entry["pattern"] == TRIP_PATTERNS[1]

    work_loc = p.get("work_location", "").strip()
    if not is_home_round and work_loc:
        lat, lon, label, gcd = _geocode_station(work_loc)
        if lat is not None:
            entry["work_station"] = label.rstrip("駅")
            entry["work_lat"] = lat
            entry["work_lon"] = lon
            entry["work_label"] = label
            entry["work_gcd"] = gcd

    home_loc = p.get("home_location", "").strip()
    if home_loc:
        lat, lon, label, gcd = _geocode_station(home_loc)
        if lat is not None:
            entry["home_station"] = label.rstrip("駅")
            entry["home_lat"] = lat
            entry["home_lon"] = lon
            entry["home_label"] = label
            entry["home_gcd"] = gcd

    if is_home_round and entry["home_lat"] is not None:
        entry["work_lat"] = entry["home_lat"]
        entry["work_lon"] = entry["home_lon"]
        entry["work_station"] = entry["home_station"]
        entry["work_label"] = entry["home_label"]
        entry["work_gcd"] = entry["home_gcd"]

    return entry


# ---------------------------------------------------------------------------
# 駅名検索ウィジェット
# ---------------------------------------------------------------------------
def _station_picker(label, key, default=""):
    names = _sorted_station_names()
    if default and default in names:
        idx = names.index(default)
    else:
        idx = None
    return st.selectbox(label, names, index=idx, placeholder="駅名を入力して選択", key=f"{key}_s")


# ---------------------------------------------------------------------------
# イベントキャッシュ無効化
# ---------------------------------------------------------------------------
def _invalidate_event_cache(event_code):
    cache_key = f"_event_cache_{event_code}"
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    # 参加者変更時は検索結果もクリア
    for k in ["_search_results", "_hp_results", "_hp_station"]:
        if k in st.session_state:
            del st.session_state[k]


# ---------------------------------------------------------------------------
# カスタムCSS
# ---------------------------------------------------------------------------
def _inject_custom_css():
    st.markdown("""
    <style>
    button[kind="primary"] {
        font-weight: bold;
    }

    /* 再実行時の白い靄を無効化 */
    [data-testid="stAppViewBlockContainer"] [data-stale="true"] {
        opacity: 1 !important;
    }

    .step-badge {
        display: inline-block;
        background: #1a73e8;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        font-weight: bold;
        font-size: 14px;
        margin-right: 8px;
    }
    .step-badge-muted {
        display: inline-block;
        background: #e0e0e0;
        color: #616161;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        font-weight: bold;
        font-size: 14px;
        margin-right: 8px;
    }
    .step-title {
        font-size: 18px;
        font-weight: 600;
        color: #212121;
    }
    .step-title-muted {
        font-size: 18px;
        font-weight: 600;
        color: #9e9e9e;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 地図描画（Leaflet.js — 軽量、スマホ対応）
# ---------------------------------------------------------------------------
_STATION_COLORS = ["#e53935", "#ff9800", "#9c27b0"]
# 参加者ごとの色パレット（人数が増えても対応）
_PERSON_COLORS = [
    "#1e88e5", "#43a047", "#f4511e", "#8e24aa",
    "#00acc1", "#c0ca33", "#6d4c41", "#d81b60",
    "#3949ab", "#00897b", "#ffb300", "#546e7a",
]


def _render_map(top_stations, geocoded):
    """Leaflet.js で上位3駅・参加者の職場/自宅を凡例付きで表示。"""
    marker_data = []
    all_points = []

    # 1位の座標（地図中心用）
    center = [top_stations[0]["lat"], top_stations[0]["lon"]] if top_stations else [35.68, 139.76]

    for i, s in enumerate(top_stations[:3]):
        marker_data.append({
            "lat": s["lat"], "lon": s["lon"], "color": _STATION_COLORS[i],
            "label": f"{i+1}位: {s['name']}", "radius": 8,
        })
        all_points.append([s["lat"], s["lon"]])

    legend_items = []
    for i, s in enumerate(top_stations[:3]):
        legend_items.append({"color": _STATION_COLORS[i], "label": f"{i+1}位: {s['name']}"})

    for pi, g in enumerate(geocoded):
        pcolor = _PERSON_COLORS[pi % len(_PERSON_COLORS)]
        if g.get("work_lat") is not None:
            marker_data.append({
                "lat": g["work_lat"], "lon": g["work_lon"], "color": pcolor,
                "label": f"{g['name']} 職場", "radius": 6,
            })
            all_points.append([g["work_lat"], g["work_lon"]])
            legend_items.append({"color": pcolor, "label": f"{g['name']} 職場"})
        if g.get("home_lat") is not None:
            marker_data.append({
                "lat": g["home_lat"], "lon": g["home_lon"], "color": pcolor,
                "label": f"{g['name']} 自宅", "radius": 6,
            })
            all_points.append([g["home_lat"], g["home_lon"]])
            legend_items.append({"color": pcolor, "label": f"{g['name']} 自宅"})

    html = _MAP_TEMPLATE.replace(
        "/*MARKERS*/", json.dumps(marker_data, ensure_ascii=False),
    ).replace(
        "/*LEGEND*/", json.dumps(legend_items, ensure_ascii=False),
    ).replace(
        "/*BOUNDS*/", json.dumps(all_points),
    ).replace(
        "/*CENTER*/", json.dumps(center),
    )
    st.components.v1.html(html, height=460)


_MAP_TEMPLATE = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="map" style="height:440px;width:100%;border-radius:8px;"></div>
<style>
.map-legend{background:rgba(255,255,255,0.95);padding:10px 14px;border-radius:6px;
  box-shadow:0 1px 5px rgba(0,0,0,.3);font-size:12px;line-height:1.8;
  max-height:220px;overflow-y:auto;}
.map-legend .title{font-weight:bold;margin-bottom:4px;}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;
  margin-right:6px;vertical-align:middle;}
</style>
<script>
(function(){
  var center=/*CENTER*/;
  var map=L.map('map',{center:center,zoom:12});
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{
    attribution:'&copy; OpenStreetMap &copy; CARTO',maxZoom:18}).addTo(map);
  var markers=/*MARKERS*/;
  var placed=[];
  markers.forEach(function(m){
    var lat=m.lat,lon=m.lon;
    var dup=0;
    for(var j=0;j<placed.length;j++){
      if(Math.abs(lat-placed[j][0])<0.0005&&Math.abs(lon-placed[j][1])<0.0005)dup++;
    }
    if(dup>0){
      var angle=dup*(2*Math.PI/6);
      lat+=0.0012*Math.cos(angle);
      lon+=0.0012*Math.sin(angle);
    }
    placed.push([m.lat,m.lon]);
    L.circleMarker([lat,lon],{radius:m.radius,fillColor:m.color,
      color:'#fff',weight:2,fillOpacity:0.9}).addTo(map).bindTooltip(m.label);
  });
  var legend=L.control({position:'bottomright'});
  legend.onAdd=function(){
    var div=L.DomUtil.create('div','map-legend');
    var h='<div class="title">凡例</div>';
    var items=/*LEGEND*/;
    items.forEach(function(it){
      h+='<div><span class="dot" style="background:'+it.color+'"></span>'+it.label+'</div>';
    });
    div.innerHTML=h;return div;
  };
  legend.addTo(map);
  var bounds=/*BOUNDS*/;
  function fitMap(){
    map.invalidateSize();
    var el=map.getContainer();
    if(el.clientHeight>0&&el.clientWidth>0){
      if(bounds.length>1)map.fitBounds(bounds,{padding:[40,40]});
      else if(bounds.length==1)map.setView(bounds[0],13);
    }else{setTimeout(fitMap,100);}
  }
  setTimeout(fitMap,50);
})();
</script>
"""


# ---------------------------------------------------------------------------
# UI フラグメント（部分更新で高速化）
# ---------------------------------------------------------------------------
@st.fragment
def _add_participant_fragment(event_id, event_code, participant_count):
    st.subheader("参加者を追加")
    st.caption("自分の情報を入力、または幹事がまとめて全員分を追加できます")

    if "_form_ver" not in st.session_state:
        st.session_state._form_ver = 0
    fv = st.session_state._form_ver

    new_pattern = st.selectbox("移動パターン", TRIP_PATTERNS, key=f"add_pattern_{fv}")
    is_home_round_form = new_pattern == TRIP_PATTERNS[1]

    new_name = st.text_input("名前", placeholder="参加者の名前", key=f"add_name_{fv}")
    new_home = _station_picker("自宅最寄駅", f"add_home_{fv}")
    if is_home_round_form:
        new_work = None
    else:
        new_work = _station_picker("職場最寄駅", f"add_work_{fv}")

    if st.button("参加者を追加", type="primary", use_container_width=True):
        if not new_name.strip():
            st.error("名前を入力してください。")
        elif not new_home:
            st.error("自宅の最寄駅を選択してください。")
        elif not is_home_round_form and not new_work:
            st.error("職場の最寄駅を選択してください。")
        else:
            work_val = "" if is_home_round_form else (new_work or "")
            add_participant(event_id, new_name.strip(), new_pattern, work_val, new_home)
            st.session_state._form_ver += 1
            _invalidate_event_cache(event_code)
            remaining = 2 - (participant_count + 1)
            if remaining > 0:
                next_hint = f"あと{remaining}人追加すると検索できます。"
            else:
                next_hint = "ページ下部の「最適スポットを検索」ボタンで検索できます。"
            st.session_state["_success_msg"] = (
                f"{new_name.strip()} さんを追加しました。\n\n"
                f"続けて他の参加者を追加するか、{next_hint}"
            )
            st.rerun()


@st.fragment
def _participant_card_fragment(p, event_code):
    editing = st.session_state.get("_editing_id") == p["id"]
    is_hr = p["pattern"] == TRIP_PATTERNS[1]

    if editing:
        if "_edit_ver" not in st.session_state:
            st.session_state._edit_ver = 0
        ev = st.session_state._edit_ver
        with st.container(border=True):
            st.markdown(f"**{p['name']} を編集中**")
            edit_pattern = st.selectbox(
                "移動パターン", TRIP_PATTERNS,
                index=TRIP_PATTERNS.index(p["pattern"]),
                key=f"edit_pattern_{p['id']}_{ev}",
            )
            edit_is_hr = edit_pattern == TRIP_PATTERNS[1]

            edit_name = st.text_input(
                "名前", value=p["name"],
                key=f"edit_name_{p['id']}_{ev}",
            )
            edit_home = _station_picker(
                "自宅最寄駅", f"edit_home_{p['id']}_{ev}",
                default=p.get("home_location", ""),
            )
            if edit_is_hr:
                edit_work = None
            else:
                edit_work = _station_picker(
                    "職場最寄駅", f"edit_work_{p['id']}_{ev}",
                    default=p.get("work_location", ""),
                )

            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("保存", key=f"save_{p['id']}", type="primary", use_container_width=True):
                    if not edit_name.strip():
                        st.error("名前を入力してください。")
                    elif not edit_home:
                        st.error("自宅の最寄駅を選択してください。")
                    elif not edit_is_hr and not edit_work:
                        st.error("職場の最寄駅を選択してください。")
                    else:
                        work_val = "" if edit_is_hr else (edit_work or "")
                        update_participant(p["id"], edit_name.strip(), edit_pattern, work_val, edit_home)
                        del st.session_state["_editing_id"]
                        st.session_state._edit_ver += 1
                        _invalidate_event_cache(event_code)
                        st.session_state["_success_msg"] = f"{edit_name.strip()} さんの情報を更新しました。"
                        st.rerun()
            with col_cancel:
                if st.button("キャンセル", key=f"cancel_{p['id']}", use_container_width=True):
                    del st.session_state["_editing_id"]
                    st.session_state._edit_ver = st.session_state.get("_edit_ver", 0) + 1
                    st.rerun(scope="fragment")
    else:
        with st.container(border=True):
            col_info, col_edit, col_del = st.columns([5, 1, 1])
            with col_info:
                work_display = "(自宅から)" if is_hr else (p.get("work_location") or "-")
                st.markdown(f"**{p['name']}**")
                st.caption(f"自宅: {p.get('home_location') or '-'} / 職場: {work_display}")
            with col_edit:
                if st.button("編集", key=f"btn_edit_{p['id']}", use_container_width=True):
                    st.session_state["_editing_id"] = p["id"]
                    st.session_state._edit_ver = st.session_state.get("_edit_ver", 0) + 1
                    st.rerun()
            with col_del:
                if st.button("削除", key=f"btn_del_{p['id']}", use_container_width=True):
                    delete_participant(p["id"])
                    _invalidate_event_cache(event_code)
                    st.rerun()


# ---------------------------------------------------------------------------
# UI: トップページ（イベント作成）
# ---------------------------------------------------------------------------
def page_top():
    _inject_custom_css()
    st.title("飲み会スポットファインダー")
    st.caption("参加者の職場と自宅から、最も集まりやすく帰りやすい駅を見つけます")

    st.markdown("---")

    st.markdown("""
    <div style="display:flex; gap:32px; margin-bottom:24px; align-items:center; flex-wrap:wrap;">
        <div><span class="step-badge">1</span><span class="step-title">飲み会を作成</span></div>
        <div style="color:#bdbdbd; font-size:20px;">→</div>
        <div><span class="step-badge-muted">2</span><span class="step-title-muted">URLを共有</span></div>
        <div style="color:#bdbdbd; font-size:20px;">→</div>
        <div><span class="step-badge-muted">3</span><span class="step-title-muted">みんなで入力</span></div>
        <div style="color:#bdbdbd; font-size:20px;">→</div>
        <div><span class="step-badge-muted">4</span><span class="step-title-muted">最適スポット発見</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("新しい飲み会を作成")
    col_input, col_btn, col_empty = st.columns([2, 1, 2])
    with col_input:
        title = st.text_input("飲み会の名前", placeholder="例: 歓迎会、忘年会", label_visibility="collapsed")
    with col_btn:
        if st.button("作成", type="primary", use_container_width=True):
            with st.spinner("作成中..."):
                event = create_event(title.strip() or "飲み会")
            code = event["event_code"]
            # キャッシュを事前設定して再フェッチを回避
            st.session_state[f"_event_cache_{code}"] = {
                "event": event,
                "participants": [],
            }
            st.query_params["event"] = code
            st.rerun()


# ---------------------------------------------------------------------------
# UI: イベントページ（参加者入力 & 結果表示）
# ---------------------------------------------------------------------------
def page_event(event_code: str, event: dict | None = None, db_participants: list | None = None):
    _inject_custom_css()
    if not event:
        st.error("イベントが見つかりません。URLを確認してください。")
        if st.button("トップに戻る"):
            del st.query_params["event"]
            st.rerun()
        return

    if db_participants is None:
        db_participants = []

    st.title(f"{event['title']}")
    st.caption("飲み会スポットファインダー")

    # 共有URL表示
    base_url = os.environ.get("APP_URL", st.context.headers.get("Origin", ""))
    share_url = f"{base_url}/?event={event_code}" if base_url else f"?event={event_code}"

    participant_count = len(db_participants)

    # ステップ表示
    step2_active = participant_count == 0
    step3_active = 0 < participant_count < 2
    step4_active = participant_count >= 2

    def _badge(n, active):
        return "step-badge" if active else "step-badge-muted"
    def _title(active):
        return "step-title" if active else "step-title-muted"

    st.markdown(f"""
    <div style="display:flex; gap:24px; margin-bottom:16px; align-items:center; flex-wrap:wrap;">
        <div><span class="step-badge" style="background:#4caf50;">✓</span><span class="step-title" style="color:#4caf50;">作成済み</span></div>
        <div style="color:#bdbdbd; font-size:20px;">→</div>
        <div><span class="{_badge(2, step2_active)}">2</span><span class="{_title(step2_active)}">URLを共有</span></div>
        <div style="color:#bdbdbd; font-size:20px;">→</div>
        <div><span class="{_badge(3, step3_active or step4_active)}">3</span><span class="{_title(step3_active or step4_active)}">みんなで入力（現在 {participant_count}人）</span></div>
        <div style="color:#bdbdbd; font-size:20px;">→</div>
        <div><span class="{_badge(4, step4_active)}">4</span><span class="{_title(step4_active)}">最適スポット発見</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**このURLを参加者に共有してください**（右のボタンでコピー）")
    st.code(share_url, language=None)

    st.markdown("---")

    # --- 成功メッセージ表示 ---
    if "_success_msg" in st.session_state:
        st.success(st.session_state.pop("_success_msg"))

    # --- 参加者追加フォーム (fragment: 入力中は部分更新のみ) ---
    _add_participant_fragment(event["id"], event_code, participant_count)

    # --- 参加者一覧 ---
    st.markdown("---")
    st.subheader(f"参加者一覧（{participant_count}人）")

    if db_participants:
        for p in db_participants:
            _participant_card_fragment(p, event_code)
    else:
        st.info("まだ参加者がいません。上のフォームから追加してください。")

    # --- 重み設定 ---
    st.sidebar.header("検索設定")
    balance = st.sidebar.slider(
        "職場 <-> 自宅 重視バランス",
        0.0, 1.0, 0.5, 0.05,
        help="左: 職場からの近さ重視 / 右: 自宅への近さ重視",
    )
    work_weight = 1.0 - balance
    home_weight = balance
    st.sidebar.caption(f"職場重視: {work_weight:.0%} | 自宅重視: {home_weight:.0%}")

    fairness_weight = st.sidebar.slider(
        "効率 <-> 公平 バランス",
        0.0, 1.0, 0.3, 0.05,
        help="左: 合計移動コスト最小 / 右: 参加者間の差を小さく",
    )

    # --- 検索実行 ---
    st.markdown("---")
    if participant_count < 2:
        st.button("最適スポットを検索", disabled=True, use_container_width=True,
                  help="2人以上の参加者が必要です")
        st.caption(f"あと{2 - participant_count}人追加すると検索できます")
        search_clicked = False
    else:
        search_clicked = st.button("最適スポットを検索", type="primary", use_container_width=True)

    if search_clicked:
        db_participants = get_participants(event["id"])
        if len(db_participants) < 2:
            st.error("2人以上の参加者を追加してください。")
            return

        with st.spinner("最適スポットを検索しています..."):
            geocoded = []
            for p in db_participants:
                entry = geocode_participant(p)
                if entry["work_lat"] is not None or entry["home_lat"] is not None:
                    geocoded.append(entry)

        if len(geocoded) < 2:
            st.error("場所を特定できた参加者が2人未満です。入力内容を確認してください。")
            return

        with st.spinner("最適スポットを計算しています..."):
            stations = find_candidate_stations(geocoded)

            if not stations:
                st.error("周辺に駅が見つかりませんでした。")
                return

            stations = _prefilter_stations(stations, geocoded, work_weight, home_weight, top_n=30)

            scored = score_stations(stations, geocoded, work_weight, home_weight,
                                    fairness_weight=fairness_weight)

        st.session_state["_search_results"] = {
            "scored": scored,
            "geocoded": geocoded,
        }

    # session_stateから結果を取得して表示
    if "_search_results" not in st.session_state:
        return

    scored = st.session_state["_search_results"]["scored"]
    geocoded = st.session_state["_search_results"]["geocoded"]
    unit = "分"

    # =====================================================================
    # 結果表示
    # =====================================================================
    st.markdown("---")
    st.subheader("検索結果")

    top_n = min(10, len(scored))
    top_stations = scored[:top_n]

    best = top_stations[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("おすすめ1位", best["name"])
    k2.metric("平均移動時間", f"{best['avg_total_val']:.1f} {unit}")
    k3.metric("最大移動者", f"{best['max_person_val']:.1f} {unit}")

    tab_ranking, tab_map, tab_detail, tab_restaurant = st.tabs(["ランキング", "地図", "詳細比較", "お店を探す"])

    with tab_ranking:
        ranking_rows = []
        for i, s in enumerate(top_stations):
            row = {
                "順位": i + 1,
                "駅名": s["name"],
                f"平均({unit})": s["avg_total_val"],
                f"最大({unit})": s["max_person_val"],
                "偏差": s["std_dev"],
            }
            for d, g in zip(s["details"], geocoded):
                is_hr = g.get("pattern", "").startswith("自宅")
                from_label = "自宅→" if is_hr else "職場→"
                to_label = "→自宅"
                row[f"{d['name']}({from_label})"] = f"{d['work_val']:.1f}{unit}"
                row[f"{d['name']}({to_label})"] = f"{d['home_val']:.1f}{unit}"
            ranking_rows.append(row)

        ranking_df = pd.DataFrame(ranking_rows)
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        st.caption("※移動時間は主要路線の駅間所要時間をベースに推定。乗換待ち時間等により実際とは異なる場合があります。")

    with tab_map:
        _render_map(top_stations, geocoded)

    with tab_detail:
        st.markdown("### 上位3駅の参加者別移動時間・経路")

        for i, s in enumerate(top_stations[:3]):
            medal = ["1位", "2位", "3位"][i]
            with st.expander(f"{medal}: {s['name']}（平均 {s['avg_total_val']:.1f}{unit}）", expanded=(i == 0)):
                detail_df = pd.DataFrame(s["details"])
                detail_df.columns = ["名前", f"職場/自宅→駅({unit})", f"駅→自宅({unit})", f"合計({unit})"]
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                chart_df = pd.DataFrame({
                    "名前": [d["name"] for d in s["details"]],
                    f"職場/自宅→駅({unit})": [d["work_val"] for d in s["details"]],
                    f"駅→自宅({unit})": [d["home_val"] for d in s["details"]],
                }).set_index("名前")
                st.bar_chart(chart_df)

                # --- 経路詳細 ---
                sn = s["name"]
                sg = s.get("gcd")
                st.markdown("#### 経路")
                for d, g in zip(s["details"], geocoded):
                    is_hr = g.get("pattern", "").startswith("自宅")
                    from_label = "自宅" if is_hr else "職場"

                    # 出発 → 飲み会駅
                    wg = g.get("work_gcd")
                    ws = g.get("work_station")
                    if wg and sg and wg != sg:
                        path_to, time_to = _find_route(wg, sg)
                        if path_to:
                            route_str = _format_route(path_to)
                            st.markdown(f"**{d['name']}**（{from_label}→飲み会）{route_str}（{time_to}{unit}）")
                        else:
                            st.markdown(f"**{d['name']}**（{from_label}→飲み会）{ws or '?'} → {sn}（{d['work_val']:.1f}{unit}）")
                    elif wg and sg and wg == sg:
                        st.markdown(f"**{d['name']}**（{from_label}→飲み会）{sn}駅（0{unit}）")
                    elif d["work_val"] > 0:
                        st.markdown(f"**{d['name']}**（{from_label}→飲み会）{ws or '?'} → {sn}（{d['work_val']:.1f}{unit}）")

                    # 飲み会駅 → 自宅
                    hg = g.get("home_gcd")
                    hs = g.get("home_station")
                    if hg and sg and hg != sg:
                        path_home, time_home = _find_route(sg, hg)
                        if path_home:
                            route_str = _format_route(path_home)
                            st.markdown(f"**{d['name']}**（飲み会→自宅）{route_str}（{time_home}{unit}）")
                        else:
                            st.markdown(f"**{d['name']}**（飲み会→自宅）{sn} → {hs or '?'}（{d['home_val']:.1f}{unit}）")
                    elif hg and sg and hg == sg:
                        st.markdown(f"**{d['name']}**（飲み会→自宅）{sn}駅（0{unit}）")
                    elif d["home_val"] > 0:
                        st.markdown(f"**{d['name']}**（飲み会→自宅）{sn} → {hs or '?'}（{d['home_val']:.1f}{unit}）")
                    st.markdown("---")

        st.markdown("### 公平性分析")
        st.caption("各駅で最も移動時間が長い人と短い人の差（小さいほど公平）")

        fairness_data = []
        for s in top_stations[:5]:
            vals = [d["total_val"] for d in s["details"]]
            fairness_data.append({
                "駅名": s["name"],
                f"平均({unit})": s["avg_total_val"],
                f"最短({unit})": min(vals),
                f"最長({unit})": max(vals),
                f"差({unit})": round(max(vals) - min(vals), 1),
                "標準偏差": s["std_dev"],
            })
        st.dataframe(pd.DataFrame(fairness_data), use_container_width=True, hide_index=True)

    with tab_restaurant:
        if not HOTPEPPER_API_KEY:
            st.info("お店検索は現在準備中です。")
        else:
            st.markdown("### 上位駅の周辺でお店を探す")
            # 駅選択
            station_options = [f"{i+1}位: {s['name']}" for i, s in enumerate(top_stations[:3])]
            selected_station_label = st.radio("駅を選択", station_options, horizontal=True)
            selected_idx = station_options.index(selected_station_label)
            sel_station = top_stations[selected_idx]

            # 検索条件
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                want_free_drink = st.checkbox("飲み放題あり", value=True)
            with col_f2:
                want_private_room = st.checkbox("個室あり")
            with col_f3:
                num_people = st.number_input("人数", min_value=0, value=len(geocoded), step=1,
                                             help="0で人数条件なし")
            keyword = st.text_input("キーワード（任意）", placeholder="例: 焼き鳥、イタリアン", key="hp_keyword")

            if st.button("お店を検索", type="primary", use_container_width=True, key="hp_search"):
                with st.spinner(f"{sel_station['name']}駅周辺のお店を検索中..."):
                    shops = _search_hotpepper(
                        sel_station["lat"], sel_station["lon"],
                        keyword=keyword,
                        free_drink=1 if want_free_drink else 0,
                        private_room=1 if want_private_room else 0,
                        party_capacity=num_people if num_people > 0 else 0,
                    )
                st.session_state["_hp_results"] = shops
                st.session_state["_hp_station"] = sel_station["name"]

            # 結果表示
            if "_hp_results" in st.session_state:
                shops = st.session_state["_hp_results"]
                station_name = st.session_state.get("_hp_station", "")
                if not shops:
                    st.warning("条件に合うお店が見つかりませんでした。条件を変えて再検索してください。")
                else:
                    st.markdown(f"**{station_name}駅**周辺で **{len(shops)}件** 見つかりました")
                    for shop in shops:
                        with st.container(border=True):
                            col_img, col_info = st.columns([1, 2])
                            with col_img:
                                if shop["photo"]:
                                    st.image(shop["photo"], use_container_width=True)
                            with col_info:
                                st.markdown(f"**{shop['name']}**")
                                st.caption(shop["catch"] if shop["catch"] else "")
                                tags = []
                                if shop["genre"]:
                                    tags.append(f"🍽️ {shop['genre']}")
                                if shop["budget"]:
                                    tags.append(f"💰 {shop['budget']}")
                                if shop["capacity"]:
                                    tags.append(f"👥 宴会最大{shop['capacity']}名")
                                if shop["private_room"] and "あり" in shop["private_room"]:
                                    tags.append("🚪 個室あり")
                                if shop["free_drink"] and "あり" in shop["free_drink"]:
                                    tags.append("🍺 飲み放題あり")
                                if tags:
                                    st.markdown(" ｜ ".join(tags))
                                st.caption(f"📍 {shop['access']}")
                                if shop["url"]:
                                    st.link_button("ホットペッパーで予約", shop["url"],
                                                   use_container_width=False)
                    st.caption("powered by ホットペッパーグルメ Webサービス")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    event_code = st.query_params.get("event")

    if event_code:
        cache_key = f"_event_cache_{event_code}"
        if cache_key not in st.session_state:
            progress = st.progress(0, text="イベント情報を取得中...")
            _event_data = get_event(event_code)
            progress.progress(50, text="参加者情報を取得中...")
            _participants = get_participants(_event_data["id"]) if _event_data else []
            progress.progress(100, text="準備完了")
            progress.empty()
            st.session_state[cache_key] = {
                "event": _event_data,
                "participants": _participants,
            }
        else:
            cached = st.session_state[cache_key]
            _event_data = cached["event"]
            _participants = cached["participants"]
        page_event(event_code, _event_data, _participants)
    else:
        page_top()


if __name__ == "__main__":
    main()
