import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import heapq
import time
import os
import string
import random

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

@st.cache_resource
def _get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Supabase の設定がありません。環境変数 SUPABASE_URL / SUPABASE_KEY を設定してください。")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def _generate_code(length=6):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
st.set_page_config(page_title="飲み会スポットファインダー", layout="wide")

GSI_GEOCODE_URL = "https://msearch.gsi.go.jp/address-search/AddressSearch"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TRAIN_SPEED_KMH = 30
WALKING_SPEED_KMH = 4
AVG_TRAIN_SPEED_KMH = 35

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 駅データ.jp CSV 読み込み → STATION_DB / グラフ構築
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="駅データを読み込んでいます...")
def _load_ekidata():
    """statione.csv, join.csv, line.csv を読み込み、駅辞書・路線グラフ・路線名マップを構築する。"""

    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    station_file = os.path.join(_DATA_DIR, "statione.csv")
    join_file = os.path.join(_DATA_DIR, "join.csv")
    line_file = os.path.join(_DATA_DIR, "line.csv")

    ldf = pd.read_csv(line_file, dtype={"line_cd": int, "line_name": str, "e_status": int})
    ldf = ldf[ldf["e_status"] == 0]
    line_cd_to_name = dict(zip(ldf["line_cd"], ldf["line_name"]))

    sdf = pd.read_csv(station_file, dtype={"station_cd": int, "station_name": str,
                                            "lat": float, "lon": float, "e_status": int,
                                            "line_cd": int})
    sdf = sdf[sdf["e_status"] == 0].copy()

    cd_to_info = {}
    for _, r in sdf.iterrows():
        cd_to_info[int(r["station_cd"])] = (r["station_name"], float(r["lat"]), float(r["lon"]))

    station_db = {}
    for _, r in sdf.iterrows():
        name = r["station_name"]
        if name not in station_db:
            station_db[name] = (float(r["lat"]), float(r["lon"]))

    station_lines: dict[str, list[str]] = {}
    for _, r in sdf.iterrows():
        name = r["station_name"]
        lc = int(r["line_cd"])
        ln = line_cd_to_name.get(lc)
        if ln and ln not in station_lines.get(name, []):
            station_lines.setdefault(name, []).append(ln)

    jdf = pd.read_csv(join_file, dtype={"line_cd": int, "station_cd1": int, "station_cd2": int})

    graph: dict[str, list[tuple[str, float]]] = {}
    for _, r in jdf.iterrows():
        cd1, cd2 = int(r["station_cd1"]), int(r["station_cd2"])
        if cd1 not in cd_to_info or cd2 not in cd_to_info:
            continue
        name1, lat1, lon1 = cd_to_info[cd1]
        name2, lat2, lon2 = cd_to_info[cd2]
        if name1 == name2:
            continue
        dist = _haversine(lat1, lon1, lat2, lon2)
        time_min = max(round(dist / AVG_TRAIN_SPEED_KMH * 60, 1), 1)
        graph.setdefault(name1, []).append((name2, time_min))
        graph.setdefault(name2, []).append((name1, time_min))

    if "station_g_cd" in sdf.columns:
        groups = sdf.groupby("station_g_cd")["station_name"].apply(set)
        for names in groups:
            name_list = list(names)
            for j in range(len(name_list)):
                for k in range(j + 1, len(name_list)):
                    n1, n2 = name_list[j], name_list[k]
                    graph.setdefault(n1, []).append((n2, 5))
                    graph.setdefault(n2, []).append((n1, 5))

    return station_db, graph, station_lines


def _get_ekidata():
    """駅データを遅延読み込み（初回のみ計算、以降はキャッシュ）。"""
    return _load_ekidata()

def _station_db():
    return _get_ekidata()[0]

def _graph():
    return _get_ekidata()[1]

def _station_lines():
    return _get_ekidata()[2]

def _station_names_set():
    return frozenset(_station_db().keys())


# ---------------------------------------------------------------------------
# グラフ経路探索
# ---------------------------------------------------------------------------
def _dijkstra(start: str, end: str) -> int | None:
    if start == end:
        return 0
    g = _graph()
    if start not in g or end not in g:
        return None
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == end:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for v, w in g.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return None


def _dijkstra_all(start: str) -> dict[str, float]:
    """start から全到達可能駅への最短所要時間（分）を返す。"""
    g = _graph()
    if start not in g:
        return {}
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in g.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _find_nearest_graph_station(lat: float, lon: float) -> tuple[str | None, float]:
    best_name, best_dist = None, float("inf")
    sdb = _station_db()
    for name, (slat, slon) in sdb.items():
        if abs(slat - lat) > 0.5:
            continue
        d = haversine(lat, lon, slat, slon)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, best_dist


def estimate_travel_time(from_lat: float, from_lon: float,
                         to_lat: float, to_lon: float) -> float:
    from_st, from_walk_km = _find_nearest_graph_station(from_lat, from_lon)
    to_st, to_walk_km = _find_nearest_graph_station(to_lat, to_lon)

    if from_st and to_st:
        train_time = _dijkstra(from_st, to_st)
        if train_time is not None:
            walk_min = (from_walk_km + to_walk_km) / WALKING_SPEED_KMH * 60
            return round(walk_min + train_time, 1)

    dist = haversine(from_lat, from_lon, to_lat, to_lon)
    return round(dist / TRAIN_SPEED_KMH * 60, 1)


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
# ジオコーディング
# ---------------------------------------------------------------------------
def _geocode_gsi(query: str) -> tuple[float | None, float | None, str]:
    try:
        resp = requests.get(GSI_GEOCODE_URL, params={"q": query}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            coords = data[0]["geometry"]["coordinates"]
            title = data[0]["properties"].get("title", "")
            return float(coords[1]), float(coords[0]), title
    except Exception:
        pass
    return None, None, ""


def _geocode_station(station_name: str) -> tuple[float | None, float | None, str]:
    name = station_name.rstrip("駅").strip()
    if not name:
        return None, None, ""

    sdb = _station_db()
    if name in sdb:
        lat, lon = sdb[name]
        return lat, lon, f"{name}駅"

    query = f'[out:json][timeout:10];node["railway"="station"]["name"~"^{name}$"];out body 1;'
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=15)
        if resp.status_code == 200 and resp.text.strip().startswith("{"):
            elems = resp.json().get("elements", [])
            if elems:
                e = elems[0]
                return e["lat"], e["lon"], f"{name}駅"
    except Exception:
        pass
    return None, None, ""


def geocode(address: str) -> tuple[float | None, float | None, str]:
    if not address or not address.strip():
        return None, None, ""
    address = address.strip()

    is_address = any(k in address for k in ["区", "市", "町", "丁目", "番地", "都", "府", "県"])

    if "駅" in address:
        lat, lon, label = _geocode_station(address)
        if lat is not None:
            return lat, lon, label

    if is_address:
        lat, lon, label = _geocode_gsi(address)
        if lat is not None:
            return lat, lon, label

    lat, lon, label = _geocode_station(address + "駅")
    if lat is not None:
        return lat, lon, label

    lat, lon, label = _geocode_gsi(address)
    if lat is not None:
        return lat, lon, label

    return None, None, ""


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

    stations = []
    seen = set()
    sdb = _station_db()
    sl = _station_lines()

    for name, (slat, slon) in sdb.items():
        dist = haversine(center_lat, center_lon, slat, slon)
        if dist <= search_radius:
            seen.add(name)
            lines = sl.get(name, [])
            stations.append({
                "name": name, "lat": slat, "lon": slon,
                "operator": "", "line": "・".join(lines),
            })

    if len(stations) < 5:
        radius_m = int(search_radius * 1000)
        query = f'[out:json][timeout:20];node["railway"="station"]["name"](around:{radius_m},{center_lat},{center_lon});out body;'
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=25)
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                for elem in resp.json().get("elements", []):
                    name = elem.get("tags", {}).get("name")
                    if not name or name in seen:
                        continue
                    dist = haversine(center_lat, center_lon, elem["lat"], elem["lon"])
                    if dist <= search_radius:
                        seen.add(name)
                        stations.append({
                            "name": name, "lat": elem["lat"], "lon": elem["lon"],
                            "operator": elem.get("tags", {}).get("operator", ""),
                            "line": "",
                        })
        except Exception:
            pass

    return stations


def _station_travel_time(from_name, to_name, from_lat, from_lon, to_lat, to_lon) -> float:
    if from_name is None or to_name is None:
        return 0.0
    if from_name == to_name:
        return 0.0
    t = _dijkstra(from_name, to_name)
    if t is not None:
        return round(t, 1)
    if from_lat is not None and to_lat is not None:
        dist = haversine(from_lat, from_lon, to_lat, to_lon)
        return round(dist / TRAIN_SPEED_KMH * 60, 1)
    return 0.0


def score_stations(stations, participants, work_weight, home_weight,
                   fairness_weight=0.0, mode="train") -> list[dict]:
    # 参加者ごとに1回だけDijkstraで全駅への距離を事前計算
    work_dists = {}  # name -> {station: time}
    home_dists = {}
    for p in participants:
        ws = p.get("work_station")
        hs = p.get("home_station")
        if ws and ws not in work_dists:
            work_dists[ws] = _dijkstra_all(ws)
        if hs and hs not in home_dists:
            home_dists[hs] = _dijkstra_all(hs)

    scored = []
    for st_info in stations:
        total_cost = 0
        max_val = 0
        details = []
        sn = st_info["name"]
        for p in participants:
            # 職場→候補駅
            if p.get("work_lat") is not None:
                ws = p.get("work_station")
                if ws and ws in work_dists and sn in work_dists[ws]:
                    work_val = round(work_dists[ws][sn], 1)
                elif ws:
                    work_val = _station_travel_time(
                        ws, sn, p.get("work_lat"), p.get("work_lon"),
                        st_info["lat"], st_info["lon"])
                else:
                    work_val = 0
            else:
                work_val = 0

            # 候補駅→自宅
            if p.get("home_lat") is not None:
                hs = p.get("home_station")
                if hs and hs in home_dists and sn in home_dists[hs]:
                    # グラフは無向なので逆方向も同じ距離
                    home_val = round(home_dists[hs][sn], 1)
                elif hs:
                    home_val = _station_travel_time(
                        sn, hs, st_info["lat"], st_info["lon"],
                        p.get("home_lat"), p.get("home_lon"))
                else:
                    home_val = 0
            else:
                home_val = 0

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
# 地図
# ---------------------------------------------------------------------------
_RANK_COLORS = [
    "#d50000", "#e65100", "#f57f17", "#ffab00", "#ffd600",
    "#aeea00", "#64dd17", "#00c853", "#00bfa5", "#0091ea",
]


_RANK_MARKER_COLORS = ["#d50000", "#e65100", "#f57f17"]

# 参加者ごとの色パレット（職場・自宅ペアで同系色）
_PERSON_COLORS = [
    {"work": "#1565c0", "home": "#42a5f5"},  # 青系
    {"work": "#c62828", "home": "#ef5350"},  # 赤系
    {"work": "#2e7d32", "home": "#66bb6a"},  # 緑系
    {"work": "#6a1b9a", "home": "#ab47bc"},  # 紫系
    {"work": "#e65100", "home": "#ff9800"},  # オレンジ系
    {"work": "#00838f", "home": "#26c6da"},  # シアン系
    {"work": "#4e342e", "home": "#8d6e63"},  # 茶系
    {"work": "#37474f", "home": "#78909c"},  # グレー系
]


def make_result_map(participants, top_stations, center_lat, center_lon, mode="train") -> go.Figure:
    fig = go.Figure()

    for i, p in enumerate(participants):
        colors = _PERSON_COLORS[i % len(_PERSON_COLORS)]
        is_hr = p.get("pattern", "").startswith("自宅")

        # 職場マーカー
        if not is_hr and p.get("work_lat") is not None:
            station = p.get("work_station") or p.get("work_label") or ""
            fig.add_trace(go.Scattermapbox(
                lat=[p["work_lat"]], lon=[p["work_lon"]],
                mode="markers+text",
                marker=dict(size=16, color=colors["work"], opacity=1.0),
                text=[f"{p['name']} 職場"],
                textposition="top center",
                textfont=dict(size=12, color=colors["work"]),
                name=f"{p['name']} 職場（{station}）",
                showlegend=True,
                hovertext=f"{p['name']} 職場: {station}",
            ))

        # 自宅マーカー
        if p.get("home_lat") is not None:
            station = p.get("home_station") or p.get("home_label") or ""
            fig.add_trace(go.Scattermapbox(
                lat=[p["home_lat"]], lon=[p["home_lon"]],
                mode="markers+text",
                marker=dict(size=16, color=colors["home"], opacity=1.0),
                text=[f"{p['name']} 自宅"],
                textposition="top center",
                textfont=dict(size=12, color=colors["home"]),
                name=f"{p['name']} 自宅（{station}）",
                showlegend=True,
                hovertext=f"{p['name']} 自宅: {station}",
            ))

    # おすすめ駅マーカー（上位3駅のみ）
    unit = "分"
    for i, s in enumerate(top_stations[:3]):
        color = _RANK_MARKER_COLORS[i]
        rank_label = ["1位", "2位", "3位"][i]
        # 白縁
        fig.add_trace(go.Scattermapbox(
            lat=[s["lat"]], lon=[s["lon"]],
            mode="markers",
            marker=dict(size=32 if i == 0 else 26, color="white", opacity=1.0),
            showlegend=False, hoverinfo="skip",
        ))
        # 色付きマーカー
        fig.add_trace(go.Scattermapbox(
            lat=[s["lat"]], lon=[s["lon"]],
            mode="markers+text",
            marker=dict(size=26 if i == 0 else 20, color=color, opacity=1.0),
            text=[f"{rank_label} {s['name']}（平均{s['avg_total_val']:.0f}{unit}）"],
            textposition="top center",
            textfont=dict(size=15 if i == 0 else 13, color="#212121",
                          family="Arial Black"),
            name=f"{rank_label}: {s['name']}",
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10,
        ),
        height=550,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.9)", font=dict(size=12),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 路線図（ネットワーク図）
# ---------------------------------------------------------------------------
def make_route_diagram(participants: list[dict], top_stations: list[dict]) -> go.Figure:
    """参加者の駅とおすすめ駅をネットワーク図で表示。"""
    fig = go.Figure()

    # ノード収集: 参加者の駅 + おすすめ駅
    nodes = {}  # name -> {"x", "y", "type", "label", "color", "size"}
    edges = []  # (from_name, to_name, time, color)

    # おすすめ駅を中央に配置
    for i, s in enumerate(top_stations[:3]):
        color = _RANK_MARKER_COLORS[i]
        rank = ["1位", "2位", "3位"][i]
        nodes[f"rec_{s['name']}"] = {
            "x": 0.5, "y": 0.5 - i * 0.2,
            "name": s["name"],
            "label": f"★{rank} {s['name']}",
            "color": color, "size": 28,
            "type": "recommend",
        }

    # 参加者の駅を左右に配置
    n_people = len(participants)
    for pi, p in enumerate(participants):
        colors = _PERSON_COLORS[pi % len(_PERSON_COLORS)]
        y_base = 1.0 - (pi + 0.5) / n_people  # 均等配置

        is_hr = p.get("pattern", "").startswith("自宅")

        # 職場（左側）
        if not is_hr and p.get("work_station"):
            ws = p["work_station"]
            node_key = f"work_{p['name']}"
            nodes[node_key] = {
                "x": 0.0, "y": y_base,
                "name": ws,
                "label": f"{p['name']} 職場\n({ws})",
                "color": colors["work"], "size": 18,
                "type": "work",
            }
            # おすすめ駅への接続線
            for i, s in enumerate(top_stations[:3]):
                t = _dijkstra(ws, s["name"])
                time_str = f"{round(t)}分" if t is not None else "?"
                edges.append((node_key, f"rec_{s['name']}", time_str, colors["work"]))

        # 自宅（右側）
        if p.get("home_station"):
            hs = p["home_station"]
            node_key = f"home_{p['name']}"
            nodes[node_key] = {
                "x": 1.0, "y": y_base,
                "name": hs,
                "label": f"{p['name']} 自宅\n({hs})",
                "color": colors["home"], "size": 18,
                "type": "home",
            }
            for i, s in enumerate(top_stations[:3]):
                t = _dijkstra(s["name"], hs)
                time_str = f"{round(t)}分" if t is not None else "?"
                edges.append((f"rec_{s['name']}", node_key, time_str, colors["home"]))

    # 接続線を描画
    for from_key, to_key, time_str, color in edges:
        if from_key not in nodes or to_key not in nodes:
            continue
        fn, tn = nodes[from_key], nodes[to_key]
        fig.add_trace(go.Scatter(
            x=[fn["x"], tn["x"]], y=[fn["y"], tn["y"]],
            mode="lines",
            line=dict(color=color, width=1.5),
            opacity=0.3,
            showlegend=False, hoverinfo="skip",
        ))
        # 所要時間ラベル（線の中点）
        mid_x = (fn["x"] + tn["x"]) / 2
        mid_y = (fn["y"] + tn["y"]) / 2
        fig.add_trace(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode="text",
            text=[time_str],
            textfont=dict(size=10, color="#757575"),
            showlegend=False, hoverinfo="skip",
        ))

    # ノードを描画
    for node in nodes.values():
        # 白縁
        fig.add_trace(go.Scatter(
            x=[node["x"]], y=[node["y"]],
            mode="markers",
            marker=dict(size=node["size"] + 8, color="white"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[node["x"]], y=[node["y"]],
            mode="markers+text",
            marker=dict(size=node["size"], color=node["color"]),
            text=[node["label"]],
            textposition="top center" if node["type"] == "recommend" else
                         ("middle left" if node["x"] < 0.3 else "middle right"),
            textfont=dict(size=12, color=node["color"]),
            showlegend=False,
            hovertext=node["name"],
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.3, 1.3]),
        yaxis=dict(visible=False, range=[-0.15, 1.15]),
        height=450,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# DB操作
# ---------------------------------------------------------------------------
TRIP_PATTERNS = ["職場→飲み会→自宅", "自宅→飲み会→自宅"]


def create_event(title: str) -> str:
    sb = _get_supabase()
    code = _generate_code()
    sb.table("events").insert({"event_code": code, "title": title}).execute()
    return code


def get_event(code: str) -> dict | None:
    sb = _get_supabase()
    res = sb.table("events").select("*").eq("event_code", code).execute()
    if res.data:
        return res.data[0]
    return None


def get_participants(event_id: str) -> list[dict]:
    sb = _get_supabase()
    res = sb.table("participants").select("*").eq("event_id", event_id).order("created_at").execute()
    return res.data or []


def add_participant(event_id: str, name: str, pattern: str, work: str, home: str) -> None:
    sb = _get_supabase()
    sb.table("participants").insert({
        "event_id": event_id,
        "name": name,
        "pattern": pattern,
        "work_location": work,
        "home_location": home,
    }).execute()


def delete_participant(participant_id: str) -> None:
    sb = _get_supabase()
    sb.table("participants").delete().eq("id", participant_id).execute()


# ---------------------------------------------------------------------------
# ジオコード参加者
# ---------------------------------------------------------------------------
def geocode_participant(p: dict, is_train: bool) -> dict:
    entry = {
        "name": p["name"],
        "pattern": p.get("pattern", TRIP_PATTERNS[0]),
        "work_station": None, "work_lat": None, "work_lon": None, "work_label": None,
        "home_station": None, "home_lat": None, "home_lon": None, "home_label": None,
    }
    is_home_round = entry["pattern"] == TRIP_PATTERNS[1]

    # 職場
    work_loc = p.get("work_location", "").strip()
    if not is_home_round and work_loc:
        if is_train:
            lat, lon, label = _geocode_station(work_loc)
            if lat is not None:
                entry["work_station"] = label.rstrip("駅")
        else:
            lat, lon, label = geocode(work_loc)
        if lat is not None:
            entry["work_lat"] = lat
            entry["work_lon"] = lon
            entry["work_label"] = label
            if entry["work_station"] is None:
                nearest, _ = _find_nearest_graph_station(lat, lon)
                entry["work_station"] = nearest

    # 自宅
    home_loc = p.get("home_location", "").strip()
    if home_loc:
        if is_train:
            lat, lon, label = _geocode_station(home_loc)
            if lat is not None:
                entry["home_station"] = label.rstrip("駅")
        else:
            lat, lon, label = geocode(home_loc)
        if lat is not None:
            entry["home_lat"] = lat
            entry["home_lon"] = lon
            entry["home_label"] = label
            if entry["home_station"] is None:
                nearest, _ = _find_nearest_graph_station(lat, lon)
                entry["home_station"] = nearest

    # 自宅往復
    if is_home_round and entry["home_lat"] is not None:
        entry["work_lat"] = entry["home_lat"]
        entry["work_lon"] = entry["home_lon"]
        entry["work_station"] = entry["home_station"]
        entry["work_label"] = entry["home_label"]

    return entry


# ---------------------------------------------------------------------------
# カスタムCSS
# ---------------------------------------------------------------------------
def _inject_custom_css():
    st.markdown("""
    <style>
    /* プライマリボタン（作成・検索） */
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"],
    button[kind="primary"] {
        font-weight: bold;
    }

    /* 参加者追加フォームのボタン */
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
        background-color: #1a73e8 !important;
        color: white !important;
        border: none !important;
    }

    /* 削除ボタン */
    button[kind="secondary"]:has(> div > p:only-child) {
        color: #d32f2f !important;
        border-color: #d32f2f !important;
    }

    /* 再実行時の白い靄を無効化 */
    [data-testid="stAppViewBlockContainer"] [data-stale="true"] {
        opacity: 1 !important;
    }

    /* ステップ番号のスタイル */
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
# UI: トップページ（イベント作成）
# ---------------------------------------------------------------------------
def page_top():
    _inject_custom_css()
    st.title("飲み会スポットファインダー")
    st.caption("参加者の職場と自宅から、最も集まりやすく帰りやすい駅を見つけます")

    st.markdown("---")

    # ステップ表示
    st.markdown("""
    <div style="display:flex; gap:32px; margin-bottom:24px; align-items:center;">
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
            code = create_event(title.strip() or "飲み会")
            st.session_state["_redirect_event"] = code
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

    # ステップ表示（現在の進行状況に応じてハイライト）
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

    # 共有URL（コピーボタン付き）
    st.markdown("**このURLを参加者に共有してください**（右のボタンでコピー）")
    st.code(share_url, language=None)

    st.markdown("---")

    # --- 参加者追加フォーム ---
    st.subheader("参加者を追加")
    st.caption("自分の情報を入力、または幹事がまとめて全員分を追加できます")

    # パターン選択をフォーム外に置いて動的に職場欄を表示/非表示
    new_pattern = st.selectbox("移動パターン", TRIP_PATTERNS, key="add_pattern")
    is_home_round_form = new_pattern == TRIP_PATTERNS[1]

    station_names = sorted(_station_db().keys())

    new_name = st.text_input("名前", placeholder="あなたの名前", key="add_name")
    new_home = st.selectbox("自宅最寄駅", options=station_names,
                            index=None, key="add_home",
                            placeholder="駅名を入力して選択...")
    if is_home_round_form:
        new_work = None
    else:
        new_work = st.selectbox("職場最寄駅", options=station_names,
                                index=None, key="add_work",
                                placeholder="駅名を入力して選択...")

    if st.button("参加者を追加", type="primary", use_container_width=True):
        if not new_name.strip():
            st.error("名前を入力してください。")
        elif not new_home:
            st.error("自宅の最寄駅を選択してください。")
        elif not is_home_round_form and not new_work:
            st.error("職場の最寄駅を選択してください。")
        else:
            work_val = "" if is_home_round_form else (new_work or "")
            add_participant(event["id"], new_name.strip(), new_pattern, work_val, new_home)
            for k in ["add_name", "add_home", "add_work", "add_pattern"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # --- 参加者一覧（DB） ---
    st.markdown("---")
    st.subheader(f"参加者一覧（{participant_count}人）")

    if db_participants:
        # ヘッダー行
        hcols = st.columns([1.5, 2, 2, 2, 0.5])
        with hcols[0]:
            st.markdown("**名前**")
        with hcols[1]:
            st.markdown("**パターン**")
        with hcols[2]:
            st.markdown("**自宅**")
        with hcols[3]:
            st.markdown("**職場**")

        for p in db_participants:
            is_hr = p["pattern"] == TRIP_PATTERNS[1]
            cols = st.columns([1.5, 2, 2, 2, 0.5])
            with cols[0]:
                st.text(p["name"])
            with cols[1]:
                st.text(p["pattern"])
            with cols[2]:
                st.text(p["home_location"] or "-")
            with cols[3]:
                st.text("(自宅のみ)" if is_hr else (p["work_location"] or "-"))
            with cols[4]:
                if st.button("✕", key=f"del_{p['id']}", help="削除"):
                    delete_participant(p["id"])
                    st.rerun()
    else:
        st.warning("まだ参加者がいません。上のフォームから追加してください。", icon="👆")

    # --- 重み設定 ---
    st.sidebar.header("検索設定")
    balance = st.sidebar.slider(
        "職場↔自宅 重視バランス",
        0.0, 1.0, 0.5, 0.05,
        help="左: 職場からの近さ重視 / 右: 自宅への近さ重視",
    )
    work_weight = 1.0 - balance
    home_weight = balance
    st.sidebar.caption(f"職場重視: {work_weight:.0%} | 自宅重視: {home_weight:.0%}")

    fairness_weight = st.sidebar.slider(
        "効率↔公平 バランス",
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

    if not search_clicked:
        return

    # 最新の参加者データ取得
    db_participants = get_participants(event["id"])
    if len(db_participants) < 2:
        st.error("2人以上の参加者を追加してください。")
        return

    # --- 検索処理 ---
    with st.spinner("最適スポットを検索しています..."):
        # ジオコーディング
        geocoded = []
        for p in db_participants:
            entry = geocode_participant(p, True)
            if entry["work_lat"] is not None or entry["home_lat"] is not None:
                geocoded.append(entry)

    if len(geocoded) < 2:
        st.error("場所を特定できた参加者が2人未満です。入力内容を確認してください。")
        return

    with st.spinner("最適スポットを計算しています..."):
        # 候補駅検索
        center_lat, center_lon, _ = compute_bounding_circle(geocoded)
        stations = find_candidate_stations(geocoded)

        if not stations:
            st.error("周辺に駅が見つかりませんでした。")
            return

        # スコアリング
        scored = score_stations(stations, geocoded, work_weight, home_weight,
                                fairness_weight=fairness_weight, mode="train")

    unit = "分"

    # =====================================================================
    # 結果表示
    # =====================================================================
    st.markdown("---")
    st.subheader("検索結果")

    top_n = min(10, len(scored))
    top_stations = scored[:top_n]

    best = top_stations[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("おすすめ1位", best["name"])
    k2.metric("平均移動時間", f"{best['avg_total_val']:.1f} {unit}")
    k3.metric("最大移動者", f"{best['max_person_val']:.1f} {unit}")
    k4.metric("候補駅数", f"{len(stations)} 駅")

    tab_route, tab_map, tab_ranking, tab_detail = st.tabs(["路線図", "地図", "ランキング", "詳細比較"])

    # 共通: 凡例HTML生成
    def _build_legends():
        person_items = ""
        for i, g in enumerate(geocoded):
            colors = _PERSON_COLORS[i % len(_PERSON_COLORS)]
            is_hr = g.get("pattern", "").startswith("自宅")
            if is_hr:
                person_items += f'<span style="margin-right:12px;"><span style="color:{colors["home"]};">●</span> {g["name"]}(自宅)</span>'
            else:
                person_items += f'<span style="margin-right:12px;"><span style="color:{colors["work"]};">■</span> {g["name"]} 職場 <span style="color:{colors["home"]};">●</span> 自宅</span>'
        rank_items = (
            f'<span style="margin-right:12px;"><span style="color:#d50000;">★</span> 1位</span>'
            f'<span style="margin-right:12px;"><span style="color:#e65100;">★</span> 2位</span>'
            f'<span><span style="color:#f57f17;">★</span> 3位</span>'
        )
        return person_items, rank_items

    person_legend, rank_legend = _build_legends()
    legend_style = "padding:8px 12px; border-radius:6px; font-size:13px; font-weight:600; display:flex; flex-wrap:wrap; gap:4px; align-items:center;"

    def _show_legends():
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(
                f'<div style="{legend_style} background:#f0f4ff; border:1px solid #c5cae9;">'
                f'<span style="margin-right:8px; color:#555;">参加者:</span>{person_legend}</div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div style="{legend_style} background:#fff8e1; border:1px solid #ffe082;">'
                f'<span style="margin-right:8px; color:#555;">おすすめ:</span>{rank_legend}</div>',
                unsafe_allow_html=True)

    # --- 路線図タブ ---
    with tab_route:
        _show_legends()
        fig_route = make_route_diagram(geocoded, top_stations[:3])
        st.plotly_chart(fig_route, use_container_width=True)

    # --- 地図タブ ---
    with tab_map:
        _show_legends()
        fig = make_result_map(geocoded, top_stations, center_lat, center_lon, mode="train")
        st.plotly_chart(fig, use_container_width=True)

    with tab_ranking:
        ranking_rows = []
        for i, s in enumerate(top_stations):
            row = {
                "順位": i + 1,
                "駅名": s["name"],
                "路線": s.get("line", ""),
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

    with tab_detail:
        st.markdown("### 上位3駅の参加者別移動時間")

        for i, s in enumerate(top_stations[:3]):
            medal = ["🥇", "🥈", "🥉"][i]
            line_info = f"　{s['line']}" if s.get("line") else ""
            with st.expander(f"{medal} {i+1}位: {s['name']}{line_info}（平均 {s['avg_total_val']:.1f}{unit}）", expanded=(i == 0)):
                detail_df = pd.DataFrame(s["details"])
                detail_df.columns = ["名前", f"出発→駅({unit})", f"駅→自宅({unit})", f"合計({unit})"]
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                fig_bar = go.Figure()
                names = [d["name"] for d in s["details"]]
                work_vals = [d["work_val"] for d in s["details"]]
                home_vals = [d["home_val"] for d in s["details"]]

                fig_bar.add_trace(go.Bar(
                    x=names, y=work_vals, name="出発→駅",
                    marker_color="#1565c0",
                ))
                fig_bar.add_trace(go.Bar(
                    x=names, y=home_vals, name="駅→自宅",
                    marker_color="#2e7d32",
                ))
                fig_bar.update_layout(
                    barmode="stack", height=300,
                    yaxis_title="移動時間 (分)",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

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


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    # イベント作成後のリダイレクト処理
    if "_redirect_event" in st.session_state:
        code = st.session_state.pop("_redirect_event")
        st.query_params["event"] = code
        st.rerun()

    event_code = st.query_params.get("event")

    if event_code:
        # 初回のみプログレスバー表示（以降はキャッシュ）
        cache_key = f"_loaded_{event_code}"
        if cache_key not in st.session_state:
            progress = st.progress(0, text="イベント情報を取得中...")
            _event_data = get_event(event_code)
            progress.progress(50, text="参加者情報を取得中...")
            _participants = get_participants(_event_data["id"]) if _event_data else []
            progress.progress(100, text="準備完了")
            progress.empty()
            st.session_state[cache_key] = True
        else:
            _event_data = get_event(event_code)
            _participants = get_participants(_event_data["id"]) if _event_data else []
        page_event(event_code, _event_data, _participants)
    else:
        page_top()


if __name__ == "__main__":
    main()
