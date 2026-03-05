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


STATION_DB, _GRAPH, STATION_LINES = _load_ekidata()


# ---------------------------------------------------------------------------
# グラフ経路探索
# ---------------------------------------------------------------------------
def _dijkstra(start: str, end: str) -> int | None:
    if start == end:
        return 0
    if start not in _GRAPH or end not in _GRAPH:
        return None
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == end:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for v, w in _GRAPH.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return None


def _find_nearest_graph_station(lat: float, lon: float) -> tuple[str | None, float]:
    best_name, best_dist = None, float("inf")
    for name, (slat, slon) in STATION_DB.items():
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

    if name in STATION_DB:
        lat, lon = STATION_DB[name]
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

    for name, (slat, slon) in STATION_DB.items():
        dist = haversine(center_lat, center_lon, slat, slon)
        if dist <= search_radius:
            seen.add(name)
            lines = STATION_LINES.get(name, [])
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
    scored = []
    for st_info in stations:
        total_cost = 0
        max_val = 0
        details = []
        for p in participants:
            if mode == "train":
                work_val = _station_travel_time(
                    p.get("work_station"), st_info["name"],
                    p.get("work_lat"), p.get("work_lon"),
                    st_info["lat"], st_info["lon"],
                ) if p.get("work_lat") is not None else 0
                home_val = _station_travel_time(
                    st_info["name"], p.get("home_station"),
                    st_info["lat"], st_info["lon"],
                    p.get("home_lat"), p.get("home_lon"),
                ) if p.get("home_lat") is not None else 0
            else:
                work_val = round(haversine(
                    p["work_lat"], p["work_lon"], st_info["lat"], st_info["lon"]
                ), 1) if p.get("work_lat") is not None else 0
                home_val = round(haversine(
                    st_info["lat"], st_info["lon"], p["home_lat"], p["home_lon"]
                ), 1) if p.get("home_lat") is not None else 0

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


def make_result_map(participants, top_stations, center_lat, center_lon, mode="train") -> go.Figure:
    fig = go.Figure()

    for p in participants:
        is_hr = p.get("pattern", "").startswith("自宅")
        if not is_hr and p.get("work_lat") is not None:
            label = p.get("work_station") or p.get("work_label") or "職場"
            fig.add_trace(go.Scattermapbox(
                lat=[p["work_lat"]], lon=[p["work_lon"]],
                mode="markers+text",
                marker=dict(size=14, color="#1565c0", opacity=0.9),
                text=[f"{p['name']}({label})"],
                textposition="top center",
                textfont=dict(size=11, color="#0d47a1"),
                name=f"{p['name']} 職場",
                showlegend=False,
            ))
    for p in participants:
        if p.get("home_lat") is not None:
            label = p.get("home_station") or p.get("home_label") or "自宅"
            fig.add_trace(go.Scattermapbox(
                lat=[p["home_lat"]], lon=[p["home_lon"]],
                mode="markers+text",
                marker=dict(size=14, color="#2e7d32", opacity=0.9),
                text=[f"{p['name']}({label})"],
                textposition="top center",
                textfont=dict(size=11, color="#1b5e20"),
                name=f"{p['name']} 自宅",
                showlegend=False,
            ))

    unit = "分" if mode == "train" else "km"
    for i, s in enumerate(top_stations[:10]):
        color = _RANK_COLORS[i] if i < len(_RANK_COLORS) else "#888888"
        fig.add_trace(go.Scattermapbox(
            lat=[s["lat"]], lon=[s["lon"]],
            mode="markers",
            marker=dict(size=28 if i == 0 else 22 if i < 3 else 18,
                        color="white", opacity=0.95),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[s["lat"]], lon=[s["lon"]],
            mode="markers+text",
            marker=dict(size=22 if i == 0 else 17 if i < 3 else 13,
                        color=color, opacity=0.95),
            text=[f"{'★' if i == 0 else str(i+1)} {s['name']} ({s['avg_total_val']:.1f}{unit})"
                  + (f"<br>{s['line']}" if s.get('line') and i < 3 else "")],
            textposition="top center",
            textfont=dict(
                size=14 if i == 0 else 12 if i < 3 else 10,
                color="#212121",
            ),
            name=f"{i+1}位: {s['name']}",
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10,
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.9)", font=dict(size=12),
        ),
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

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("新しい飲み会を作成")
        title = st.text_input("飲み会の名前", placeholder="例: 歓迎会、忘年会")
        if st.button("作成してURLを発行", type="primary", use_container_width=True):
            code = create_event(title.strip() or "飲み会")
            st.session_state["_redirect_event"] = code
            st.rerun()

    with col2:
        st.subheader("コードで参加")
        st.caption("幹事から共有されたコードを入力")
        join_code = st.text_input("イベントコード", placeholder="例: Xk9mZ3", label_visibility="collapsed")
        if st.button("参加する", use_container_width=True):
            if join_code.strip():
                event = get_event(join_code.strip())
                if event:
                    st.query_params["event"] = join_code.strip()
                    st.rerun()
                else:
                    st.error("イベントが見つかりません。コードを確認してください。")


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

    # 共有URL（目立つように表示）
    st.info(f"**このURLを参加者に共有してください**\n\n`{share_url}`", icon="🔗")

    st.markdown("---")

    # 駅名リスト（レコメンド用）
    station_names = sorted(STATION_DB.keys())

    # --- 参加者追加フォーム ---
    st.subheader("自分の情報を入力")
    st.caption("最寄駅を選択してください（入力すると候補が絞り込まれます）")

    # パターン選択をフォーム外に置いて動的に職場欄を表示/非表示
    new_pattern = st.selectbox("移動パターン", TRIP_PATTERNS, key="add_pattern")
    is_home_round_form = new_pattern == TRIP_PATTERNS[1]

    new_name = st.text_input("名前", placeholder="あなたの名前", key="add_name")
    new_home = st.selectbox("自宅最寄駅", options=[""] + station_names,
                            index=0, key="add_home",
                            format_func=lambda x: "駅を選択..." if x == "" else x)
    if is_home_round_form:
        new_work = ""
    else:
        new_work = st.selectbox("職場最寄駅", options=[""] + station_names,
                                index=0, key="add_work",
                                format_func=lambda x: "駅を選択..." if x == "" else x)

    if st.button("参加者を追加", type="primary", use_container_width=True):
        if not new_name.strip():
            st.error("名前を入力してください。")
        elif not new_home:
            st.error("自宅の最寄駅を選択してください。")
        elif not is_home_round_form and not new_work:
            st.error("職場の最寄駅を選択してください。")
        else:
            work_val = "" if is_home_round_form else new_work
            add_participant(event["id"], new_name.strip(), new_pattern, work_val, new_home)
            # 入力をリセット
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

    # --- ジオコーディング ---
    progress = st.progress(0)
    status = st.empty()
    geocoded = []

    for i, p in enumerate(db_participants):
        status.text(f"検索中... {p['name']}")
        entry = geocode_participant(p, True)
        if entry["work_lat"] is not None or entry["home_lat"] is not None:
            geocoded.append(entry)
        else:
            st.warning(f"{p['name']} の場所が特定できませんでした。")
        progress.progress((i + 1) / (len(db_participants) + 2))
        time.sleep(0.3)

    if len(geocoded) < 2:
        st.error("場所を特定できた参加者が2人未満です。入力内容を確認してください。")
        progress.empty()
        status.empty()
        return

    # --- 候補駅検索 ---
    status.text("候補駅を検索中...")
    center_lat, center_lon, _ = compute_bounding_circle(geocoded)
    stations = find_candidate_stations(geocoded)
    progress.progress(0.9)

    if not stations:
        st.error("周辺に駅が見つかりませんでした。")
        progress.empty()
        status.empty()
        return

    # --- スコアリング ---
    status.text("各駅のスコアを計算中...")
    scored = score_stations(stations, geocoded, work_weight, home_weight,
                            fairness_weight=fairness_weight, mode="train")
    progress.progress(1.0)
    progress.empty()
    status.empty()

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

    tab_map, tab_ranking, tab_detail = st.tabs(["地図", "ランキング", "詳細比較"])

    with tab_map:
        fig = make_result_map(geocoded, top_stations, center_lat, center_lon, mode="train")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("青丸: 職場 / 緑丸: 自宅 / 色付き丸: おすすめ駅（順位順）")

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
