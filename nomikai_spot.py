import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import heapq
import time
import os

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
st.set_page_config(page_title="飲み会スポットファインダー", layout="wide")

GSI_GEOCODE_URL = "https://msearch.gsi.go.jp/address-search/AddressSearch"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TRAIN_SPEED_KMH = 30  # 駅間移動の概算速度（グラフ未対応時のフォールバック）
WALKING_SPEED_KMH = 4  # 徒歩速度
AVG_TRAIN_SPEED_KMH = 35  # 隣接駅間の平均速度（所要時間推定用）

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 駅データ.jp CSV 読み込み → STATION_DB / グラフ構築
# ---------------------------------------------------------------------------
@st.cache_data
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

    # --- 路線データ読み込み ---
    ldf = pd.read_csv(line_file, dtype={"line_cd": int, "line_name": str, "e_status": int})
    ldf = ldf[ldf["e_status"] == 0]
    line_cd_to_name = dict(zip(ldf["line_cd"], ldf["line_name"]))

    # --- 駅データ読み込み ---
    sdf = pd.read_csv(station_file, dtype={"station_cd": int, "station_name": str,
                                            "lat": float, "lon": float, "e_status": int,
                                            "line_cd": int})
    # 運用中のみ
    sdf = sdf[sdf["e_status"] == 0].copy()

    # station_cd → (name, lat, lon) のマップ
    cd_to_info = {}
    for _, r in sdf.iterrows():
        cd_to_info[int(r["station_cd"])] = (r["station_name"], float(r["lat"]), float(r["lon"]))

    # 駅名 → (lat, lon) 辞書（同名駅は最初のものを採用）
    station_db = {}
    for _, r in sdf.iterrows():
        name = r["station_name"]
        if name not in station_db:
            station_db[name] = (float(r["lat"]), float(r["lon"]))

    # 駅名 → 路線名リスト
    station_lines: dict[str, list[str]] = {}
    for _, r in sdf.iterrows():
        name = r["station_name"]
        lc = int(r["line_cd"])
        ln = line_cd_to_name.get(lc)
        if ln and ln not in station_lines.get(name, []):
            station_lines.setdefault(name, []).append(ln)

    # --- 接続データ読み込み → グラフ構築 ---
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
        # 所要時間推定: 距離 ÷ 平均速度
        dist = _haversine(lat1, lon1, lat2, lon2)
        time_min = max(round(dist / AVG_TRAIN_SPEED_KMH * 60, 1), 1)
        graph.setdefault(name1, []).append((name2, time_min))
        graph.setdefault(name2, []).append((name1, time_min))

    # 同名駅（乗換）: station_g_cd が同じ駅同士を乗換時間5分で接続
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
    """start→end の最短所要時間（分）を返す。到達不可なら None。"""
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
    """座標から最も近い駅名と距離(km)を返す。"""
    best_name, best_dist = None, float("inf")
    for name, (slat, slon) in STATION_DB.items():
        # 緯度差で粗いフィルタ（約0.09度 ≈ 10km）
        if abs(slat - lat) > 0.5:
            continue
        d = haversine(lat, lon, slat, slon)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, best_dist


def estimate_travel_time(from_lat: float, from_lon: float,
                         to_lat: float, to_lon: float) -> float:
    """2地点間の推定移動時間（分）。路線グラフ優先、フォールバックで直線距離÷30km/h。"""
    from_st, from_walk_km = _find_nearest_graph_station(from_lat, from_lon)
    to_st, to_walk_km = _find_nearest_graph_station(to_lat, to_lon)

    if from_st and to_st:
        train_time = _dijkstra(from_st, to_st)
        if train_time is not None:
            walk_min = (from_walk_km + to_walk_km) / WALKING_SPEED_KMH * 60
            return round(walk_min + train_time, 1)

    # フォールバック
    dist = haversine(from_lat, from_lon, to_lat, to_lon)
    return round(dist / TRAIN_SPEED_KMH * 60, 1)


# ---------------------------------------------------------------------------
# 距離計算（Haversine）
# ---------------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2点間の距離(km)を計算。"""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# ジオコーディング（国土地理院 + Overpass 駅名検索）
# ---------------------------------------------------------------------------
def _geocode_gsi(query: str) -> tuple[float | None, float | None, str]:
    """国土地理院 API で住所/地名を座標に変換。"""
    try:
        resp = requests.get(GSI_GEOCODE_URL, params={"q": query}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            coords = data[0]["geometry"]["coordinates"]
            title = data[0]["properties"].get("title", "")
            return float(coords[1]), float(coords[0]), title  # lon,lat → lat,lon
    except Exception:
        pass
    return None, None, ""


def _geocode_station(station_name: str) -> tuple[float | None, float | None, str]:
    """ビルトインDB → Overpass API の順で駅名を検索。"""
    name = station_name.rstrip("駅").strip()
    if not name:
        return None, None, ""

    # 1) ビルトインDB
    if name in STATION_DB:
        lat, lon = STATION_DB[name]
        return lat, lon, f"{name}駅"

    # 2) Overpass API フォールバック
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
    """住所/駅名を座標に変換。複数の方法を試して最適な結果を返す。"""
    if not address or not address.strip():
        return None, None, ""
    address = address.strip()

    # 明らかな住所パターン（区、市、町、丁目、番地を含む）
    is_address = any(k in address for k in ["区", "市", "町", "丁目", "番地", "都", "府", "県"])

    # 「駅」を含む場合は駅名として検索
    if "駅" in address:
        lat, lon, label = _geocode_station(address)
        if lat is not None:
            return lat, lon, label

    # 住所っぽい場合は国土地理院を優先
    if is_address:
        lat, lon, label = _geocode_gsi(address)
        if lat is not None:
            return lat, lon, label

    # 短い地名は駅名の可能性が高い → 先に駅検索
    lat, lon, label = _geocode_station(address + "駅")
    if lat is not None:
        return lat, lon, label

    # 最後に国土地理院で地名検索
    lat, lon, label = _geocode_gsi(address)
    if lat is not None:
        return lat, lon, label

    return None, None, ""


# ---------------------------------------------------------------------------
# 参加者を囲む円の算出 & 候補駅フィルタ
# ---------------------------------------------------------------------------
def compute_bounding_circle(
    participants: list[dict],
) -> tuple[float, float, float]:
    """全参加者の職場・自宅を囲む最小円（中心lat, lon, 半径km）を返す。"""
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


def find_candidate_stations(
    participants: list[dict], margin: float = 1.2,
) -> list[dict]:
    """参加者を囲む円の内側（+margin）にある駅を候補として返す。
    margin=1.2 なら円の半径を20%拡大して少し余裕を持たせる。
    """
    center_lat, center_lon, radius = compute_bounding_circle(participants)
    search_radius = max(radius * margin, 3.0)  # 最低3km

    stations = []
    seen = set()

    # 1) ビルトインDBから円内の駅
    for name, (slat, slon) in STATION_DB.items():
        dist = haversine(center_lat, center_lon, slat, slon)
        if dist <= search_radius:
            seen.add(name)
            lines = STATION_LINES.get(name, [])
            stations.append({
                "name": name, "lat": slat, "lon": slon,
                "operator": "", "line": "・".join(lines),
            })

    # 2) Overpass API で補完（ビルトインDBだけでは足りない場合）
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
                    # 円内チェック
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


# ---------------------------------------------------------------------------
# 駅スコアリング
# ---------------------------------------------------------------------------
def _station_travel_time(from_name: str | None, to_name: str | None,
                         from_lat: float | None, from_lon: float | None,
                         to_lat: float | None, to_lon: float | None) -> float:
    """駅名ベースでDijkstra探索。グラフに無い場合は直線距離フォールバック。"""
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


def score_stations(
    stations: list[dict],
    participants: list[dict],
    work_weight: float,
    home_weight: float,
    fairness_weight: float = 0.0,
    mode: str = "train",
) -> list[dict]:
    """各駅について参加者全員の移動コストを計算しスコアリング。
    mode="train": 路線グラフベースの所要時間（分）
    mode="distance": 直線距離（km）
    fairness_weight: 0=効率重視 ～ 1=公平重視（標準偏差へのペナルティ係数）
    """
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
        # 最終スコア = 効率コスト + 公平性ペナルティ
        # fairness_weight=1 のとき標準偏差を平均と同等の重みで加算
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


def make_result_map(
    participants: list[dict],
    top_stations: list[dict],
    center_lat: float,
    center_lon: float,
    mode: str = "train",
) -> go.Figure:
    """参加者の位置と推薦駅を地図上に表示。"""
    fig = go.Figure()

    # 職場マーカー（自宅往復の人はスキップ）
    for p in participants:
        is_hr = p.get("pattern", "").startswith("自宅")
        if not is_hr and p.get("work_lat") is not None:
            label = p.get("work_station") or p.get("work_label") or "職場"
            fig.add_trace(go.Scattermapbox(
                lat=[p["work_lat"]], lon=[p["work_lon"]],
                mode="markers+text",
                marker=dict(size=14, color="#1565c0",
                            opacity=0.9),
                text=[f"{p['name']}({label})"],
                textposition="top center",
                textfont=dict(size=11, color="#0d47a1"),
                name=f"{p['name']} 職場",
                showlegend=False,
            ))
    # 自宅マーカー
    for p in participants:
        if p.get("home_lat") is not None:
            label = p.get("home_station") or p.get("home_label") or "自宅"
            fig.add_trace(go.Scattermapbox(
                lat=[p["home_lat"]], lon=[p["home_lon"]],
                mode="markers+text",
                marker=dict(size=14, color="#2e7d32",
                            opacity=0.9),
                text=[f"{p['name']}({label})"],
                textposition="top center",
                textfont=dict(size=11, color="#1b5e20"),
                name=f"{p['name']} 自宅",
                showlegend=False,
            ))

    # 推薦駅マーカー（白縁 → 色付き の二重マーカーで視認性UP）
    unit = "分" if mode == "train" else "km"
    for i, s in enumerate(top_stations[:10]):
        color = _RANK_COLORS[i] if i < len(_RANK_COLORS) else "#888888"
        # 白い縁取り（少し大きめ）
        fig.add_trace(go.Scattermapbox(
            lat=[s["lat"]], lon=[s["lon"]],
            mode="markers",
            marker=dict(size=28 if i == 0 else 22 if i < 3 else 18,
                        color="white", opacity=0.95),
            showlegend=False, hoverinfo="skip",
        ))
        # 色付きマーカー
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
# メイン
# ---------------------------------------------------------------------------
def main():
    st.title("飲み会スポットファインダー")
    st.caption("参加者の職場と自宅から、最も集まりやすく帰りやすい駅を見つけます")

    # --- モード選択 ---
    mode = st.radio(
        "計算モード",
        ["電車（路線グラフ）", "直線距離"],
        horizontal=True,
        help="電車モード: 主要路線の所要時間で計算（駅名のみ入力）\n直線距離モード: 直線距離で計算（駅名・住所・地名OK）",
    )
    is_train = mode.startswith("電車")

    TRIP_PATTERNS = ["職場→飲み会→自宅", "自宅→飲み会→自宅"]

    # --- セッションステート初期化 ---
    if "participants" not in st.session_state:
        st.session_state.participants = [
            {"name": "Aさん", "work": "", "home": "", "pattern": TRIP_PATTERNS[0]},
            {"name": "Bさん", "work": "", "home": "", "pattern": TRIP_PATTERNS[0]},
            {"name": "Cさん", "work": "", "home": "", "pattern": TRIP_PATTERNS[0]},
        ]

    # --- 参加者入力 ---
    st.subheader("参加者情報を入力")
    if is_train:
        st.caption("最寄駅名を入力してください（例: 東京、渋谷、新宿）")
        work_ph, home_ph = "職場最寄駅（例: 東京）", "自宅最寄駅（例: 吉祥寺）"
    else:
        st.caption("駅名、地名、住所のいずれかを入力してください（例: 渋谷、新宿区西新宿2丁目）")
        work_ph, home_ph = "職場（駅名 or 住所）", "自宅（駅名 or 住所）"

    for i, p in enumerate(st.session_state.participants):
        cols = st.columns([1, 1.5, 1.8, 1.8, 0.4])
        with cols[0]:
            st.session_state.participants[i]["name"] = st.text_input(
                "名前", value=p["name"], key=f"name_{i}", label_visibility="collapsed",
                placeholder="名前",
            )
        with cols[1]:
            cur_pattern = st.selectbox(
                "移動パターン", TRIP_PATTERNS, key=f"pattern_{i}",
                index=TRIP_PATTERNS.index(p.get("pattern", TRIP_PATTERNS[0])),
                label_visibility="collapsed",
            )
            st.session_state.participants[i]["pattern"] = cur_pattern
        is_home_round = cur_pattern == TRIP_PATTERNS[1]
        with cols[2]:
            st.session_state.participants[i]["home"] = st.text_input(
                "自宅", value=p["home"], key=f"home_{i}", label_visibility="collapsed",
                placeholder=home_ph,
            )
        with cols[3]:
            if is_home_round:
                st.markdown("<div style='line-height:2.4;color:#999;font-size:0.9em'>← 自宅のみで計算</div>",
                            unsafe_allow_html=True)
            else:
                st.session_state.participants[i]["work"] = st.text_input(
                    "職場", value=p["work"], key=f"work_{i}", label_visibility="collapsed",
                    placeholder=work_ph,
                )
        with cols[4]:
            if st.button("✕", key=f"del_{i}", help="削除"):
                st.session_state.participants.pop(i)
                st.rerun()

    btn_cols = st.columns([1, 1.5, 4])
    with btn_cols[0]:
        if st.button("＋ 参加者を追加"):
            n = len(st.session_state.participants) + 1
            st.session_state.participants.append(
                {"name": f"{chr(64+n)}さん", "work": "", "home": "", "pattern": TRIP_PATTERNS[0]}
            )
            st.rerun()

    # --- 重み設定 ---
    st.sidebar.header("検索設定")
    balance = st.sidebar.slider(
        "職場↔自宅 重視バランス",
        0.0, 1.0, 0.5, 0.05,
        help="左: 職場からの近さ重視（行きやすさ）/ 右: 自宅への近さ重視（帰りやすさ）",
    )
    work_weight = 1.0 - balance
    home_weight = balance
    st.sidebar.caption(f"職場重視: {work_weight:.0%}　|　自宅重視: {home_weight:.0%}")

    fairness_weight = st.sidebar.slider(
        "効率↔公平 バランス",
        0.0, 1.0, 0.3, 0.05,
        help="左: 全員の合計移動コスト最小（効率重視）/ 右: 参加者間の移動差を小さく（公平重視）",
    )
    if fairness_weight < 0.2:
        st.sidebar.caption("⚡ 効率重視: 合計コスト最小の駅を優先")
    elif fairness_weight > 0.8:
        st.sidebar.caption("⚖️ 公平重視: 参加者間の差が小さい駅を優先")
    else:
        st.sidebar.caption("⚡⚖️ 効率と公平のバランス")

    # --- 検索実行 ---
    with btn_cols[1]:
        search_clicked = st.button("最適スポットを検索", type="primary", use_container_width=True)

    if not search_clicked:
        st.info("参加者情報を入力して「最適スポットを検索」を押してください。")
        return

    # 入力バリデーション
    valid_participants = []
    for p in st.session_state.participants:
        if not p["name"].strip():
            continue
        is_home_round = p.get("pattern", "") == TRIP_PATTERNS[1]
        if is_home_round:
            if p["home"].strip():
                valid_participants.append(p)
        else:
            if p["work"].strip() or p["home"].strip():
                valid_participants.append(p)
    if len(valid_participants) < 2:
        st.error("2人以上の参加者情報を入力してください。")
        return

    # --- 場所解決 ---
    progress = st.progress(0)
    status = st.empty()
    geocoded = []
    total_queries = sum(
        (0 if p.get("pattern", "") == TRIP_PATTERNS[1] else (1 if p["work"].strip() else 0))
        + (1 if p["home"].strip() else 0)
        for p in valid_participants
    )
    done = 0

    for p in valid_participants:
        is_home_round = p.get("pattern", "") == TRIP_PATTERNS[1]
        entry = {
            "name": p["name"],
            "pattern": p.get("pattern", TRIP_PATTERNS[0]),
            "work_station": None, "work_lat": None, "work_lon": None, "work_label": None,
            "home_station": None, "home_lat": None, "home_lon": None, "home_label": None,
        }

        # 職場（自宅往復の場合はスキップ）
        if not is_home_round and p["work"].strip():
            status.text(f"検索中… {p['name']}の職場")
            if is_train:
                lat, lon, label = _geocode_station(p["work"])
                if lat is not None:
                    entry["work_station"] = label.rstrip("駅")
            else:
                lat, lon, label = geocode(p["work"])
            if lat is not None:
                entry["work_lat"] = lat
                entry["work_lon"] = lon
                entry["work_label"] = label
                if entry["work_station"] is None:
                    nearest, _ = _find_nearest_graph_station(lat, lon)
                    entry["work_station"] = nearest
            else:
                st.warning(f"⚠ {p['name']}の職場「{p['work']}」が見つかりませんでした。")
            done += 1
            progress.progress(done / (total_queries + 2))
            time.sleep(0.5)

        # 自宅
        if p["home"].strip():
            status.text(f"検索中… {p['name']}の自宅")
            if is_train:
                lat, lon, label = _geocode_station(p["home"])
                if lat is not None:
                    entry["home_station"] = label.rstrip("駅")
            else:
                lat, lon, label = geocode(p["home"])
            if lat is not None:
                entry["home_lat"] = lat
                entry["home_lon"] = lon
                entry["home_label"] = label
                if entry["home_station"] is None:
                    nearest, _ = _find_nearest_graph_station(lat, lon)
                    entry["home_station"] = nearest
            else:
                st.warning(f"⚠ {p['name']}の自宅「{p['home']}」が見つかりませんでした。")
            done += 1
            progress.progress(done / (total_queries + 2))
            time.sleep(0.5)

        # 自宅往復の場合: 出発地も自宅
        if is_home_round and entry["home_lat"] is not None:
            entry["work_lat"] = entry["home_lat"]
            entry["work_lon"] = entry["home_lon"]
            entry["work_station"] = entry["home_station"]
            entry["work_label"] = entry["home_label"]

        geocoded.append(entry)

    geocoded = [g for g in geocoded if g["work_lat"] is not None or g["home_lat"] is not None]
    if len(geocoded) < 2:
        st.error("場所を特定できた参加者が2人未満です。入力内容を確認してください。")
        progress.empty()
        status.empty()
        return

    # --- 候補駅検索（参加者を囲む円の内側） ---
    status.text("候補駅を検索中…")
    center_lat, center_lon, _radius = compute_bounding_circle(geocoded)
    stations = find_candidate_stations(geocoded)
    progress.progress(0.9)

    if not stations:
        st.error("周辺に駅が見つかりませんでした。参加者の場所を確認してください。")
        progress.empty()
        status.empty()
        return

    # --- スコアリング ---
    score_mode = "train" if is_train else "distance"
    status.text("各駅のスコアを計算中…")
    scored = score_stations(stations, geocoded, work_weight, home_weight,
                            fairness_weight=fairness_weight, mode=score_mode)
    progress.progress(1.0)
    progress.empty()
    status.empty()

    unit = "分" if is_train else "km"

    # =====================================================================
    # 結果表示
    # =====================================================================
    st.markdown("---")
    st.subheader("検索結果")

    top_n = min(10, len(scored))
    top_stations = scored[:top_n]

    # KPI
    best = top_stations[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("おすすめ1位", best["name"])
    k2.metric("平均移動" + ("時間" if is_train else "距離"), f"{best['avg_total_val']:.1f} {unit}")
    k3.metric("最大移動者", f"{best['max_person_val']:.1f} {unit}")
    k4.metric("候補駅数", f"{len(stations)} 駅")

    # タブ
    tab_map, tab_ranking, tab_detail = st.tabs(["地図", "ランキング", "詳細比較"])

    # --- 地図 ---
    with tab_map:
        fig = make_result_map(geocoded, top_stations, center_lat, center_lon, mode=score_mode)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("青丸: 職場 / 緑丸: 自宅 / 色付き丸: おすすめ駅（順位順）")

    # --- ランキング ---
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
        if is_train:
            st.caption("※移動時間は主要路線の駅間所要時間をベースに推定。乗換待ち時間等により実際とは異なる場合があります。")
        else:
            st.caption("※直線距離での概算です。実際の移動距離・時間は路線や経路により異なります。")

    # --- 詳細比較 ---
    with tab_detail:
        st.markdown(f"### 上位3駅の参加者別移動{'時間' if is_train else '距離'}")

        for i, s in enumerate(top_stations[:3]):
            medal = ["🥇", "🥈", "🥉"][i]
            line_info = f"　{s['line']}" if s.get("line") else ""
            with st.expander(f"{medal} {i+1}位: {s['name']}{line_info}（平均 {s['avg_total_val']:.1f}{unit}）", expanded=(i == 0)):
                detail_df = pd.DataFrame(s["details"])
                detail_df.columns = ["名前", f"出発→駅({unit})", f"駅→自宅({unit})", f"合計({unit})"]
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                # 棒グラフ
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
                    yaxis_title=f"{'移動時間 (分)' if is_train else '距離 (km)'}",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # 公平性分析
        st.markdown("### 公平性分析")
        st.caption(f"各駅で最も移動{'時間' if is_train else '距離'}が長い人と短い人の差（小さいほど公平）")

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


if __name__ == "__main__":
    main()
