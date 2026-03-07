"""FastAPI backend for 飲み会スポットファインダー."""
import os
import string
import random
import pickle
import heapq
import json
import urllib.request
import urllib.parse
from math import radians, sin, cos, sqrt, atan2

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
HOTPEPPER_API_KEY = os.environ.get("HOTPEPPER_API_KEY", "")
ADSENSE_CLIENT = os.environ.get("ADSENSE_CLIENT", "")
GA_ID = os.environ.get("GA_ID", "")

TRAIN_SPEED_KMH = 30
AVG_TRAIN_SPEED_KMH = 35
TRANSFER_PENALTY = 5.0

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# ---------------------------------------------------------------------------
# Station data (loaded once at startup)
# ---------------------------------------------------------------------------
_ekidata = None


def _load_ekidata():
    global _ekidata
    if _ekidata is not None:
        return _ekidata
    pkl = os.path.join(_DATA_DIR, "ekidata_cache.pkl")
    if not os.path.exists(pkl):
        raise RuntimeError("ekidata_cache.pkl not found. Run precompute.py first.")
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple) and len(data) >= 7:
        _ekidata = data[:7]
    elif isinstance(data, tuple) and len(data) >= 5:
        _ekidata = data[:5] + ({}, {})
    else:
        raise RuntimeError("ekidata_cache.pkl outdated.")
    return _ekidata


def _station_db():
    return _load_ekidata()[0]


def _graph():
    return _load_ekidata()[1]


def _station_names_arr():
    return _load_ekidata()[2]


def _station_coords():
    return _load_ekidata()[3]


def _edge_lines():
    return _load_ekidata()[4]


def _name_to_gcd():
    return _load_ekidata()[5]


def _gcd_to_name():
    return _load_ekidata()[6]


# ---------------------------------------------------------------------------
# Supabase REST
# ---------------------------------------------------------------------------
def _sb_request(method, path, body=None, params=None):
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
    return "".join(random.choices(chars, k=length))


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------
def _dijkstra(graph, start, targets=None, edge_lines=None):
    if edge_lines is None:
        edge_lines = {}
    state_dist = {(start, ""): 0.0}
    best = {start: 0.0}
    heap = [(0.0, start, "")]
    remaining = set(targets) if targets else None
    if remaining:
        remaining.discard(start)
    while heap:
        d, u, u_line = heapq.heappop(heap)
        if d > state_dist.get((u, u_line), float("inf")):
            continue
        if remaining is not None and u in remaining and d <= best.get(u, float("inf")):
            remaining.discard(u)
            if not remaining:
                break
        for v, w in graph.get(u, []):
            v_lines = edge_lines.get((u, v), [])
            if not v_lines:
                v_lines = [""]
            for v_line in v_lines:
                penalty = TRANSFER_PENALTY if u_line and v_line and u_line != v_line else 0
                nd = d + w + penalty
                state = (v, v_line)
                if nd < state_dist.get(state, float("inf")):
                    state_dist[state] = nd
                    if nd < best.get(v, float("inf")):
                        best[v] = nd
                    heapq.heappush(heap, (nd, v, v_line))
    return best


def _dijkstra_with_path(graph, start, target, edge_lines=None):
    if edge_lines is None:
        edge_lines = {}
    state_dist = {(start, ""): 0.0}
    prev = {(start, ""): None}
    best_target_d = float("inf")
    best_target_state = None
    heap = [(0.0, start, "")]
    while heap:
        d, u, u_line = heapq.heappop(heap)
        state = (u, u_line)
        if d > state_dist.get(state, float("inf")):
            continue
        if u == target and d < best_target_d:
            best_target_d = d
            best_target_state = state
            continue
        if d >= best_target_d:
            continue
        for v, w in graph.get(u, []):
            v_lines = edge_lines.get((u, v), [])
            if not v_lines:
                v_lines = [""]
            for v_line in v_lines:
                penalty = TRANSFER_PENALTY if u_line and v_line and u_line != v_line else 0
                nd = d + w + penalty
                next_state = (v, v_line)
                if nd < state_dist.get(next_state, float("inf")):
                    state_dist[next_state] = nd
                    prev[next_state] = state
                    heapq.heappush(heap, (nd, v, v_line))
    if best_target_state is None:
        return None, None
    # 経路復元: (gcd, line) のペアを返す
    path_states = []
    s = best_target_state
    while s is not None:
        path_states.append(s)
        s = prev.get(s)
    path_states.reverse()
    path = [s[0] for s in path_states]
    return path, round(best_target_d, 1), path_states


def _batch_dijkstra(sources, targets):
    targets_key = tuple(sorted(set(targets)))
    graph = _graph()
    el = _edge_lines()
    result = {}
    for src in sources:
        if src not in graph:
            result[src] = {}
            continue
        dist = _dijkstra(graph, src, set(targets_key), el)
        result[src] = {t: round(dist[t], 1) for t in targets_key if t in dist}
    return result


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
def _geocode_station(station_name):
    name = station_name.rstrip("駅").strip()
    if not name:
        return None, None, "", None
    sdb = _station_db()
    ntg = _name_to_gcd()
    # 完全一致（"京橋（東京メトロ銀座線）" など表示名そのまま）
    if name in sdb:
        lat, lon = sdb[name]
        gcd = ntg.get(name)
        return lat, lon, f"{name}駅", gcd
    # 括弧なし名での前方一致（"京橋" → "京橋（...）" の最初のヒット）
    for key in sdb:
        base = key.split("（")[0]
        if base == name:
            lat, lon = sdb[key]
            gcd = ntg.get(key)
            return lat, lon, f"{key}駅", gcd
    return None, None, "", None


TRIP_PATTERNS = ["職場→飲み会→自宅", "自宅→飲み会→自宅"]


def geocode_participant(p):
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
# Candidate stations & scoring
# ---------------------------------------------------------------------------
def find_candidate_stations(participants):
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
    search_radius = max(max_radius * 1.2, 3.0)

    sdb = _station_db()
    coords = _station_coords()
    names = _station_names_arr()

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


def score_stations(stations, participants, work_weight, home_weight, fairness_weight=0.0):
    person_sources = set()
    for p in participants:
        wg = p.get("work_gcd")
        if wg:
            person_sources.add(wg)
        hg = p.get("home_gcd")
        if hg:
            person_sources.add(hg)

    candidate_gcds = [s["gcd"] for s in stations if s.get("gcd")]
    dist_table = _batch_dijkstra(person_sources, candidate_gcds)

    scored = []
    for st_info in stations:
        total_cost = 0
        max_val = 0
        details = []
        sg = st_info.get("gcd")
        for p in participants:
            if p.get("work_lat") is not None:
                wg = p.get("work_gcd")
                if wg and sg and wg == sg:
                    work_val = 0.0
                elif wg and sg:
                    work_val = dist_table.get(wg, {}).get(sg)
                    if work_val is None:
                        dist = haversine(p["work_lat"], p["work_lon"], st_info["lat"], st_info["lon"])
                        work_val = round(dist / TRAIN_SPEED_KMH * 60, 1)
                else:
                    dist = haversine(p["work_lat"], p["work_lon"], st_info["lat"], st_info["lon"])
                    work_val = round(dist / TRAIN_SPEED_KMH * 60, 1)
            else:
                work_val = 0

            if p.get("home_lat") is not None:
                hg = p.get("home_gcd")
                if hg and sg and hg == sg:
                    home_val = 0.0
                elif hg and sg:
                    home_val = dist_table.get(hg, {}).get(sg)
                    if home_val is None:
                        dist = haversine(st_info["lat"], st_info["lon"], p["home_lat"], p["home_lon"])
                        home_val = round(dist / TRAIN_SPEED_KMH * 60, 1)
                else:
                    dist = haversine(st_info["lat"], st_info["lon"], p["home_lat"], p["home_lon"])
                    home_val = round(dist / TRAIN_SPEED_KMH * 60, 1)
            else:
                home_val = 0

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
                "work_val": round(work_val, 1),
                "home_val": round(home_val, 1),
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
# Route formatting
# ---------------------------------------------------------------------------
def _format_route(path_gcds, path_states=None):
    if not path_gcds or len(path_gcds) < 2:
        gtn = _gcd_to_name()
        return gtn.get(path_gcds[0], "?") if path_gcds else ""
    gtn = _gcd_to_name()

    # path_states があれば各ノードの路線名を使う（ダイクストラの結果）
    if path_states and len(path_states) == len(path_gcds):
        lines = [s[1] for s in path_states]
    else:
        # フォールバック: edge_lines の最初の路線を使う
        el = _edge_lines()
        lines = [""]
        for j in range(len(path_gcds) - 1):
            edge_l = el.get((path_gcds[j], path_gcds[j + 1]), [])
            lines.append(edge_l[0] if edge_l else "")

    # 同じ路線が続く区間をまとめる
    segments = []
    current_line = lines[1] if len(lines) > 1 else ""
    seg_start = path_gcds[0]
    for j in range(1, len(path_gcds) - 1):
        next_line = lines[j + 1] if j + 1 < len(lines) else ""
        if next_line != current_line:
            segments.append((seg_start, path_gcds[j], current_line))
            seg_start = path_gcds[j]
            current_line = next_line
    segments.append((seg_start, path_gcds[-1], current_line))
    parts = [gtn.get(segments[0][0], "?")]
    for _start, end, line in segments:
        name = gtn.get(end, "?")
        if line:
            parts.append(f" →({line})→ {name}")
        else:
            parts.append(f" → {name}")
    return "".join(parts)


def _find_route(source_gcd, target_gcd):
    if not source_gcd or not target_gcd or source_gcd == target_gcd:
        return None, 0.0, None
    graph = _graph()
    if source_gcd not in graph:
        return None, None, None
    return _dijkstra_with_path(graph, source_gcd, target_gcd, _edge_lines())


# ---------------------------------------------------------------------------
# Hot Pepper
# ---------------------------------------------------------------------------
_HP_API_URL = "https://webservice.recruit.co.jp/hotpepper/gourmet/v1/"


def _search_hotpepper(lat, lon, keyword="", count=100,
                      free_drink=0, private_room=0, party_capacity=0):
    if not HOTPEPPER_API_KEY:
        return []
    params = {
        "key": HOTPEPPER_API_KEY,
        "lat": lat, "lng": lon,
        "range": 3, "order": 4,
        "count": count, "format": "json",
    }
    if keyword:
        params["keyword"] = keyword
    if free_drink:
        params["free_drink"] = 1
    if private_room:
        params["private_room"] = 1
    if party_capacity:
        params["party_capacity"] = party_capacity
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
# Pydantic models
# ---------------------------------------------------------------------------
class CreateEventReq(BaseModel):
    title: str = "飲み会"


class UpdateEventReq(BaseModel):
    title: str


class AddParticipantReq(BaseModel):
    event_id: str
    name: str
    pattern: str
    work_location: str = ""
    home_location: str = ""


class UpdateParticipantReq(BaseModel):
    name: str
    pattern: str
    work_location: str = ""
    home_location: str = ""


class SearchReq(BaseModel):
    event_id: str
    work_weight: float = 0.5
    home_weight: float = 0.5
    fairness_weight: float = 0.3


class RestaurantReq(BaseModel):
    lat: float
    lon: float
    keyword: str = ""
    free_drink: int = 0
    private_room: int = 0
    party_capacity: int = 0


class RouteReq(BaseModel):
    source_gcd: int
    target_gcd: int


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/stations")
def api_stations(q: str = ""):
    """Station name autocomplete. Returns up to 30 matches."""
    names = _station_names_arr()
    if not q:
        return []
    q = q.rstrip("駅").strip()
    if not q:
        return []
    results = [n for n in names if q in n]
    return results[:30]


@app.get("/api/config")
def api_config():
    """Return public config for frontend."""
    return {
        "adsense_client": ADSENSE_CLIENT,
        "ga_id": GA_ID,
        "has_hotpepper": bool(HOTPEPPER_API_KEY),
    }


@app.post("/api/events")
def api_create_event(req: CreateEventReq):
    code = _generate_code()
    rows = _sb_request("POST", "events", body={"event_code": code, "title": req.title or "飲み会"})
    return rows[0]


@app.patch("/api/events/{code}")
def api_update_event(code: str, req: UpdateEventReq):
    rows = _sb_request("GET", "events", params={
        "select": "id", "event_code": f"eq.{code}", "limit": "1",
    })
    if not rows:
        raise HTTPException(status_code=404, detail="Event not found")
    _sb_request("PATCH", f"events?id=eq.{rows[0]['id']}", body={"title": req.title})
    return {"ok": True}


@app.get("/api/events/{code}")
def api_get_event(code: str):
    rows = _sb_request("GET", "events", params={
        "select": "*", "event_code": f"eq.{code}", "limit": "1",
    })
    if not rows:
        raise HTTPException(status_code=404, detail="Event not found")
    event = rows[0]
    participants = _sb_request("GET", "participants", params={
        "select": "*", "event_id": f"eq.{event['id']}", "order": "created_at",
    })
    return {"event": event, "participants": participants}


@app.post("/api/participants")
def api_add_participant(req: AddParticipantReq):
    rows = _sb_request("POST", "participants", body={
        "event_id": req.event_id,
        "name": req.name,
        "pattern": req.pattern,
        "work_location": req.work_location,
        "home_location": req.home_location,
    })
    return rows[0]


@app.patch("/api/participants/{pid}")
def api_update_participant(pid: str, req: UpdateParticipantReq):
    rows = _sb_request("PATCH", f"participants?id=eq.{pid}", body={
        "name": req.name,
        "pattern": req.pattern,
        "work_location": req.work_location,
        "home_location": req.home_location,
    })
    return rows[0] if rows else {"ok": True}


@app.delete("/api/participants/{pid}")
def api_delete_participant(pid: str):
    _sb_request("DELETE", f"participants?id=eq.{pid}")
    return {"ok": True}


@app.post("/api/search")
def api_search(req: SearchReq):
    participants = _sb_request("GET", "participants", params={
        "select": "*", "event_id": f"eq.{req.event_id}", "order": "created_at",
    })
    if len(participants) < 2:
        raise HTTPException(status_code=400, detail="2人以上の参加者が必要です")

    geocoded = []
    for p in participants:
        entry = geocode_participant(p)
        if entry["work_lat"] is not None or entry["home_lat"] is not None:
            geocoded.append(entry)

    if len(geocoded) < 2:
        raise HTTPException(status_code=400, detail="場所を特定できた参加者が2人未満です")

    stations = find_candidate_stations(geocoded)
    if not stations:
        raise HTTPException(status_code=400, detail="周辺に駅が見つかりませんでした")

    stations = _prefilter_stations(stations, geocoded, req.work_weight, req.home_weight, top_n=30)
    scored = score_stations(stations, geocoded, req.work_weight, req.home_weight,
                            fairness_weight=req.fairness_weight)

    top_n = min(10, len(scored))
    top_stations = scored[:top_n]

    # Build route info for top 3
    for s in top_stations[:3]:
        sg = s.get("gcd")
        routes = []
        for d, g in zip(s["details"], geocoded):
            person_routes = {"name": d["name"], "to_route": None, "home_route": None}
            is_hr = g.get("pattern", "").startswith("自宅")
            person_routes["from_label"] = "自宅" if is_hr else "職場"

            wg = g.get("work_gcd")
            if wg and sg and wg != sg:
                path_to, time_to, states_to = _find_route(wg, sg)
                if path_to:
                    person_routes["to_route"] = _format_route(path_to, states_to)
                    person_routes["to_time"] = time_to
                else:
                    ws = g.get("work_station") or "?"
                    person_routes["to_route"] = f"{ws} → {s['name']}"
                    person_routes["to_time"] = d["work_val"]
            elif wg and sg and wg == sg:
                person_routes["to_route"] = f"{s['name']}駅"
                person_routes["to_time"] = 0
            elif d["work_val"] > 0:
                ws = g.get("work_station") or "?"
                person_routes["to_route"] = f"{ws} → {s['name']}"
                person_routes["to_time"] = d["work_val"]

            hg = g.get("home_gcd")
            if hg and sg and hg != sg:
                path_home, time_home, states_home = _find_route(sg, hg)
                if path_home:
                    person_routes["home_route"] = _format_route(path_home, states_home)
                    person_routes["home_time"] = time_home
                else:
                    hs = g.get("home_station") or "?"
                    person_routes["home_route"] = f"{s['name']} → {hs}"
                    person_routes["home_time"] = d["home_val"]
            elif hg and sg and hg == sg:
                person_routes["home_route"] = f"{s['name']}駅"
                person_routes["home_time"] = 0
            elif d["home_val"] > 0:
                hs = g.get("home_station") or "?"
                person_routes["home_route"] = f"{s['name']} → {hs}"
                person_routes["home_time"] = d["home_val"]

            routes.append(person_routes)
        s["routes"] = routes

    return {
        "scored": top_stations,
        "geocoded": [
            {
                "name": g["name"],
                "pattern": g["pattern"],
                "work_lat": g.get("work_lat"),
                "work_lon": g.get("work_lon"),
                "work_station": g.get("work_station"),
                "home_lat": g.get("home_lat"),
                "home_lon": g.get("home_lon"),
                "home_station": g.get("home_station"),
            }
            for g in geocoded
        ],
    }


@app.post("/api/restaurants")
def api_restaurants(req: RestaurantReq):
    shops = _search_hotpepper(
        req.lat, req.lon,
        keyword=req.keyword,
        count=100,
        free_drink=req.free_drink,
        private_room=req.private_room,
        party_capacity=req.party_capacity,
    )
    return shops


# ---------------------------------------------------------------------------
# Static files & SPA fallback
# ---------------------------------------------------------------------------
_static_dir = os.path.join(_DATA_DIR, "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/llms.txt")
def serve_llms_txt():
    return FileResponse(os.path.join(_static_dir, "llms.txt"), media_type="text/plain; charset=utf-8")


@app.get("/ads.txt")
def serve_ads_txt():
    return FileResponse(os.path.join(_static_dir, "ads.txt"), media_type="text/plain; charset=utf-8")


@app.get("/")
@app.get("/{path:path}")
def serve_spa(path: str = ""):
    if path.startswith("api/"):
        raise HTTPException(status_code=404)
    return FileResponse(os.path.join(_static_dir, "index.html"))
