#!/usr/bin/env python3
"""Pre-compute station data pickle for faster startup.

Builds the station database and graph structure only.
Shortest paths are computed on-demand at search time.

Graph uses station_g_cd (physical station ID) as node keys
to avoid phantom connections between same-named stations in different cities.
"""
import os
import pickle
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
AVG_TRAIN_SPEED_KMH = 35


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def main():
    station_file = os.path.join(DATA_DIR, "statione.csv")
    join_file = os.path.join(DATA_DIR, "join.csv")
    line_file = os.path.join(DATA_DIR, "line.csv")

    print("Reading CSV files...")
    ldf = pd.read_csv(line_file, dtype={"line_cd": int, "line_name": str, "e_status": int})
    ldf = ldf[ldf["e_status"] == 0]
    line_names = dict(zip(ldf["line_cd"].astype(int), ldf["line_name"]))

    sdf = pd.read_csv(station_file, dtype={
        "station_cd": int, "station_name": str,
        "lat": float, "lon": float, "e_status": int, "line_cd": int,
    })
    sdf = sdf[sdf["e_status"] == 0].copy()

    # station_g_cd: 同一物理駅を束ねる一意ID
    has_gcd = "station_g_cd" in sdf.columns
    if has_gcd:
        sdf["station_g_cd"] = sdf["station_g_cd"].astype(int)
    else:
        # フォールバック: station_cd をそのまま使用
        sdf["station_g_cd"] = sdf["station_cd"].astype(int)

    cd_to_info = dict(zip(
        sdf["station_cd"].astype(int),
        zip(sdf["station_name"], sdf["lat"].astype(float), sdf["lon"].astype(float)),
    ))
    cd_to_gcd = dict(zip(sdf["station_cd"].astype(int), sdf["station_g_cd"].astype(int)))

    # g_cd → (name, lat, lon) 代表情報（最初の出現）
    gcd_info = {}
    for scd, name, lat, lon, gcd in zip(
        sdf["station_cd"], sdf["station_name"], sdf["lat"], sdf["lon"], sdf["station_g_cd"]
    ):
        gcd = int(gcd)
        if gcd not in gcd_info:
            gcd_info[gcd] = (name, float(lat), float(lon))

    # 同名駅の検出: station_g_cd が異なる同名駅（例: 京橋）
    # gcd_info は gcd → (name, lat, lon)
    name_to_gcds = {}  # name → [gcd, ...]
    for gcd, (name, lat, lon) in gcd_info.items():
        name_to_gcds.setdefault(name, []).append(gcd)
    dup_names = {name for name, gcds in name_to_gcds.items() if len(gcds) > 1}

    # 都道府県コード → 都道府県名（短縮形: 東京, 大阪 など）
    PREF_NAMES = {
        1: "北海道", 2: "青森", 3: "岩手", 4: "宮城", 5: "秋田", 6: "山形", 7: "福島",
        8: "茨城", 9: "栃木", 10: "群馬", 11: "埼玉", 12: "千葉", 13: "東京", 14: "神奈川",
        15: "新潟", 16: "富山", 17: "石川", 18: "福井", 19: "山梨", 20: "長野",
        21: "岐阜", 22: "静岡", 23: "愛知", 24: "三重",
        25: "滋賀", 26: "京都", 27: "大阪", 28: "兵庫", 29: "奈良", 30: "和歌山",
        31: "鳥取", 32: "島根", 33: "岡山", 34: "広島", 35: "山口",
        36: "徳島", 37: "香川", 38: "愛媛", 39: "高知",
        40: "福岡", 41: "佐賀", 42: "長崎", 43: "熊本", 44: "大分", 45: "宮崎", 46: "鹿児島", 47: "沖縄",
    }
    # 都道府県コード → 都道府県名（正式名: 住所から除去用）
    PREF_FULL = {
        1:"北海道",2:"青森県",3:"岩手県",4:"宮城県",5:"秋田県",6:"山形県",7:"福島県",
        8:"茨城県",9:"栃木県",10:"群馬県",11:"埼玉県",12:"千葉県",13:"東京都",14:"神奈川県",
        15:"新潟県",16:"富山県",17:"石川県",18:"福井県",19:"山梨県",20:"長野県",
        21:"岐阜県",22:"静岡県",23:"愛知県",24:"三重県",25:"滋賀県",26:"京都府",27:"大阪府",
        28:"兵庫県",29:"奈良県",30:"和歌山県",31:"鳥取県",32:"島根県",33:"岡山県",34:"広島県",
        35:"山口県",36:"徳島県",37:"香川県",38:"愛媛県",39:"高知県",40:"福岡県",41:"佐賀県",
        42:"長崎県",43:"熊本県",44:"大分県",45:"宮崎県",46:"鹿児島県",47:"沖縄県",
    }

    import re

    def _extract_city(address, pref_cd):
        """住所から市区町村を抽出（政令指定都市は市+区）"""
        addr = address
        pref_full = PREF_FULL.get(pref_cd, "")
        if pref_full and addr.startswith(pref_full):
            addr = addr[len(pref_full):]
        m = re.match(r"(.+?市)(.+?区)", addr)
        if m:
            return m.group(1) + m.group(2)
        m = re.match(r"(.+?[市区町村郡])", addr)
        return m.group(1) if m else ""

    # gcd → 都道府県名, 市区町村, 代表路線名
    gcd_to_pref = {}
    gcd_to_city = {}
    gcd_to_line = {}
    for scd, gcd_val, pref_cd, addr, lc in zip(
        sdf["station_cd"].astype(int), sdf["station_g_cd"].astype(int),
        sdf["pref_cd"].astype(int),
        sdf["address"].fillna("").astype(str),
        sdf["line_cd"].astype(int),
    ):
        gv = int(gcd_val)
        if gv not in gcd_to_pref:
            pref_name = PREF_NAMES.get(int(pref_cd), "")
            if pref_name:
                gcd_to_pref[gv] = pref_name
            city = _extract_city(str(addr), int(pref_cd))
            gcd_to_city[gv] = city
            ln = line_names.get(int(lc), "")
            if ln:
                gcd_to_line[gv] = ln

    # station_db: display_name → (lat, lon)
    # 同名駅の区別: 都道府県 → 市区町村 → 路線名 の順にフォールバック
    #  - 異なる都道府県: "京橋（東京）" vs "京橋（大阪）"
    #  - 同一都道府県・異なる市区町村: "柚木（富士市）" vs "柚木（静岡市葵区）"
    #  - 同一市区町村: "琴似（JR函館本線）" vs "琴似（札幌市営地下鉄東西線）"

    # まず同名駅をグループ化して適切なサフィックスを決定
    gcd_suffix = {}  # gcd → suffix string
    for name, gcds in name_to_gcds.items():
        if len(gcds) < 2:
            continue
        # 都道府県で区別できるかチェック
        pref_map = {}  # pref → [gcd, ...]
        for gcd in gcds:
            p = gcd_to_pref.get(gcd, "")
            pref_map.setdefault(p, []).append(gcd)

        for pref, pref_gcds in pref_map.items():
            if len(pref_gcds) == 1:
                # この都道府県に1つだけ → 都道府県名で十分
                gcd_suffix[pref_gcds[0]] = pref
            else:
                # 同一都道府県に複数 → 市区町村で区別を試みる
                city_map = {}  # city → [gcd, ...]
                for gcd in pref_gcds:
                    c = gcd_to_city.get(gcd, "")
                    city_map.setdefault(c, []).append(gcd)

                for city, city_gcds in city_map.items():
                    if len(city_gcds) == 1:
                        # 市区町村で区別できた
                        gcd_suffix[city_gcds[0]] = city if city else pref
                    else:
                        # 同一市区町村 → 路線名にフォールバック
                        for gcd in city_gcds:
                            ln = gcd_to_line.get(gcd, "")
                            gcd_suffix[gcd] = ln if ln else pref

    station_db = {}
    name_to_gcd = {}
    gcd_to_name = {}
    for gcd, (name, lat, lon) in gcd_info.items():
        if name in dup_names:
            suffix = gcd_suffix.get(gcd, "")
            display = f"{name}（{suffix}）" if suffix else name
        else:
            display = name
        # 表示名がまだ衝突する場合はスキップ（最初の出現を優先）
        if display not in station_db:
            station_db[display] = (float(lat), float(lon))
            name_to_gcd[display] = gcd
        gcd_to_name[gcd] = display

    print(f"Loaded {len(station_db)} station names, {len(gcd_info)} physical stations")

    # グラフ構築: g_cd をノードキーに使用
    jdf = pd.read_csv(join_file, dtype={"line_cd": int, "station_cd1": int, "station_cd2": int})
    graph = {}
    edge_lines = {}
    for cd1, cd2, lc in zip(
        jdf["station_cd1"].astype(int).values,
        jdf["station_cd2"].astype(int).values,
        jdf["line_cd"].astype(int).values,
    ):
        if cd1 not in cd_to_info or cd2 not in cd_to_info:
            continue
        gcd1 = cd_to_gcd.get(cd1)
        gcd2 = cd_to_gcd.get(cd2)
        if gcd1 is None or gcd2 is None or gcd1 == gcd2:
            continue
        _, lat1, lon1 = cd_to_info[cd1]
        _, lat2, lon2 = cd_to_info[cd2]
        dist = haversine(lat1, lon1, lat2, lon2)
        time_min = max(round(dist / AVG_TRAIN_SPEED_KMH * 60, 1), 1)
        graph.setdefault(gcd1, []).append((gcd2, time_min))
        graph.setdefault(gcd2, []).append((gcd1, time_min))
        ln = line_names.get(lc, "")
        if ln:
            edge_lines.setdefault((gcd1, gcd2), [])
            if ln not in edge_lines[(gcd1, gcd2)]:
                edge_lines[(gcd1, gcd2)].append(ln)
            edge_lines.setdefault((gcd2, gcd1), [])
            if ln not in edge_lines[(gcd2, gcd1)]:
                edge_lines[(gcd2, gcd1)].append(ln)

    # g_cd をノードキーにしているため、同一 station_g_cd の駅は
    # 自動的に同一ノードになる（乗換コスト0）。
    # 異なる station_g_cd 間の徒歩乗換は不要（路線接続のみ）。

    print(f"Built graph with {len(graph)} nodes, {len(edge_lines)} edge-line mappings")

    sorted_names = sorted(station_db.keys())
    coords = np.array([(station_db[n][0], station_db[n][1]) for n in sorted_names])

    pickle_file = os.path.join(DATA_DIR, "ekidata_cache.pkl")
    with open(pickle_file, "wb") as f:
        pickle.dump((station_db, graph, sorted_names, coords, edge_lines,
                      name_to_gcd, gcd_to_name), f)

    size_mb = os.path.getsize(pickle_file) / 1024 / 1024
    print(f"Saved {pickle_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
