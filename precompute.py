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

    # station_db: name → (lat, lon) UI用（最初の出現のみ）
    station_db = {}
    for name, lat, lon in zip(sdf["station_name"], sdf["lat"], sdf["lon"]):
        if name not in station_db:
            station_db[name] = (float(lat), float(lon))

    # name_to_gcd: name → g_cd（UI名から内部IDへの変換、最初の出現）
    name_to_gcd = {}
    for gcd, (name, lat, lon) in gcd_info.items():
        if name not in name_to_gcd:
            name_to_gcd[name] = gcd

    # gcd_to_name: g_cd → name（内部IDからUI名への変換）
    gcd_to_name = {gcd: info[0] for gcd, info in gcd_info.items()}

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
            edge_lines[(gcd1, gcd2)] = ln
            edge_lines[(gcd2, gcd1)] = ln

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
