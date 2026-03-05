#!/usr/bin/env python3
"""Pre-compute station data pickle for faster startup.

Builds the station database and graph structure only.
Shortest paths are computed on-demand at search time.
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

    sdf = pd.read_csv(station_file, dtype={
        "station_cd": int, "station_name": str,
        "lat": float, "lon": float, "e_status": int, "line_cd": int,
    })
    sdf = sdf[sdf["e_status"] == 0].copy()

    cd_to_info = dict(zip(
        sdf["station_cd"].astype(int),
        zip(sdf["station_name"], sdf["lat"].astype(float), sdf["lon"].astype(float)),
    ))

    station_db = {}
    for name, lat, lon in zip(sdf["station_name"], sdf["lat"], sdf["lon"]):
        if name not in station_db:
            station_db[name] = (float(lat), float(lon))

    print(f"Loaded {len(station_db)} stations")

    # 路線名マッピング
    line_names = dict(zip(ldf["line_cd"].astype(int), ldf["line_name"]))

    jdf = pd.read_csv(join_file, dtype={"line_cd": int, "station_cd1": int, "station_cd2": int})
    graph = {}
    edge_lines = {}  # (駅名1, 駅名2) -> 路線名
    for cd1, cd2, lc in zip(
        jdf["station_cd1"].astype(int).values,
        jdf["station_cd2"].astype(int).values,
        jdf["line_cd"].astype(int).values,
    ):
        if cd1 not in cd_to_info or cd2 not in cd_to_info:
            continue
        name1, lat1, lon1 = cd_to_info[cd1]
        name2, lat2, lon2 = cd_to_info[cd2]
        if name1 == name2:
            continue
        dist = haversine(lat1, lon1, lat2, lon2)
        time_min = max(round(dist / AVG_TRAIN_SPEED_KMH * 60, 1), 1)
        graph.setdefault(name1, []).append((name2, time_min))
        graph.setdefault(name2, []).append((name1, time_min))
        ln = line_names.get(lc, "")
        if ln:
            edge_lines[(name1, name2)] = ln
            edge_lines[(name2, name1)] = ln

    if "station_g_cd" in sdf.columns:
        groups = sdf.groupby("station_g_cd")["station_name"].apply(set)
        for names_set in groups:
            name_list = list(names_set)
            for j in range(len(name_list)):
                for k in range(j + 1, len(name_list)):
                    n1, n2 = name_list[j], name_list[k]
                    graph.setdefault(n1, []).append((n2, 5))
                    graph.setdefault(n2, []).append((n1, 5))
                    edge_lines[(n1, n2)] = "乗換"
                    edge_lines[(n2, n1)] = "乗換"

    print(f"Built graph with {len(graph)} nodes, {len(edge_lines)} edge-line mappings")

    sorted_names = sorted(station_db.keys())
    coords = np.array([(station_db[n][0], station_db[n][1]) for n in sorted_names])

    pickle_file = os.path.join(DATA_DIR, "ekidata_cache.pkl")
    with open(pickle_file, "wb") as f:
        pickle.dump((station_db, graph, sorted_names, coords, edge_lines), f)

    size_mb = os.path.getsize(pickle_file) / 1024 / 1024
    print(f"Saved {pickle_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
