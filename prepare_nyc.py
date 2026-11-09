"""Build MoST input files for Taxi_NYC from the UCTB pickle (data/taxi_nyc/Taxi_NYC.zip).

Follows the paper setup: 15-minute steps, 2016-01-01 .. 2018-12-31, chronological 7:1:2 split.
POI / satellite features are not available here, so zero placeholders are written; disable those
modalities at run time with  --data-state "[0,0,1,1]"  (order: poi, satellite, location, TS).

Usage: python prepare_nyc.py [--root datasets]
"""
import argparse
import json
import os
import pickle
import zipfile

import numpy as np
import pandas as pd

CITY = "Taxi_NYC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/taxi_nyc/Taxi_NYC.zip")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2019-01-01")  # exclusive
    ap.add_argument("--agg", type=int, default=3, help="5-min steps per output step (3 -> 15 min)")
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as z, z.open(CITY + ".pkl") as f:
        raw = pickle.load(f)

    t0 = pd.Timestamp(raw["TimeRange"][0])
    fit = pd.Timedelta(minutes=raw["TimeFitness"])
    series = raw["Node"]["TrafficNode"]  # [T, N] at 5 min
    stations = raw["Node"]["StationInfo"]  # [id, date, lat, lon, name]

    s = int((pd.Timestamp(args.start) - t0) / fit)
    e = int((pd.Timestamp(args.end) - t0) / fit)
    x = series[s:e].astype(np.float32)
    T, N = x.shape
    T -= T % args.agg
    x = x[:T].reshape(T // args.agg, args.agg, N).sum(1)  # [L, N] at 15 min
    L = len(x)
    times = pd.date_range(args.start, periods=L, freq=fit * args.agg).strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype("U19")
    index = np.array([int(st[0]) for st in stations])
    print(f"series {x.shape}, {times[0]} .. {times[-1]}, mean {x.mean():.3f}")

    out = os.path.join(args.root, "data", CITY)
    os.makedirs(out, exist_ok=True)
    n_tr, n_va = int(L * 0.7), int(L * 0.1)
    parts = {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, L)}
    for name, sl in parts.items():
        np.savez(os.path.join(out, f"{CITY}_{name}1.npz"), data=x[sl], start_time=times[sl], index=index)
        print(name, x[sl].shape, times[sl][0], "->", times[sl][-1])

    # locations, keys in the same order as `index`
    loc = {str(int(st[0])): {"lat": float(st[2]), "lon": float(st[3]), "name": st[4]} for st in stations}
    with open(os.path.join(out, CITY + ".json"), "w") as f:
        json.dump(loc, f, indent=1)

    # Gaussian-kernel adjacency on haversine distance between zone centres
    lat = np.radians([st[2] for st in stations])
    lon = np.radians([st[3] for st in stations])
    dlat = lat[:, None] - lat[None]
    dlon = lon[:, None] - lon[None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))  # km
    sigma = dist[~np.eye(N, dtype=bool)].std()
    adj = np.exp(-(dist / sigma) ** 2).astype(np.float32)
    np.save(os.path.join(out, f"{CITY}_rn_adj.npy"), adj)
    print(f"adj {adj.shape}, sigma {sigma:.2f} km")

    # mean/std from the training split (required by Engine.train even when is_norm=0)
    ms_path = os.path.join(args.root, "data", "mean_std.npy")
    ms = np.load(ms_path, allow_pickle=True).item() if os.path.exists(ms_path) else {}
    ms[CITY] = [float(x[parts["train"]].mean()), float(x[parts["train"]].std())]
    np.save(ms_path, ms, allow_pickle=True)

    # zero placeholders for missing modalities (disabled via state)
    poi_dir = os.path.join(args.root, "poi")
    sat_dir = os.path.join(args.root, "picture", CITY)
    os.makedirs(poi_dir, exist_ok=True)
    os.makedirs(sat_dir, exist_ok=True)
    z768, z1000 = json.dumps([0.0] * 768), json.dumps([0.0] * 1000)
    pd.DataFrame({"list_id": index, "sentence_vector": [z768] * N}).to_csv(
        os.path.join(poi_dir, f"{CITY}_poi_1000_vectors.csv"), index=False)
    pd.DataFrame({"filename": [f"{i}.png" for i in index], "feature_vector": [z1000] * N}).to_csv(
        os.path.join(sat_dir, "image_features.csv"), index=False)
    print("done ->", args.root)


if __name__ == "__main__":
    main()
