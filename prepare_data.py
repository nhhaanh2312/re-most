import argparse
import json
import os
import pickle
import zipfile

import numpy as np
import pandas as pd

CITIES = {
    "Taxi_NYC": {"zip": "data/taxi_nyc/Taxi_NYC.zip", "start": "2016-01-01", "end": "2019-01-01"},
    "Taxi_Chicago": {"zip": "data/taxi_chi/Taxi_Chicago.zip", "start": "2013-01-01", "end": "2016-01-01"},
}


def prepare(city, root, minutes):
    cfg = CITIES[city]
    with zipfile.ZipFile(cfg["zip"]) as z, z.open(city + ".pkl") as f:
        raw = pickle.load(f)

    t0 = pd.Timestamp(raw["TimeRange"][0])
    fit = pd.Timedelta(minutes=raw["TimeFitness"])
    series = raw["Node"]["TrafficNode"] 
    stations = raw["Node"]["StationInfo"]  
    agg = minutes // raw["TimeFitness"] 

    s = int((pd.Timestamp(cfg["start"]) - t0) / fit)
    e = int((pd.Timestamp(cfg["end"]) - t0) / fit)
    x = series[s:e].astype(np.float32)
    T, N = x.shape
    T -= T % agg
    x = x[:T].reshape(T // agg, agg, N).sum(1)  # [L, N] at the output interval
    L = len(x)
    times = pd.date_range(cfg["start"], periods=L, freq=fit * agg).strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype("U19")
    index = np.array([int(st[0]) for st in stations])
    print(f"[{city}] series {x.shape}, {times[0]} .. {times[-1]}, mean {x.mean():.3f}")

    out = os.path.join(root, "data", city)
    os.makedirs(out, exist_ok=True)
    n_tr, n_va = int(L * 0.7), int(L * 0.1)
    parts = {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, L)}
    for name, sl in parts.items():
        np.savez(os.path.join(out, f"{city}_{name}1.npz"), data=x[sl], start_time=times[sl], index=index)
        print(f"[{city}]", name, x[sl].shape, times[sl][0], "->", times[sl][-1])

    loc = {str(int(st[0])): {"lat": float(st[2]), "lon": float(st[3]), "name": st[4]} for st in stations}
    with open(os.path.join(out, city + ".json"), "w") as f:
        json.dump(loc, f, indent=1)

    lat = np.radians([st[2] for st in stations])
    lon = np.radians([st[3] for st in stations])
    dlat = lat[:, None] - lat[None]
    dlon = lon[:, None] - lon[None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # km
    sigma = dist[~np.eye(N, dtype=bool)].std()
    adj = np.exp(-(dist / sigma) ** 2).astype(np.float32)
    np.save(os.path.join(out, f"{city}_rn_adj.npy"), adj)
    print(f"[{city}] adj {adj.shape}, sigma {sigma:.2f} km")

    ms_path = os.path.join(root, "data", "mean_std.npy")
    ms = np.load(ms_path, allow_pickle=True).item() if os.path.exists(ms_path) else {}
    ms[city] = [float(x[parts["train"]].mean()), float(x[parts["train"]].std())]
    np.save(ms_path, np.array(ms, dtype=object), allow_pickle=True)

    poi_dir = os.path.join(root, "poi")
    sat_dir = os.path.join(root, "picture", city)
    os.makedirs(poi_dir, exist_ok=True)
    os.makedirs(sat_dir, exist_ok=True)
    z768, z1000 = json.dumps([0.0] * 768), json.dumps([0.0] * 1000)
    pd.DataFrame({"list_id": index, "sentence_vector": [z768] * N}).to_csv(
        os.path.join(poi_dir, f"{city}_poi_1000_vectors.csv"), index=False)
    pd.DataFrame({"filename": [f"{i}.png" for i in index], "feature_vector": [z1000] * N}).to_csv(
        os.path.join(sat_dir, "image_features.csv"), index=False)
    print(f"[{city}] done -> {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=[*CITIES, "all"])
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--minutes", type=int, default=15, help="output step in minutes")
    args = ap.parse_args()

    cities = list(CITIES) if args.city == "all" else [args.city]
    for city in cities:
        prepare(city, args.root, args.minutes)


if __name__ == "__main__":
    main()
