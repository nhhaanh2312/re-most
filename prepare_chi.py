import argparse
import json
import os
import pickle
import zipfile

import numpy as np
import pandas as pd

CITY = "Taxi_Chicago"


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 moved these classes; NumPy 1 uses numpy.core.
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_mean_std(path):
    if not os.path.exists(path):
        return {}
    try:
        return np.load(path, allow_pickle=True).item()
    except ModuleNotFoundError as error:
        if not error.name.startswith("numpy._core"):
            raise
        with open(path, "rb") as source:
            version = np.lib.format.read_magic(source)
            if version == (1, 0):
                np.lib.format.read_array_header_1_0(source)
            elif version == (2, 0):
                np.lib.format.read_array_header_2_0(source)
            else:
                raise ValueError(f"Unsupported mean/std NPY version: {version}")
            return _NumpyCompatUnpickler(source).load().item()


def _portable_array_reduce(array):
    # Reconstruct via public numpy.ndarray, avoiding numpy._core in the pickle.
    return np.ndarray, (array.shape, array.dtype), array.__reduce__()[2]


def save_mean_std(path, stats):
    array = np.array(stats, dtype=object)
    with open(path, "wb") as target:
        np.lib.format.write_array_header_1_0(
            target, {"descr": "|O", "fortran_order": False, "shape": ()})
        # Python 3.8 and NumPy 1 can also read files produced under NumPy 2.
        writer = pickle.Pickler(target, protocol=3)
        writer.dispatch_table = {np.ndarray: _portable_array_reduce}
        writer.dump(array)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/taxi_chi/Taxi_Chicago.zip")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2016-01-01")  # exclusive
    ap.add_argument("--agg", type=int, default=1, help="15-min steps per output step (default: 1, no aggregation)")
    args = ap.parse_args()
    if args.agg < 1:
        ap.error("--agg must be a positive integer")

    with zipfile.ZipFile(args.zip) as z, z.open(CITY + ".pkl") as f:
        raw = pickle.load(f)

    t0 = pd.Timestamp(raw["TimeRange"][0])
    fit = pd.Timedelta(minutes=raw["TimeFitness"])
    series = raw["Node"]["TrafficNode"]  # [T, N] at 15 min
    stations = raw["Node"]["StationInfo"]  # [id, date, lat, lon, name]

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if not t0 <= start < end <= t0 + len(series) * fit:
        ap.error(f"date range must lie within [{t0}, {t0 + len(series) * fit}] with start < end")
    if (start - t0) % fit or (end - t0) % fit:
        ap.error(f"--start and --end must align to the source interval ({fit})")
    s = int((start - t0) / fit)
    e = int((end - t0) / fit)
    x = series[s:e].astype(np.float32)
    T, N = x.shape
    T -= T % args.agg
    if T // args.agg < 10:
        ap.error("date range must contain at least 10 output steps for a 7:1:2 split")
    x = x[:T].reshape(T // args.agg, args.agg, N).sum(1)  # [L, N] at the output interval
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
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # km
    sigma = dist[~np.eye(N, dtype=bool)].std()
    adj = np.exp(-(dist / sigma) ** 2).astype(np.float32)
    np.save(os.path.join(out, f"{CITY}_rn_adj.npy"), adj)
    print(f"adj {adj.shape}, sigma {sigma:.2f} km")

    # mean/std from the training split (required by Engine.train even when is_norm=0)
    ms_path = os.path.join(args.root, "data", "mean_std.npy")
    ms = load_mean_std(ms_path)
    ms[CITY] = [float(x[parts["train"]].mean()), float(x[parts["train"]].std())]
    save_mean_std(ms_path, ms)

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
