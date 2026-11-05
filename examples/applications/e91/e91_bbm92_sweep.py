"""
Sweep E91 vs BBM92, on a noiseless network and on NV device + heralded link.

One CSV row per run. The noiseless network has no distance parameter, so it is
run ONCE per protocol (distance_km left empty) and plotted as a reference line.

The CSV is written after every run, and runs already present in it are
skipped: if the script is interrupted, launching it again resumes where it
stopped. To start from scratch (e.g. after changing NUM_ROUNDS), delete the CSV.

Use e91_bbm92_plot.py to plot; it only reads the CSV and can be run on a
partial file too.
"""
import os
import time

import numpy
import pandas as pd
import netsquid as ns

# NOTE: adjust the module name to the file that contains your E91/BBM92 code.
from e91v1_1 import (
    PROTOCOLS,
    create_perfect_network,
    create_nv_heralded_network,
    calibrate_expected_relation,
    run_protocol,
    analyze,
    run_key_extraction_pipeline,
)

OUT_CSV = "e91_bbm92_sweep.csv"
DISTANCES_KM = numpy.linspace(1, 50, 11)
NUM_ROUNDS = 10000        # per run; lower it (e.g. 1000) for a quick test
CALIB_ROUNDS = 500
SAMPLE_FRACTION = 0.25


def run_one(network: str, protocol: str, distance_km=None) -> dict:
    if network == "noiseless":
        cfg = create_perfect_network()
    else:
        cfg = create_nv_heralded_network(distance_km=distance_km)
    config = PROTOCOLS[protocol]

    expected_relation = calibrate_expected_relation(cfg, config, n_calib_rounds=CALIB_ROUNDS)
    alice_result, bob_result = run_protocol(cfg, config, NUM_ROUNDS)
    stats = analyze(alice_result, bob_result, config, expected_relation)
    pipe = run_key_extraction_pipeline(
        alice_result, bob_result, config, expected_relation, sample_fraction=SAMPLE_FRACTION
    )

    elapsed_s = stats["total_elapsed_ns"] * 1e-9
    return {
        "network": network,
        "protocol": protocol,
        "distance_km": distance_km,
        "n_rounds": stats["n_total"],
        "n_sifted": stats["n_sifted"],
        "sifted_rate": stats["sifted_rate"],
        # QBER on the whole sifted key (more statistics) with its Wilson CI
        "qber": stats["qber"],
        "qber_ci_low": stats["qber_ci_low"],
        "qber_ci_high": stats["qber_ci_high"],
        # QBER the protocol would actually estimate, from the sacrificed sample only
        "qber_sample": pipe["qber_estimated"],
        "qber_sample_ci_low": pipe["qber_estimated_ci_low"],
        "qber_sample_ci_high": pipe["qber_estimated_ci_high"],
        # None (empty in the CSV) for BBM92, which has no CHSH test
        "S": stats["S"],
        "S_margin": stats["S_margin"],
        "avg_entanglement_latency_ms": stats["avg_entanglement_latency_ns"] / 1e6,
        "avg_total_latency_ms": stats["avg_total_latency_ns"] / 1e6,
        "n_raw_key": pipe["n_raw_key"],
        "throughput_bps": pipe["n_raw_key"] / elapsed_s if elapsed_s > 0 else None,
    }


def job_key(network, protocol, distance_km):
    return (network, protocol, None if distance_km is None else round(float(distance_km), 6))


def load_done() -> set:
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    return {
        job_key(r.network, r.protocol, None if pd.isna(r.distance_km) else r.distance_km)
        for r in df.itertuples()
    }


def append_row(row: dict):
    pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)


def build_jobs():
    # Noiseless references first (cheap), then each distance for both protocols,
    # so a partial CSV always has both protocols at the same distances.
    jobs = [("noiseless", protocol, None) for protocol in PROTOCOLS]
    for d in DISTANCES_KM:
        jobs += [("nv", protocol, float(d)) for protocol in PROTOCOLS]
    return jobs


if __name__ == "__main__":
    ns.set_qstate_formalism(ns.QFormalism.KET)

    jobs = build_jobs()
    done = load_done()
    todo = [j for j in jobs if job_key(*j) not in done]
    print(f"{len(jobs)} runs in total, {len(jobs) - len(todo)} already in {OUT_CSV}, {len(todo)} to do "
          f"({NUM_ROUNDS} rounds each)")

    for i, (network, protocol, d) in enumerate(todo, 1):
        where = "" if d is None else f"{d:g} km"
        print(f"[{i}/{len(todo)}] {network:9s} {protocol:5s} {where}", flush=True)
        t0 = time.time()
        row = run_one(network, protocol, d)
        append_row(row)
        tp = row["throughput_bps"]
        print(f"    QBER = {row['qber'] * 100:.2f}%   "
              f"throughput = {'n/a' if tp is None else f'{tp:.1f}'} bit/s   "
              f"({time.time() - t0:.0f} s)", flush=True)

    print(f"\nDone. Results in {OUT_CSV}")