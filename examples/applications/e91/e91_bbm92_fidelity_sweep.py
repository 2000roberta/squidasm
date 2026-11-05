"""
Fidelity sweep: E91 vs BBM92 on NV device + heralded link, at a fixed distance.

The parameter that is varied is the link's emission_fidelity. The x-axis used
by e91_bbm92_fidelity_plot.py is the DELIVERED fidelity, measured at each point
with the simulator-only oracle (density matrix): it differs from the nominal
parameter and does not depend on the NV model, so it can be compared with other
platforms. The oracle is used only to label the x-axis, never by the protocol.

One CSV row per (emission_fidelity, protocol). The CSV is written after every
run, and runs already in it are skipped, so an interrupted sweep resumes where
it stopped. If you change NUM_ROUNDS or DISTANCE_KM, delete the CSV first.
"""
import os
import time

import pandas as pd
import netsquid as ns

# NOTE: adjust the module name to the file that contains your E91/BBM92 code.
# create_nv_heralded_network must accept the emission_fidelity argument.
from e91v1_1 import (
    PROTOCOLS,
    create_nv_heralded_network,
    run_oracle_diagnostic,
    calibrate_expected_relation,
    run_protocol,
    analyze,
    run_key_extraction_pipeline,
)

OUT_CSV = "e91_bbm92_fidelity_sweep.csv"
DISTANCE_KM = 10.0
EMISSION_FIDELITIES = [0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
NUM_ROUNDS = 10000        # per run; lower it (e.g. 1000) for a quick test
CALIB_ROUNDS = 500
SAMPLE_FRACTION = 0.25
ORACLE_ROUNDS = 20

_fidelity_cache = {}


def delivered_fidelity(cfg, emission_fidelity: float) -> float:
    """Average fidelity of the delivered pair to its closest Bell state (oracle, DM).
    Computed once per emission_fidelity and shared by both protocols."""
    if emission_fidelity not in _fidelity_cache:
        _, _, avg_fidelity = run_oracle_diagnostic(cfg, n_rounds=ORACLE_ROUNDS)
        _fidelity_cache[emission_fidelity] = float(avg_fidelity)
    return _fidelity_cache[emission_fidelity]


def run_one(emission_fidelity: float, protocol: str) -> dict:
    cfg = create_nv_heralded_network(distance_km=DISTANCE_KM, emission_fidelity=emission_fidelity)
    config = PROTOCOLS[protocol]

    fidelity = delivered_fidelity(cfg, emission_fidelity)

    expected_relation = calibrate_expected_relation(cfg, config, n_calib_rounds=CALIB_ROUNDS)
    alice_result, bob_result = run_protocol(cfg, config, NUM_ROUNDS)
    stats = analyze(alice_result, bob_result, config, expected_relation)
    pipe = run_key_extraction_pipeline(
        alice_result, bob_result, config, expected_relation, sample_fraction=SAMPLE_FRACTION
    )

    elapsed_s = stats["total_elapsed_ns"] * 1e-9
    return {
        "protocol": protocol,
        "emission_fidelity": emission_fidelity,
        "fidelity": fidelity,
        "distance_km": DISTANCE_KM,
        "n_rounds": stats["n_total"],
        "n_sifted": stats["n_sifted"],
        # QBER on the whole sifted key, with its Wilson CI
        "qber": stats["qber"],
        "qber_ci_low": stats["qber_ci_low"],
        "qber_ci_high": stats["qber_ci_high"],
        # QBER the protocol would actually estimate: sacrificed sample only
        "qber_sample": pipe["qber_estimated"],
        "qber_sample_ci_low": pipe["qber_estimated_ci_low"],
        "qber_sample_ci_high": pipe["qber_estimated_ci_high"],
        "avg_total_latency_ms": stats["avg_total_latency_ns"] / 1e6,
        "n_raw_key": pipe["n_raw_key"],
        "throughput_bps": pipe["n_raw_key"] / elapsed_s if elapsed_s > 0 else None,
        # CHSH: solo E91 (per BBM92 S è None -> cella vuota nel CSV)
        "s_abs": abs(stats["S"]) if stats["S"] is not None else None,
        "s_margin": stats["S_margin"],
        "s_ci_low": abs(stats["S"]) - stats["S_margin"] if stats["S"] is not None else None,
        "s_ci_high": abs(stats["S"]) + stats["S_margin"] if stats["S"] is not None else None,
    }


def job_key(protocol, emission_fidelity):
    return (protocol, round(float(emission_fidelity), 6))


def load_done() -> set:
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    return {job_key(r.protocol, r.emission_fidelity) for r in df.itertuples()}


def append_row(row: dict):
    pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)


if __name__ == "__main__":
    # DM formalism is switched on only inside run_oracle_diagnostic and restored afterwards.
    ns.set_qstate_formalism(ns.QFormalism.KET)

    jobs = [(p, f) for f in EMISSION_FIDELITIES for p in PROTOCOLS]
    done = load_done()
    todo = [j for j in jobs if job_key(*j) not in done]
    print(f"{len(jobs)} runs in total, {len(jobs) - len(todo)} already in {OUT_CSV}, {len(todo)} to do "
          f"({NUM_ROUNDS} rounds each, {DISTANCE_KM:g} km)")

    for i, (protocol, f) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {protocol:5s} emission_fidelity = {f:.2f}", flush=True)
        t0 = time.time()
        row = run_one(f, protocol)
        append_row(row)
        tp = row["throughput_bps"]
        print(f"    delivered fidelity = {row['fidelity']:.4f}   QBER = {row['qber'] * 100:.2f}%   "
              f"throughput = {'n/a' if tp is None else f'{tp:.1f}'} bit/s   ({time.time() - t0:.0f} s)",
              flush=True)

    print(f"\nDone. Results in {OUT_CSV}")