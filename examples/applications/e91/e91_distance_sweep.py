"""
Distance sweep for E91 V3 (NV device + heralded link).

Runs the full pipeline (calibration -> production -> sifting -> raw key
extraction) once per distance and saves one row of metrics per distance
to CSV. The oracle diagnostic is skipped here (it's simulator-only and
already validated separately) to keep the sweep faster.

Run this ONCE, then use e91_plot_latencyVSdistance.py as many times as you want
to iterate on the plot without re-running any simulation.
"""
import numpy
import pandas as pd
import netsquid as ns

# NOTE: adjust this import if your V3 file has a different name/path
# (e.g. "from e91v1_1 import ..." if that's what you called it).
from e91v1_1 import (
    create_nv_heralded_network,
    DEFAULT_CONFIG,
    calibrate_expected_relation,
    run_e91_v2,
    analyze,
    run_key_extraction_pipeline,
)


def run_at_distance(
    distance_km: float,
    n_calib_rounds: int = 500,
    num_rounds: int = 2000,
    sample_fraction: float = 0.25,
) -> dict:
    cfg = create_nv_heralded_network(distance_km=distance_km)
    config = DEFAULT_CONFIG

    expected_relation = calibrate_expected_relation(cfg, config, n_calib_rounds=n_calib_rounds)

    alice_result, bob_result = run_e91_v2(cfg, config, num_rounds)
    stats = analyze(alice_result, bob_result, config, expected_relation)
    pipeline = run_key_extraction_pipeline(
        alice_result, bob_result, config, expected_relation, sample_fraction=sample_fraction
    )

    throughput_bps = None
    if stats["total_elapsed_ns"] and stats["total_elapsed_ns"] > 0:
        throughput_bps = pipeline["n_raw_key"] / (stats["total_elapsed_ns"] * 1e-9)

    avg_ent_ms = stats["avg_entanglement_latency_ns"] / 1e6 if stats["avg_entanglement_latency_ns"] is not None else None
    avg_tot_ms = stats["avg_total_latency_ns"] / 1e6 if stats["avg_total_latency_ns"] is not None else None
    classical_overhead_ms = (avg_tot_ms - avg_ent_ms) if (avg_tot_ms is not None and avg_ent_ms is not None) else None

    return {
        "distance_km": distance_km,
        "n_total": stats["n_total"],
        "n_sifted": stats["n_sifted"],
        "sifted_rate": stats["sifted_rate"],
        "qber": stats["qber"],
        "qber_ci_low": stats["qber_ci_low"],
        "qber_ci_high": stats["qber_ci_high"],
        "S": stats["S"],
        "S_margin": stats["S_margin"],
        "avg_entanglement_latency_ms": avg_ent_ms,
        "avg_total_latency_ms": avg_tot_ms,
        "classical_overhead_ms": classical_overhead_ms,
        "n_raw_key": pipeline["n_raw_key"],
        "qber_estimated": pipeline["qber_estimated"],
        "residual_mismatch_rate": (
            pipeline["residual_mismatches"] / pipeline["n_raw_key"] if pipeline["n_raw_key"] > 0 else None
        ),
        "throughput_bps": throughput_bps,
    }


if __name__ == "__main__":
    # DM formalism is switched on internally, only for the oracle call --
    # not used here since the sweep skips the oracle diagnostic.
    ns.set_qstate_formalism(ns.QFormalism.KET)

    DISTANCES_KM = numpy.linspace(1, 50, 11)  # adjust range/step count as needed

    rows = []
    for dist in DISTANCES_KM:
        print(f"Running distance = {dist:.1f} km ...")
        row = run_at_distance(distance_km=dist)
        rows.append(row)
        print(
            f"  QBER={row['qber']*100:.2f}%  S={row['S']:.3f}  "
            f"latency={row['avg_total_latency_ms']:.4f} ms  "
            f"throughput={row['throughput_bps']:.2f} bit/s"
        )

    df = pd.DataFrame(rows)
    df.to_csv("e91_distance_sweep.csv", index=False)
    print("\nSaved results to e91_distance_sweep.csv")