import numpy
import netsquid as ns
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from netsquid.qubits.ketstates import BellIndex
from netsquid_netbuilder.util.fidelity import calculate_fidelity_epr

from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.util import get_qubit_state
from squidasm.run.stack.config import (
    GenericQDeviceConfig,
    LinkConfig,
    StackConfig,
    StackNetworkConfig,
    CLinkConfig,
    DefaultCLinkConfig,
)

# ============================================================
# E91 - VERSION 2
#
# 1. AliceProgram / BobProgram do ONLY quantum operations + basis
#    exchange. They have no notion of "key" / "test" / "discard" --
#    that classification is sifting POLICY, decided later, from the
#    publicly exchanged basis indices, by whoever is doing the
#    post-processing (here: the analysis functions below). Neither
#    node needs to know in real time what a round will be used for.
#
# 2. Basis exchange is a single symmetric function, exchange_basis(),
#    used identically by both Alice and Bob. No assumption about which
#    side the classical channel serves first.
#
# 3. All protocol-specific parameters (measurement angles, which basis
#    pairs are "key" vs "test") live in a ProtocolConfig object passed
#    into the programs and into analysis, instead of module-level
#    globals -- V3 can change angles/pairs without touching the
#    programs themselves.
#
# 4. Bit comparison uses a pure predicate, expected_equal(a, b, parity),
#    instead of mutating Bob's outcome. Raw outcomes are never modified.
# ============================================================


@dataclass(frozen=True)
class ProtocolConfig:
    alice_angles: Tuple[float, ...]
    bob_angles: Tuple[float, ...]
    key_pairs: Tuple[Tuple[int, int], ...]
    test_pairs: Tuple[Tuple[int, int], ...]

    def classify(self, a_idx: int, b_idx: int) -> str:
        pair = (a_idx, b_idx)
        if pair in self.key_pairs:
            return "key"
        elif pair in self.test_pairs:
            return "test"
        return "discard"


DEFAULT_CONFIG = ProtocolConfig(
    alice_angles=(0.0, numpy.pi / 4, numpy.pi / 2),
    bob_angles=(numpy.pi / 4, numpy.pi / 2, 3 * numpy.pi / 4),
    key_pairs=((1, 0), (2, 1)),
    test_pairs=((0, 0), (0, 2), (2, 0), (2, 2)),
)


def expected_equal(a: int, b: int, parity: int) -> bool:
    """Pure predicate: does (a, b) match the calibrated expected relation?

    parity == 1  -> Bell state gives correlated outcomes, expect a == b
    parity == -1 -> Bell state gives anti-correlated outcomes, expect a != b

    Never mutates either bit. Both raw outcomes stay untouched so later
    stages (error correction, etc.) can still work from original data.
    """
    if parity == 1:
        return a == b
    else:
        return a != b


def create_perfect_network() -> StackNetworkConfig:
    """Noiseless generic device + noiseless link: pure protocol validation."""
    qdevice_cfg = GenericQDeviceConfig.perfect_config()
    alice_stack = StackConfig(name="Alice", qdevice_typ="generic", qdevice_cfg=qdevice_cfg)
    bob_stack = StackConfig(name="Bob", qdevice_typ="generic", qdevice_cfg=qdevice_cfg)
    link = LinkConfig.perfect_config(stack1="Alice", stack2="Bob")
    clink = CLinkConfig(
        stack1="Alice", stack2="Bob",
        typ="default",
        cfg=DefaultCLinkConfig(delay=0.0),
    )
    return StackNetworkConfig(stacks=[alice_stack, bob_stack], links=[link], clinks=[clink])


def exchange_basis(csocket, my_idx: int):
    """Symmetric public basis announcement.

    Both Alice and Bob call this SAME function with their own index and
    get back the peer's. Send is fire-and-forget, recv blocks until the
    peer's message shows up -- send-then-recv on both ends is safe
    regardless of any ordering guarantee (or lack of one) in the
    underlying classical channel, and there is no separate code path
    per role to keep in sync.
    """
    csocket.send(str(my_idx))
    peer_idx = int((yield from csocket.recv()))
    return peer_idx


# ============================================================
# PRODUCTION PROGRAMS - quantum operations + basis exchange only.
# No sifting policy, no notion of "key"/"test"/"discard" round.
# ============================================================

class AliceProgram(Program):
    PEER_NAME = "Bob"

    def __init__(self, config: ProtocolConfig):
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.config = config

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_program",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        csocket = context.csockets[self.PEER_NAME]
        connection = context.connection

        a_idx = int(numpy.random.randint(0, len(self.config.alice_angles)))

        epr = epr_socket.create_keep()[0]
        epr.rot_Y(angle=self.config.alice_angles[a_idx])
        a = epr.measure()
        yield from connection.flush()

        b_idx = yield from exchange_basis(csocket, a_idx)

        return {"basis": a_idx, "peer_basis": b_idx, "outcome": int(a)}


class BobProgram(Program):
    PEER_NAME = "Alice"

    def __init__(self, config: ProtocolConfig):
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.config = config

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_program",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        csocket = context.csockets[self.PEER_NAME]
        connection = context.connection

        b_idx = int(numpy.random.randint(0, len(self.config.bob_angles)))

        epr = epr_socket.recv_keep()[0]
        epr.rot_Y(angle=self.config.bob_angles[b_idx])
        b = epr.measure()
        yield from connection.flush()

        a_idx = yield from exchange_basis(csocket, b_idx)

        return {"basis": b_idx, "peer_basis": a_idx, "outcome": int(b)}


# ============================================================
# ORACLE / CALIBRATION-ONLY PROGRAMS (unchanged principle from V1)
# ============================================================

class _OracleAliceProgram(Program):
    PEER_NAME = "Bob"

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="oracle_alice_program",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        epr = epr_socket.create_keep()[0]
        yield from connection.flush()

        dm = get_qubit_state(epr, node_name="Alice", full_state=True)
        fidelities = {idx: calculate_fidelity_epr(dm, idx) for idx in BellIndex}
        best_idx = max(fidelities, key=fidelities.get)

        return {"bell_index": str(best_idx), "fidelity": fidelities[best_idx]}


class _OracleBobProgram(Program):
    PEER_NAME = "Alice"

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="oracle_bob_program",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        epr_socket.recv_keep()
        yield from connection.flush()

        return {}


def run_oracle_diagnostic(cfg: StackNetworkConfig, n_rounds: int = 20):
    """Simulator-only diagnostic. Never used by production analysis."""
    alice_result, _ = run(
        config=cfg,
        programs={"Alice": _OracleAliceProgram(), "Bob": _OracleBobProgram()},
        num_times=n_rounds,
    )
    counts: Dict[str, int] = {}
    for r in alice_result:
        counts[r["bell_index"]] = counts.get(r["bell_index"], 0) + 1
    dominant = max(counts, key=counts.get)
    avg_fidelity = numpy.mean([r["fidelity"] for r in alice_result])
    return dominant, counts, avg_fidelity


# ============================================================
# SIFTING / ANALYSIS - all protocol policy lives here, parametrized
# by ProtocolConfig. Alice/Bob programs know nothing of this.
# ============================================================

def calibrate_expected_relation(
    cfg: StackNetworkConfig, config: ProtocolConfig, n_calib_rounds: int = 300
) -> Dict[Tuple[int, int], int]:
    """Empirical, oracle-free calibration of same/anti-correlated parity
    for each key-pair basis combination, via majority vote."""
    alice_result, bob_result = run(
        config=cfg,
        programs={"Alice": AliceProgram(config), "Bob": BobProgram(config)},
        num_times=n_calib_rounds,
    )

    expected_relation: Dict[Tuple[int, int], int] = {}
    for pair in config.key_pairs:
        same = diff = 0
        for ar, br in zip(alice_result, bob_result):
            if (ar["basis"], br["basis"]) == pair:
                if ar["outcome"] == br["outcome"]:
                    same += 1
                else:
                    diff += 1
        if same + diff > 0:
            expected_relation[pair] = 1 if same >= diff else -1

    return expected_relation


def run_e91_v2(cfg: StackNetworkConfig, config: ProtocolConfig, num_rounds: int):
    return run(
        config=cfg,
        programs={"Alice": AliceProgram(config), "Bob": BobProgram(config)},
        num_times=num_rounds,
    )


def analyze(
    alice_result,
    bob_result,
    config: ProtocolConfig,
    expected_relation: Dict[Tuple[int, int], int],
):
    n_total = len(alice_result)
    n_sifted = n_test = n_discarded = 0
    matches = 0
    correlations_data = {pair: {"same": 0, "diff": 0} for pair in config.test_pairs}

    for ar, br in zip(alice_result, bob_result):
        pair = (ar["basis"], br["basis"])
        round_type = config.classify(*pair)

        if round_type == "key":
            n_sifted += 1
            parity = expected_relation.get(pair)
            # NOTE: comparing Alice's and Bob's outcome for the full raw
            # key here (not just a sacrificed subsample) is a
            # benchmarking simplification -- we're simulating both
            # nodes, so the driver "sees" both sides. A real deployment
            # would only reveal a random subsample for this estimate.
            if parity is not None and expected_equal(ar["outcome"], br["outcome"], parity):
                matches += 1
        elif round_type == "test":
            n_test += 1
            d = correlations_data[pair]
            if ar["outcome"] == br["outcome"]:
                d["same"] += 1
            else:
                d["diff"] += 1
        else:
            n_discarded += 1

    if n_sifted > 0:
        qber = 1 - matches / n_sifted
        qber_margin = 1.96 * numpy.sqrt(qber * (1 - qber) / n_sifted)
    else:
        qber, qber_margin = None, None

    correlations, n_per_pair = {}, {}
    for pair in config.test_pairs:
        d = correlations_data[pair]
        n = d["same"] + d["diff"]
        n_per_pair[pair] = n
        correlations[pair] = (d["same"] - d["diff"]) / n if n > 0 else None

    if all(v is not None for v in correlations.values()):
        S = (
            correlations[config.test_pairs[0]]
            - correlations[config.test_pairs[1]]
            + correlations[config.test_pairs[2]]
            + correlations[config.test_pairs[3]]
        )
        var_S = sum(
            (1 - correlations[p] ** 2) / n_per_pair[p]
            for p in config.test_pairs if n_per_pair[p] > 0
        )
        S_margin = 1.96 * numpy.sqrt(var_S)
    else:
        S, S_margin = None, None

    return {
        "n_total": n_total,
        "n_sifted": n_sifted,
        "n_test": n_test,
        "n_discarded": n_discarded,
        "sifted_rate": n_sifted / n_total if n_total else 0.0,
        "qber": qber,
        "qber_margin": qber_margin,
        "S": S,
        "S_margin": S_margin,
    }


# ============================================================
# V2.5 - RAW KEY EXTRACTION PIPELINE
#
# Scope: parameter estimation on a sacrificed random sample, and
# construction of the final aligned raw key from the remaining bits.
# Error correction (Cascade) and privacy amplification are explicitly
# OUT of scope here -- they're V4 territory. What comes out of this
# pipeline is a "raw key": same length on both sides, parity-aligned,
# but not yet guaranteed identical (residual QBER) and not yet
# information-theoretically secret.
# ============================================================

def extract_key_rounds(alice_result, bob_result, config: ProtocolConfig):
    """Raw (a_bit, b_bit, basis_pair) triples for KEY_PAIRS rounds only.
    Nothing is corrected or discarded here -- this is the untouched
    private data each side has right after the quantum phase."""
    key_rounds = []
    for ar, br in zip(alice_result, bob_result):
        pair = (ar["basis"], br["basis"])
        if config.classify(*pair) == "key":
            key_rounds.append((ar["outcome"], br["outcome"], pair))
    return key_rounds


def sample_for_estimation(key_rounds, sample_fraction: float):
    """Randomly split the raw key rounds into a sacrificed sample (used
    only to estimate QBER, then discarded) and the remaining rounds
    that become the actual raw key. Indices are used so this could be
    done via a public announcement of indices over a classical channel
    in a real deployment -- here it's just index bookkeeping."""
    n = len(key_rounds)
    n_sample = max(1, int(round(n * sample_fraction)))
    indices = numpy.random.permutation(n)
    sample_idx = set(indices[:n_sample].tolist())

    sample = [kr for i, kr in enumerate(key_rounds) if i in sample_idx]
    remaining = [kr for i, kr in enumerate(key_rounds) if i not in sample_idx]
    return sample, remaining


def estimate_qber(sample, expected_relation: Dict[Tuple[int, int], int]):
    """QBER estimated ONLY from the sacrificed sample -- this is the
    number a real protocol would actually have available, since the
    remaining (secret) bits are never compared directly."""
    n = 0
    mismatches = 0
    for a_bit, b_bit, pair in sample:
        parity = expected_relation.get(pair)
        if parity is None:
            continue
        n += 1
        if not expected_equal(a_bit, b_bit, parity):
            mismatches += 1

    if n == 0:
        return None, None
    qber = mismatches / n
    qber_margin = 1.96 * numpy.sqrt(qber * (1 - qber) / n)
    return qber, qber_margin


def build_raw_key(remaining, expected_relation: Dict[Tuple[int, int], int]):
    """Build the final aligned raw key from the non-sacrificed rounds.

    Bob's bits are realigned to Alice's convention using the calibrated
    parity -- this is a DELIBERATE, explicit, one-time transformation to
    produce a usable key, not a silent in-place mutation of the raw
    per-round data (extract_key_rounds() above still returns the
    untouched originals). Returns two same-length lists; they are not
    guaranteed identical (residual QBER is exactly what error
    correction, out of scope here, would fix).
    """
    alice_key, bob_key_aligned = [], []
    for a_bit, b_bit, pair in remaining:
        parity = expected_relation.get(pair)
        if parity is None:
            continue
        b_aligned = b_bit if parity == 1 else 1 - b_bit
        alice_key.append(a_bit)
        bob_key_aligned.append(b_aligned)
    return alice_key, bob_key_aligned


def run_key_extraction_pipeline(
    alice_result, bob_result, config: ProtocolConfig,
    expected_relation: Dict[Tuple[int, int], int],
    sample_fraction: float = 0.15,
):
    key_rounds = extract_key_rounds(alice_result, bob_result, config)
    sample, remaining = sample_for_estimation(key_rounds, sample_fraction)

    qber_est, qber_est_margin = estimate_qber(sample, expected_relation)
    alice_key, bob_key_aligned = build_raw_key(remaining, expected_relation)

    # Simulation-only cross-check: residual mismatches in the FINAL raw
    # key. A real deployment cannot compute this without giving up the
    # key's secrecy -- it exists here purely to validate the pipeline.
    residual_mismatches = sum(1 for a, b in zip(alice_key, bob_key_aligned) if a != b)

    return {
        "n_key_rounds": len(key_rounds),
        "n_sacrificed": len(sample),
        "n_raw_key": len(alice_key),
        "qber_estimated": qber_est,
        "qber_estimated_margin": qber_est_margin,
        "residual_mismatches": residual_mismatches,
        "alice_key": alice_key,
        "bob_key_aligned": bob_key_aligned,
    }


if __name__ == "__main__":
    ns.set_qstate_formalism(ns.QFormalism.DM)

    cfg = create_perfect_network()
    config = DEFAULT_CONFIG

    print("=" * 60)
    print("ORACLE DIAGNOSTIC (simulator-only, not used by production code)")
    print("=" * 60)
    dominant_bell_state, bell_state_counts, avg_fid = run_oracle_diagnostic(cfg, n_rounds=20)
    print(f"Dominant Bell state observed: {dominant_bell_state} {bell_state_counts}")
    print(f"Average fidelity to dominant state: {avg_fid:.4f}")
    print()

    print("=" * 60)
    print("CALIBRATION (empirical, no oracle access)")
    print("=" * 60)
    expected_relation = calibrate_expected_relation(cfg, config, n_calib_rounds=300)
    for pair, parity in expected_relation.items():
        label = "correlated (a==b)" if parity == 1 else "anti-correlated (a!=b)"
        print(f"Basis pair {pair}: {label}")
    print()

    print("=" * 60)
    print("E91 V2 - Classical basis exchange, sifting as post-processing")
    print("Expected: |S| ~= 2.828, QBER ~= 0%")
    print("=" * 60)

    NUM_ROUNDS = 1000
    alice_result, bob_result = run_e91_v2(cfg, config, NUM_ROUNDS)
    stats = analyze(alice_result, bob_result, config, expected_relation)

    print(f"Total rounds:       {stats['n_total']}")
    print(f"Sifted key length:  {stats['n_sifted']} ({stats['sifted_rate']*100:.1f}% of rounds)")
    print(f"CHSH test rounds:   {stats['n_test']}")
    print(f"Discarded rounds:   {stats['n_discarded']}")
    if stats["qber"] is not None:
        print(f"QBER:  {stats['qber']*100:.2f}% +/- {stats['qber_margin']*100:.2f}%")
    if stats["S"] is not None:
        print(f"CHSH S: {stats['S']:.4f} +/- {stats['S_margin']:.4f}  (|S| target: 2.8284)")

    if stats["S"] is not None and abs(abs(stats["S"]) - 2 * numpy.sqrt(2)) > 3 * stats["S_margin"]:
        print("\nWARNING: |S| is far from 2*sqrt(2) beyond the statistical margin.")
        print("Check the basis exchange logic before moving to V3.")

    print()
    print("=" * 60)
    print("V2.5 - Raw key extraction (parameter estimation + alignment)")
    print("Out of scope here: error correction, privacy amplification (V4)")
    print("=" * 60)

    pipeline = run_key_extraction_pipeline(
        alice_result, bob_result, config, expected_relation, sample_fraction=0.15
    )

    print(f"Key rounds (pre-sampling): {pipeline['n_key_rounds']}")
    print(f"Sacrificed for estimation: {pipeline['n_sacrificed']}")
    print(f"Raw key length:            {pipeline['n_raw_key']}")
    if pipeline["qber_estimated"] is not None:
        print(f"QBER (from sample only):   {pipeline['qber_estimated']*100:.2f}% +/- {pipeline['qber_estimated_margin']*100:.2f}%")
    print(f"Residual mismatches in raw key (sim-only cross-check): {pipeline['residual_mismatches']}/{pipeline['n_raw_key']}")

    if pipeline["residual_mismatches"] == 0:
        print("Raw key identical on both sides (expected with no noise -- V1/V2 baseline).")
    else:
        print("Raw key NOT yet identical -- this residual gap is exactly what error correction (V4) resolves.")