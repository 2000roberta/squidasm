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
    NVQDeviceConfig,
    HeraldedLinkConfig,
    LinkConfig,
    StackConfig,
    StackNetworkConfig,
    CLinkConfig,
    DefaultCLinkConfig,
)

# ============================================================
# E91 / BBM92 on SquidASM.
# The protocol is selected by a ProtocolConfig (measurement angles,
# "key" pairs, "test" pairs). AliceProgram / BobProgram do only quantum
# operations + basis exchange and do not know which protocol is running;
# sifting is post-processing.
# ============================================================


@dataclass(frozen=True)
class ProtocolConfig:
    name: str
    alice_angles: Tuple[float, ...]
    bob_angles: Tuple[float, ...]
    key_pairs: Tuple[Tuple[int, int], ...]
    test_pairs: Tuple[Tuple[int, int], ...]
    # clifford_safe: bool = False   # True solo se ogni correlatore usato è a basi accoppiate

    @property
    def has_bell_test(self) -> bool:
        # CHSH needs exactly 4 correlators (2 settings per side)
        return len(self.test_pairs) == 4

    def classify(self, a_idx: int, b_idx: int) -> str:
        pair = (a_idx, b_idx)
        if pair in self.key_pairs:
            return "key"
        elif pair in self.test_pairs:
            return "test"
        return "discard"


E91_CONFIG = ProtocolConfig(
    name="e91",
    alice_angles=(0.0, numpy.pi / 4, numpy.pi / 2),
    bob_angles=(numpy.pi / 4, numpy.pi / 2, 3 * numpy.pi / 4),
    key_pairs=((1, 0), (2, 1)),
    test_pairs=((0, 0), (0, 2), (2, 0), (2, 2)),
)

# BBM92: two bases (Z and X) on both sides; same basis -> key, different -> discard.
BBM92_CONFIG = ProtocolConfig(
    name="bbm92",
    alice_angles=(0.0, numpy.pi / 2),
    bob_angles=(0.0, numpy.pi / 2),
    key_pairs=((0, 0), (1, 1)),
    test_pairs=(),
    # clifford_safe: bool = False   # True solo se ogni correlatore usato è a basi accoppiate
)

PROTOCOLS = {"e91": E91_CONFIG, "bbm92": BBM92_CONFIG}
DEFAULT_CONFIG = E91_CONFIG  # kept so existing imports (sweep script) keep working


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion.

    Unlike the normal-approximation interval (z * sqrt(p(1-p)/n)), this
    does NOT collapse to a zero-width margin when k=0 or k=n -- exactly
    the failure mode that gave a false "QBER: 0.00% +/- 0.00%" from a
    32-sample batch with 0 observed mismatches, when the true rate
    (confirmed against the full raw key) was ~9%. Returns
    (point_estimate, lower_bound, upper_bound).
    """
    if n == 0:
        return None, None, None
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half_width = (z * numpy.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
    return phat, max(0.0, center - half_width), min(1.0, center + half_width)


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
    """Noiseless generic device + noiseless link: kept from V1/V2 as a
    reference point -- run it alongside V3's noisy config to confirm the
    pipeline itself hasn't regressed, isolating any change in behavior
    to the noise sources introduced below."""
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


def create_nv_heralded_network(distance_km: float = 10.0) -> StackNetworkConfig:
    """NV center device + heralded double-click link -- the realistic
    noise sources: imperfect gates/measurement on the NV device, photon
    loss and emission infidelity on the heralded link. The link can
    herald different Bell states depending on which detector clicks
    first (B00/B01/B10/B11) -- this is exactly why the empirical,
    majority-vote calibration from V1/V2 matters here instead of a
    hardcoded parity: if the link produces a STATISTICAL MIX of Bell
    states rather than one dominant state, that mix itself becomes part
    of the noise the majority vote correctly absorbs into a higher QBER
    for the affected basis pair, without needing to know per-round which
    state was actually produced.
    """
    nv_cfg = NVQDeviceConfig()
    alice_stack = StackConfig(name="Alice", qdevice_typ="nv", qdevice_cfg=nv_cfg)
    bob_stack = StackConfig(name="Bob", qdevice_typ="nv", qdevice_cfg=nv_cfg)

    herald_cfg = HeraldedLinkConfig(
        length=distance_km,
        emission_fidelity=0.95,
    )
    link = LinkConfig(stack1="Alice", stack2="Bob", typ="heralded", cfg=herald_cfg)
    clink = CLinkConfig(
        stack1="Alice", stack2="Bob",
        typ="default",
        cfg=DefaultCLinkConfig(delay=distance_km / 200000 * 1e9),
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

# def apply_basis_change(q, angle: float):
    """Rotate q so a Z-measurement measures in the basis at `angle`.
    0 -> no gate; pi/2 -> H (both Clifford, so BBM92 needs no arbitrary
    rotation). Any other angle (E91) falls back to rot_Y."""
    if numpy.isclose(angle, 0.0):
        return
    if numpy.isclose(angle, numpy.pi / 2):
        q.H()
    else:
        q.rot_Y(angle=angle)

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

        t_start = ns.sim_time()

        a_idx = int(numpy.random.randint(0, len(self.config.alice_angles)))

        epr = epr_socket.create_keep()[0]
        epr.rot_Y(angle=self.config.alice_angles[a_idx])
        # if self.config.clifford_safe:
        #    apply_basis_change(epr, self.config.alice_angles[a_idx])
        # else:
        #    epr.rot_Y(angle=self.config.alice_angles[a_idx])

        a = epr.measure()
        yield from connection.flush()

        t_entanglement_done = ns.sim_time()

        b_idx = yield from exchange_basis(csocket, a_idx)

        t_round_done = ns.sim_time()

        return {
            "basis": a_idx,
            "peer_basis": b_idx,
            "outcome": int(a),
            "entanglement_latency_ns": t_entanglement_done - t_start,
            "total_latency_ns": t_round_done - t_start,
        }


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

        t_start = ns.sim_time()

        b_idx = int(numpy.random.randint(0, len(self.config.bob_angles)))

        epr = epr_socket.recv_keep()[0]
        epr.rot_Y(angle=self.config.bob_angles[b_idx])
        # if self.config.clifford_safe:
        #    apply_basis_change(epr, self.config.bob_angles[b_idx])
        # else:
        #    epr.rot_Y(angle=self.config.bob_angles[b_idx])
        b = epr.measure()
        yield from connection.flush()

        t_entanglement_done = ns.sim_time()

        a_idx = yield from exchange_basis(csocket, b_idx)

        t_round_done = ns.sim_time()

        return {
            "basis": b_idx,
            "peer_basis": a_idx,
            "outcome": int(b),
            "entanglement_latency_ns": t_entanglement_done - t_start,
            "total_latency_ns": t_round_done - t_start,
        }


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

        # Discard measurement: the outcome is unused, this exists ONLY to
        # free the qubit from simulated quantum memory. Without it, qubits
        # from every round of num_times accumulate in the DM-tracked joint
        # state, and the density matrix blows up in size round over round
        # (fine for 2 rounds, pathological by round ~20) -- this is what
        # caused the oracle diagnostic to hang.
        epr.measure()
        yield from connection.flush()

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

        epr = epr_socket.recv_keep()[0]
        yield from connection.flush()

        # Same fix as the Alice side: discard-measure to free the qubit
        # from simulated memory instead of leaving it allocated across
        # every round of num_times.
        epr.measure()
        yield from connection.flush()

        return {}


def run_oracle_diagnostic(cfg: StackNetworkConfig, n_rounds: int = 20):
    """Simulator-only diagnostic. Never used by production analysis.

    DM formalism is only needed HERE (density-matrix based fidelity
    extraction). Calibration and production runs never call
    get_qubit_state, so they run in the default (much cheaper) KET
    formalism -- switching only for this call keeps the bulk of the
    simulation (500-1000+ rounds) fast, especially with a noisy NV
    device + heralded link where DM tracking is expensive.
    """
    ns.set_qstate_formalism(ns.QFormalism.DM)
    alice_result, _ = run(
        config=cfg,
        programs={"Alice": _OracleAliceProgram(), "Bob": _OracleBobProgram()},
        num_times=n_rounds,
    )
    ns.set_qstate_formalism(ns.QFormalism.KET)

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
    cfg: StackNetworkConfig, config: ProtocolConfig, n_calib_rounds: int = 1500
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


def run_protocol(cfg: StackNetworkConfig, config: ProtocolConfig, num_rounds: int):
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

    # Latency: summed (not averaged) total_latency_ns across ALL rounds --
    # this is the real elapsed time the batch took, used below to derive
    # throughput (raw key bits per second), not just per-round averages.
    entanglement_latencies_ns = []
    total_latencies_ns = []

    for ar, br in zip(alice_result, bob_result):
        pair = (ar["basis"], br["basis"])
        round_type = config.classify(*pair)

        entanglement_latencies_ns.append(ar["entanglement_latency_ns"])
        total_latencies_ns.append(ar["total_latency_ns"])

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
        mismatches = n_sifted - matches
        qber, qber_ci_low, qber_ci_high = wilson_interval(mismatches, n_sifted)
    else:
        qber, qber_ci_low, qber_ci_high = None, None, None

    correlations, n_per_pair = {}, {}
    for pair in config.test_pairs:
        d = correlations_data[pair]
        n = d["same"] + d["diff"]
        n_per_pair[pair] = n
        correlations[pair] = (d["same"] - d["diff"]) / n if n > 0 else None

    if config.has_bell_test and all(v is not None for v in correlations.values()):
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
        "qber_ci_low": qber_ci_low,
        "qber_ci_high": qber_ci_high,
        "S": S,
        "S_margin": S_margin,
        "avg_entanglement_latency_ns": numpy.mean(entanglement_latencies_ns) if entanglement_latencies_ns else None,
        "avg_total_latency_ns": numpy.mean(total_latencies_ns) if total_latencies_ns else None,
        "total_elapsed_ns": sum(total_latencies_ns),
    }

QBER_THRESHOLD = 0.11  # asymptotic one-way threshold: 1 - 2*h2(Q) = 0 at Q ~ 11%


def security_check(stats, config: ProtocolConfig):
    """Returns (ok, message).

    E91:   Bell violation must be established, i.e. the lower bound of the
           95% CI on |S| must exceed 2.
    BBM92: there is no S; the upper bound of the 95% CI on the QBER must stay
           below the asymptotic threshold (trusted-device assumption).
    """
    if config.has_bell_test:
        if stats["S"] is None:
            return False, "S not computable (missing test rounds)"
        lower = abs(stats["S"]) - stats["S_margin"]
        if lower > 2:
            return True, f"Bell violation established (CI lower bound on |S| = {lower:.3f} > 2)"
        return False, (f"CI lower bound on |S| = {lower:.3f} <= 2: Bell violation NOT confidently "
                       f"established, key not certifiable without more test rounds")
    if stats["qber_ci_high"] is None:
        return False, "QBER not computable (no key rounds)"
    if stats["qber_ci_high"] < QBER_THRESHOLD:
        return True, (f"QBER upper bound {stats['qber_ci_high']*100:.2f}% is below the "
                      f"{QBER_THRESHOLD*100:.0f}% threshold")
    return False, (f"QBER upper bound {stats['qber_ci_high']*100:.2f}% is not below the "
                   f"{QBER_THRESHOLD*100:.0f}% threshold: no secret key can be guaranteed")


def qber_by_key_pair(alice_result, bob_result, config: ProtocolConfig, expected_relation):
    """QBER separately for each key basis pair (e.g. Z-Z and X-X in BBM92)."""
    counts = {pair: [0, 0] for pair in config.key_pairs}
    for ar, br in zip(alice_result, bob_result):
        pair = (ar["basis"], br["basis"])
        if pair in counts and pair in expected_relation:
            counts[pair][0] += 1
            if not expected_equal(ar["outcome"], br["outcome"], expected_relation[pair]):
                counts[pair][1] += 1
    return {p: wilson_interval(m, n) for p, (n, m) in counts.items() if n > 0}


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
    remaining (secret) bits are never compared directly.

    Uses the Wilson interval (see wilson_interval() above) instead of
    the normal approximation, specifically because sample sizes here
    are small (tens of rounds) and QBER estimates of exactly 0% from
    small samples are common and were shown to be badly misleading
    (0.00% +/- 0.00% from a sample where the true rate, confirmed
    against the full raw key, was ~9%).
    """
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
        return None, None, None
    return wilson_interval(mismatches, n)


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

    qber_est, qber_est_ci_low, qber_est_ci_high = estimate_qber(sample, expected_relation)
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
        "qber_estimated_ci_low": qber_est_ci_low,
        "qber_estimated_ci_high": qber_est_ci_high,
        "residual_mismatches": residual_mismatches,
        "alice_key": alice_key,
        "bob_key_aligned": bob_key_aligned,
    }


def run_full_pipeline(cfg: StackNetworkConfig, config: ProtocolConfig, label: str,
                       n_calib_rounds: int = 1500, num_rounds: int = 10000,
                       sample_fraction: float = 0.15):
    print("=" * 60)
    print(f"{label}")
    print("=" * 60)

    print("--- Oracle diagnostic (simulator-only) ---")
    dominant_bell_state, bell_state_counts, avg_fid = run_oracle_diagnostic(cfg, n_rounds=20)
    print(f"Dominant Bell state observed: {dominant_bell_state} {bell_state_counts}")
    print(f"Average fidelity to dominant state: {avg_fid:.4f}")
    print()

    print("--- Calibration (empirical, no oracle access) ---")
    expected_relation = calibrate_expected_relation(cfg, config, n_calib_rounds=n_calib_rounds)
    for pair, parity in expected_relation.items():
        label_p = "correlated (a==b)" if parity == 1 else "anti-correlated (a!=b)"
        print(f"Basis pair {pair}: {label_p}")
    print()

    print("--- Production run + sifting ---")
    alice_result, bob_result = run_protocol(cfg, config, num_rounds)
    stats = analyze(alice_result, bob_result, config, expected_relation)

    qber_split = qber_by_key_pair(alice_result, bob_result, config, expected_relation)
    for pair, (point, lo, hi) in qber_split.items():
        print(f"QBER basis {pair}: {point*100:.2f}% (CI [{lo*100:.2f}%, {hi*100:.2f}%])")

    print(f"Total rounds:       {stats['n_total']}")
    print(f"Sifted key length:  {stats['n_sifted']} ({stats['sifted_rate']*100:.1f}% of rounds)")
    if config.has_bell_test:
        print(f"CHSH test rounds:   {stats['n_test']}")
    print(f"Discarded rounds:   {stats['n_discarded']}")
    if stats["qber"] is not None:
        print(f"QBER:  {stats['qber']*100:.2f}%  (95% CI: [{stats['qber_ci_low']*100:.2f}%, {stats['qber_ci_high']*100:.2f}%])")
    if stats["S"] is not None:
        print(f"CHSH S: {stats['S']:.4f} +/- {stats['S_margin']:.4f}  (noiseless reference: 2.8284)")
    ok, msg = security_check(stats, config)
    print(f"{'OK' if ok else 'WARNING'}: {msg}")
    print()

    print("--- Raw key extraction (parameter estimation + alignment) ---")
    pipeline = run_key_extraction_pipeline(
        alice_result, bob_result, config, expected_relation, sample_fraction=sample_fraction
    )
    print(f"Key rounds (pre-sampling): {pipeline['n_key_rounds']}")
    print(f"Sacrificed for estimation: {pipeline['n_sacrificed']}")
    print(f"Raw key length:            {pipeline['n_raw_key']}")
    if pipeline["qber_estimated"] is not None:
        print(f"QBER (from sample only):   {pipeline['qber_estimated']*100:.2f}%  "
              f"(95% CI: [{pipeline['qber_estimated_ci_low']*100:.2f}%, {pipeline['qber_estimated_ci_high']*100:.2f}%])")
    print(f"Residual mismatches in raw key (sim-only cross-check): {pipeline['residual_mismatches']}/{pipeline['n_raw_key']}")
    if pipeline["n_raw_key"] > 0:
        residual_rate = pipeline['residual_mismatches'] / pipeline['n_raw_key']
        print(f"Residual mismatch rate: {residual_rate*100:.2f}% -- this is what error correction would fix.")

    # Throughput: raw key bits per second of ACTUAL simulated protocol
    # operation -- uses total_elapsed_ns from ALL num_rounds (test and
    # discard rounds cost time too, not just key rounds), not just an
    # average per-round latency times round count. This is what lets you
    # compare "bits/sec" across distances/configs later, not just raw
    # counts, which are misleading once latency varies with distance.
    if stats["total_elapsed_ns"] and stats["total_elapsed_ns"] > 0:
        throughput_bps = pipeline["n_raw_key"] / (stats["total_elapsed_ns"] * 1e-9)
        print(f"Raw key throughput: {throughput_bps:.4f} bits/s "
              f"({pipeline['n_raw_key']} raw key bits / {stats['total_elapsed_ns']/1e9:.4f} s elapsed)")
    print()

    return stats, pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="e91")
    args = parser.parse_args()
    config = PROTOCOLS[args.protocol]

    # Reference run: same pipeline, zero noise -- confirms the pipeline
    # itself is unchanged before looking at the noisy numbers below.
    perfect_cfg = create_perfect_network()
    run_full_pipeline(perfect_cfg, config, f"[{config.name}] REFERENCE: perfect network (no noise)")

    # NV device + heralded link
    DISTANCE_KM = 10.0
    noisy_cfg = create_nv_heralded_network(distance_km=DISTANCE_KM)
    # Larger num_rounds and sample_fraction than the default: with real
    # noise, QBER needs enough sacrificed samples to be estimated with a
    # useful margin (the 32-sample estimate earlier had a Wilson
    # interval spanning roughly 0-15% at ~8% true QBER -- not useless,
    # but wide). This trades more simulation time for a tighter CI.
    run_full_pipeline(
        noisy_cfg, config, f"[{config.name}] NV device + heralded link ({DISTANCE_KM} km)",
        num_rounds=10000, sample_fraction=0.25,
    )

    print()
    print(f"Run 'python e91v1_1.py --protocol bbm92' to select BBM92 protocol (default: E91).")