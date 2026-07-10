import numpy
import netsquid as ns
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
# E91 - VERSION 1 (ideal protocol validation)
#
# Design principles enforced in this version:
#
# 1. NO ORACLE ACCESS IN PRODUCTION PROGRAMS. AliceProgram / BobProgram
#    (the classes that represent what a real node would run) never touch
#    the density matrix. Density-matrix / fidelity access is isolated in
#    _OracleAliceProgram / _OracleBobProgram, used only in a separate,
#    clearly-labelled calibration step, never mixed into protocol logic
#    or into the statistics we report as "protocol performance".
#
# 2. NO HARDCODED BELL-STATE ASSUMPTION. Which pair of outcomes counts
#    as "correlated" for a given basis combination is determined
#    empirically from a calibration batch (majority vote), not assumed
#    from a specific Bell state. This makes the analysis correct
#    regardless of whether the link produces Phi+, Psi-, or anything
#    else, and carries over unchanged to V3 (heralded link).
#
# 3. SINGLE SIMULATION PER BATCH. Each phase (oracle diagnostic,
#    calibration, production) is a single run(..., num_times=N) call.
#    No Python-level loop recreates the NetSquid environment per round.
# ============================================================

ALICE_ANGLES = [0.0, numpy.pi / 4, numpy.pi / 2]              # a1, a2, a3
BOB_ANGLES = [numpy.pi / 4, numpy.pi / 2, 3 * numpy.pi / 4]    # b1, b2, b3

KEY_PAIRS = [(1, 0), (2, 1)]                        # matching angles -> raw key
TEST_PAIRS = [(0, 0), (0, 2), (2, 0), (2, 2)]       # CHSH S estimation


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


# ============================================================
# PRODUCTION PROGRAMS - what a real node would actually run.
# No density matrix, no global state. Basis is chosen locally and
# randomly INSIDE run(), so a single run(num_times=N) call gives N
# independent rounds without recreating the simulation.
# ============================================================

class AliceProgram(Program):
    PEER_NAME = "Bob"

    def __init__(self):
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)

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
        connection = context.connection

        basis_idx = int(numpy.random.randint(0, 3))

        epr = epr_socket.create_keep()[0]
        theta = ALICE_ANGLES[basis_idx]
        epr.rot_Y(angle=theta)
        a = epr.measure()

        yield from connection.flush()

        return {"basis": basis_idx, "outcome": int(a)}


class BobProgram(Program):
    PEER_NAME = "Alice"

    def __init__(self):
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)

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
        connection = context.connection

        basis_idx = int(numpy.random.randint(0, 3))

        epr = epr_socket.recv_keep()[0]
        theta = BOB_ANGLES[basis_idx]
        epr.rot_Y(angle=theta)
        b = epr.measure()

        yield from connection.flush()

        return {"basis": basis_idx, "outcome": int(b)}


# ============================================================
# ORACLE / CALIBRATION-ONLY PROGRAMS.
# These are never used to collect protocol statistics. They exist
# purely so that, once, offline, we can report which physical Bell
# state the link is actually producing -- useful for a sanity-check
# printout and for writing up results, but the production analysis
# below does NOT depend on this being correct or even being run.
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

        # Bob must complete the entanglement handshake, but never touches
        # the density matrix himself -- only Alice's side is read, and only
        # in this oracle-only class.
        epr_socket.recv_keep()
        yield from connection.flush()

        return {}


def run_oracle_diagnostic(cfg: StackNetworkConfig, n_rounds: int = 20):
    """Simulator-only diagnostic: which Bell state is the link actually producing?

    NEVER call this as part of a benchmark run whose numbers you intend to
    report as protocol performance -- it exists only to sanity-check /
    document the physical setup, exactly like a lab would run state
    tomography on a source once during characterization, not during
    every data-taking run.
    """
    alice_result, _ = run(
        config=cfg,
        programs={"Alice": _OracleAliceProgram(), "Bob": _OracleBobProgram()},
        num_times=n_rounds,
    )
    counts = {}
    for r in alice_result:
        counts[r["bell_index"]] = counts.get(r["bell_index"], 0) + 1
    dominant = max(counts, key=counts.get)
    avg_fidelity = numpy.mean([r["fidelity"] for r in alice_result])
    return dominant, counts, avg_fidelity


# ============================================================
# EMPIRICAL CALIBRATION (no oracle): determine, for each KEY_PAIRS
# basis combination, whether Alice's and Bob's outcomes are naturally
# correlated (same) or anti-correlated (different), by majority vote
# over a calibration batch. This is Bell-state agnostic by construction.
# ============================================================

def calibrate_expected_relation(cfg: StackNetworkConfig, n_calib_rounds: int = 300):
    alice_result, bob_result = run(
        config=cfg,
        programs={"Alice": AliceProgram(), "Bob": BobProgram()},
        num_times=n_calib_rounds,
    )

    records = [
        {"a_idx": ar["basis"], "b_idx": br["basis"], "a": ar["outcome"], "b": br["outcome"]}
        for ar, br in zip(alice_result, bob_result)
    ]

    expected_relation = {}
    for pair in KEY_PAIRS:
        subset = [r for r in records if (r["a_idx"], r["b_idx"]) == pair]
        if not subset:
            continue
        same = sum(1 for r in subset if r["a"] == r["b"])
        diff = len(subset) - same
        # +1: outcomes are naturally correlated (a == b) for this pair
        # -1: outcomes are naturally anti-correlated (a != b) for this pair
        expected_relation[pair] = 1 if same >= diff else -1

    return expected_relation


# ============================================================
# PRODUCTION RUN + ANALYSIS
# ============================================================

def run_e91_v1(cfg: StackNetworkConfig, num_rounds: int):
    alice_result, bob_result = run(
        config=cfg,
        programs={"Alice": AliceProgram(), "Bob": BobProgram()},
        num_times=num_rounds,
    )
    return [
        {"a_idx": ar["basis"], "b_idx": br["basis"], "a": ar["outcome"], "b": br["outcome"]}
        for ar, br in zip(alice_result, bob_result)
    ]


def analyze(records, expected_relation):
    # --- Sifted key + QBER, corrected for the empirically calibrated parity ---
    key_a, key_b_corrected = [], []
    for r in records:
        pair = (r["a_idx"], r["b_idx"])
        if pair in expected_relation:
            parity = expected_relation[pair]
            b_corrected = r["b"] if parity == 1 else 1 - r["b"]
            key_a.append(r["a"])
            key_b_corrected.append(b_corrected)

    n_sifted = len(key_a)
    if n_sifted > 0:
        mismatches = sum(1 for a, b in zip(key_a, key_b_corrected) if a != b)
        qber = mismatches / n_sifted
        qber_margin = 1.96 * numpy.sqrt(qber * (1 - qber) / n_sifted)
    else:
        qber, qber_margin = None, None

    # --- CHSH S: computed from raw (uncorrected) correlations. If the
    # link's global parity is flipped relative to Phi+, S comes out with
    # the opposite sign but the same magnitude -- so we report |S| against
    # the Tsirelson bound, and keep the signed value for transparency. ---
    correlations, n_per_pair = {}, {}
    for pair in TEST_PAIRS:
        subset = [r for r in records if (r["a_idx"], r["b_idx"]) == pair]
        n_per_pair[pair] = len(subset)
        if subset:
            same = sum(1 for r in subset if r["a"] == r["b"])
            diff = len(subset) - same
            correlations[pair] = (same - diff) / len(subset)
        else:
            correlations[pair] = None

    if all(v is not None for v in correlations.values()):
        S = (
            correlations[(0, 0)]
            - correlations[(0, 2)]
            + correlations[(2, 0)]
            + correlations[(2, 2)]
        )
        var_S = sum(
            (1 - correlations[p] ** 2) / n_per_pair[p]
            for p in TEST_PAIRS if n_per_pair[p] > 0
        )
        S_margin = 1.96 * numpy.sqrt(var_S)
    else:
        S, S_margin = None, None

    return {
        "n_total": len(records),
        "n_sifted": n_sifted,
        "sifted_rate": n_sifted / len(records) if records else 0.0,
        "qber": qber,
        "qber_margin": qber_margin,
        "S": S,
        "S_margin": S_margin,
    }


if __name__ == "__main__":
    ns.set_qstate_formalism(ns.QFormalism.DM)

    cfg = create_perfect_network()

    print("=" * 60)
    print("ORACLE DIAGNOSTIC (simulator-only, not used by production code)")
    print("=" * 60)
    dominant_bell_state, bell_state_counts, avg_fid = run_oracle_diagnostic(cfg, n_rounds=20)
    print(f"Dominant Bell state observed: {dominant_bell_state} {bell_state_counts}")
    print(f"Average fidelity to dominant state: {avg_fid:.4f}")
    print("(For information only -- production analysis below does not use this.)")
    print()

    print("=" * 60)
    print("CALIBRATION (empirical, no oracle access)")
    print("=" * 60)
    expected_relation = calibrate_expected_relation(cfg, n_calib_rounds=300)
    for pair, parity in expected_relation.items():
        label = "correlated (a==b)" if parity == 1 else "anti-correlated (a!=b)"
        print(f"Basis pair {pair}: {label}")
    print()

    print("=" * 60)
    print("E91 V1 - Ideal protocol validation (no device/link noise)")
    print("Expected: |S| ~= 2.828, QBER ~= 0%")
    print("=" * 60)

    NUM_ROUNDS = 1000
    records = run_e91_v1(cfg, NUM_ROUNDS)
    stats = analyze(records, expected_relation)

    print(f"Sifted key length: {stats['n_sifted']} ({stats['sifted_rate']*100:.1f}% of rounds)")
    if stats["qber"] is not None:
        print(f"QBER:  {stats['qber']*100:.2f}% +/- {stats['qber_margin']*100:.2f}%")
    if stats["S"] is not None:
        print(f"CHSH S: {stats['S']:.4f} +/- {stats['S_margin']:.4f}  (|S| target: 2.8284)")

    if stats["S"] is not None and abs(abs(stats["S"]) - 2 * numpy.sqrt(2)) > 3 * stats["S_margin"]:
        print("\nWARNING: |S| is far from 2*sqrt(2) beyond the statistical margin.")
        print("Check the rotation/angle convention before moving to V2.")