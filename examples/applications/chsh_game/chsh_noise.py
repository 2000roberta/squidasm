import logging
import numpy
import netsquid as ns
from netsquid.qubits.ketstates import BellIndex
from netsquid_netbuilder.util.fidelity import calculate_fidelity_epr
from netqasm.sdk import Qubit

import csv

from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.util import get_qubit_state
from squidasm.run.stack.config import (
    NVQDeviceConfig,
    HeraldedLinkConfig,
    LinkConfig,
    StackConfig,
    StackNetworkConfig,
    CLinkConfig,
    DefaultCLinkConfig,
)

#network configuration

def create_nv_perfect_link_network() -> StackNetworkConfig:
    """NV center device + perfect link."""
    nv_cfg = NVQDeviceConfig()
    alice_stack = StackConfig(name="Alice", qdevice_typ="nv", qdevice_cfg=nv_cfg)
    bob_stack   = StackConfig(name="Bob",   qdevice_typ="nv", qdevice_cfg=nv_cfg)
    link  = LinkConfig.perfect_config(stack1="Alice", stack2="Bob")
    clink = CLinkConfig(
        stack1="Alice", stack2="Bob",
        typ="default",
        cfg=DefaultCLinkConfig(delay=0.0),
    )
    return StackNetworkConfig(stacks=[alice_stack, bob_stack], links=[link], clinks=[clink])

def create_nv_heralded_network(distance_km: float = 1.0) -> StackNetworkConfig:
    """
    NV center device + heralded double-click link.
    The heralded link simulates a realistic optical fiber link where
    entanglement is confirmed by a midpoint heralding station.
 
    Default fiber parameters:
    - p_loss_length = 0.2 dB/km
    - speed_of_light = 200000 km/s
 
    :param distance_km: Total fiber length between the two nodes in km.
    """
    nv_cfg = NVQDeviceConfig()
 
    alice_stack = StackConfig(name="Alice", qdevice_typ="nv", qdevice_cfg=nv_cfg)
    bob_stack   = StackConfig(name="Bob",   qdevice_typ="nv", qdevice_cfg=nv_cfg)
 
    herald_cfg = HeraldedLinkConfig(
        length=distance_km,      # km — swept parameter
        emission_fidelity=0.95,  # realistic NV photon emission fidelity
        # p_loss_length=0.2 dB/km and speed_of_light=200000 km/s from defaults
    )
 
    link  = LinkConfig(stack1="Alice", stack2="Bob", typ="heralded", cfg=herald_cfg)
    clink = CLinkConfig(
        stack1="Alice", stack2="Bob",
        typ="default",
        cfg=DefaultCLinkConfig(delay=distance_km / 200000 * 1e9),  # ns
    )
 
    return StackNetworkConfig(stacks=[alice_stack, bob_stack], links=[link], clinks=[clink])

#helper function

def game_won(x, y, a, b) -> bool:
    if x == 1 and y == 1:
        return a != b
    else:
        return a == b

#programs

class AliceProgram(Program):
    PEER_NAME = "Bob"

    def __init__(self, x: int):
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.x = x

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_program",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )
    
    @staticmethod
    def measure_basis_0(q: Qubit):
        return q.measure()

    @staticmethod
    def measure_basis_1(q: Qubit):
        q.rot_Y(angle=-numpy.pi / 2)
        q.rot_Z(angle=numpy.pi)
        return q.measure()

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        t_start = ns.sim_time() #measure start time

        epr = epr_socket.create_keep()[0]
        yield from connection.flush()

        # Fidelity before measurement, only possible in simulations
        #heralded link can generate different Bell states (B00, B01, B10, B11),
        #depending on which photon is detected first at the heralding station
        #So we take the maximum fidelity over all Bell states
        dm = get_qubit_state(epr, node_name="Alice", full_state=True)
        fidelity = max(calculate_fidelity_epr(dm, idx) for idx in BellIndex)

        # CHSH strategy
        if self.x == 0:
            a = epr.measure()
        else:
            epr.rot_Y(angle=-numpy.pi / 2)
            epr.rot_Z(angle=numpy.pi)
            a = epr.measure()

        yield from connection.flush()

        latency_ns = ns.sim_time() - t_start #total latency: EPR creation+measurement

        # Stampa risultato 
        return {"x": self.x, "a": int(a), "fidelity": fidelity, "latency_ns": latency_ns}


class BobProgram(Program):
    PEER_NAME = "Alice"

    def __init__(self, y: int):
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.y = y

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_program",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )
    
    @staticmethod
    def measure_basis_0(q: Qubit):
        q.rot_Y(angle=-numpy.pi / 4)
        return q.measure()

    @staticmethod
    def measure_basis_1(q: Qubit):
        q.rot_Y(angle=numpy.pi / 4)
        return q.measure()

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        epr = epr_socket.recv_keep()[0]
        yield from connection.flush()

        # CHSH strategy
        if self.y == 0:
            epr.rot_Y(angle=-numpy.pi / 4)
        else:
            epr.rot_Y(angle=numpy.pi / 4)

        b = epr.measure()
        yield from connection.flush()

        return {"y": self.y, "b": int(b)}


if __name__ == "__main__":

    ns.set_qstate_formalism(ns.QFormalism.DM) #density matrix formalism, required for fidelity calculation

    #STEP 1: NV DEVICE + PERFECT LINK
    print("=" * 60)
    print("STEP 1: NV DEVICE + PERFECT LINK")
    print("Isolating the effect of the realistic NV device.")
    print("=" * 60)

    NUM_ROUNDS_STEP1 = 200
    cfg_nv = create_nv_perfect_link_network()

    wins     = 0
    fid_list = []
    lat_list = []

    for _ in range(NUM_ROUNDS_STEP1):
        x = numpy.random.randint(0, 2)
        y = numpy.random.randint(0, 2)

        alice_program = AliceProgram(x)
        bob_program   = BobProgram(y)
        alice_program.logger.setLevel(logging.ERROR)
        bob_program.logger.setLevel(logging.ERROR)

        alice_result, bob_result = run(
            config=cfg_nv,
            programs={"Alice": alice_program, "Bob": bob_program},
            num_times=1,
        )

        a   = alice_result[0]["a"]
        b   = bob_result[0]["b"]
        fid_list.append(alice_result[0]["fidelity"])
        lat_list.append(alice_result[0]["latency_ns"])

        if game_won(x, y, a, b):
            wins += 1

    p        = wins / NUM_ROUNDS_STEP1
    margin   = 1.96 * numpy.sqrt(p * (1 - p) / NUM_ROUNDS_STEP1) * 100
    win_rate = p * 100
    avg_fid  = numpy.mean(fid_list)
    avg_lat  = numpy.mean(lat_list) / 1e6

    print(f"Win rate:     {wins}/{NUM_ROUNDS_STEP1} = {win_rate:.1f}% ± {margin:.1f}%")
    print(f"Avg fidelity: {avg_fid:.4f}")
    print(f"Avg latency:  {avg_lat:.3f} ms")
    print()
    

    #STEP 2: NV DEVICE + HERALDED LINK
    NUM_ROUNDS = 500
    NUM_STEPS = 50
    distances_km = numpy.linspace(1.0, 100.0, NUM_STEPS) #from 1 to 100 km

    win_rates   = []
    margins     = []
    fidelities  = []
    latencies   = []
 
    print("=" * 60)
    print("STEP 2: NV DEVICE + HERALDED LINK")
    print("Sweeping over fiber distance between the two nodes.")
    print("=" * 60)
    print(f"{'Dist(km)':>10}  {'Win%':>8}  {'±':>6}  {'Fidelity':>10}  {'Latency(ms)':>12}")
    print("-" * 55)

    for dist in distances_km:
        cfg = create_nv_heralded_network(distance_km=dist)
 
        wins     = 0
        fid_list = []
        lat_list = []
 
        for _ in range(NUM_ROUNDS):
            x = numpy.random.randint(0, 2)
            y = numpy.random.randint(0, 2)
 
            alice_program = AliceProgram(x)
            bob_program   = BobProgram(y)
            alice_program.logger.setLevel(logging.ERROR)
            bob_program.logger.setLevel(logging.ERROR)
 
            alice_result, bob_result = run(
                config=cfg,
                programs={"Alice": alice_program, "Bob": bob_program},
                num_times=1,
            )
 
            a   = alice_result[0]["a"]
            b   = bob_result[0]["b"]
            fid_list.append(alice_result[0]["fidelity"])
            lat_list.append(alice_result[0]["latency_ns"])
 
            if game_won(x, y, a, b):
                wins += 1
 
        p          = wins / NUM_ROUNDS
        margin     = 1.96 * numpy.sqrt(p * (1 - p) / NUM_ROUNDS)
        win_rate   = p * 100
        margin_pct = margin * 100
        avg_fid    = numpy.mean(fid_list)
        avg_lat_ms = numpy.mean(lat_list) / 1e6
 
        win_rates.append(win_rate)
        margins.append(margin_pct)
        fidelities.append(avg_fid)
        latencies.append(avg_lat_ms)
 
        print(f"{dist:>10.1f}  {win_rate:>7.1f}%  ±{margin_pct:>4.1f}%  {avg_fid:>10.4f}  {avg_lat_ms:>11.3f}ms")

    #CSV file
    with open("risultati_nv_heralded.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance_km", "win_rate", "margin", "fidelity", "latency_ms"])
        for i in range(len(distances_km)):
            writer.writerow([
                distances_km[i],
                win_rates[i],
                margins[i],
                fidelities[i],
                latencies[i],
            ])
 
    print("\nResults saved to risultati_nv_heralded.csv")