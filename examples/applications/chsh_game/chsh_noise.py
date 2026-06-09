import logging
from multiprocessing import connection, context

import numpy
from netqasm.sdk import Qubit

from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import create_two_node_network
from squidasm.util.util import get_qubit_state

import csv

import netsquid as ns
from netsquid.qubits.ketstates import BellIndex
from netsquid_netbuilder.util.fidelity import calculate_fidelity_epr


def game_won(x, y, a, b):
    if x == 1 and y == 1:
        if a != b:
            return "Alice and Bob won the game, since x * y = 1 and a ^ b = 1"
        else:
            return "Alice and Bob lost the game, since x * y = 1 and a ^ b = 0"
    else:
        if a == b:
            return "Alice and Bob won the game, since x * y = 0 and a ^ b = 0"
        else:
            return "Alice and Bob lost the game, since x * y = 0 and a ^ b = 1"
        

# def calc_bell_fidelity(dm: numpy.ndarray) -> float: #calcola fidelity rispetto a phi+
    
    # Ottieni la density matrix del qubit da NetSquid
    # dm = ns.qubits.reduced_dm(qubit)

    # Stato di Bell phi+ ideale: (|00> + |11>) / sqrt(2)
    #phi_plus = numpy.array([1, 0, 0, 1]) / numpy.sqrt(2)
    #rho_ideal = numpy.outer(phi_plus, phi_plus)  # matrice 4x4

    rho_ideal = numpy.array([[0.5, 0], [0, 0.5]])  # matrice 2x2

    # Fidelity = <phi+| rho_reale |phi+> = Tr(rho_ideal * rho_reale)
    fidelity = numpy.real(numpy.trace(rho_ideal @ dm))
    return float(fidelity)


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
        q.H()
        return q.measure()

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        epr = epr_socket.create_keep()[0]
        yield from connection.flush()
        self.logger.info("Finished EPR pair creation")

        #calcolo fidelity prima della misura
        #ns_qubit = connection.storage_manager.get_qubit(epr) #accedo al qubit fisico netsquid tramite l'handle netqasm
        dm = get_qubit_state(epr, node_name="Alice", full_state=True)
        fidelity = calculate_fidelity_epr(dm, BellIndex.B00)

        # CHSH strategy: measure in one of 2 bases depending on x.
        self.logger.info(f"Measuring in basis x = {self.x}")
        if self.x == 0:
            a = self.measure_basis_0(epr)
        else:
            a = self.measure_basis_1(epr)

        yield from connection.flush()
        self.logger.info(f"Measured a: {a}")

        return {"x": self.x, "a": int(a), "fidelity": fidelity}


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
        self.logger.info("Finished EPR pair creation")

        # CHSH strategy: measure in one of 2 bases depending on y.
        self.logger.info(f"Measuring in basis y = {self.y}")
        if self.y == 0:
            b = self.measure_basis_0(epr)
        else:
            b = self.measure_basis_1(epr)

        yield from connection.flush()
        self.logger.info(f"Measured b: {b}")

        return {"y": self.y, "b": int(b)}


if __name__ == "__main__":

    ns.set_qstate_formalism(ns.QFormalism.DM) #attivo il formalismo density matrix

    NUM_ROUNDS = 200
    NUM_STEPS  = 11
    noise_levels = numpy.linspace(0.0, 1.0, NUM_STEPS)
    win_rates_link    = []
    win_rates_device  = []
    margins_link      = []
    margins_device    = []
    fidelities_link   = [] #fidelity media per ogni livello di link noise
    fidelities_device = [] #fidelity media per ogni livello di qdevice noise

    print("=== SWEEP LINK NOISE ===")
    print(f"{'Rumore':>8}  {'Vittorie':>10}  {'%':>6}  {'Fidelity':>10}")
    print("-" * 50)


    for noise in noise_levels:
        # Crea la rete con il livello di rumore corrente
        cfg = create_two_node_network(node_names=["Alice", "Bob"], link_noise=noise, qdevice_noise=0.0)

        wins = 0
        fid_list = []

        for _ in range(NUM_ROUNDS):
            # generate x & y randomly
            x = numpy.random.randint(0, 2)
            y = numpy.random.randint(0, 2)

            # Create instances of programs to run
            alice_program = AliceProgram(x)
            bob_program   = BobProgram(y)

            # toggle logging. Set to logging.INFO for logging of events.
            alice_program.logger.setLevel(logging.ERROR)
            bob_program.logger.setLevel(logging.ERROR)

            # Run the simulation. Programs argument is a mapping of network node labels to programs to run on that node
            alice_result, bob_result = run(
                config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=1,
            )

            a = alice_result[0]["a"]
            b = bob_result[0]["b"]
            fid_list.append(alice_result[0]["fidelity"])

            if "won" in game_won(x, y, a, b):
                wins += 1

        #rate = wins / NUM_ROUNDS * 100
        #win_rates.append(rate)
        #print(f"{noise:>8.1f}  {wins:>5}/{NUM_ROUNDS}  {rate:>5.1f}%")

        #calcolo intervallo di confidenza al 95%
        p = wins / NUM_ROUNDS
        margin = 1.96 * numpy.sqrt(p * (1 - p) / NUM_ROUNDS)

        rate = p * 100
        margin_pct = margin * 100

        avg_fid    = numpy.mean(fid_list)

        win_rates_link.append(rate)
        margins_link.append(margin_pct)
        fidelities_link.append(avg_fid)

        print(f"{noise:>8.2f}  {wins:>5}/{NUM_ROUNDS}  {rate:>5.1f}% ± {margin_pct:.1f}%   fid={avg_fid:.4f}")

    print("=== SWEEP QDEVICE NOISE ===")
    print(f"{'Rumore':>8}  {'Vittorie':>10}  {'%':>6}  {'Fidelity':>10}")
    print("-" * 50)


    for noise in noise_levels:
        # Crea la rete con il livello di rumore corrente
        cfg = create_two_node_network(node_names=["Alice", "Bob"], link_noise=0.0, qdevice_noise=noise)

        wins = 0
        fid_list = []

        for _ in range(NUM_ROUNDS):
            # generate x & y randomly
            x = numpy.random.randint(0, 2)
            y = numpy.random.randint(0, 2)

            # Create instances of programs to run
            alice_program = AliceProgram(x)
            bob_program   = BobProgram(y)

            # toggle logging. Set to logging.INFO for logging of events.
            alice_program.logger.setLevel(logging.ERROR)
            bob_program.logger.setLevel(logging.ERROR)

            # Run the simulation. Programs argument is a mapping of network node labels to programs to run on that node
            alice_result, bob_result = run(
                config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=1,
            )

            a = alice_result[0]["a"]
            b = bob_result[0]["b"]
            fid_list.append(alice_result[0]["fidelity"])

            if "won" in game_won(x, y, a, b):
                wins += 1

        #rate = wins / NUM_ROUNDS * 100
        #win_rates.append(rate)
        #print(f"{noise:>8.1f}  {wins:>5}/{NUM_ROUNDS}  {rate:>5.1f}%")

        #calcolo intervallo di confidenza al 95%
        p = wins / NUM_ROUNDS
        margin = 1.96 * numpy.sqrt(p * (1 - p) / NUM_ROUNDS)

        rate = p * 100
        margin_pct = margin * 100

        avg_fid    = numpy.mean(fid_list)

        win_rates_device.append(rate)
        margins_device.append(margin_pct)
        fidelities_device.append(avg_fid)

        print(f"{noise:>8.2f}  {wins:>5}/{NUM_ROUNDS}  {rate:>5.1f}% ± {margin_pct:.1f}%   fid={avg_fid:.4f}")


    #crea file dati
    with open("risultati_fidelity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noise", "win_rate_link", "margin_link", "fidelity_link", "win_rate_qdevice", "margin_qdevice", "fidelity_qdevice"])
        for i in range(len(noise_levels)):
            writer.writerow([
                noise_levels[i],
                win_rates_link[i],
                margins_link[i],
                fidelities_link[i],
                win_rates_device[i],
                margins_device[i],
                fidelities_device[i],
            ])

    print("Risultati salvati in risultati_fidelity.csv")