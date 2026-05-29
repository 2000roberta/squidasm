import logging

import numpy
from netqasm.sdk import Qubit

from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import create_two_node_network

import csv


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

        # CHSH strategy: measure in one of 2 bases depending on x.
        self.logger.info(f"Measuring in basis x = {x}")
        if x == 0:
            a = self.measure_basis_0(epr)
        else:
            a = self.measure_basis_1(epr)

        yield from connection.flush()
        self.logger.info(f"Measured a: {a}")

        return {"x": x, "a": a}


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

    NUM_ROUNDS = 200
    NUM_STEPS  = 11
    noise_levels = numpy.linspace(0.0, 1.0, NUM_STEPS)
    win_rates = []
    margins = []

    print(f"{'Rumore':>8}  {'Vittorie':>10}  {'%':>6}")
    print("-" * 30)

    for noise in noise_levels:
        # Crea la rete con il livello di rumore corrente
        cfg = create_two_node_network(node_names=["Alice", "Bob"], link_noise=noise, qdevice_noise=noise)

        wins = 0
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
            if "won" in game_won(x, y, a, b):
                wins += 1

        #rate = wins / NUM_ROUNDS * 100
        #win_rates.append(rate)
        #print(f"{noise:>8.1f}  {wins:>5}/{NUM_ROUNDS}  {rate:>5.1f}%")

        #calcolo intervallo di confidenza al 95%
        p=wins/NUM_ROUNDS
        margin=1.96 * numpy.sqrt(p*(1-p)/NUM_ROUNDS)

        rate = p * 100
        margin_pct = margin * 100

        win_rates.append(rate)
        margins.append(margin_pct)
        print(f"{noise:>8.1f}  {wins:>5}/{NUM_ROUNDS}  {rate:>5.1f}% ± {margin_pct:.1f}%")

    #crea file dati
    with open("risultati.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noise", "win_rate", "margin"])
        for noise, rate, margin in zip(noise_levels, win_rates, margins):
            writer.writerow([noise, rate, margin])

    print("Risultati salvati in risultati.csv")