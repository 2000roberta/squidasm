"""Quantum Coordination Game — adattamento del gioco CHSH.

Due router (Alice e Bob) devono assegnare job a uno di due server
senza comunicare. Condividono una coppia EPR e usano la strategia
quantistica ottimale del CHSH per coordinarsi.

Input:
    x, y ∈ {0,1}  — tipo del job (0=leggero, 1=pesante)
                    ottenuto campionando X ~ Exp(mu=1) e confrontando
                    con la soglia mediana theta = ln(2) ≈ 0.693

Output:
    a, b ∈ {0,1}  — server scelto (0=server1, 1=server2)

Payoff V(x, y, a, b):
    (0,0) → 0          indifferente, job leggeri
    (0,1) → +1 se a==b, -1 se a!=b
    (1,0) → +1 se a==b, -1 se a!=b
    (1,1) → +2 se a!=b, -1 se a==b   (separare due pesanti: massimo beneficio)

Payoff medio classico atteso (strategia deterministica ottimale): 1.0
"""

import logging

import numpy
from netqasm.sdk import Qubit
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import create_two_node_network

# ---------------------------------------------------------------------------
# Parametri del modello
# ---------------------------------------------------------------------------

MU = 1.0                    # service rate (come nel simulatore principale)
THETA = numpy.log(2) / MU  # soglia mediana: P(X < THETA) = 0.5


# ---------------------------------------------------------------------------
# Funzione di payoff
# ---------------------------------------------------------------------------

def coordination_payoff(x: int, y: int, a: int, b: int) -> int:
    """Payoff del coordination game.

    Args:
        x: tipo job di Alice (0=leggero, 1=pesante)
        y: tipo job di Bob   (0=leggero, 1=pesante)
        a: server scelto da Alice (0=server1, 1=server2)
        b: server scelto da Bob   (0=server1, 1=server2)

    Returns:
        Payoff intero in {-1, 0, +1, +2}.
    """
    if x == 0 and y == 0:
        # Entrambi leggeri: indifferente
        return 0
    elif (x == 0 and y == 1) or (x == 1 and y == 0):
        # Un leggero e un pesante: meglio tenerli insieme
        return +1 if a == b else -1
    else:
        # x == 1 and y == 1: entrambi pesanti, meglio separarli
        return +2 if a != b else -1


# ---------------------------------------------------------------------------
# Programma di Alice
# ---------------------------------------------------------------------------

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
        """Base computazionale — usata quando x=0 (job leggero)."""
        return q.measure()

    @staticmethod
    def measure_basis_1(q: Qubit):
        """Base di Hadamard — usata quando x=1 (job pesante)."""
        q.H()
        return q.measure()

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        epr = epr_socket.create_keep()[0]
        yield from connection.flush()
        self.logger.info("Coppia EPR creata")

        if self.x == 0:
            a = self.measure_basis_0(epr)
        else:
            a = self.measure_basis_1(epr)
        yield from connection.flush()

        self.logger.info(f"x={self.x}, a={int(a)}")
        return {"x": self.x, "a": int(a)}


# ---------------------------------------------------------------------------
# Programma di Bob
# ---------------------------------------------------------------------------

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
        """Rotazione Y di -π/4 — usata quando y=0 (job leggero)."""
        q.rot_Y(angle=-numpy.pi / 4)
        return q.measure()

    @staticmethod
    def measure_basis_1(q: Qubit):
        """Rotazione Y di +π/4 — usata quando y=1 (job pesante)."""
        q.rot_Y(angle=numpy.pi / 4)
        return q.measure()

    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        epr = epr_socket.recv_keep()[0]
        yield from connection.flush()
        self.logger.info("Coppia EPR ricevuta")

        if self.y == 0:
            b = self.measure_basis_0(epr)
        else:
            b = self.measure_basis_1(epr)
        yield from connection.flush()

        self.logger.info(f"y={self.y}, b={int(b)}")
        return {"y": self.y, "b": int(b)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = create_two_node_network(node_names=["Alice", "Bob"])

    N = 100  # numero di round della simulazione
    total_payoff = 0
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}  # frequenza casi
    payoff_by_case = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    print(f"Soglia theta = ln(2) = {THETA:.4f}")
    print(f"Avvio simulazione su {N} round...\n")

    for i in range(N):
        # Campiona i tempi di servizio e discretizza
        X_A = numpy.random.exponential(scale=1.0 / MU)
        X_B = numpy.random.exponential(scale=1.0 / MU)
        x = 1 if X_A >= THETA else 0
        y = 1 if X_B >= THETA else 0

        alice_program = AliceProgram(x)
        bob_program = BobProgram(y)
        alice_program.logger.setLevel(logging.ERROR)
        bob_program.logger.setLevel(logging.ERROR)

        alice_result, bob_result = run(
            config=cfg,
            programs={"Alice": alice_program, "Bob": bob_program},
            num_times=1,
        )

        a = int(alice_result[0]["a"])
        b = int(bob_result[0]["b"])
        v = coordination_payoff(x, y, a, b)

        total_payoff += v
        counts[(x, y)] += 1
        payoff_by_case[(x, y)] += v

    # Risultati
    print("=" * 50)
    print(f"Payoff medio quantistico:  {total_payoff / N:.3f}")
    print(f"Payoff medio classico att: 1.000")
    print(f"Vantaggio quantistico:     {total_payoff / N - 1.0:+.3f}")
    print()
    print("Dettaglio per caso (x,y):")
    for case in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        x, y = case
        n = counts[case]
        if n > 0:
            avg = payoff_by_case[case] / n
            label = ("leggero", "leggero") if case == (0, 0) else \
                    ("leggero", "pesante") if case == (0, 1) else \
                    ("pesante", "leggero") if case == (1, 0) else \
                    ("pesante", "pesante")
            print(f"  ({x},{y}) {label[0]:8s}+{label[1]:8s}: "
                  f"n={n:3d}, payoff medio={avg:+.3f}")
    print("=" * 50)