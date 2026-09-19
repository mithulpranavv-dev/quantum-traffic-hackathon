"""
qaoa_solver.py
--------------
Solves the QUBO built in qubo_formulation.py with the Quantum
Approximate Optimization Algorithm (QAOA), run on a local statevector
simulator via Qiskit -- no real quantum hardware or account needed.

Why QAOA sits here (see README.md for the longer version):
    The signal-phase choice at every intersection is a single binary
    variable, and the whole network's objective is exactly a QUBO
    (quadratic in those binaries). QAOA is the standard NISQ-era
    algorithm for exactly this class of problem: it alternates a
    "cost" unitary (built from the QUBO's Ising form) with a "mixer"
    unitary (which lets the state explore other bitstrings), and a
    classical optimizer tunes the (gamma, beta) angles of each layer to
    push the quantum state's probability mass onto the bitstring(s)
    with lowest energy = least total (congestion + poor coordination).

Pipeline
    QUBO (h, Q, offset)
        -> Ising Hamiltonian  H = sum_i c_i Z_i + sum_ij c_ij Z_i Z_j + const
        -> parameterised QAOA circuit |psi(gamma, beta)>
        -> classical optimizer (COBYLA) minimises <psi|H|psi>
        -> sample the optimized state, decode bitstrings back to x_i,
           keep the best-scoring one against the ORIGINAL QUBO.

Everything here runs on `qiskit.quantum_info.Statevector`, i.e. exact
linear-algebra simulation -- appropriate because n <= 8 qubits (256
amplitudes at most), which is instant on a laptop and needs no shots/
sampling noise to demonstrate the method.

If qiskit is not installed, `solve_qaoa` transparently falls back to
`_numpy_qaoa`, a dependency-free re-implementation of the *exact same*
circuit using NumPy tensor products. This keeps the dashboard/demo
runnable even if the Qiskit install fails mid-hackathon; the console
output/README make clear which backend actually ran.
"""

from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np
from scipy.optimize import minimize

from src.qubo_formulation import evaluate_qubo, brute_force_solve

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp

    QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when qiskit is missing
    QISKIT_AVAILABLE = False


# ---------------------------------------------------------------------
# QUBO -> Ising conversion (shared by both backends)
# ---------------------------------------------------------------------
def qubo_to_ising(
    h: Dict[int, float], Q: Dict[Tuple[int, int], float], offset: float
) -> Tuple[List[int], Dict[int, float], Dict[Tuple[int, int], float], float]:
    """Convert QUBO (minimise over x in {0,1}) to Ising form (minimise
    over z in {-1,+1}) via the standard substitution x_i = (1 - z_i) / 2.

    Returns (nodes, linear_z, quad_z, new_offset) where `nodes` fixes
    the node-order -> qubit-index mapping used everywhere below.
    """
    nodes = sorted(h.keys())
    linear_z: Dict[int, float] = {n: 0.0 for n in nodes}
    quad_z: Dict[Tuple[int, int], float] = {}
    new_offset = offset

    for i, hi in h.items():
        linear_z[i] += -hi / 2.0
        new_offset += hi / 2.0

    for (i, j), qij in Q.items():
        linear_z[i] += -qij / 4.0
        linear_z[j] += -qij / 4.0
        quad_z[(i, j)] = quad_z.get((i, j), 0.0) + qij / 4.0
        new_offset += qij / 4.0

    return nodes, linear_z, quad_z, new_offset


# ---------------------------------------------------------------------
# Qiskit backend
# ---------------------------------------------------------------------
def _build_cost_hamiltonian(nodes: List[int], linear_z, quad_z) -> "SparsePauliOp":
    n = len(nodes)
    idx = {node: k for k, node in enumerate(nodes)}
    pauli_list = []

    for node, coeff in linear_z.items():
        if abs(coeff) < 1e-12:
            continue
        label = ["I"] * n
        label[idx[node]] = "Z"
        pauli_list.append(("".join(reversed(label)), coeff))

    for (i, j), coeff in quad_z.items():
        if abs(coeff) < 1e-12:
            continue
        label = ["I"] * n
        label[idx[i]] = "Z"
        label[idx[j]] = "Z"
        pauli_list.append(("".join(reversed(label)), coeff))

    if not pauli_list:  # degenerate all-zero Hamiltonian edge case
        pauli_list = [("I" * n, 0.0)]

    return SparsePauliOp.from_list(pauli_list)


def _qaoa_circuit(n: int, idx, linear_z, quad_z, gammas: np.ndarray, betas: np.ndarray) -> "QuantumCircuit":
    qc = QuantumCircuit(n)
    qc.h(range(n))  # start in uniform superposition |+>^n
    for gamma, beta in zip(gammas, betas):
        # cost unitary  exp(-i * gamma * H_cost)
        for node, coeff in linear_z.items():
            if abs(coeff) > 1e-12:
                qc.rz(2 * gamma * coeff, idx[node])
        for (i, j), coeff in quad_z.items():
            if abs(coeff) > 1e-12:
                qc.rzz(2 * gamma * coeff, idx[i], idx[j])
        # mixer unitary  exp(-i * beta * sum_i X_i)
        for q in range(n):
            qc.rx(2 * beta, q)
    return qc


def _qiskit_qaoa(nodes, linear_z, quad_z, reps: int, maxiter: int, seed: int):
    n = len(nodes)
    idx = {node: k for k, node in enumerate(nodes)}
    hamiltonian = _build_cost_hamiltonian(nodes, linear_z, quad_z)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=2 * reps)

    def expectation(params):
        gammas, betas = params[:reps], params[reps:]
        qc = _qaoa_circuit(n, idx, linear_z, quad_z, gammas, betas)
        sv = Statevector.from_instruction(qc)
        return float(np.real(sv.expectation_value(hamiltonian)))

    result = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter})

    gammas, betas = result.x[:reps], result.x[reps:]
    final_qc = _qaoa_circuit(n, idx, linear_z, quad_z, gammas, betas)
    sv = Statevector.from_instruction(final_qc)
    probs = sv.probabilities()  # index i -> qubit 0 is the LSB of i

    return probs, idx, result.fun, "qiskit-statevector"


# ---------------------------------------------------------------------
# Dependency-free NumPy backend (identical math, no qiskit import)
# ---------------------------------------------------------------------
_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _op_on_qubit(op, q, n):
    mats = [_I] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _op_on_two_qubits(op, q1, q2, n):
    """Build the ZZ operator on qubits q1,q2 as a product of two
    single-qubit Z operators embedded in the n-qubit space."""
    z1 = _op_on_qubit(op, q1, n)
    z2 = _op_on_qubit(op, q2, n)
    return z1 @ z2


def _numpy_qaoa(nodes, linear_z, quad_z, reps: int, maxiter: int, seed: int):
    """A from-scratch statevector simulation of the exact same QAOA
    circuit, using dense NumPy matrices. Only used if qiskit isn't
    importable -- fine for n<=8 qubits (<=256x256 matrices)."""
    n = len(nodes)
    idx = {node: k for k, node in enumerate(nodes)}
    dim = 2 ** n

    H = np.zeros((dim, dim), dtype=complex)
    for node, coeff in linear_z.items():
        if abs(coeff) > 1e-12:
            H += coeff * _op_on_qubit(_Z, idx[node], n)
    for (i, j), coeff in quad_z.items():
        if abs(coeff) > 1e-12:
            H += coeff * _op_on_two_qubits(_Z, idx[i], idx[j], n)

    X_sum = sum(_op_on_qubit(_X, q, n) for q in range(n))
    Z_single = {q: _op_on_qubit(_Z, q, n) for q in range(n)}
    ZZ_pair = {(i, j): _op_on_two_qubits(_Z, idx[i], idx[j], n) for (i, j) in quad_z}

    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi0 = plus
    for _ in range(n - 1):
        psi0 = np.kron(psi0, plus)

    def expm_diag_apply(diag_generator_matrix, angle, state):
        # generator_matrix is diagonal in the computational basis (built
        # purely from Z/ZZ terms), so exp(-i*angle*G) is just a
        # per-amplitude phase -- far cheaper than a generic matrix exp.
        phases = np.exp(-1j * angle * np.diag(diag_generator_matrix).real)
        return phases * state

    from scipy.linalg import expm

    def apply_mixer(angle, state):
        return expm(-1j * angle * X_sum) @ state

    def run_circuit(gammas, betas):
        state = psi0.copy()
        for gamma, beta in zip(gammas, betas):
            state = expm_diag_apply(H, gamma, state)
            state = apply_mixer(beta, state)
        return state

    def expectation(params):
        gammas, betas = params[:reps], params[reps:]
        state = run_circuit(gammas, betas)
        return float(np.real(np.conj(state) @ H @ state))

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=2 * reps)
    result = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter})

    gammas, betas = result.x[:reps], result.x[reps:]
    final_state = run_circuit(gammas, betas)
    probs = np.abs(final_state) ** 2

    return probs, idx, result.fun, "numpy-statevector"


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def solve_qaoa(
    h: Dict[int, float],
    Q: Dict[Tuple[int, int], float],
    offset: float,
    reps: int = 2,
    maxiter: int = 120,
    seed: int = 7,
    top_k: int = 5,
    verify_with_brute_force: bool = False,
):
    """Run QAOA on the given QUBO and return the best decoded solution.

    Returns a dict with:
        x            : {node: 0/1}  best bitstring found
        energy       : QUBO energy of that bitstring (lower = better)
        backend      : which simulator actually ran ("qiskit-statevector"
                       or "numpy-statevector")
        top_bitstrings: the `top_k` most probable measured bitstrings and
                        their QAOA-assigned probability, for the dashboard
                        to show "the quantum state's confidence"
        optimal_energy / optimal_x : brute-force ground truth, only
                        computed if `verify_with_brute_force=True`
                        (cheap since n<=8)
    """
    nodes, linear_z, quad_z, ising_offset = qubo_to_ising(h, Q, offset)
    n = len(nodes)

    if QISKIT_AVAILABLE:
        probs, idx, _, backend = _qiskit_qaoa(nodes, linear_z, quad_z, reps, maxiter, seed)
    else:
        probs, idx, _, backend = _numpy_qaoa(nodes, linear_z, quad_z, reps, maxiter, seed)

    order = np.argsort(-probs)[: max(top_k, 1)]
    candidates = []
    for state_int in order:
        bits = [(state_int >> k) & 1 for k in range(n)]  # qubit 0 = LSB
        x = {node: bits[idx[node]] for node in nodes}
        energy = evaluate_qubo(h, Q, offset, x)
        candidates.append({"x": x, "probability": float(probs[state_int]), "energy": energy})

    best = min(candidates, key=lambda c: c["energy"])

    result = {
        "x": best["x"],
        "energy": best["energy"],
        "backend": backend,
        "reps": reps,
        "top_bitstrings": candidates,
    }

    if verify_with_brute_force and n <= 12:
        opt_x, opt_e = brute_force_solve(h, Q, offset)
        result["optimal_x"] = opt_x
        result["optimal_energy"] = opt_e

    return result
