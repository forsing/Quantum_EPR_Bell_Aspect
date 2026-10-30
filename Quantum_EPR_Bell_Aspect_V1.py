#!/usr/bin/env python3

# Quantum_EPR_Bell_Aspect


"""
QUANTUM EPR-BELL-ASPECT V1 — Loto 7/39

Python 3.11
Qiskit 1.4.4
Dependencies: qiskit, numpy, scipy

Quantum model:
    learned entangled two-qubit blocks;
    complete Born probabilities 00, 01, 10, 11;
    one one-qubit block for the unpaired label;
    previous-draw occupancy selects the quantum codeword.

Pair circuit:
    RY(entangling_angle) on qubit 0;
    CX(0,1);
    RZ(phase) on qubit 0;
    RY(local_angle_0) on qubit 0;
    RY(local_angle_1) on qubit 1.

Qiskit Statevector computes every quantum codebook during fitting.
Analytic amplitude derivatives are independently checked.

Pairing:
    deterministic greedy matching using absolute occupancy
    correlations from the fitting prefix only;
    frozen during validation or test;
    no matching based on future observations.

Distribution:
    product of complete block Born probabilities;
    conditioned on exactly seven occupied labels;
    exact normalization, outcome marginals and maximum by DP.

Bell/CHSH:
    an ideal Bell-state self-check validates the quantum primitive;
    Bell violation is not the training objective;
    it is not evidence of quantum behavior in CSV data.

No classifier or classical output regressor.
No random initialization, sampling, injected noise, row shuffling,
rounding, duplicate repair or cosmetic combination filters.

Best valid parameters are checkpointed.
Optimizer convergence is reported separately from numerical validity.
"""


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import qiskit

from scipy.optimize import minimize
from scipy.special import logsumexp

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, Statevector


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
)

NUMBER_COUNT = 39
DRAW_SIZE = 7

RIDGES = (0.03, 0.3, 3.0)
INITIALIZATIONS = (0, 1)

OPTIMIZER_STEPS = 500
GRADIENT_LIMIT = 5e-5

BASE_PAIR_ANGLES = np.array(
    [0.9, 0.4, 0.3, -0.5],
    dtype=float,
)

BASE_SINGLE_ANGLE = 0.8
SINGLE_PHASE = math.pi / 4.0

CHECKPOINT_VERSION = "EPR_BELL_ASPECT_BLOCKS_V1_20261029"

CHECKPOINT_DIR = (
    Path(__file__).resolve().parent
    / "Quantum_EPR_Bell_Aspect_V1_checkpoints"
)

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

X = np.asarray(Pauli("X").to_matrix(), dtype=complex)
Y = np.asarray(Pauli("Y").to_matrix(), dtype=complex)
Z = np.asarray(Pauli("Z").to_matrix(), dtype=complex)

Y0 = np.asarray(Pauli("IY").to_matrix(), dtype=complex)
Y1 = np.asarray(Pauli("YI").to_matrix(), dtype=complex)
Z0 = np.asarray(Pauli("IZ").to_matrix(), dtype=complex)

_cx_circuit = QuantumCircuit(2)
_cx_circuit.cx(0, 1)
CX01 = np.asarray(Operator(_cx_circuit).data, dtype=complex)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def safe_log(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)

    require(
        np.all(np.isfinite(probabilities))
        and np.all(probabilities >= 0),
        "Neispravne verovatnoće.",
    )

    result = np.full_like(probabilities, -np.inf)
    positive = probabilities > 0
    result[positive] = np.log(probabilities[positive])

    return result


def rotation(generator, angle):
    return (
        math.cos(float(angle) / 2.0) * I4
        - 1j * math.sin(float(angle) / 2.0) * generator
    )


def pair_state(angles, derivatives=False):
    angles = np.asarray(angles, dtype=float)

    require(
        angles.shape == (4,)
        and np.all(np.isfinite(angles)),
        "Neispravni uglovi dvokubitnog kola.",
    )

    circuit = QuantumCircuit(2)

    circuit.ry(float(angles[0]), 0)
    circuit.cx(0, 1)
    circuit.rz(float(angles[1]), 0)
    circuit.ry(float(angles[2]), 0)
    circuit.ry(float(angles[3]), 1)

    state = np.asarray(
        Statevector.from_instruction(circuit).data,
        dtype=complex,
    )

    probabilities = np.abs(state) ** 2

    require(
        np.all(np.isfinite(state))
        and abs(float(probabilities.sum()) - 1.0) < 1e-10,
        "Qiskit dvokubitno stanje nije normalizovano.",
    )

    if not derivatives:
        return state, probabilities, None

    gates = (
        rotation(Y0, angles[0]),
        CX01,
        rotation(Z0, angles[1]),
        rotation(Y0, angles[2]),
        rotation(Y1, angles[3]),
    )

    prefix = [np.array([1, 0, 0, 0], dtype=complex)]

    for gate in gates:
        prefix.append(gate @ prefix[-1])

    require(
        np.max(np.abs(prefix[-1] - state)) < 1e-10,
        "Izvodni amplitudni proračun nije saglasan sa Qiskitom.",
    )

    derivative = np.empty((4, 4), dtype=float)

    for parameter, (position, generator) in enumerate(
        ((0, Y0), (2, Z0), (3, Y0), (4, Y1))
    ):
        tangent = -0.5j * generator @ prefix[position + 1]

        for gate in gates[position + 1:]:
            tangent = gate @ tangent

        derivative[:, parameter] = 2.0 * np.real(
            np.conj(state) * tangent
        )

    require(
        np.max(np.abs(derivative.sum(axis=0))) < 1e-9,
        "Izvodi Born verovatnoća ne čuvaju normalizaciju.",
    )

    return state, probabilities, derivative


def single_state(angle, derivatives=False):
    angle = float(angle)

    require(
        math.isfinite(angle),
        "Neispravan ugao jednokubitnog kola.",
    )

    circuit = QuantumCircuit(1)
    circuit.ry(angle, 0)
    circuit.rz(SINGLE_PHASE, 0)

    state = np.asarray(
        Statevector.from_instruction(circuit).data,
        dtype=complex,
    )

    probabilities = np.abs(state) ** 2

    require(
        abs(float(probabilities.sum()) - 1.0) < 1e-10,
        "Jednokubitno stanje nije normalizovano.",
    )

    if not derivatives:
        return state, probabilities, None

    phases = np.array(
        [
            np.exp(-0.5j * SINGLE_PHASE),
            np.exp(0.5j * SINGLE_PHASE),
        ]
    )

    tangent = phases * np.array(
        [
            -0.5 * math.sin(angle / 2.0),
            0.5 * math.cos(angle / 2.0),
        ]
    )

    derivative = (
        2.0 * np.real(np.conj(state) * tangent)
    )[:, None]

    return state, probabilities, derivative


def training_pairing(latest):
    """
    Graph preprocessing uses only the supplied fitting histories.
    It is not an output-frequency ranking.
    """
    latest = np.asarray(latest, dtype=float)
    n, m = latest.shape

    require(
        n > 0
        and np.all((latest == 0) | (latest == 1)),
        "Neispravan ulaz za izbor parova.",
    )

    mean = latest.mean(axis=0)
    covariance = latest.T @ latest / n - np.outer(mean, mean)

    variance = mean * (1.0 - mean)
    denominator = np.sqrt(
        variance[:, None] * variance[None, :]
    )

    correlation = np.zeros_like(covariance)

    np.divide(
        covariance,
        denominator,
        out=correlation,
        where=denominator > 0,
    )

    candidates = sorted(
        (
            (-abs(float(correlation[first, second])), first, second)
            for first in range(m)
            for second in range(first + 1, m)
        )
    )

    unused = set(range(m))
    blocks = []

    for _, first, second in candidates:
        if first in unused and second in unused:
            blocks.append((first, second))
            unused.remove(first)
            unused.remove(second)

    blocks.extend((number,) for number in sorted(unused))
    blocks.sort()

    flattened = [
        number
        for block in blocks
        for number in block
    ]

    require(
        sorted(flattened) == list(range(m)),
        "Parovi ne pokrivaju sve oznake tačno jednom.",
    )

    return tuple(blocks)


class QuantumBlocks:
    def __init__(self, blocks):
        self.blocks = tuple(tuple(block) for block in blocks)
        self.m = sum(len(block) for block in self.blocks)
        self.block_count = len(self.blocks)

        require(
            sorted(number for block in self.blocks for number in block)
            == list(range(self.m))
            and all(len(block) in (1, 2) for block in self.blocks),
            "Neispravna struktura kvantnih blokova.",
        )

        self.entries = []
        self.nout = []
        self.degrees = np.zeros((self.block_count, 4), dtype=int)
        position = 0

        for block_index, block in enumerate(self.blocks):
            outcome_count = 1 << len(block)
            width = 4 if len(block) == 2 else 1

            self.nout.append(outcome_count)

            for outcome in range(outcome_count):
                self.degrees[block_index, outcome] = (
                    int(outcome & 1)
                    + int((outcome >> 1) & 1)
                )

            for code in range(outcome_count):
                self.entries.append(
                    (block_index, code, position, width)
                )
                position += width

        self.parameter_count = position

    def initial(self, initialization=0):
        if initialization == 0:
            return np.zeros(self.parameter_count)

        return 0.005 * np.sin(
            np.arange(self.parameter_count) + 0.7
        )

    def encode(self, occupancy):
        occupancy = np.asarray(occupancy)

        require(
            occupancy.ndim == 2
            and occupancy.shape[1] == self.m
            and np.all((occupancy == 0) | (occupancy == 1)),
            "Neispravne binarne oznake za kvantne blokove.",
        )

        result = np.empty(
            (len(occupancy), self.block_count),
            dtype=np.int64,
        )

        for block_index, block in enumerate(self.blocks):
            result[:, block_index] = occupancy[:, block[0]]

            if len(block) == 2:
                result[:, block_index] += (
                    2 * occupancy[:, block[1]]
                )

        return result

    def state_for_entry(self, parameters, block_index, start, width,
                        derivatives=False):
        if width == 4:
            angles = (
                BASE_PAIR_ANGLES
                + parameters[start:start + width]
            )

            return pair_state(angles, derivatives)

        angle = BASE_SINGLE_ANGLE + parameters[start]

        return single_state(angle, derivatives)

    def lookup(self, parameters, derivatives=False):
        parameters = np.asarray(parameters, dtype=float)

        require(
            parameters.shape == (self.parameter_count,)
            and np.all(np.isfinite(parameters)),
            "Neispravni kvantni parametri.",
        )

        probabilities = np.zeros((self.block_count, 4, 4))

        jacobian = (
            np.zeros((self.block_count, 4, 4, 4))
            if derivatives
            else None
        )

        for block_index, code, start, width in self.entries:
            _, local_probability, local_derivative = self.state_for_entry(
                parameters,
                block_index,
                start,
                width,
                derivatives,
            )

            count = self.nout[block_index]

            require(
                np.all(local_probability > 0),
                "Kodna reč ima nultu numeričku Born verovatnoću.",
            )

            probabilities[block_index, code, :count] = (
                local_probability
            )

            if derivatives:
                jacobian[
                    block_index,
                    code,
                    :count,
                    :width,
                ] = local_derivative

        return probabilities, jacobian

    def select(self, lookup, codes):
        require(
            codes.ndim == 2
            and codes.shape[1] == self.block_count,
            "Neispravni indeksi kodnih reči.",
        )

        for block_index, count in enumerate(self.nout):
            require(
                np.all(
                    (codes[:, block_index] >= 0)
                    & (codes[:, block_index] < count)
                ),
                "Kodna reč je van opsega.",
            )

        probabilities = lookup[
            np.arange(self.block_count)[None, :],
            codes,
        ]

        require(
            np.max(np.abs(
                probabilities.sum(axis=2) - 1.0
            )) < 1e-10,
            "Izabrane blokovske verovatnoće nisu normalizovane.",
        )

        return probabilities


def conditional_partition(probabilities, source, k, marginals=False):
    probabilities = np.asarray(probabilities, dtype=float)

    require(
        probabilities.ndim == 3
        and probabilities.shape[1:] == (source.block_count, 4),
        "Neispravan oblik zajedničkih Born verovatnoća.",
    )

    n = len(probabilities)
    block_count = source.block_count
    log_probability = safe_log(probabilities)

    forward = np.full((n, block_count + 1, k + 1), -np.inf)
    forward[:, 0, 0] = 0.0

    for block in range(block_count):
        for outcome in range(source.nout[block]):
            degree = int(source.degrees[block, outcome])

            if degree > k:
                continue

            forward[:, block + 1, degree:] = np.logaddexp(
                forward[:, block + 1, degree:],
                forward[:, block, :k + 1 - degree]
                + log_probability[:, block, outcome, None],
            )

    log_z = forward[:, block_count, k]

    require(
        np.all(np.isfinite(log_z))
        and np.max(log_z) <= 1e-8,
        "Neispravna verovatnoća uslova tačno sedam.",
    )

    if not marginals:
        return log_z, None

    backward = np.full_like(forward, -np.inf)
    backward[:, block_count, 0] = 0.0

    for block in range(block_count - 1, -1, -1):
        for outcome in range(source.nout[block]):
            degree = int(source.degrees[block, outcome])

            if degree > k:
                continue

            backward[:, block, degree:] = np.logaddexp(
                backward[:, block, degree:],
                backward[:, block + 1, :k + 1 - degree]
                + log_probability[:, block, outcome, None],
            )

    require(
        np.max(np.abs(
            backward[:, 0, k] - log_z
        )) < 1e-8,
        "Prednja i zadnja normalizacija nisu saglasne.",
    )

    posterior = np.zeros_like(probabilities)

    for block in range(block_count):
        for outcome in range(source.nout[block]):
            degree = int(source.degrees[block, outcome])

            if degree > k:
                continue

            terms = (
                forward[:, block, :k + 1 - degree]
                + backward[:, block + 1, k - degree::-1]
            )

            posterior[:, block, outcome] = np.exp(
                log_probability[:, block, outcome]
                + logsumexp(terms, axis=1)
                - log_z
            )

    require(
        np.all(np.isfinite(posterior))
        and np.min(posterior) >= -1e-9
        and np.max(posterior) <= 1.0 + 1e-9
        and np.max(np.abs(
            posterior.sum(axis=2) - 1.0
        )) < 1e-7,
        "Uslovne blokovske margine nisu ispravne.",
    )

    expected_count = np.sum(
        posterior * source.degrees[None, :, :],
        axis=(1, 2),
    )

    require(
        np.max(np.abs(expected_count - k)) < 1e-7,
        "Zbir uslovnih zauzeća nije sedam.",
    )

    return log_z, posterior


def objective(parameters, source, codes, outcomes, k, ridge,
              gradient=True):
    lookup, jacobian = source.lookup(
        parameters,
        derivatives=gradient,
    )

    probabilities = source.select(lookup, codes)

    log_z, posterior = conditional_partition(
        probabilities,
        source,
        k,
        marginals=gradient,
    )

    rows = np.arange(len(outcomes))[:, None]
    blocks = np.arange(source.block_count)[None, :]

    observed_probabilities = probabilities[rows, blocks, outcomes]

    require(
        np.all(observed_probabilities > 0),
        "Posmatrana kombinacija ima nultu Born verovatnoću.",
    )

    logp = (
        np.log(observed_probabilities).sum(axis=1)
        - log_z
    )

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna verovatnoća kompletne sedmorke.",
    )

    loss = (
        -float(logp.mean())
        + 0.5 * ridge * float(np.mean(parameters * parameters))
    )

    require(np.isfinite(loss), "Funkcija cilja nije konačna.")

    if not gradient:
        return loss

    residual = posterior.copy()
    residual[rows, blocks, outcomes] -= 1.0

    probability_gradient = np.zeros_like(probabilities)

    np.divide(
        residual,
        probabilities,
        out=probability_gradient,
        where=probabilities > 0,
    )

    probability_gradient /= len(outcomes)

    lookup_gradient = np.zeros_like(lookup)

    for block in range(source.block_count):
        np.add.at(
            lookup_gradient[block],
            codes[:, block],
            probability_gradient[:, block, :],
        )

    result = np.zeros_like(parameters)

    for block, code, start, width in source.entries:
        count = source.nout[block]

        result[start:start + width] = np.sum(
            lookup_gradient[block, code, :count, None]
            * jacobian[block, code, :count, :width],
            axis=0,
        )

    result += ridge * parameters / len(parameters)

    require(
        np.all(np.isfinite(result)),
        "Gradijent nije konačan.",
    )

    return loss, result


def outcome_numbers(block, outcome):
    selected = []

    if outcome & 1:
        selected.append(block[0])

    if len(block) == 2 and outcome & 2:
        selected.append(block[1])

    return tuple(selected)


def exact_maximum(probabilities, source, k):
    log_probability = safe_log(probabilities)

    values = np.full(k + 1, -np.inf)
    paths = [None] * (k + 1)

    values[0] = 0.0
    paths[0] = ()

    for block_index, block in enumerate(source.blocks):
        following = np.full_like(values, -np.inf)
        next_paths = [None] * (k + 1)

        for count in range(k + 1):
            if paths[count] is None:
                continue

            for outcome in range(source.nout[block_index]):
                degree = int(source.degrees[block_index, outcome])
                new_count = count + degree

                if new_count > k:
                    continue

                score = (
                    values[count]
                    + log_probability[block_index, outcome]
                )

                if not np.isfinite(score):
                    continue

                candidate = tuple(sorted(
                    paths[count] + outcome_numbers(block, outcome)
                ))

                old_path = next_paths[new_count]
                old_score = following[new_count]

                if (
                    old_path is None
                    or score > old_score
                    or (score == old_score and candidate < old_path)
                ):
                    following[new_count] = score
                    next_paths[new_count] = candidate

        values = following
        paths = next_paths

    require(
        paths[k] is not None
        and len(paths[k]) == k
        and len(set(paths[k])) == k,
        "Tačan maksimum nije pronađen.",
    )

    log_z = conditional_partition(
        probabilities[None, :, :],
        source,
        k,
    )[0][0]

    logp = float(values[k] - log_z)

    require(
        -math.log(math.comb(source.m, k)) - 1e-7
        <= logp <= 1e-7,
        "Neispravna verovatnoća maksimuma.",
    )

    return paths[k], logp


def inclusion_marginals(posterior, source):
    result = np.zeros((len(posterior), source.m))

    for block_index, block in enumerate(source.blocks):
        for outcome in range(source.nout[block_index]):
            for number in outcome_numbers(block, outcome):
                result[:, number] += posterior[:, block_index, outcome]

    return result


def factorize_blocks(probabilities, source):
    result = probabilities.copy()

    for block, labels in enumerate(source.blocks):
        if len(labels) != 2:
            continue

        p = probabilities[:, block, :]
        first_one = p[:, 1] + p[:, 3]
        second_one = p[:, 2] + p[:, 3]

        result[:, block, 0] = (
            (1.0 - first_one) * (1.0 - second_one)
        )
        result[:, block, 1] = first_one * (1.0 - second_one)
        result[:, block, 2] = (1.0 - first_one) * second_one
        result[:, block, 3] = first_one * second_one

    return result


def load_draws(path):
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()

    rows = []
    blank = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not item.strip() for item in row):
                blank += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: potrebno je 7 kolona.",
            )

            try:
                numbers = tuple(int(item.strip()) for item in row)
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(numbers)) == DRAW_SIZE
                and min(numbers) >= 1
                and max(numbers) <= NUMBER_COUNT,
                f"{path.name}, red {line_number}: neispravna sedmorka.",
            )

            rows.append(
                tuple(value - 1 for value in sorted(numbers))
            )

    require(len(rows) >= 100, "Nedovoljno istorijskih kola.")

    draws = np.asarray(rows, dtype=np.int64)
    occupancy = np.zeros((len(draws), NUMBER_COUNT), dtype=np.int64)

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, occupancy, fingerprint


def checkpoint_path(latest, targets, source, ridge, initialization):
    digest = hashlib.sha256()

    digest.update(repr(
        (
            CHECKPOINT_VERSION,
            source.blocks,
            float(ridge),
            initialization,
            OPTIMIZER_STEPS,
            qiskit.__version__,
            np.__version__,
        )
    ).encode("utf-8"))

    for array in (latest, targets):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())

    return CHECKPOINT_DIR / (digest.hexdigest() + ".npz")


def save_checkpoint(path, parameters, iterations, done, message):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")

    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                parameters=np.asarray(parameters, dtype=float),
                iterations=np.array(int(iterations)),
                done=np.array(int(done)),
                message=np.array(str(message)),
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, shape):
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as saved:
            parameters = saved["parameters"].astype(float)
            iterations = int(saved["iterations"].item())
            done = bool(saved["done"].item())
            message = str(saved["message"].item())

        require(
            parameters.shape == shape
            and np.all(np.isfinite(parameters))
            and iterations >= 0,
            "Neispravna kontrolna tačka.",
        )

        return parameters, iterations, done, message

    except (
        OSError, ValueError, KeyError, RuntimeError,
        EOFError, BadZipFile,
    ) as error:
        say(
            f"Kontrolna tačka nije učitana: {path.name}",
            str(error),
        )
        return None


@dataclass
class Fitted:
    source: QuantumBlocks
    parameters: np.ndarray
    lookup: np.ndarray
    loss: float
    gradient_norm: float
    converged: bool
    message: str


def fit(latest, target_occupancy, ridge, label):
    source = QuantumBlocks(training_pairing(latest))
    codes = source.encode(latest)
    outcomes = source.encode(target_occupancy)

    accepted = []

    say(
        "",
        label,
        f"λ={ridge:g} | prelaza={len(latest)}",
        f"Kvantnih parametara: {source.parameter_count}",
    )

    for initialization in INITIALIZATIONS:
        initial = source.initial(initialization)

        path = checkpoint_path(
            latest,
            target_occupancy,
            source,
            ridge,
            initialization,
        )

        saved = load_checkpoint(path, initial.shape)

        parameters = initial.copy()
        used_iterations = 0
        done = False
        message = "Novi proračun."

        if saved is not None:
            parameters, used_iterations, done, message = saved

        started = time.perf_counter()

        try:
            best_loss, best_gradient = objective(
                parameters, source, codes, outcomes,
                DRAW_SIZE, ridge,
            )

        except (
            RuntimeError, FloatingPointError, np.linalg.LinAlgError,
        ) as error:
            if saved is None:
                say(
                    f"Početak {initialization} nije validan:",
                    str(error),
                )
                continue

            parameters = initial.copy()
            used_iterations = 0
            done = False

            best_loss, best_gradient = objective(
                parameters, source, codes, outcomes,
                DRAW_SIZE, ridge,
            )

        best_parameters = parameters.copy()
        best_gradient = best_gradient.copy()

        added_iterations = 0
        optimizer_success = False

        if done:
            say(
                f"Početak {initialization}: učitan završen proračun."
            )

        else:
            save_checkpoint(
                path, best_parameters, used_iterations, False,
                "Početna ili nastavljena validna tačka.",
            )

            def function(current):
                nonlocal best_loss, best_parameters, best_gradient

                loss, gradient = objective(
                    current, source, codes, outcomes,
                    DRAW_SIZE, ridge,
                )

                if loss < best_loss:
                    best_loss = float(loss)
                    best_parameters = current.copy()
                    best_gradient = gradient.copy()

                return loss, gradient

            def callback(current):
                nonlocal added_iterations
                added_iterations += 1

                if added_iterations % 25 == 0:
                    total = used_iterations + added_iterations

                    save_checkpoint(
                        path, best_parameters, total, False,
                        "Sačuvana najbolja validna tačka.",
                    )

                    say(
                        f"Početak {initialization}"
                        f" | iteracija {total}"
                        f" | proteklo {time.perf_counter() - started:.1f} s",
                        f"Najbolji cilj: {best_loss:.9f}",
                        f"Max|grad|: {np.max(np.abs(best_gradient)):.3e}",
                    )

            remaining = max(
                0, OPTIMIZER_STEPS - used_iterations
            )

            if remaining:
                try:
                    result = minimize(
                        function,
                        parameters,
                        method="L-BFGS-B",
                        jac=True,
                        callback=callback,
                        options={
                            "maxiter": remaining,
                            "maxls": 60,
                            "maxcor": 20,
                            "ftol": 1e-13,
                            "gtol": 1e-6,
                        },
                    )

                    optimizer_success = bool(result.success)
                    message = str(result.message)
                    function(result.x)

                except (
                    RuntimeError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ) as error:
                    message = (
                        "Pretraga prekinuta; zadržana validna tačka: "
                        + str(error)
                    )
            else:
                message = "Dostignut sačuvani budžet iteracija."

            used_iterations += added_iterations

        loss, gradient = objective(
            best_parameters, source, codes, outcomes,
            DRAW_SIZE, ridge,
        )

        maximum_gradient = float(np.max(np.abs(gradient)))
        converged = maximum_gradient <= GRADIENT_LIMIT

        lookup, _ = source.lookup(best_parameters)

        if not done:
            message = (
                "OPTIMIZER_SUCCESS: "
                if optimizer_success
                else "OPTIMIZER_NOT_CONFIRMED: "
            ) + message

            save_checkpoint(
                path, best_parameters, used_iterations,
                True, message,
            )

        accepted.append(
            Fitted(
                source=source,
                parameters=best_parameters.copy(),
                lookup=lookup.copy(),
                loss=float(loss),
                gradient_norm=maximum_gradient,
                converged=converged,
                message=message,
            )
        )

        say(
            f"Početak {initialization}: zadržan cilj {loss:.9f}",
            f"Max|grad|: {maximum_gradient:.3e}",
            "Gradijentni uslov: "
            + ("dostignut" if converged else "NIJE dostignut"),
            message,
        )

    require(
        bool(accepted),
        f"{label}: nema nijedne numerički ispravne tačke.",
    )

    return min(accepted, key=lambda item: item.loss)


def evaluate(model, latest, target_occupancy,
             hits=False, factorized=False):
    source = model.source
    codes = source.encode(latest)
    outcomes = source.encode(target_occupancy)

    probabilities = source.select(model.lookup, codes)

    if factorized:
        probabilities = factorize_blocks(probabilities, source)

    log_z, _ = conditional_partition(
        probabilities, source, DRAW_SIZE
    )

    rows = np.arange(len(outcomes))[:, None]
    blocks = np.arange(source.block_count)[None, :]

    chosen = probabilities[rows, blocks, outcomes]

    require(
        np.all(chosen > 0),
        "Nulta test verovatnoća.",
    )

    logp = np.log(chosen).sum(axis=1) - log_z

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna verovatnoća na testu.",
    )

    histogram = np.zeros(DRAW_SIZE + 1, dtype=np.int64)

    if hits:
        for row in range(len(latest)):
            predicted, _ = exact_maximum(
                probabilities[row],
                source,
                DRAW_SIZE,
            )

            actual = np.flatnonzero(target_occupancy[row])

            histogram[
                len(set(predicted).intersection(actual.tolist()))
            ] += 1

        require(
            int(histogram.sum()) == len(latest),
            "Histogram pogodaka nije potpun.",
        )

    return -float(logp.mean()), histogram


def self_checks():
    say(
        "",
        "Pokrećem ugrađene kvantne i matematičke provere.",
    )

    started = time.perf_counter()

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)

    bell_state = Statevector.from_instruction(bell).data

    b0 = (Z + X) / math.sqrt(2.0)
    b1 = (Z - X) / math.sqrt(2.0)

    chsh_operator = (
        np.kron(b0, Z)
        + np.kron(b1, Z)
        + np.kron(b0, X)
        - np.kron(b1, X)
    )

    chsh = float(np.vdot(
        bell_state,
        chsh_operator @ bell_state,
    ).real)

    require(
        abs(chsh - 2.0 * math.sqrt(2.0)) < 1e-11,
        "Idealna Bell/CHSH provera nije prošla.",
    )

    source = QuantumBlocks(((0, 3), (1, 4), (2,)))
    parameters = source.initial(1)

    latest = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
        ],
        dtype=np.int64,
    )

    target = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=np.int64,
    )

    codes = source.encode(latest)
    outcomes = source.encode(target)

    k = 2
    ridge = 0.03

    _, analytic = objective(
        parameters, source, codes, outcomes, k, ridge
    )

    epsilon = 1e-6
    numerical = np.empty_like(parameters)

    for parameter in range(len(parameters)):
        positive = parameters.copy()
        negative = parameters.copy()

        positive[parameter] += epsilon
        negative[parameter] -= epsilon

        numerical[parameter] = (
            objective(
                positive, source, codes, outcomes,
                k, ridge, gradient=False,
            )
            - objective(
                negative, source, codes, outcomes,
                k, ridge, gradient=False,
            )
        ) / (2.0 * epsilon)

    gradient_error = float(
        np.max(np.abs(analytic - numerical))
    )

    require(
        gradient_error < 3e-6,
        f"Provera gradijenta nije prošla: {gradient_error:.3e}",
    )

    lookup, _ = source.lookup(parameters)
    probabilities = source.select(lookup, codes)

    log_z, posterior = conditional_partition(
        probabilities, source, k, marginals=True
    )

    subsets = list(combinations(range(source.m), k))
    all_y = np.zeros((len(subsets), source.m), dtype=np.int64)

    for row, selected in enumerate(subsets):
        all_y[row, list(selected)] = 1

    all_outcomes = source.encode(all_y)

    for sample in range(len(latest)):
        direct_scores = np.zeros(len(subsets))

        for block in range(source.block_count):
            direct_scores += np.log(
                probabilities[
                    sample,
                    block,
                    all_outcomes[:, block],
                ]
            )

        direct_log_z = float(logsumexp(direct_scores))
        normalized = np.exp(direct_scores - direct_log_z)

        require(
            abs(direct_log_z - log_z[sample]) < 1e-11
            and abs(float(normalized.sum()) - 1.0) < 1e-12,
            "Potpuna provera normalizacije nije prošla.",
        )

        direct_posterior = np.zeros((source.block_count, 4))

        for row, probability in enumerate(normalized):
            for block in range(source.block_count):
                direct_posterior[
                    block, all_outcomes[row, block]
                ] += probability

        require(
            np.max(np.abs(
                direct_posterior - posterior[sample]
            )) < 1e-10,
            "Potpuna provera zajedničkih margina nije prošla.",
        )

        best, score = exact_maximum(
            probabilities[sample], source, k
        )

        maximum_index = int(np.argmax(direct_scores))

        require(
            best == subsets[maximum_index]
            and abs(
                score
                - (direct_scores[maximum_index] - direct_log_z)
            ) < 1e-10,
            "Provera tačnog maksimuma nije prošla.",
        )

    for boundary_k in (1, source.m):
        z, boundary_posterior = conditional_partition(
            probabilities[:1],
            source,
            boundary_k,
            marginals=True,
        )

        inclusion = inclusion_marginals(
            boundary_posterior,
            source,
        )

        require(
            np.all(np.isfinite(z))
            and abs(
                float(inclusion.sum()) - boundary_k
            ) < 1e-9,
            "Granična provera nije prošla.",
        )

    pairing = training_pairing(latest)

    require(
        sorted(number for block in pairing for number in block)
        == list(range(source.m)),
        "Provera izbora parova nije prošla.",
    )

    # Local Bob operations must not change Alice's marginal.
    angles = BASE_PAIR_ANGLES + np.array([0.1, -0.2, 0.15, 0.05])
    state, p, _ = pair_state(angles)

    alice_one = float(p[1] + p[3])

    for angle in (0.0, 0.31, 1.17):
        local = QuantumCircuit(2)
        local.ry(angle, 1)

        transformed = Statevector(state).evolve(local).probabilities()

        require(
            abs(
                float(transformed[1] + transformed[3])
                - alice_one
            ) < 1e-11,
            "Provera lokalnih marginalnih verovatnoća nije prošla.",
        )

    say(
        "Ugrađene provere su prošle.",
        f"Idealni Bell CHSH: {chsh:.12f}",
        f"Najveća greška gradijenta: {gradient_error:.3e}",
        "Provereni su amplitude, normalizacija, zajedničke margine i maksimum.",
        f"Trajanje provera: {time.perf_counter() - started:.1f} s",
    )


def longest_run(numbers):
    best = 1
    current = 1

    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        best = max(best, current)

    return best


def quantum_next_diagnostics(model, latest_row, selected):
    source = model.source
    codes = source.encode(latest_row[None, :])[0]

    selected_set = set(selected)
    log_raw_probability = 0.0

    concurrences = []
    joint_differences = []

    entry_map = {
        (block, code): (start, width)
        for block, code, start, width in source.entries
    }

    for block_index, block in enumerate(source.blocks):
        code = int(codes[block_index])
        start, width = entry_map[(block_index, code)]

        state, p, _ = source.state_for_entry(
            model.parameters,
            block_index,
            start,
            width,
        )

        outcome = int(block[0] in selected_set)

        if len(block) == 2:
            outcome += 2 * int(block[1] in selected_set)

            concurrence = 2.0 * abs(
                state[0] * state[3] - state[1] * state[2]
            )

            require(
                -1e-10 <= concurrence <= 1.0 + 1e-9,
                "Neispravna dvokubitna spregnutost.",
            )

            concurrences.append(float(concurrence))

            first_one = float(p[1] + p[3])
            second_one = float(p[2] + p[3])

            joint_differences.append(
                abs(float(p[3]) - first_one * second_one)
            )

        require(
            p[outcome] > 0,
            "NEXT ima nultu kvantnu amplitudu.",
        )

        log_raw_probability += math.log(float(p[outcome]))

    return (
        log_raw_probability,
        float(np.mean(concurrences)) if concurrences else 0.0,
        float(np.max(joint_differences)) if joint_differences else 0.0,
    )


def print_result(result):
    say(
        "",
        result["name"],
        "NEXT: "
        + " ".join(f"{number:02d}" for number in result["next"]),
        f"Validacioni NLL: {result['validation']:.9f}",
        f"Odvojeni test NLL: {result['test']:.9f}",
        "Test NLL posle uklanjanja unutrašnjih parnih korelacija:"
        f" {result['factorized_test']:.9f}",
        "Pogodaka 0–7: "
        + " ".join(str(int(value)) for value in result["histogram"]),
        f"Prosečan broj pogodaka na testu: {result['mean_hits']:.6f}",
        f"Modelna verovatnoća NEXT: {result['probability']:.12g}",
        f"Zbir uslovnih uključnih margina: {result['marginal_sum']:.12f}",
        f"Prosečna spregnutost NEXT parova: {result['concurrence']:.9f}",
        f"Najveća parna Born zavisnost: {result['joint_dependence']:.9e}",
        f"Greška direktne Qiskit provere: {result['quantum_error']:.3e}",
        f"Osetljivost na istorijski ulaz: {result['sensitivity']:.9e}",
        f"Max|grad|: {result['gradient']:.3e}",
        "Gradijentni uslov: "
        + ("dostignut" if result["converged"] else "NIJE dostignut"),
    )

    for warning in result["warnings"]:
        say(f"DIJAGNOSTIKA: {warning}")


def process_file(path, draws, occupancy):
    latest = occupancy[:-1]
    target = occupancy[1:]

    n = len(target)

    split60 = int(0.60 * n)
    split70 = int(0.70 * n)
    split80 = int(0.80 * n)

    require(
        50 < split60 < split70 < split80 < n,
        "Nedovoljno istorije za hronološku podelu.",
    )

    scores = []

    for ridge in RIDGES:
        total_nll = 0.0
        total_count = 0
        incomplete_folds = 0

        for fold, (train_end, validation_end) in enumerate(
            ((split60, split70), (split70, split80)),
            1,
        ):
            model = fit(
                latest[:train_end],
                target[:train_end],
                ridge,
                f"{path.name} / validacija {fold}",
            )

            incomplete_folds += int(not model.converged)

            score = evaluate(
                model,
                latest[train_end:validation_end],
                target[train_end:validation_end],
            )[0]

            count = validation_end - train_end
            total_nll += count * score
            total_count += count

            say(f"Validacioni NLL: {score:.9f}")

        score = total_nll / total_count
        scores.append((score, ridge, incomplete_folds))

        say(
            f"KANDIDAT: λ={ridge:g}",
            f"Zbirni validacioni NLL: {score:.9f}",
        )

    validation_nll, ridge, incomplete_folds = min(
        scores,
        key=lambda item: item[0],
    )

    say(
        "",
        f"IZABRANO: λ={ridge:g}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit(
        latest[:split80],
        target[:split80],
        ridge,
        f"{path.name} / odvojeni test",
    )

    test_nll, histogram = evaluate(
        test_model,
        latest[split80:],
        target[split80:],
        hits=True,
    )

    # Diagnostic only. It does not change parameter selection.
    factorized_test_nll = evaluate(
        test_model,
        latest[split80:],
        target[split80:],
        factorized=True,
    )[0]

    say(
        f"ODVOJENI TEST: {n - split80} prelaza",
        f"Zajednički kvantni NLL: {test_nll:.9f}",
        f"Kontrola bez parnih korelacija: {factorized_test_nll:.9f}",
    )

    final_model = fit(
        latest,
        target,
        ridge,
        f"{path.name} / završno učenje svih {n} prelaza",
    )

    source = final_model.source

    # Re-execute the final quantum codebook.
    final_model.lookup = source.lookup(
        final_model.parameters
    )[0]

    next_codes = source.encode(occupancy[-1:])
    next_probabilities = source.select(
        final_model.lookup,
        next_codes,
    )

    log_z, posterior = conditional_partition(
        next_probabilities,
        source,
        DRAW_SIZE,
        marginals=True,
    )

    best, logp = exact_maximum(
        next_probabilities[0],
        source,
        DRAW_SIZE,
    )

    inclusion = inclusion_marginals(posterior, source)

    raw_logp, concurrence, joint_dependence = quantum_next_diagnostics(
        final_model,
        occupancy[-1],
        best,
    )

    quantum_error = abs(
        raw_logp - float(log_z[0]) - logp
    )

    require(
        quantum_error < 2e-9,
        "Direktna Qiskit provera NEXT verovatnoće nije prošla.",
    )

    old_probabilities = source.select(
        final_model.lookup,
        source.encode(latest[-8:]),
    )

    _, old_posterior = conditional_partition(
        old_probabilities,
        source,
        DRAW_SIZE,
        marginals=True,
    )

    old_inclusion = inclusion_marginals(
        old_posterior,
        source,
    )

    sensitivity = float(
        np.max(np.abs(old_inclusion - inclusion[0]))
    )

    warnings = []

    if incomplete_folds:
        warnings.append(
            "Izabrani kandidat ima validacione treninge"
            " bez potvrđenog gradijentnog uslova."
        )

    if not test_model.converged:
        warnings.append(
            "Model odvojenog testa nema potvrđen gradijentni uslov."
        )

    if not final_model.converged:
        warnings.append(
            "Završni model nema potvrđenu konvergenciju;"
            " NEXT je maksimum njegove validne raspodele."
        )

    run = longest_run(best)

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if best[-1] - best[0] <= 9:
        warnings.append(
            "Svih sedam NEXT brojeva je u rasponu od deset brojeva."
        )

    repeats = int(np.count_nonzero(
        np.all(draws == np.asarray(best)[None, :], axis=1)
    ))

    if repeats:
        warnings.append(
            f"NEXT se pojavljuje {repeats} puta u istoriji."
        )

    if best == tuple(int(value) for value in draws[-1]):
        warnings.append(
            "NEXT je jednak poslednjem istorijskom kolu."
        )

    if sensitivity < 1e-8:
        warnings.append(
            "Uslovne margine slabo reaguju na proverene kontekste."
        )

    if joint_dependence < 1e-8:
        warnings.append(
            "NEXT parovi imaju gotovo faktorizovane Born raspodele."
        )

    if test_nll >= factorized_test_nll:
        warnings.append(
            "Na odvojenom testu zajedničke parne korelacije"
            " nisu poboljšale NLL u odnosu na kontrolu"
            " sa istim uključenim marginalnim verovatnoćama blokova."
        )

    say(
        "",
        "Završni parovi:",
        " | ".join(
            "-".join(str(number + 1) for number in block)
            for block in source.blocks
        ),
        "Parovi su određeni samo iz završnog treninga.",
    )

    result = {
        "name": path.name,
        "next": tuple(value + 1 for value in best),
        "validation": validation_nll,
        "test": test_nll,
        "factorized_test": factorized_test_nll,
        "histogram": histogram,
        "mean_hits": float(
            histogram @ np.arange(DRAW_SIZE + 1)
            / (n - split80)
        ),
        "probability": math.exp(logp),
        "marginal_sum": float(inclusion.sum()),
        "concurrence": concurrence,
        "joint_dependence": joint_dependence,
        "quantum_error": quantum_error,
        "sensitivity": sensitivity,
        "gradient": final_model.gradient_norm,
        "converged": final_model.converged,
        "warnings": warnings,
    }

    print_result(result)

    return result


def main():
    np.seterr(
        divide="raise",
        over="raise",
        invalid="raise",
        under="ignore",
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    say(
        "QUANTUM EPR–BELL–ASPECT V1 — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "Devetnaest spregnutih dvokubitnih blokova i jedan kubit.",
        "Uče se kvantni uglovi za istorijski uslovljene kodne reči.",
        "Qiskit računa amplitude tokom svake ocene funkcije cilja.",
        "Koriste se sva četiri zajednička ishoda svakog para.",
        "Parovi se biraju isključivo iz trening prefiksa.",
        "Izbor parova je klasična priprema strukture modela.",
        "Nema klasičnog izlaznog regresora.",
        "Raspodela se tačno uslovljava na sedam različitih brojeva.",
        "Normalizacija, margine i maksimum računaju se dinamički.",
        "Nema pretrage 15 miliona sedmorki.",
        "Nema slučajnog uzorkovanja, šuma ili mešanja redova.",
        "Nema gustog vektora stanja sa 39 kubita.",
        "Bellova provera potvrđuje kvantni primitiv, ne loto predikciju.",
        "Spregnutost postoji unutar blokova; blokove dodatno povezuje uslov sedam.",
        "Limit iteracija ne odbacuje validnu sačuvanu raspodelu.",
        "Konvergencija se prijavljuje odvojeno.",
        f"Kontrolne tačke: {CHECKPOINT_DIR}",
    )

    require(
        len(set(path.resolve() for path in FILES)) == len(FILES),
        "Ulazne putanje nisu različite.",
    )

    datasets = []
    hashes = []

    for path in FILES:
        require(path.is_file(), f"Fajl nije pronađen: {path}")

        draws, occupancy, fingerprint = load_draws(path)
        datasets.append((path, draws, occupancy))
        hashes.append(fingerprint)

    require(
        len(set(hashes)) == len(hashes),
        "Najmanje dva ulaza imaju isti sadržaj.",
    )

    self_checks()

    results = []

    for path, draws, occupancy in datasets:
        results.append(
            process_file(path, draws, occupancy)
        )

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            if results[first]["next"] == results[second]["next"]:
                results[first]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[second]["name"]
                    + "; proveriti uticaj istorije i degeneraciju."
                )

                results[second]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[first]["name"]
                    + "; proveriti uticaj istorije i degeneraciju."
                )

    say("", "ZAVRŠNA TRI NEXT REZULTATA")

    for result in results:
        print_result(result)

    say(
        "",
        "Obrada je završena.",
        "NEXT je tačan maksimum raspodele za sačuvane parametre.",
        "Nepotvrđena konvergencija ostaje označena.",
        "Prediktivna vrednost procenjuje se odvojenim testom.",
    )


if __name__ == "__main__":
    try:
        main()

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM EPR-BELL-ASPECT V1 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Devetnaest spregnutih dvokubitnih blokova i jedan kubit.
Uče se kvantni uglovi za istorijski uslovljene kodne reči.
Qiskit računa amplitude tokom svake ocene funkcije cilja.
Koriste se sva četiri zajednička ishoda svakog para.
Parovi se biraju isključivo iz trening prefiksa.
Izbor parova je klasična priprema strukture modela.
Nema klasičnog izlaznog regresora.
Raspodela se tačno uslovljava na sedam različitih brojeva.
Normalizacija, margine i maksimum računaju se dinamički.
Nema pretrage 15 miliona sedmorki.
Nema slučajnog uzorkovanja, šuma ili mešanja redova.
Nema gustog vektora stanja sa 39 kubita.
Bellova provera potvrđuje kvantni primitiv, ne loto predikciju.
Spregnutost postoji unutar blokova; blokove dodatno povezuje uslov sedam.
Limit iteracija ne odbacuje validnu sačuvanu raspodelu.
Konvergencija se prijavljuje odvojeno.
Kontrolne tačke: /Quantum_EPR_Bell_Aspect/Quantum_EPR_Bell_Aspect_V1_checkpoints

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

Pokrećem ugrađene kvantne i matematičke provere.
Ugrađene provere su prošle.
Idealni Bell CHSH: 2.828427124746
Najveća greška gradijenta: 3.562e-10
Provereni su amplitude, normalizacija, zajedničke margine i maksimum.
Trajanje provera: 0.1 s

loto7_4710_k86.csv / validacija 1
λ=0.03 | prelaza=2825
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.6 s
Najbolji cilj: 16.498380774
Max|grad|: 2.239e-03
Početak 0 | iteracija 50 | proteklo 5.2 s
Najbolji cilj: 16.497964804
Max|grad|: 3.310e-04
Početak 0 | iteracija 75 | proteklo 7.6 s
Najbolji cilj: 16.497902539
Max|grad|: 2.275e-04
Početak 0 | iteracija 100 | proteklo 10.0 s
Najbolji cilj: 16.497873618
Max|grad|: 4.364e-04
Početak 0 | iteracija 125 | proteklo 12.3 s
Najbolji cilj: 16.497831272
Max|grad|: 1.662e-04
Početak 0 | iteracija 150 | proteklo 14.7 s
Najbolji cilj: 16.497812038
Max|grad|: 1.314e-04
Početak 0 | iteracija 175 | proteklo 17.0 s
Najbolji cilj: 16.497806919
Max|grad|: 7.190e-05
Početak 0 | iteracija 200 | proteklo 19.4 s
Najbolji cilj: 16.497804673
Max|grad|: 3.418e-05
Početak 0 | iteracija 225 | proteklo 21.6 s
Najbolji cilj: 16.497804285
Max|grad|: 2.739e-05
Početak 0 | iteracija 250 | proteklo 23.9 s
Najbolji cilj: 16.497804075
Max|grad|: 3.572e-05
Početak 0 | iteracija 275 | proteklo 26.1 s
Najbolji cilj: 16.497803916
Max|grad|: 3.369e-05
Početak 0 | iteracija 300 | proteklo 28.5 s
Najbolji cilj: 16.497803857
Max|grad|: 4.871e-06
Početak 0 | iteracija 325 | proteklo 31.0 s
Najbolji cilj: 16.497803843
Max|grad|: 8.111e-06
Početak 0 | iteracija 350 | proteklo 33.4 s
Najbolji cilj: 16.497803839
Max|grad|: 2.109e-06
Početak 0: zadržan cilj 16.497803836
Max|grad|: 7.316e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.6 s
Najbolji cilj: 16.498403076
Max|grad|: 2.606e-03
Početak 1 | iteracija 50 | proteklo 5.0 s
Najbolji cilj: 16.497964320
Max|grad|: 4.355e-04
Početak 1 | iteracija 75 | proteklo 7.1 s
Najbolji cilj: 16.497901931
Max|grad|: 2.453e-04
Početak 1 | iteracija 100 | proteklo 9.5 s
Najbolji cilj: 16.497874086
Max|grad|: 2.360e-04
Početak 1 | iteracija 125 | proteklo 11.7 s
Najbolji cilj: 16.497831423
Max|grad|: 8.674e-05
Početak 1 | iteracija 150 | proteklo 14.0 s
Najbolji cilj: 16.497812552
Max|grad|: 7.152e-05
Početak 1 | iteracija 175 | proteklo 16.1 s
Najbolji cilj: 16.497806548
Max|grad|: 1.670e-04
Početak 1 | iteracija 200 | proteklo 18.4 s
Najbolji cilj: 16.497804603
Max|grad|: 1.899e-05
Početak 1 | iteracija 225 | proteklo 20.6 s
Najbolji cilj: 16.497804114
Max|grad|: 1.226e-05
Početak 1 | iteracija 250 | proteklo 22.9 s
Najbolji cilj: 16.497803966
Max|grad|: 9.543e-06
Početak 1 | iteracija 275 | proteklo 25.2 s
Najbolji cilj: 16.497803885
Max|grad|: 7.295e-06
Početak 1 | iteracija 300 | proteklo 27.5 s
Najbolji cilj: 16.497803839
Max|grad|: 1.052e-06
Početak 1: zadržan cilj 16.497803839
Max|grad|: 9.379e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.592691722

loto7_4710_k86.csv / validacija 2
λ=0.03 | prelaza=3296
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 3.0 s
Najbolji cilj: 16.503081818
Max|grad|: 1.457e-03
Početak 0 | iteracija 50 | proteklo 5.6 s
Najbolji cilj: 16.502784799
Max|grad|: 1.538e-04
Početak 0 | iteracija 75 | proteklo 8.2 s
Najbolji cilj: 16.502733384
Max|grad|: 3.243e-04
Početak 0 | iteracija 100 | proteklo 10.7 s
Najbolji cilj: 16.502706517
Max|grad|: 1.163e-04
Početak 0 | iteracija 125 | proteklo 13.2 s
Najbolji cilj: 16.502671952
Max|grad|: 1.010e-04
Početak 0 | iteracija 150 | proteklo 15.8 s
Najbolji cilj: 16.502658602
Max|grad|: 7.200e-05
Početak 0 | iteracija 175 | proteklo 18.4 s
Najbolji cilj: 16.502656055
Max|grad|: 5.545e-05
Početak 0 | iteracija 200 | proteklo 20.9 s
Najbolji cilj: 16.502654576
Max|grad|: 2.660e-05
Početak 0 | iteracija 225 | proteklo 23.5 s
Najbolji cilj: 16.502654158
Max|grad|: 1.029e-05
Početak 0 | iteracija 250 | proteklo 26.0 s
Najbolji cilj: 16.502654061
Max|grad|: 1.839e-05
Početak 0 | iteracija 275 | proteklo 28.5 s
Najbolji cilj: 16.502653955
Max|grad|: 1.499e-05
Početak 0 | iteracija 300 | proteklo 31.0 s
Najbolji cilj: 16.502653922
Max|grad|: 5.532e-06
Početak 0 | iteracija 325 | proteklo 33.6 s
Najbolji cilj: 16.502653917
Max|grad|: 1.827e-06
Početak 0: zadržan cilj 16.502653915
Max|grad|: 9.653e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.8 s
Najbolji cilj: 16.503083420
Max|grad|: 3.464e-03
Početak 1 | iteracija 50 | proteklo 5.2 s
Najbolji cilj: 16.502784088
Max|grad|: 4.324e-04
Početak 1 | iteracija 75 | proteklo 7.6 s
Najbolji cilj: 16.502733537
Max|grad|: 1.142e-04
Početak 1 | iteracija 100 | proteklo 10.1 s
Najbolji cilj: 16.502708039
Max|grad|: 2.495e-04
Početak 1 | iteracija 125 | proteklo 12.6 s
Najbolji cilj: 16.502674247
Max|grad|: 2.408e-04
Početak 1 | iteracija 150 | proteklo 15.0 s
Najbolji cilj: 16.502660229
Max|grad|: 4.627e-05
Početak 1 | iteracija 175 | proteklo 17.5 s
Najbolji cilj: 16.502656249
Max|grad|: 7.543e-05
Početak 1 | iteracija 200 | proteklo 20.0 s
Najbolji cilj: 16.502654813
Max|grad|: 3.432e-05
Početak 1 | iteracija 225 | proteklo 22.4 s
Najbolji cilj: 16.502654262
Max|grad|: 4.000e-05
Početak 1 | iteracija 250 | proteklo 25.0 s
Najbolji cilj: 16.502654074
Max|grad|: 9.409e-06
Početak 1 | iteracija 275 | proteklo 27.5 s
Najbolji cilj: 16.502653998
Max|grad|: 8.478e-06
Početak 1 | iteracija 300 | proteklo 30.0 s
Najbolji cilj: 16.502653941
Max|grad|: 8.232e-06
Početak 1 | iteracija 325 | proteklo 32.5 s
Najbolji cilj: 16.502653924
Max|grad|: 3.854e-06
Početak 1 | iteracija 350 | proteklo 35.1 s
Najbolji cilj: 16.502653918
Max|grad|: 1.124e-06
Početak 1: zadržan cilj 16.502653917
Max|grad|: 9.678e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.634080154
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.613385938

loto7_4710_k86.csv / validacija 1
λ=0.3 | prelaza=2825
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.5 s
Najbolji cilj: 16.508982135
Max|grad|: 2.000e-03
Početak 0 | iteracija 50 | proteklo 4.8 s
Najbolji cilj: 16.508218247
Max|grad|: 1.551e-03
Početak 0 | iteracija 75 | proteklo 7.1 s
Najbolji cilj: 16.508043504
Max|grad|: 1.137e-04
Početak 0 | iteracija 100 | proteklo 9.3 s
Najbolji cilj: 16.508036887
Max|grad|: 2.741e-05
Početak 0 | iteracija 125 | proteklo 11.7 s
Najbolji cilj: 16.508036586
Max|grad|: 1.432e-05
Početak 0 | iteracija 150 | proteklo 14.1 s
Najbolji cilj: 16.508036579
Max|grad|: 2.044e-06
Početak 0: zadržan cilj 16.508036579
Max|grad|: 8.465e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.6 s
Najbolji cilj: 16.509003807
Max|grad|: 2.343e-03
Početak 1 | iteracija 50 | proteklo 5.1 s
Najbolji cilj: 16.508200336
Max|grad|: 6.116e-04
Početak 1 | iteracija 75 | proteklo 7.4 s
Najbolji cilj: 16.508041928
Max|grad|: 1.042e-04
Početak 1 | iteracija 100 | proteklo 9.7 s
Najbolji cilj: 16.508036822
Max|grad|: 2.654e-05
Početak 1 | iteracija 125 | proteklo 11.9 s
Najbolji cilj: 16.508036585
Max|grad|: 1.005e-05
Početak 1 | iteracija 150 | proteklo 14.1 s
Najbolji cilj: 16.508036579
Max|grad|: 1.319e-06
Početak 1: zadržan cilj 16.508036579
Max|grad|: 6.916e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.591059830

loto7_4710_k86.csv / validacija 2
λ=0.3 | prelaza=3296
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.8 s
Najbolji cilj: 16.513297761
Max|grad|: 3.589e-03
Početak 0 | iteracija 50 | proteklo 5.3 s
Najbolji cilj: 16.512617141
Max|grad|: 7.635e-04
Početak 0 | iteracija 75 | proteklo 7.9 s
Najbolji cilj: 16.512525372
Max|grad|: 2.191e-04
Početak 0 | iteracija 100 | proteklo 10.4 s
Najbolji cilj: 16.512520882
Max|grad|: 2.081e-05
Početak 0 | iteracija 125 | proteklo 12.9 s
Najbolji cilj: 16.512520678
Max|grad|: 5.249e-06
Početak 0: zadržan cilj 16.512520674
Max|grad|: 8.574e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.8 s
Najbolji cilj: 16.513293039
Max|grad|: 3.559e-03
Početak 1 | iteracija 50 | proteklo 5.2 s
Najbolji cilj: 16.512608900
Max|grad|: 6.926e-04
Početak 1 | iteracija 75 | proteklo 7.8 s
Najbolji cilj: 16.512525564
Max|grad|: 1.733e-04
Početak 1 | iteracija 100 | proteklo 10.2 s
Najbolji cilj: 16.512520872
Max|grad|: 5.246e-05
Početak 1 | iteracija 125 | proteklo 12.7 s
Najbolji cilj: 16.512520678
Max|grad|: 2.897e-06
Početak 1: zadržan cilj 16.512520673
Max|grad|: 7.731e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.616334802
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.603697316

loto7_4710_k86.csv / validacija 1
λ=3 | prelaza=2825
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.5 s
Najbolji cilj: 16.592943839
Max|grad|: 2.255e-03
Početak 0 | iteracija 50 | proteklo 5.0 s
Najbolji cilj: 16.592897811
Max|grad|: 5.959e-06
Početak 0: zadržan cilj 16.592897808
Max|grad|: 6.190e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.5 s
Najbolji cilj: 16.592943155
Max|grad|: 1.035e-03
Početak 1 | iteracija 50 | proteklo 4.9 s
Najbolji cilj: 16.592897811
Max|grad|: 1.740e-05
Početak 1: zadržan cilj 16.592897808
Max|grad|: 8.844e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.588788892

loto7_4710_k86.csv / validacija 2
λ=3 | prelaza=3296
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.9 s
Najbolji cilj: 16.595052077
Max|grad|: 9.394e-04
Početak 0 | iteracija 50 | proteklo 5.9 s
Najbolji cilj: 16.595006401
Max|grad|: 1.297e-05
Početak 0: zadržan cilj 16.595006399
Max|grad|: 6.833e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.9 s
Najbolji cilj: 16.595051050
Max|grad|: 7.796e-04
Početak 1 | iteracija 50 | proteklo 5.7 s
Najbolji cilj: 16.595006401
Max|grad|: 6.917e-06
Početak 1: zadržan cilj 16.595006399
Max|grad|: 5.374e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.586906515
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.587847703

IZABRANO: λ=3
Validacioni NLL: 16.587847703

loto7_4710_k86.csv / odvojeni test
λ=3 | prelaza=3767
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 3.2 s
Najbolji cilj: 16.601307253
Max|grad|: 1.167e-03
Početak 0 | iteracija 50 | proteklo 6.2 s
Najbolji cilj: 16.601274293
Max|grad|: 4.232e-06
Početak 0: zadržan cilj 16.601274292
Max|grad|: 7.019e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 3.1 s
Najbolji cilj: 16.601300091
Max|grad|: 9.492e-04
Početak 1 | iteracija 50 | proteklo 6.1 s
Najbolji cilj: 16.601274294
Max|grad|: 3.964e-06
Početak 1: zadržan cilj 16.601274292
Max|grad|: 6.360e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
ODVOJENI TEST: 942 prelaza
Zajednički kvantni NLL: 16.591897199
Kontrola bez parnih korelacija: 16.563769237

loto7_4710_k86.csv / završno učenje svih 4709 prelaza
λ=3 | prelaza=4709
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 3.9 s
Najbolji cilj: 16.605050305
Max|grad|: 5.443e-04
Početak 0 | iteracija 50 | proteklo 7.2 s
Najbolji cilj: 16.605016776
Max|grad|: 1.761e-06
Početak 0: zadržan cilj 16.605016775
Max|grad|: 9.823e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 3.8 s
Najbolji cilj: 16.605049193
Max|grad|: 5.343e-04
Početak 1 | iteracija 50 | proteklo 7.4 s
Najbolji cilj: 16.605016776
Max|grad|: 1.815e-06
Početak 1: zadržan cilj 16.605016775
Max|grad|: 4.052e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Završni parovi:
1-36 | 2-26 | 3-31 | 4-30 | 5-17 | 6-15 | 7-13 | 8-39 | 9 | 10-11 | 12-22 | 14-25 | 16-23 | 18-27 | 19-38 |20-35 | 21-33 | 24-37 | 28-29 | 32-34
Parovi su određeni samo iz završnog treninga.

loto7_4710_k86.csv
NEXT: 02 x 08 y 26 z 34
Validacioni NLL: 16.587847703
Odvojeni test NLL: 16.591897199
Test NLL posle uklanjanja unutrašnjih parnih korelacija: 16.563769237
Pogodaka 0–7: 207 398 253 74 10 0 0 0
Prosečan broj pogodaka na testu: 1.237792
Modelna verovatnoća NEXT: 2.17490315148e-07
Zbir uslovnih uključnih margina: 7.000000000000
Prosečna spregnutost NEXT parova: 0.670282044
Najveća parna Born zavisnost: 2.491055991e-02
Greška direktne Qiskit provere: 0.000e+00
Osetljivost na istorijski ulaz: 5.497838327e-02
Max|grad|: 4.052e-07
Gradijentni uslov: dostignut
DIJAGNOSTIKA: Na odvojenom testu zajedničke parne korelacije nisu poboljšale NLL u odnosu na kontrolu sa istim uključenim marginalnim verovatnoćama blokova.

loto7_4710_k86_loto_2977.csv / validacija 1
λ=0.03 | prelaza=1785
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.8 s
Najbolji cilj: 16.465281729
Max|grad|: 1.924e-03
Početak 0 | iteracija 50 | proteklo 3.4 s
Najbolji cilj: 16.464385963
Max|grad|: 2.910e-04
Početak 0 | iteracija 75 | proteklo 5.1 s
Najbolji cilj: 16.464290932
Max|grad|: 1.552e-04
Početak 0 | iteracija 100 | proteklo 6.7 s
Najbolji cilj: 16.464233826
Max|grad|: 3.895e-04
Početak 0 | iteracija 125 | proteklo 8.4 s
Najbolji cilj: 16.464181762
Max|grad|: 1.994e-04
Početak 0 | iteracija 150 | proteklo 10.0 s
Najbolji cilj: 16.464141792
Max|grad|: 4.603e-04
Početak 0 | iteracija 175 | proteklo 11.6 s
Najbolji cilj: 16.464135871
Max|grad|: 6.606e-05
Početak 0 | iteracija 200 | proteklo 13.3 s
Najbolji cilj: 16.464133228
Max|grad|: 3.311e-05
Početak 0 | iteracija 225 | proteklo 15.0 s
Najbolji cilj: 16.464132907
Max|grad|: 1.236e-05
Početak 0 | iteracija 250 | proteklo 16.8 s
Najbolji cilj: 16.464132761
Max|grad|: 1.214e-05
Početak 0 | iteracija 275 | proteklo 18.5 s
Najbolji cilj: 16.464132702
Max|grad|: 1.377e-05
Početak 0 | iteracija 300 | proteklo 20.3 s
Najbolji cilj: 16.464132653
Max|grad|: 4.511e-06
Početak 0 | iteracija 325 | proteklo 21.9 s
Najbolji cilj: 16.464132632
Max|grad|: 7.667e-06
Početak 0 | iteracija 350 | proteklo 23.6 s
Najbolji cilj: 16.464132626
Max|grad|: 9.323e-07
Početak 0: zadržan cilj 16.464132626
Max|grad|: 9.323e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.8 s
Najbolji cilj: 16.465295833
Max|grad|: 1.914e-03
Početak 1 | iteracija 50 | proteklo 3.4 s
Najbolji cilj: 16.464388072
Max|grad|: 2.492e-04
Početak 1 | iteracija 75 | proteklo 5.1 s
Najbolji cilj: 16.464288428
Max|grad|: 2.016e-04
Početak 1 | iteracija 100 | proteklo 6.7 s
Najbolji cilj: 16.464233374
Max|grad|: 1.623e-04
Početak 1 | iteracija 125 | proteklo 8.5 s
Najbolji cilj: 16.464181769
Max|grad|: 3.100e-04
Početak 1 | iteracija 150 | proteklo 10.1 s
Najbolji cilj: 16.464143324
Max|grad|: 4.041e-04
Početak 1 | iteracija 175 | proteklo 11.7 s
Najbolji cilj: 16.464136213
Max|grad|: 6.128e-05
Početak 1 | iteracija 200 | proteklo 13.3 s
Najbolji cilj: 16.464133355
Max|grad|: 2.004e-05
Početak 1 | iteracija 225 | proteklo 14.9 s
Najbolji cilj: 16.464132950
Max|grad|: 1.530e-05
Početak 1 | iteracija 250 | proteklo 16.6 s
Najbolji cilj: 16.464132737
Max|grad|: 1.571e-05
Početak 1 | iteracija 275 | proteklo 18.2 s
Najbolji cilj: 16.464132688
Max|grad|: 5.473e-06
Početak 1 | iteracija 300 | proteklo 19.8 s
Najbolji cilj: 16.464132640
Max|grad|: 4.272e-06
Početak 1 | iteracija 325 | proteklo 21.4 s
Najbolji cilj: 16.464132627
Max|grad|: 2.367e-06
Početak 1: zadržan cilj 16.464132622
Max|grad|: 9.103e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.621438729

loto7_4710_k86_loto_2977.csv / validacija 2
λ=0.03 | prelaza=2083
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.0 s
Najbolji cilj: 16.473583336
Max|grad|: 1.669e-03
Početak 0 | iteracija 50 | proteklo 3.9 s
Najbolji cilj: 16.472917970
Max|grad|: 3.510e-04
Početak 0 | iteracija 75 | proteklo 5.7 s
Najbolji cilj: 16.472832111
Max|grad|: 2.222e-04
Početak 0 | iteracija 100 | proteklo 7.5 s
Najbolji cilj: 16.472790001
Max|grad|: 2.565e-04
Početak 0 | iteracija 125 | proteklo 9.3 s
Najbolji cilj: 16.472737457
Max|grad|: 1.981e-04
Početak 0 | iteracija 150 | proteklo 11.1 s
Najbolji cilj: 16.472706513
Max|grad|: 1.857e-04
Početak 0 | iteracija 175 | proteklo 12.9 s
Najbolji cilj: 16.472700336
Max|grad|: 9.962e-05
Početak 0 | iteracija 200 | proteklo 14.6 s
Najbolji cilj: 16.472697835
Max|grad|: 1.388e-05
Početak 0 | iteracija 225 | proteklo 16.4 s
Najbolji cilj: 16.472697519
Max|grad|: 1.385e-05
Početak 0 | iteracija 250 | proteklo 18.2 s
Najbolji cilj: 16.472697391
Max|grad|: 1.360e-05
Početak 0 | iteracija 275 | proteklo 20.0 s
Najbolji cilj: 16.472697327
Max|grad|: 8.891e-06
Početak 0 | iteracija 300 | proteklo 21.9 s
Najbolji cilj: 16.472697257
Max|grad|: 3.208e-06
Početak 0 | iteracija 325 | proteklo 23.7 s
Najbolji cilj: 16.472697249
Max|grad|: 2.782e-06
Početak 0: zadržan cilj 16.472697248
Max|grad|: 9.794e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.9 s
Najbolji cilj: 16.473534500
Max|grad|: 1.397e-03
Početak 1 | iteracija 50 | proteklo 3.9 s
Najbolji cilj: 16.472917817
Max|grad|: 2.729e-04
Početak 1 | iteracija 75 | proteklo 5.7 s
Najbolji cilj: 16.472832542
Max|grad|: 2.932e-04
Početak 1 | iteracija 100 | proteklo 7.5 s
Najbolji cilj: 16.472790757
Max|grad|: 1.482e-04
Početak 1 | iteracija 125 | proteklo 9.3 s
Najbolji cilj: 16.472736326
Max|grad|: 1.431e-04
Početak 1 | iteracija 150 | proteklo 11.0 s
Najbolji cilj: 16.472705817
Max|grad|: 2.053e-04
Početak 1 | iteracija 175 | proteklo 12.8 s
Najbolji cilj: 16.472700223
Max|grad|: 7.859e-05
Početak 1 | iteracija 200 | proteklo 14.7 s
Najbolji cilj: 16.472697783
Max|grad|: 2.154e-05
Početak 1 | iteracija 225 | proteklo 16.4 s
Najbolji cilj: 16.472697487
Max|grad|: 1.179e-05
Početak 1 | iteracija 250 | proteklo 18.1 s
Najbolji cilj: 16.472697369
Max|grad|: 1.741e-05
Početak 1 | iteracija 275 | proteklo 19.9 s
Najbolji cilj: 16.472697305
Max|grad|: 6.928e-06
Početak 1 | iteracija 300 | proteklo 21.7 s
Najbolji cilj: 16.472697256
Max|grad|: 2.098e-06
Početak 1 | iteracija 325 | proteklo 23.5 s
Najbolji cilj: 16.472697247
Max|grad|: 1.024e-06
Početak 1: zadržan cilj 16.472697247
Max|grad|: 9.721e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.692899993
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.657109310

loto7_4710_k86_loto_2977.csv / validacija 1
λ=0.3 | prelaza=1785
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.8 s
Najbolji cilj: 16.477004969
Max|grad|: 1.296e-03
Početak 0 | iteracija 50 | proteklo 3.4 s
Najbolji cilj: 16.475742458
Max|grad|: 1.153e-03
Početak 0 | iteracija 75 | proteklo 4.9 s
Najbolji cilj: 16.475442094
Max|grad|: 7.852e-05
Početak 0 | iteracija 100 | proteklo 6.6 s
Najbolji cilj: 16.475436231
Max|grad|: 2.094e-05
Početak 0 | iteracija 125 | proteklo 8.3 s
Najbolji cilj: 16.475435962
Max|grad|: 5.643e-06
Početak 0 | iteracija 150 | proteklo 9.9 s
Najbolji cilj: 16.475435952
Max|grad|: 1.613e-06
Početak 0: zadržan cilj 16.475435952
Max|grad|: 7.098e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.8 s
Najbolji cilj: 16.477004206
Max|grad|: 1.504e-03
Početak 1 | iteracija 50 | proteklo 3.4 s
Najbolji cilj: 16.475704734
Max|grad|: 9.355e-04
Početak 1 | iteracija 75 | proteklo 5.0 s
Najbolji cilj: 16.475441398
Max|grad|: 1.319e-04
Početak 1 | iteracija 100 | proteklo 6.6 s
Najbolji cilj: 16.475436236
Max|grad|: 2.378e-05
Početak 1 | iteracija 125 | proteklo 8.3 s
Najbolji cilj: 16.475435960
Max|grad|: 6.281e-06
Početak 1 | iteracija 150 | proteklo 9.9 s
Najbolji cilj: 16.475435952
Max|grad|: 1.709e-06
Početak 1: zadržan cilj 16.475435952
Max|grad|: 7.202e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.604767836

loto7_4710_k86_loto_2977.csv / validacija 2
λ=0.3 | prelaza=2083
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.0 s
Najbolji cilj: 16.485045207
Max|grad|: 1.546e-03
Početak 0 | iteracija 50 | proteklo 3.9 s
Najbolji cilj: 16.483931970
Max|grad|: 1.926e-03
Početak 0 | iteracija 75 | proteklo 5.7 s
Najbolji cilj: 16.483728562
Max|grad|: 2.291e-04
Početak 0 | iteracija 100 | proteklo 7.5 s
Najbolji cilj: 16.483723002
Max|grad|: 2.749e-05
Početak 0 | iteracija 125 | proteklo 9.2 s
Najbolji cilj: 16.483722701
Max|grad|: 5.474e-06
Početak 0 | iteracija 150 | proteklo 11.1 s
Najbolji cilj: 16.483722688
Max|grad|: 3.249e-06
Početak 0: zadržan cilj 16.483722687
Max|grad|: 9.438e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.1 s
Najbolji cilj: 16.485033860
Max|grad|: 1.348e-03
Početak 1 | iteracija 50 | proteklo 4.1 s
Najbolji cilj: 16.483930770
Max|grad|: 1.359e-03
Početak 1 | iteracija 75 | proteklo 6.0 s
Najbolji cilj: 16.483728913
Max|grad|: 6.542e-05
Početak 1 | iteracija 100 | proteklo 7.8 s
Najbolji cilj: 16.483722970
Max|grad|: 2.465e-05
Početak 1 | iteracija 125 | proteklo 9.7 s
Najbolji cilj: 16.483722700
Max|grad|: 9.716e-06
Početak 1 | iteracija 150 | proteklo 11.5 s
Najbolji cilj: 16.483722687
Max|grad|: 1.100e-06
Početak 1: zadržan cilj 16.483722686
Max|grad|: 7.591e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.665911438
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.635288256

loto7_4710_k86_loto_2977.csv / validacija 1
λ=3 | prelaza=1785
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.8 s
Najbolji cilj: 16.565459940
Max|grad|: 6.453e-04
Početak 0 | iteracija 50 | proteklo 3.5 s
Najbolji cilj: 16.565399343
Max|grad|: 3.567e-06
Početak 0: zadržan cilj 16.565399341
Max|grad|: 6.966e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.8 s
Najbolji cilj: 16.565458811
Max|grad|: 5.500e-04
Početak 1 | iteracija 50 | proteklo 3.6 s
Najbolji cilj: 16.565399344
Max|grad|: 5.794e-06
Početak 1: zadržan cilj 16.565399341
Max|grad|: 9.048e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.574444637

loto7_4710_k86_loto_2977.csv / validacija 2
λ=3 | prelaza=2083
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.1 s
Najbolji cilj: 16.573002955
Max|grad|: 1.515e-03
Početak 0 | iteracija 50 | proteklo 3.9 s
Najbolji cilj: 16.572942343
Max|grad|: 2.029e-05
Početak 0: zadržan cilj 16.572942340
Max|grad|: 8.648e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.0 s
Najbolji cilj: 16.573004033
Max|grad|: 6.441e-04
Početak 1 | iteracija 50 | proteklo 3.9 s
Najbolji cilj: 16.572942343
Max|grad|: 5.182e-06
Početak 1: zadržan cilj 16.572942340
Max|grad|: 9.981e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.636838032
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.605588903

IZABRANO: λ=3
Validacioni NLL: 16.605588903

loto7_4710_k86_loto_2977.csv / odvojeni test
λ=3 | prelaza=2380
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.3 s
Najbolji cilj: 16.577693404
Max|grad|: 6.460e-04
Početak 0 | iteracija 50 | proteklo 4.3 s
Najbolji cilj: 16.577659410
Max|grad|: 4.746e-06
Početak 0: zadržan cilj 16.577659407
Max|grad|: 7.141e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.2 s
Najbolji cilj: 16.577699590
Max|grad|: 3.253e-04
Početak 1 | iteracija 50 | proteklo 4.2 s
Najbolji cilj: 16.577659409
Max|grad|: 2.642e-05
Početak 1: zadržan cilj 16.577659407
Max|grad|: 9.602e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
ODVOJENI TEST: 596 prelaza
Zajednički kvantni NLL: 16.610914985
Kontrola bez parnih korelacija: 16.581915633

loto7_4710_k86_loto_2977.csv / završno učenje svih 2976 prelaza
λ=3 | prelaza=2976
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 2.6 s
Najbolji cilj: 16.592562202
Max|grad|: 8.845e-04
Početak 0 | iteracija 50 | proteklo 5.0 s
Najbolji cilj: 16.592526700
Max|grad|: 4.812e-06
Početak 0: zadržan cilj 16.592526698
Max|grad|: 6.768e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 2.5 s
Najbolji cilj: 16.592575812
Max|grad|: 1.002e-03
Početak 1 | iteracija 50 | proteklo 4.9 s
Najbolji cilj: 16.592526701
Max|grad|: 5.254e-06
Početak 1: zadržan cilj 16.592526698
Max|grad|: 9.613e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Završni parovi:
1 | 2-21 | 3-26 | 4-20 | 5-10 | 6-38 | 7-32 | 8-35 | 9-39 | 11-12 | 13-15 | 14-16 | 17-28 | 18-27 | 19-23 |22-36 | 24-37 | 25-29 | 30-34 | 31-33
Parovi su određeni samo iz završnog treninga.

loto7_4710_k86_loto_2977.csv
NEXT: 08 x 22 y 32 z 39
Validacioni NLL: 16.605588903
Odvojeni test NLL: 16.610914985
Test NLL posle uklanjanja unutrašnjih parnih korelacija: 16.581915633
Pogodaka 0–7: 117 271 163 43 2 0 0 0
Prosečan broj pogodaka na testu: 1.231544
Modelna verovatnoća NEXT: 1.99843922327e-07
Zbir uslovnih uključnih margina: 7.000000000000
Prosečna spregnutost NEXT parova: 0.659099520
Najveća parna Born zavisnost: 1.302385178e-02
Greška direktne Qiskit provere: 0.000e+00
Osetljivost na istorijski ulaz: 5.971459840e-02
Max|grad|: 6.768e-07
Gradijentni uslov: dostignut
DIJAGNOSTIKA: Na odvojenom testu zajedničke parne korelacije nisu poboljšale NLL u odnosu na kontrolu sa istim uključenim marginalnim verovatnoćama blokova.

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=0.03 | prelaza=1039
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.413972936
Max|grad|: 4.063e-03
Početak 0 | iteracija 50 | proteklo 2.6 s
Najbolji cilj: 16.413028592
Max|grad|: 8.989e-04
Početak 0 | iteracija 75 | proteklo 3.7 s
Najbolji cilj: 16.412909670
Max|grad|: 1.003e-03
Početak 0 | iteracija 100 | proteklo 4.9 s
Najbolji cilj: 16.412824359
Max|grad|: 1.869e-04
Početak 0 | iteracija 125 | proteklo 6.0 s
Najbolji cilj: 16.412752418
Max|grad|: 3.248e-04
Početak 0 | iteracija 150 | proteklo 7.2 s
Najbolji cilj: 16.412697344
Max|grad|: 3.138e-04
Početak 0 | iteracija 175 | proteklo 8.4 s
Najbolji cilj: 16.412678580
Max|grad|: 7.625e-05
Početak 0 | iteracija 200 | proteklo 9.5 s
Najbolji cilj: 16.412673147
Max|grad|: 3.281e-05
Početak 0 | iteracija 225 | proteklo 10.6 s
Najbolji cilj: 16.412672106
Max|grad|: 9.855e-05
Početak 0 | iteracija 250 | proteklo 11.8 s
Najbolji cilj: 16.412671593
Max|grad|: 2.333e-05
Početak 0 | iteracija 275 | proteklo 13.0 s
Najbolji cilj: 16.412671399
Max|grad|: 9.395e-06
Početak 0 | iteracija 300 | proteklo 14.1 s
Najbolji cilj: 16.412671265
Max|grad|: 4.090e-06
Početak 0 | iteracija 325 | proteklo 15.3 s
Najbolji cilj: 16.412671226
Max|grad|: 4.159e-06
Početak 0 | iteracija 350 | proteklo 16.4 s
Najbolji cilj: 16.412671214
Max|grad|: 5.555e-06
Početak 0 | iteracija 375 | proteklo 17.6 s
Najbolji cilj: 16.412671206
Max|grad|: 1.495e-06
Početak 0: zadržan cilj 16.412671205
Max|grad|: 8.123e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.3 s
Najbolji cilj: 16.413928844
Max|grad|: 2.388e-03
Početak 1 | iteracija 50 | proteklo 2.5 s
Najbolji cilj: 16.413024385
Max|grad|: 5.423e-04
Početak 1 | iteracija 75 | proteklo 3.7 s
Najbolji cilj: 16.412910684
Max|grad|: 3.446e-04
Početak 1 | iteracija 100 | proteklo 4.9 s
Najbolji cilj: 16.412825141
Max|grad|: 2.720e-04
Početak 1 | iteracija 125 | proteklo 6.1 s
Najbolji cilj: 16.412753257
Max|grad|: 3.239e-04
Početak 1 | iteracija 150 | proteklo 7.2 s
Najbolji cilj: 16.412696456
Max|grad|: 3.398e-04
Početak 1 | iteracija 175 | proteklo 8.4 s
Najbolji cilj: 16.412678656
Max|grad|: 1.360e-04
Početak 1 | iteracija 200 | proteklo 9.6 s
Najbolji cilj: 16.412673055
Max|grad|: 2.337e-05
Početak 1 | iteracija 225 | proteklo 10.8 s
Najbolji cilj: 16.412672002
Max|grad|: 1.082e-04
Početak 1 | iteracija 250 | proteklo 11.9 s
Najbolji cilj: 16.412671548
Max|grad|: 1.604e-05
Početak 1 | iteracija 275 | proteklo 13.1 s
Najbolji cilj: 16.412671322
Max|grad|: 7.598e-06
Početak 1 | iteracija 300 | proteklo 14.2 s
Najbolji cilj: 16.412671245
Max|grad|: 4.294e-06
Početak 1 | iteracija 325 | proteklo 15.4 s
Najbolji cilj: 16.412671215
Max|grad|: 6.937e-06
Početak 1 | iteracija 350 | proteklo 16.6 s
Najbolji cilj: 16.412671208
Max|grad|: 2.863e-06
Početak 1: zadržan cilj 16.412671205
Max|grad|: 9.872e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.986037785

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=0.03 | prelaza=1212
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.438271298
Max|grad|: 8.485e-04
Početak 0 | iteracija 50 | proteklo 2.6 s
Najbolji cilj: 16.437721346
Max|grad|: 2.904e-04
Početak 0 | iteracija 75 | proteklo 3.9 s
Najbolji cilj: 16.437608899
Max|grad|: 2.210e-04
Početak 0 | iteracija 100 | proteklo 5.2 s
Najbolji cilj: 16.437541616
Max|grad|: 1.452e-04
Početak 0 | iteracija 125 | proteklo 6.4 s
Najbolji cilj: 16.437483066
Max|grad|: 3.391e-04
Početak 0 | iteracija 150 | proteklo 7.7 s
Najbolji cilj: 16.437435093
Max|grad|: 2.118e-04
Početak 0 | iteracija 175 | proteklo 9.0 s
Najbolji cilj: 16.437413481
Max|grad|: 6.509e-05
Početak 0 | iteracija 200 | proteklo 10.2 s
Najbolji cilj: 16.437411228
Max|grad|: 4.840e-05
Početak 0 | iteracija 225 | proteklo 11.5 s
Najbolji cilj: 16.437410641
Max|grad|: 2.014e-05
Početak 0 | iteracija 250 | proteklo 12.7 s
Najbolji cilj: 16.437410233
Max|grad|: 1.681e-05
Početak 0 | iteracija 275 | proteklo 14.0 s
Najbolji cilj: 16.437410011
Max|grad|: 1.614e-05
Početak 0 | iteracija 300 | proteklo 15.2 s
Najbolji cilj: 16.437409923
Max|grad|: 1.073e-05
Početak 0 | iteracija 325 | proteklo 16.5 s
Najbolji cilj: 16.437409900
Max|grad|: 3.030e-06
Početak 0 | iteracija 350 | proteklo 17.8 s
Najbolji cilj: 16.437409894
Max|grad|: 2.365e-06
Početak 0 | iteracija 375 | proteklo 19.1 s
Najbolji cilj: 16.437409891
Max|grad|: 9.026e-07
Početak 0: zadržan cilj 16.437409891
Max|grad|: 9.026e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.438344100
Max|grad|: 1.400e-03
Početak 1 | iteracija 50 | proteklo 2.8 s
Najbolji cilj: 16.437724842
Max|grad|: 1.619e-03
Početak 1 | iteracija 75 | proteklo 4.1 s
Najbolji cilj: 16.437608987
Max|grad|: 3.638e-04
Početak 1 | iteracija 100 | proteklo 5.4 s
Najbolji cilj: 16.437545775
Max|grad|: 4.138e-04
Početak 1 | iteracija 125 | proteklo 6.6 s
Najbolji cilj: 16.437488390
Max|grad|: 3.591e-04
Početak 1 | iteracija 150 | proteklo 7.8 s
Najbolji cilj: 16.437437461
Max|grad|: 1.940e-04
Početak 1 | iteracija 175 | proteklo 9.1 s
Najbolji cilj: 16.437413867
Max|grad|: 3.591e-04
Početak 1 | iteracija 200 | proteklo 10.4 s
Najbolji cilj: 16.437411323
Max|grad|: 2.344e-05
Početak 1 | iteracija 225 | proteklo 11.6 s
Najbolji cilj: 16.437410667
Max|grad|: 1.764e-05
Početak 1 | iteracija 250 | proteklo 13.0 s
Najbolji cilj: 16.437410247
Max|grad|: 1.772e-05
Početak 1 | iteracija 275 | proteklo 14.2 s
Najbolji cilj: 16.437410020
Max|grad|: 2.909e-05
Početak 1 | iteracija 300 | proteklo 15.5 s
Najbolji cilj: 16.437409922
Max|grad|: 4.967e-06
Početak 1 | iteracija 325 | proteklo 16.8 s
Najbolji cilj: 16.437409899
Max|grad|: 4.060e-06
Početak 1 | iteracija 350 | proteklo 18.1 s
Najbolji cilj: 16.437409893
Max|grad|: 1.933e-06
Početak 1 | iteracija 375 | proteklo 19.3 s
Najbolji cilj: 16.437409890
Max|grad|: 2.283e-06
Početak 1: zadržan cilj 16.437409887
Max|grad|: 9.393e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.719876760
KANDIDAT: λ=0.03
Zbirni validacioni NLL: 16.852957273

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=0.3 | prelaza=1039
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.3 s
Najbolji cilj: 16.427633738
Max|grad|: 1.884e-03
Početak 0 | iteracija 50 | proteklo 2.4 s
Najbolji cilj: 16.425955300
Max|grad|: 1.231e-03
Početak 0 | iteracija 75 | proteklo 3.6 s
Najbolji cilj: 16.425638780
Max|grad|: 4.176e-04
Početak 0 | iteracija 100 | proteklo 4.8 s
Najbolji cilj: 16.425630286
Max|grad|: 2.429e-05
Početak 0 | iteracija 125 | proteklo 5.9 s
Najbolji cilj: 16.425629854
Max|grad|: 4.756e-06
Početak 0 | iteracija 150 | proteklo 7.1 s
Najbolji cilj: 16.425629839
Max|grad|: 1.571e-06
Početak 0: zadržan cilj 16.425629838
Max|grad|: 9.516e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.3 s
Najbolji cilj: 16.427717709
Max|grad|: 5.308e-03
Početak 1 | iteracija 50 | proteklo 2.5 s
Najbolji cilj: 16.426000541
Max|grad|: 1.038e-03
Početak 1 | iteracija 75 | proteklo 3.7 s
Najbolji cilj: 16.425638703
Max|grad|: 1.433e-04
Početak 1 | iteracija 100 | proteklo 5.0 s
Najbolji cilj: 16.425630316
Max|grad|: 3.521e-05
Početak 1 | iteracija 125 | proteklo 6.3 s
Najbolji cilj: 16.425629868
Max|grad|: 4.899e-05
Početak 1 | iteracija 150 | proteklo 7.5 s
Najbolji cilj: 16.425629841
Max|grad|: 2.684e-06
Početak 1: zadržan cilj 16.425629839
Max|grad|: 9.649e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.833962097

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=0.3 | prelaza=1212
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.451386794
Max|grad|: 3.491e-03
Početak 0 | iteracija 50 | proteklo 2.8 s
Najbolji cilj: 16.450061560
Max|grad|: 8.746e-04
Početak 0 | iteracija 75 | proteklo 4.0 s
Najbolji cilj: 16.449778548
Max|grad|: 1.562e-04
Početak 0 | iteracija 100 | proteklo 5.4 s
Najbolji cilj: 16.449768795
Max|grad|: 5.509e-05
Početak 0 | iteracija 125 | proteklo 6.7 s
Najbolji cilj: 16.449768345
Max|grad|: 1.167e-05
Početak 0 | iteracija 150 | proteklo 8.0 s
Najbolji cilj: 16.449768326
Max|grad|: 1.432e-06
Početak 0: zadržan cilj 16.449768326
Max|grad|: 9.926e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.451415865
Max|grad|: 1.182e-03
Početak 1 | iteracija 50 | proteklo 2.7 s
Najbolji cilj: 16.450065008
Max|grad|: 1.874e-03
Početak 1 | iteracija 75 | proteklo 4.0 s
Najbolji cilj: 16.449778943
Max|grad|: 9.638e-05
Početak 1 | iteracija 100 | proteklo 5.3 s
Najbolji cilj: 16.449768810
Max|grad|: 6.753e-05
Početak 1 | iteracija 125 | proteklo 6.6 s
Najbolji cilj: 16.449768350
Max|grad|: 2.028e-05
Početak 1 | iteracija 150 | proteklo 8.0 s
Najbolji cilj: 16.449768327
Max|grad|: 2.516e-06
Početak 1: zadržan cilj 16.449768326
Max|grad|: 8.431e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.657873664
KANDIDAT: λ=0.3
Zbirni validacioni NLL: 16.745917881

loto7_4710_k86_loto_plus_1733.csv / validacija 1
λ=3 | prelaza=1039
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.527535320
Max|grad|: 6.850e-04
Početak 0 | iteracija 50 | proteklo 2.6 s
Najbolji cilj: 16.527425333
Max|grad|: 7.698e-06
Početak 0: zadržan cilj 16.527425325
Max|grad|: 9.927e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.527514702
Max|grad|: 2.494e-03
Početak 1 | iteracija 50 | proteklo 2.6 s
Najbolji cilj: 16.527425331
Max|grad|: 2.982e-05
Početak 1: zadržan cilj 16.527425325
Max|grad|: 7.146e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.673637468

loto7_4710_k86_loto_plus_1733.csv / validacija 2
λ=3 | prelaza=1212
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.5 s
Najbolji cilj: 16.547500598
Max|grad|: 6.802e-04
Početak 0 | iteracija 50 | proteklo 2.8 s
Najbolji cilj: 16.547407423
Max|grad|: 1.058e-05
Početak 0: zadržan cilj 16.547407418
Max|grad|: 6.821e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.4 s
Najbolji cilj: 16.547528960
Max|grad|: 6.503e-04
Početak 1 | iteracija 50 | proteklo 2.6 s
Najbolji cilj: 16.547407423
Max|grad|: 7.330e-06
Početak 1: zadržan cilj 16.547407417
Max|grad|: 6.990e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacioni NLL: 16.598912231
KANDIDAT: λ=3
Zbirni validacioni NLL: 16.636274849

IZABRANO: λ=3
Validacioni NLL: 16.636274849

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
λ=3 | prelaza=1385
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.5 s
Najbolji cilj: 16.549658661
Max|grad|: 5.848e-04
Početak 0 | iteracija 50 | proteklo 3.0 s
Najbolji cilj: 16.549585954
Max|grad|: 9.630e-06
Početak 0: zadržan cilj 16.549585951
Max|grad|: 7.918e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.6 s
Najbolji cilj: 16.549659321
Max|grad|: 5.764e-04
Početak 1 | iteracija 50 | proteklo 3.0 s
Najbolji cilj: 16.549585954
Max|grad|: 1.309e-05
Početak 1: zadržan cilj 16.549585951
Max|grad|: 9.793e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
ODVOJENI TEST: 347 prelaza
Zajednički kvantni NLL: 16.639824089
Kontrola bez parnih korelacija: 16.575834521

loto7_4710_k86_loto_plus_1733.csv / završno učenje svih 1732 prelaza
λ=3 | prelaza=1732
Kvantnih parametara: 306
Početak 0 | iteracija 25 | proteklo 1.8 s
Najbolji cilj: 16.557559219
Max|grad|: 4.241e-04
Početak 0 | iteracija 50 | proteklo 3.4 s
Najbolji cilj: 16.557497500
Max|grad|: 1.054e-05
Početak 0: zadržan cilj 16.557497497
Max|grad|: 9.418e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Početak 1 | iteracija 25 | proteklo 1.7 s
Najbolji cilj: 16.557569458
Max|grad|: 4.369e-04
Početak 1 | iteracija 50 | proteklo 3.4 s
Najbolji cilj: 16.557497500
Max|grad|: 5.546e-06
Početak 1: zadržan cilj 16.557497498
Max|grad|: 7.437e-07
Gradijentni uslov: dostignut
OPTIMIZER_SUCCESS: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Završni parovi:
1-36 | 2-28 | 3-16 | 4-17 | 5-19 | 6-10 | 7 | 8-32 | 9-25 | 11-38 | 12-22 | 13-23 | 14-18 | 15-29 | 20-26 |21-33 | 24-30 | 27-31 | 34-37 | 35-39
Parovi su određeni samo iz završnog treninga.

loto7_4710_k86_loto_plus_1733.csv
NEXT: 02 x 04 y 10 z 23
Validacioni NLL: 16.636274849
Odvojeni test NLL: 16.639824089
Test NLL posle uklanjanja unutrašnjih parnih korelacija: 16.575834521
Pogodaka 0–7: 76 134 106 29 2 0 0 0
Prosečan broj pogodaka na testu: 1.270893
Modelna verovatnoća NEXT: 2.66657218556e-07
Zbir uslovnih uključnih margina: 7.000000000000
Prosečna spregnutost NEXT parova: 0.615429852
Najveća parna Born zavisnost: 3.877346982e-02
Greška direktne Qiskit provere: 0.000e+00
Osetljivost na istorijski ulaz: 1.099310386e-01
Max|grad|: 9.418e-07
Gradijentni uslov: dostignut
DIJAGNOSTIKA: Na odvojenom testu zajedničke parne korelacije nisu poboljšale NLL u odnosu na kontrolu sa istim uključenim marginalnim verovatnoćama blokova.



ZAVRŠNA TRI NEXT REZULTATA

loto7_4710_k86.csv
NEXT: 02 03 08 23 26 32 34
Validacioni NLL: 16.587847703
Odvojeni test NLL: 16.591897199
Test NLL posle uklanjanja unutrašnjih parnih korelacija: 16.563769237
Pogodaka 0–7: 207 398 253 74 10 0 0 0
Prosečan broj pogodaka na testu: 1.237792
Modelna verovatnoća NEXT: 2.17490315148e-07
Zbir uslovnih uključnih margina: 7.000000000000
Prosečna spregnutost NEXT parova: 0.670282044
Najveća parna Born zavisnost: 2.491055991e-02
Greška direktne Qiskit provere: 0.000e+00
Osetljivost na istorijski ulaz: 5.497838327e-02
Max|grad|: 4.052e-07
Gradijentni uslov: dostignut
DIJAGNOSTIKA: Na odvojenom testu zajedničke parne korelacije nisu poboljšale NLL u odnosu na kontrolu sa istim uključenim marginalnim verovatnoćama blokova.

loto7_4710_k86_loto_2977.csv
NEXT: 08 13 22 23 32 36 39
Validacioni NLL: 16.605588903
Odvojeni test NLL: 16.610914985
Test NLL posle uklanjanja unutrašnjih parnih korelacija: 16.581915633
Pogodaka 0–7: 117 271 163 43 2 0 0 0
Prosečan broj pogodaka na testu: 1.231544
Modelna verovatnoća NEXT: 1.99843922327e-07
Zbir uslovnih uključnih margina: 7.000000000000
Prosečna spregnutost NEXT parova: 0.659099520
Najveća parna Born zavisnost: 1.302385178e-02
Greška direktne Qiskit provere: 0.000e+00
Osetljivost na istorijski ulaz: 5.971459840e-02
Max|grad|: 6.768e-07
Gradijentni uslov: dostignut
DIJAGNOSTIKA: Na odvojenom testu zajedničke parne korelacije nisu poboljšale NLL u odnosu na kontrolu sa istim uključenim marginalnim verovatnoćama blokova.

loto7_4710_k86_loto_plus_1733.csv
NEXT: 02 03 04 08 10 18 23
Validacioni NLL: 16.636274849
Odvojeni test NLL: 16.639824089
Test NLL posle uklanjanja unutrašnjih parnih korelacija: 16.575834521
Pogodaka 0–7: 76 134 106 29 2 0 0 0
Prosečan broj pogodaka na testu: 1.270893
Modelna verovatnoća NEXT: 2.66657218556e-07
Zbir uslovnih uključnih margina: 7.000000000000
Prosečna spregnutost NEXT parova: 0.615429852
Najveća parna Born zavisnost: 3.877346982e-02
Greška direktne Qiskit provere: 0.000e+00
Osetljivost na istorijski ulaz: 1.099310386e-01
Max|grad|: 9.418e-07
Gradijentni uslov: dostignut
DIJAGNOSTIKA: Na odvojenom testu zajedničke parne korelacije nisu poboljšale NLL u odnosu na kontrolu sa istim uključenim marginalnim verovatnoćama blokova.

Obrada je završena.
NEXT je tačan maksimum raspodele za sačuvane parametre.
Nepotvrđena konvergencija ostaje označena.
Prediktivna vrednost procenjuje se odvojenim testom.
"""



"""
Učiti zajedničke kvantne verovatnoće parova brojeva, umesto samo pojedinačnih verovatnoća. 


Rad	                            Glavni rezultat	                                                                                                                            Šta preneti
Einstein-Podolsky-Rosen, 1935	Razmatraju kako merenje jednog dela zajedničkog stanja omogućava zaključivanje o drugom i postavljaju pitanje potpunosti kvantnog opisa.	Modelovati zajedničko stanje, sa uslovnim verovatnoćama izvedenim iz njegovih amplituda. 
Bell, 1964	                    Pokazuje nesaglasnost određenih lokalnih modela skrivenih promenljivih sa kvantnim korelacijama.	                                        Razlikovati spregnutu raspodelu od proizvoda zasebnih očitavanja. Bellova nejednakost nije funkcija za predviđanje sledećeg kola. 
Aspect-Grangier-Roger, 1982	    Mere sva četiri zajednička polarizaciona ishoda i dobijaju \(S=2.697\pm0.015\), u saglasnosti sa kvantnim predviđanjem.	                    Koristiti kompletnu tabelu 00, 01, 10, 11, uz proveru normalizacije i korelacija. Rad navodi i preostala eksperimentalna ograničenja. 


Moja adaptacija za Loto 7/39:
1. Qiskit kolo priprema parametrizovano spregnuto dvokubitno stanje.
2. Dva kubita predstavljaju uključenost dva broja. Njihove kompleksne amplitude daju zajedničke verovatnoće 00, 01, 10, 11.
3. Istorija upravlja kvantnim uglovima; parametri se uče preko verovatnoće naredne cele sedmorke.
4. Dvobrojni blokovi povezuju se uslovom tačno sedam uključenih brojeva. Normalizacija i maksimum mogu se računati dinamičkim programiranjem.

Ograničenje je što neposrednu spregnutost modeluje unutar parova.

Bellov test služi proveri kvantnog kola. 
Njegovo narušavanje na simulatoru ne dokazuje prediktivnu vrednost niti kvantnu prirodu loto podataka. 
Odlučujuće poboljšanje je na odvojenom hronološkom testu, a ne povećavanje Bellovog parametra.

V1 sa spregnutim dvokubitnim blokovima u Qiskitu. 
Koristim sva četiri zajednička Bornova ishoda, tačno uslovljavanje na sedam brojeva i dinamičko određivanje maksimuma. 
Bellova provera služiće proveri kola; uspeh na loto podacima meriće odvojeni hronološki test.

Parove brojeva odrediću iz povezanosti u trening podacima i zamrznuti ih za validaciju ili test. 
Tako ne namećem susedne brojeve kao parove. 
Model će učiti zajedničke ishode unutar svakog para; testna kontrola uklanja te korelacije, da se vidi da li su koristile predikciji.

V1 uči zajedničke kvantne verovatnoće parova: 00, 01, 10, 11. 
Parovi se određuju samo iz trening podataka. 
Izlaz se tačno uslovljava na sedam brojeva, a maksimum se računa dinamički.
Uključeni su Qiskit Bell/CHSH provera, provere izvoda i normalizacije, kontrolne tačke i odvojeni test sa kontrolom uklonjenih parnih korelacija.
Nezavisna mala matematička provera normalizacije i maksimuma je prošla.
"""
