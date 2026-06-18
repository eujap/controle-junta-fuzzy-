from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fuzzy_controller import (
    FuzzyPositionController,
)

from src.motor_model import (
    MotorParameters,
    RobotJointPlant,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configurações gerais da simulação."""

    sample_time: float = 0.01
    total_time: float = 5.0
    reference: float = 1.0
    voltage_limit: float = 12.0

    disturbance_start: float = 2.5
    disturbance_duration: float = 0.2
    disturbance_torque: float = -0.08


@dataclass
class SimulationResult:
    """Dados produzidos pela simulação."""

    time: np.ndarray
    reference: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    error: np.ndarray
    control_voltage: np.ndarray
    disturbance_torque: np.ndarray


def run_simulation(
    config: SimulationConfig,
    motor_parameters: MotorParameters | None = None,
) -> SimulationResult:
    """Executa a simulação em malha fechada."""

    parameters = (
        motor_parameters
        or MotorParameters()
    )

    plant = RobotJointPlant(
        parameters=parameters,
        sample_time=config.sample_time,
    )

    controller = FuzzyPositionController(
        error_scale=3.0,
        derivative_scale=0.10,
        output_scale=config.voltage_limit,
        voltage_limit=config.voltage_limit,
    )

    time = np.arange(
        0.0,
        config.total_time
        + config.sample_time,
        config.sample_time,
    )

    reference = np.full_like(
        time,
        config.reference,
        dtype=float,
    )

    position = np.zeros_like(time)
    velocity = np.zeros_like(time)
    error_history = np.zeros_like(time)
    control_voltage = np.zeros_like(time)
    disturbance_history = np.zeros_like(time)

    previous_error = (
        config.reference
        - plant.theta
    )

    error_history[0] = previous_error

    disturbance_end = (
        config.disturbance_start
        + config.disturbance_duration
    )

    for index in range(
        1,
        len(time),
    ):
        interval_time = time[index - 1]

        error = (
            config.reference
            - plant.theta
        )

        error_derivative = (
            error - previous_error
        ) / config.sample_time

        voltage = controller.compute(
            error,
            error_derivative,
        )

        if (
            config.disturbance_start
            <= interval_time
            < disturbance_end
        ):
            disturbance = (
                config.disturbance_torque
            )
        else:
            disturbance = 0.0

        theta, omega, _current = plant.step(
            voltage=voltage,
            disturbance_torque=disturbance,
        )

        position[index] = theta
        velocity[index] = omega

        error_history[index] = (
            config.reference
            - theta
        )

        control_voltage[index] = voltage

        disturbance_history[index] = (
            disturbance
        )

        previous_error = error

    return SimulationResult(
        time=time,
        reference=reference,
        position=position,
        velocity=velocity,
        error=error_history,
        control_voltage=control_voltage,
        disturbance_torque=disturbance_history,
    )


def save_plots(
    result: SimulationResult,
    output_directory: str = "results",
    show_plots: bool = True,
) -> None:
    """Salva os gráficos principais."""

    directory = Path(output_directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        result.time,
        result.reference,
        "--",
        label="Referência",
    )

    plt.plot(
        result.time,
        result.position,
        label="Posição da junta",
    )

    plt.xlabel("Tempo [s]")
    plt.ylabel("Posição [rad]")

    plt.title(
        "Resposta de posição com controlador fuzzy"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        directory / "resposta_posicao.png",
        dpi=150,
    )

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        result.time,
        result.error,
        label="Erro de posição",
    )

    plt.xlabel("Tempo [s]")
    plt.ylabel("Erro [rad]")
    plt.title("Erro de posição")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        directory / "erro_posicao.png",
        dpi=150,
    )

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        result.time,
        result.control_voltage,
        label="Tensão de controle",
    )

    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensão [V]")
    plt.title("Sinal de controle")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        directory / "sinal_controle.png",
        dpi=150,
    )

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        result.time,
        result.disturbance_torque,
        label="Torque de distúrbio",
    )

    plt.xlabel("Tempo [s]")
    plt.ylabel("Torque [N.m]")

    plt.title(
        "Distúrbio aplicado à junta"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        directory / "disturbio.png",
        dpi=150,
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")