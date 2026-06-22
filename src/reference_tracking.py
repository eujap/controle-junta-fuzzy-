from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fuzzy_controller import FuzzyPositionController
from src.motor_model import MotorParameters, RobotJointPlant
from src.simulation import SimulationResult


def create_reference_signal(
    time: np.ndarray,
) -> np.ndarray:
    """
    Cria o sinal de referência variável.

    0,0 até 0,5 s  ->  0,0 rad
    0,5 até 2,0 s  ->  1,0 rad
    2,0 até 3,2 s  ->  0,5 rad
    3,2 até 4,5 s  -> -0,5 rad
    4,5 até 6,0 s  ->  0,0 rad
    """

    reference = np.zeros_like(
        time,
        dtype=float,
    )

    reference[
        (time >= 0.5)
        & (time < 2.0)
    ] = 1.0

    reference[
        (time >= 2.0)
        & (time < 3.2)
    ] = 0.5

    reference[
        (time >= 3.2)
        & (time < 4.5)
    ] = -0.5

    reference[
        time >= 4.5
    ] = 0.0

    return reference


def run_reference_tracking() -> SimulationResult:
    """Executa a simulação com diferentes referências."""

    sample_time = 0.01
    total_time = 6.0
    voltage_limit = 12.0

    disturbance_start = 5.0
    disturbance_duration = 0.2
    disturbance_end = (
        disturbance_start
        + disturbance_duration
    )

    time = np.arange(
        0.0,
        total_time + sample_time,
        sample_time,
    )

    reference = create_reference_signal(
        time
    )

    motor_parameters = MotorParameters(
        inertia=0.01,
        viscous_friction=0.10,
        torque_constant=0.01,
        back_emf_constant=0.01,
        resistance=1.0,
    )

    plant = RobotJointPlant(
        parameters=motor_parameters,
        sample_time=sample_time,
    )

    controller = FuzzyPositionController(
        error_scale=4.0,
        derivative_scale=0.20,
        output_scale=12.0,
        voltage_limit=voltage_limit,
    )

    position = np.zeros_like(time)
    velocity = np.zeros_like(time)
    error_history = np.zeros_like(time)
    control_voltage = np.zeros_like(time)
    disturbance_history = np.zeros_like(time)

    previous_error = (
        reference[0]
        - plant.theta
    )

    error_history[0] = previous_error

    for index in range(
        1,
        len(time),
    ):
        current_time = time[index]

        current_reference = (
            reference[index]
        )

        error = (
            current_reference
            - plant.theta
        )

        error_derivative = (
            error
            - previous_error
        ) / sample_time

        voltage = controller.compute(
            error=error,
            error_derivative=error_derivative,
        )

        if (
            disturbance_start
            <= current_time
            < disturbance_end
        ):
            disturbance = -0.08
        else:
            disturbance = 0.0

        theta, omega, _current = plant.step(
            voltage=voltage,
            disturbance_torque=disturbance,
        )

        position[index] = theta
        velocity[index] = omega

        error_history[index] = (
            current_reference
            - theta
        )

        control_voltage[index] = voltage
        disturbance_history[index] = disturbance

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


def save_reference_tracking_plots(
    result: SimulationResult,
    output_directory: str = "results",
) -> None:
    """Salva os gráficos do teste de referência variável."""

    directory = Path(output_directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Referência e posição
    figure_position = plt.figure(
        figsize=(11, 6)
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
    plt.ylabel("Posição angular [rad]")

    plt.title(
        "Rastreamento de referências variáveis"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure_position.savefig(
        directory
        / "referencia_variavel.png",
        dpi=200,
    )

    # Erro
    figure_error = plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        result.time,
        result.error,
        label="Erro de posição",
    )

    plt.xlabel("Tempo [s]")
    plt.ylabel("Erro [rad]")

    plt.title(
        "Erro no rastreamento de referência"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure_error.savefig(
        directory
        / "erro_referencia_variavel.png",
        dpi=200,
    )

    # Tensão
    figure_voltage = plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        result.time,
        result.control_voltage,
        label="Tensão de controle",
    )

    plt.axhline(
        12.0,
        linestyle="--",
        label="Limite superior",
    )

    plt.axhline(
        -12.0,
        linestyle="--",
        label="Limite inferior",
    )

    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensão [V]")

    plt.title(
        "Sinal de controle para referência variável"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure_voltage.savefig(
        directory
        / "controle_referencia_variavel.png",
        dpi=200,
    )

    plt.show()


def print_tracking_results(
    result: SimulationResult,
) -> None:
    """Exibe indicadores gerais do teste."""

    mean_absolute_error = float(
        np.mean(
            np.abs(result.error)
        )
    )

    maximum_absolute_error = float(
        np.max(
            np.abs(result.error)
        )
    )

    maximum_voltage = float(
        np.max(
            np.abs(
                result.control_voltage
            )
        )
    )

    saturated_samples = np.isclose(
        np.abs(result.control_voltage),
        12.0,
        atol=1e-6,
    )

    saturation_percentage = float(
        np.mean(saturated_samples)
        * 100.0
    )

    print()
    print("=" * 60)
    print("TESTE DE RASTREAMENTO DE REFERÊNCIA")
    print("=" * 60)

    print(
        "Erro absoluto médio: "
        f"{mean_absolute_error:.6f} rad"
    )

    print(
        "Maior erro absoluto: "
        f"{maximum_absolute_error:.6f} rad"
    )

    print(
        "Tensão máxima: "
        f"{maximum_voltage:.3f} V"
    )

    print(
        "Percentual em saturação: "
        f"{saturation_percentage:.3f}%"
    )

    print("=" * 60)


def main() -> None:
    """Executa o teste completo."""

    result = run_reference_tracking()

    print_tracking_results(
        result
    )

    save_reference_tracking_plots(
        result=result,
        output_directory="results",
    )


if __name__ == "__main__":
    main()