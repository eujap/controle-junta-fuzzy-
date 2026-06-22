from __future__ import annotations

import csv
import math
from itertools import product
from pathlib import Path

import numpy as np

from src.motor_model import MotorParameters
from src.performance import calculate_metrics
from src.simulation import (
    SimulationConfig,
    run_simulation,
)


def convert_optional_metric(
    value: float | None,
    penalty: float = 999.0,
) -> float:
    """
    Converte métricas não atingidas em um valor alto.

    Quando o tempo de assentamento ou de recuperação
    não é encontrado, calculate_metrics retorna None.
    """

    if value is None:
        return penalty

    return float(value)


def calculate_saturation_percentage(
    control_voltage: np.ndarray,
    voltage_limit: float,
    tolerance: float = 1e-6,
) -> float:
    """Calcula quanto tempo o atuador permaneceu saturado."""

    saturated_samples = np.isclose(
        np.abs(control_voltage),
        voltage_limit,
        atol=tolerance,
    )

    return float(
        np.mean(saturated_samples) * 100.0
    )


def calculate_score(
    overshoot: float,
    settling_time: float,
    steady_state_error: float,
    recovery_time: float,
    saturation_percentage: float,
) -> float:
    """
    Calcula uma pontuação para ordenar os resultados.

    Quanto menor a pontuação, melhor a configuração.
    """

    return (
        2.0 * overshoot / 5.0
        + 2.0 * settling_time / 1.5
        + 4.0 * steady_state_error / 0.01
        + 2.0 * recovery_time / 1.0
        + 0.5 * saturation_percentage / 100.0
    )


def run_parameter_search() -> list[dict[str, float | bool]]:
    """Testa diversas combinações de parâmetros fuzzy."""

    error_scales = [
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
    ]

    derivative_scales = [
        0.03,
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.20,
    ]

    output_scales = [
        6.0,
        8.0,
        10.0,
        11.0,
        12.0,
    ]

    combinations = list(
        product(
            error_scales,
            derivative_scales,
            output_scales,
        )
    )

    motor_parameters = MotorParameters()
    tuning_results: list[
        dict[str, float | bool]
    ] = []

    print()
    print("Iniciando sintonia automática...")
    print(
        f"Total de configurações: "
        f"{len(combinations)}"
    )

    for test_number, (
        error_scale,
        derivative_scale,
        output_scale,
    ) in enumerate(combinations, start=1):

        config = SimulationConfig(
            controller_error_scale=error_scale,
            controller_derivative_scale=(
                derivative_scale
            ),
            controller_output_scale=output_scale,
        )

        simulation_result = run_simulation(
            config=config,
            motor_parameters=motor_parameters,
        )

        metrics = calculate_metrics(
            result=simulation_result,
            config=config,
        )

        settling_time = convert_optional_metric(
            metrics.settling_time
        )

        recovery_time = convert_optional_metric(
            metrics.disturbance_recovery_time
        )

        saturation_percentage = (
            calculate_saturation_percentage(
                control_voltage=(
                    simulation_result.control_voltage
                ),
                voltage_limit=config.voltage_limit,
            )
        )

        meets_overshoot = (
            metrics.overshoot_percent < 5.0
        )

        meets_settling = (
            settling_time < 1.5
        )

        # Numericamente, consideramos erro inferior
        # a 0,01 rad como suficientemente próximo de zero.
        meets_steady_state = (
            metrics.steady_state_error < 0.01
        )

        meets_recovery = (
            recovery_time < 1.0
        )

        meets_requirements = all(
            [
                meets_overshoot,
                meets_settling,
                meets_steady_state,
                meets_recovery,
            ]
        )

        score = calculate_score(
            overshoot=metrics.overshoot_percent,
            settling_time=settling_time,
            steady_state_error=(
                metrics.steady_state_error
            ),
            recovery_time=recovery_time,
            saturation_percentage=(
                saturation_percentage
            ),
        )

        tuning_results.append(
            {
                "error_scale": error_scale,
                "derivative_scale": (
                    derivative_scale
                ),
                "output_scale": output_scale,
                "overshoot_percent": (
                    metrics.overshoot_percent
                ),
                "settling_time": settling_time,
                "steady_state_error": (
                    metrics.steady_state_error
                ),
                "recovery_time": recovery_time,
                "maximum_voltage": (
                    metrics.maximum_voltage
                ),
                "saturation_percentage": (
                    saturation_percentage
                ),
                "meets_overshoot": (
                    meets_overshoot
                ),
                "meets_settling": (
                    meets_settling
                ),
                "meets_steady_state": (
                    meets_steady_state
                ),
                "meets_recovery": (
                    meets_recovery
                ),
                "meets_requirements": (
                    meets_requirements
                ),
                "score": score,
            }
        )

        if (
            test_number % 25 == 0
            or test_number == len(combinations)
        ):
            print(
                f"Processadas {test_number}/"
                f"{len(combinations)} configurações"
            )

    tuning_results.sort(
        key=lambda result: (
            not bool(
                result["meets_requirements"]
            ),
            float(result["score"]),
        )
    )

    return tuning_results


def save_results(
    tuning_results: list[
        dict[str, float | bool]
    ],
) -> Path:
    """Salva os resultados da sintonia em CSV."""

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_directory
        / "resultados_sintonia.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                tuning_results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(tuning_results)

    return output_file


def print_best_results(
    tuning_results: list[
        dict[str, float | bool]
    ],
    quantity: int = 15,
) -> None:
    """Exibe as melhores configurações no terminal."""

    approved_quantity = sum(
        bool(result["meets_requirements"])
        for result in tuning_results
    )

    print()
    print("=" * 100)
    print(
        "MELHORES CONFIGURAÇÕES DO "
        "CONTROLADOR FUZZY"
    )
    print("=" * 100)

    print(
        f"Configurações aprovadas: "
        f"{approved_quantity}/"
        f"{len(tuning_results)}"
    )

    print()

    for index, result in enumerate(
        tuning_results[:quantity],
        start=1,
    ):
        status = (
            "ATENDE"
            if result["meets_requirements"]
            else "NÃO ATENDE"
        )

        settling_time = float(
            result["settling_time"]
        )

        recovery_time = float(
            result["recovery_time"]
        )

        settling_text = (
            "não atingido"
            if not math.isfinite(settling_time)
            or settling_time >= 999.0
            else f"{settling_time:.3f} s"
        )

        recovery_text = (
            "não atingido"
            if not math.isfinite(recovery_time)
            or recovery_time >= 999.0
            else f"{recovery_time:.3f} s"
        )

        print(
            f"{index:02d} | "
            f"Ke={float(result['error_scale']):.2f} | "
            f"Kde={float(result['derivative_scale']):.2f} | "
            f"Ku={float(result['output_scale']):.2f} | "
            f"Mp={float(result['overshoot_percent']):.3f}% | "
            f"Ts={settling_text} | "
            f"Erro={float(result['steady_state_error']):.6f} rad | "
            f"Rec={recovery_text} | "
            f"Sat={float(result['saturation_percentage']):.2f}% | "
            f"{status}"
        )

    print("=" * 100)


def main() -> None:
    """Executa a sintonia automática."""

    tuning_results = run_parameter_search()

    output_file = save_results(
        tuning_results
    )

    print_best_results(
        tuning_results,
        quantity=15,
    )

    print()
    print(
        "Resultados completos salvos em:"
    )
    print(output_file)


if __name__ == "__main__":
    main()