from __future__ import annotations

import csv
import math
from itertools import product
from pathlib import Path

from src.motor_model import MotorParameters
from src.performance import calculate_metrics
from src.simulation import (
    SimulationConfig,
    run_simulation,
)


def numeric_value(
    value: float | None,
) -> float:
    """Converte None em infinito para permitir comparações."""

    if value is None:
        return math.inf

    return value


def run_parameter_search() -> None:
    """Testa diferentes parâmetros do controlador fuzzy."""

    error_scales = [
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    ]

    derivative_scales = [
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
    ]

    output_scales = [
        6.0,
        8.0,
        10.0,
        12.0,
    ]

    motor_parameters = MotorParameters()

    results = []

    combinations = product(
        error_scales,
        derivative_scales,
        output_scales,
    )

    for (
        error_scale,
        derivative_scale,
        output_scale,
    ) in combinations:

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

        settling_time = numeric_value(
            metrics.settling_time
        )

        recovery_time = numeric_value(
            metrics.disturbance_recovery_time
        )

        # Na simulação numérica, consideramos
        # erro menor que 0,01 rad como próximo de zero.
        meets_requirements = (
            metrics.overshoot_percent < 5.0
            and settling_time < 1.5
            and metrics.steady_state_error < 0.01
            and recovery_time < 1.0
        )

        # Índice utilizado apenas para ordenar
        # as configurações encontradas.
        score = (
            metrics.overshoot_percent / 5.0
            + settling_time / 1.5
            + metrics.steady_state_error / 0.01
            + recovery_time
            + 0.05
            * metrics.maximum_voltage
            / config.voltage_limit
        )

        results.append(
            {
                "error_scale": error_scale,
                "derivative_scale": derivative_scale,
                "output_scale": output_scale,
                "overshoot": (
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
                "meets_requirements": (
                    meets_requirements
                ),
                "score": score,
            }
        )

    # Configurações aprovadas aparecem primeiro.
    results.sort(
        key=lambda item: (
            not item["meets_requirements"],
            item["score"],
        )
    )

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
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("======= MELHORES CONFIGURAÇÕES =======")

    for index, result in enumerate(
        results[:10],
        start=1,
    ):
        status = (
            "ATENDE"
            if result["meets_requirements"]
            else "NÃO ATENDE"
        )

        print(
            f"{index:02d} | "
            f"Ke={result['error_scale']:.2f} | "
            f"Kde={result['derivative_scale']:.2f} | "
            f"Ku={result['output_scale']:.2f} | "
            f"Mp={result['overshoot']:.3f}% | "
            f"Ts={result['settling_time']:.3f}s | "
            f"Erro={result['steady_state_error']:.6f} | "
            f"Rec={result['recovery_time']:.3f}s | "
            f"{status}"
        )

    print("======================================")

    print(
        "\nResultados completos salvos em:"
    )

    print(output_file)


if __name__ == "__main__":
    run_parameter_search()