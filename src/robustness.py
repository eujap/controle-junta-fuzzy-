from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.motor_model import MotorParameters
from src.performance import (
    PerformanceMetrics,
    calculate_metrics,
)
from src.simulation import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
)


def format_optional_value(
    value: float | None,
) -> str:
    """Formata métricas que podem não ser atingidas."""

    if value is None:
        return "não atingido"

    return f"{value:.3f}"


def meets_requirements(
    metrics: PerformanceMetrics,
) -> bool:
    """Verifica os requisitos de desempenho do trabalho."""

    if metrics.settling_time is None:
        return False

    if metrics.disturbance_recovery_time is None:
        return False

    return (
        metrics.overshoot_percent < 5.0
        and metrics.settling_time < 1.5
        and metrics.steady_state_error < 0.01
        and metrics.disturbance_recovery_time < 1.0
    )


def run_robustness_analysis() -> None:
    """Testa o controlador com diferentes valores de inércia."""

    config = SimulationConfig(
        controller_error_scale=4.0,
        controller_derivative_scale=0.20,
        controller_output_scale=12.0,
    )

    inertia_cases = [
        ("Carga leve", 0.008),
        ("Carga nominal", 0.010),
        ("Carga pesada", 0.015),
    ]

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    simulation_results: list[
        tuple[str, float, SimulationResult]
    ] = []

    table_results: list[dict[str, object]] = []

    print()
    print("=" * 90)
    print("TESTE DE ROBUSTEZ COM VARIAÇÃO DA INÉRCIA")
    print("=" * 90)

    for load_name, inertia in inertia_cases:
        motor_parameters = MotorParameters(
            inertia=inertia,
            viscous_friction=0.10,
            torque_constant=0.01,
            back_emf_constant=0.01,
            resistance=1.0,
        )

        result = run_simulation(
            config=config,
            motor_parameters=motor_parameters,
        )

        metrics = calculate_metrics(
            result=result,
            config=config,
        )

        approved = meets_requirements(metrics)

        simulation_results.append(
            (
                load_name,
                inertia,
                result,
            )
        )

        table_results.append(
            {
                "carga": load_name,
                "inercia": inertia,
                "sobresinal_percentual": (
                    metrics.overshoot_percent
                ),
                "tempo_assentamento": (
                    metrics.settling_time
                ),
                "erro_regime_permanente": (
                    metrics.steady_state_error
                ),
                "tempo_recuperacao": (
                    metrics.disturbance_recovery_time
                ),
                "tensao_maxima": (
                    metrics.maximum_voltage
                ),
                "atende_requisitos": approved,
            }
        )

        status = (
            "ATENDE"
            if approved
            else "NÃO ATENDE"
        )

        print(
            f"{load_name:15s} | "
            f"J={inertia:.3f} kg.m² | "
            f"Mp={metrics.overshoot_percent:.3f}% | "
            f"Ts={format_optional_value(metrics.settling_time)} s | "
            f"Erro={metrics.steady_state_error:.6f} rad | "
            f"Rec="
            f"{format_optional_value(metrics.disturbance_recovery_time)} s | "
            f"Umax={metrics.maximum_voltage:.3f} V | "
            f"{status}"
        )

    print("=" * 90)

    save_results_csv(
        results=table_results,
        output_directory=output_directory,
    )

    save_comparison_plot(
        simulations=simulation_results,
        config=config,
        output_directory=output_directory,
    )


def save_results_csv(
    results: list[dict[str, object]],
    output_directory: Path,
) -> None:
    """Salva as métricas da análise em CSV."""

    output_file = (
        output_directory
        / "resultados_robustez_inercia.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        "Tabela salva em: "
        f"{output_file}"
    )


def save_comparison_plot(
    simulations: list[
        tuple[str, float, SimulationResult]
    ],
    config: SimulationConfig,
    output_directory: Path,
) -> None:
    """Gera o gráfico comparativo das diferentes cargas."""

    figure = plt.figure(
        figsize=(11, 6)
    )

    for (
        load_name,
        inertia,
        result,
    ) in simulations:
        plt.plot(
            result.time,
            result.position,
            label=(
                f"{load_name} "
                f"(J={inertia:.3f} kg.m²)"
            ),
        )

    plt.axhline(
        y=config.reference,
        linestyle="--",
        label="Referência",
    )

    disturbance_end = (
        config.disturbance_start
        + config.disturbance_duration
    )

    plt.axvline(
        x=config.disturbance_start,
        linestyle=":",
        label="Início do distúrbio",
    )

    plt.axvline(
        x=disturbance_end,
        linestyle=":",
        label="Fim do distúrbio",
    )

    plt.xlabel("Tempo [s]")
    plt.ylabel("Posição angular [rad]")

    plt.title(
        "Robustez do controlador fuzzy "
        "para diferentes valores de inércia"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file = (
        output_directory
        / "comparacao_inercia.png"
    )

    figure.savefig(
        output_file,
        dpi=200,
    )

    print(
        "Gráfico salvo em: "
        f"{output_file}"
    )

    plt.show()


if __name__ == "__main__":
    run_robustness_analysis()