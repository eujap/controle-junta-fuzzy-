from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.simulation import (
    SimulationConfig,
    SimulationResult,
)

@dataclass(frozen=True)
class PerformanceMetrics:
    """Métricas de desempenho do sistema."""

    overshoot_percent: float
    settling_time: float | None
    steady_state_error: float
    disturbance_recovery_time: float | None
    maximum_voltage:  float


def find_settling_time(
        time: np.ndarray,
        position:np.ndarray,
        reference: float,
        tolerance: float,
) -> float | None:
    """Calcula o tempo de assentamento."""

    band = (
        tolerance
        * max(
            abs(reference),
            1e-9,
        )
    )

    lower_limit = reference - band
    upper_limit = reference + band

    for index in range(len(time)):
        remaining = position[index:]
        inside_band = np.all(
          (remaining >= lower_limit)  
        & (remaining <= upper_limit)
        )
        if inside_band:
            return float(
                time[index]
            )
    
    return None
def calculate_metrics(
        result: SimulationResult,
        config: SimulationConfig,
        tolerance: float = 0.02,
) -> PerformanceMetrics:
    """Calcular os índices exigidos"""

    before_disturbance = (
        result.time
        < config.disturbance_start
    )

    time_before = result.time[
        before_disturbance
    ]

    position_before = result.position[
        before_disturbance
    ]

    maximum_position = float(
        np.max(position_before)
    )

    if abs(config.reference) > 1e-9:
        overshoot_percent = max(
            0.0,
            (
                maximum_position
                - config.reference
            )
            / abs(config.reference)
            *100.0,
        )
    else:
        overshoot_percent = 0.0

    settling_time = find_settling_time(
        time=time_before,
        position=position_before,
        reference=config.reference,
        tolerance=tolerance,
    )
    steady_state_window_start = max(
        0.0,
        config.disturbance_start -0.5,
    )

    steady_state_mask = (
        (
            result.time
            >= steady_state_window_start
        )
        &(
            result.time
            < config.disturbance_start
        )
    )

    steady_state_error = float(
        np.mean(
            np.abs(
                config.reference
                - result.position[
                    steady_state_mask
                ]
            )
        )
    )

    disturbance_end = (
        config.disturbance_start
        + config.disturbance_duration
    )

    after_disturbance = (
        result.time
        >= disturbance_end
    )

    disturbance_recovery_time = (
        find_settling_time(
            time=(
                result.time[
                    after_disturbance
                ]
                - disturbance_end
            ),
            position=result.position[
                after_disturbance
            ],
            reference=config.reference,
            tolerance=tolerance,
        )
    )

    maximum_voltage = float(
        np.max(
            np.abs(
                result.control_voltage
            )
        )
    )

    return PerformanceMetrics(
        overshoot_percent=overshoot_percent,
        settling_time=settling_time,
        steady_state_error=steady_state_error,
        disturbance_recovery_time=(
            disturbance_recovery_time
        ),
        maximum_voltage=maximum_voltage,
    )


def print_metrics(
    metrics: PerformanceMetrics,
) -> None:
    """Mostra as métricas no terminal."""

    if metrics.settling_time is None:
        settling_text = "não atingido"
    else:
        settling_text = (
            f"{metrics.settling_time:.3f} s"
        )

    if (
        metrics.disturbance_recovery_time
        is None
    ):
        recovery_text = "não atingido"
    else:
        recovery_text = (
            f"{metrics.disturbance_recovery_time:.3f} s"
        )

    print()
    print("========== RESULTADOS ==========")

    print(
        "Sobresinal máximo: "
        f"{metrics.overshoot_percent:.3f}%"
    )

    print(
        "Tempo de assentamento: "
        f"{settling_text}"
    )

    print(
        "Erro de regime permanente: "
        f"{metrics.steady_state_error:.6f} rad"
    )

    print(
        "Recuperação após distúrbio: "
        f"{recovery_text}"
    )

    print(
        "Tensão máxima: "
        f"{metrics.maximum_voltage:.3f} V"
    )

    print("================================")