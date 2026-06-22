from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fuzzy_controller import FuzzyPositionController


def plot_membership_functions(
    output_directory: str = "results",
    show_plot: bool = True,
) -> None:
    """Gera o gráfico das funções de pertinência normalizadas."""

    directory = Path(output_directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    controller = FuzzyPositionController()

    values = np.linspace(
        -1.0,
        1.0,
        500,
    )

    membership_values = {
        label: []
        for label in controller.LABELS
    }

    for value in values:
        memberships = controller.memberships(
            float(value)
        )

        for label in controller.LABELS:
            membership_values[label].append(
                memberships[label]
            )

    figure = plt.figure(
        figsize=(10, 6)
    )

    for label in controller.LABELS:
        plt.plot(
            values,
            membership_values[label],
            label=label,
        )

    plt.xlabel("Variável normalizada")
    plt.ylabel("Grau de pertinência")

    plt.title(
        "Funções de pertinência do controlador fuzzy"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        directory / "funcoes_pertinencia.png",
        dpi=200,
    )

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def plot_control_surface(
    output_directory: str = "results",
    show_plot: bool = True,
) -> None:
    """Gera a superfície tridimensional do controlador fuzzy."""

    directory = Path(output_directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    controller = FuzzyPositionController(
        error_scale=3.0,
        derivative_scale=0.10,
        output_scale=12.0,
        voltage_limit=12.0,
    )

    # Como error_scale = 3, erros próximos de ±0,34 rad
    # já atingem os extremos normalizados ±1.
    error_values = np.linspace(
        -0.4,
        0.4,
        80,
    )

    # Como derivative_scale = 0,10,
    # derivadas de ±10 rad/s atingem ±1.
    derivative_values = np.linspace(
        -10.0,
        10.0,
        80,
    )

    error_grid, derivative_grid = np.meshgrid(
        error_values,
        derivative_values,
    )

    output_grid = np.zeros_like(
        error_grid
    )

    for row in range(
        error_grid.shape[0]
    ):
        for column in range(
            error_grid.shape[1]
        ):
            output_grid[row, column] = (
                controller.compute(
                    error=float(
                        error_grid[row, column]
                    ),
                    error_derivative=float(
                        derivative_grid[
                            row,
                            column,
                        ]
                    ),
                )
            )

    figure = plt.figure(
        figsize=(11, 8)
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    axis.plot_surface(
        error_grid,
        derivative_grid,
        output_grid,
    )

    axis.set_xlabel(
        "Erro de posição [rad]"
    )

    axis.set_ylabel(
        "Derivada do erro [rad/s]"
    )

    axis.set_zlabel(
        "Tensão de controle [V]"
    )

    axis.set_title(
        "Superfície de controle fuzzy"
    )

    figure.tight_layout()

    figure.savefig(
        directory / "superficie_controle_fuzzy.png",
        dpi=200,
    )

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def generate_fuzzy_visualizations(
    output_directory: str = "results",
    show_plots: bool = True,
) -> None:
    """Gera todas as visualizações do controlador fuzzy."""

    plot_membership_functions(
        output_directory=output_directory,
        show_plot=show_plots,
    )

    plot_control_surface(
        output_directory=output_directory,
        show_plot=show_plots,
    )


if __name__ == "__main__":
    generate_fuzzy_visualizations()