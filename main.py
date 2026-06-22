from src.motor_model import (
    MotorParameters,
    RobotJointPlant,
)

from src.performance import (
    calculate_metrics,
    print_metrics,
)

from src.simulation import (
    SimulationConfig,
    run_simulation,
    save_plots,
)

from src.fuzzy_visualization import (
    generate_fuzzy_visualizations,
)


def main() -> None:
    """Ponto de entrada do projeto."""

    config = SimulationConfig(
    controller_error_scale=4.0,
    controller_derivative_scale=0.20,
    controller_output_scale=12.0,
)

    motor_parameters = MotorParameters()

    plant = RobotJointPlant(
        parameters=motor_parameters,
        sample_time=config.sample_time,
    )

    numerator, denominator = (
        plant.transfer_function_coefficients()
    )

    print(
        "Função de transferência da planta:"
    )

    print(
        f"Numerador: {numerator}"
    )

    print(
        f"Denominador: {denominator}"
    )

    result = run_simulation(
        config=config,
        motor_parameters=motor_parameters,
    )

    metrics = calculate_metrics(
        result=result,
        config=config,
    )

    print_metrics(metrics)

    save_plots(
        result=result,
        output_directory="results",
        show_plots=True,
    )

    generate_fuzzy_visualizations(
    output_directory="results",
    show_plots=True,
)


if __name__ == "__main__":
    main()