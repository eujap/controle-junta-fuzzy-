from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class MotorParameters:
    """Parametros físicos do motor DC e junta robótica."""
    
    inertia: float = 0.015
    viscous_friction: float = 0.10
    torque_constant: float = 0.01
    back_emf_constant: float = 0.01
    resistance: float = 1.0

class RobotJointPlant:
    """Modelo discreto do motor DC acoplado á junta robótica."""

    def __init__(
            self,
            parameters: MotorParameters,
            sample_time: float =0.01,
    ) -> None:
        if sample_time <= 0:
            raise ValueError(
                "O Tempo de amostragem deve ser positivo."
            )
        
        self.parameters = parameters
        self.sample_time = sample_time

        self.theta = 0.0
        self.omega = 0.0
        self.current = 0.0
    
    def reset(
            self,
            initial_position: float = 0.0,
            initial_velocity: float = 0.0,
    ) -> None:
        """Reinicia os estados da junta."""
        self.theta = initial_position
        self.omega = initial_velocity
        self.current = 0.0
    
    def step(
            self,
            voltage: float,
            disturbance_torque: float = 0.0,
    ) -> tuple[float, float, float]:
        """
        Avança a simulação em um período de amostragem.
        
        Modelo elétrico:

            i = (V - Ke * omega) / R

        Modelo mecânico:
            J * alpha = Kt * i + torque_disturbio - b * omega
        """

        p = self.parameters

        self.current = (
            voltage
            - p.back_emf_constant * self.omega
        ) / p.resistance

        motor_torque = (
            p.torque_constant * self.current
        )

        angular_acceleration = (
            motor_torque
            + disturbance_torque
            - p.viscous_friction * self.omega
        ) / p.inertia

        # Integração de Euler semi-implícita
        self.omega += (
            angular_acceleration
            * self.sample_time
        )

        self.theta +=(
            self.omega
            * self.sample_time
        )

        return(
            self.theta,
            self.omega,
            self.current,
        )
    def transfer_function_coefficients(
            self,
    ) -> tuple[list[float], list[float]]:
        """
        Retorna os coeficientes da função de transferência.

        G(s) = Kt / 
        [R*J*s² + (R*b + Kt*Ke)*s]
        """
        p = self.parameters
        
        numerator = [
            p.torque_constant
        ]

        denominator = [
            p.resistance * p.inertia,
            (
                p.resistance
                * p.viscous_friction
                + p.torque_constant
                * p.back_emf_constant
            ),
            0.0,
        ]

        return numerator, denominator