from __future__ import annotations

import numpy as np

def triangular_menbership(
        value: float,
        left: float,
        center: float,
        right: float,
) -> float:
    """Função de pertinência triangular."""
    if value <= left or value >=right:
        return 0.0
    
    if value == center:
        return 1.0
    
    if value < center:
        return(
            value - left
        ) / (
            center - left
        )
    
    return (
        right - value
    ) / (
        right - center
    )

def left_shoulder_membership(
        value: float,
        full_membership_until: float,
        zero_membership_from: float,
) -> float:
    """Função de pertinência do tipo ombro esquerdo."""
    if value <= full_membership_until:
        return 1.0
    
    if value >= zero_membership_from:
        return 0.0
    
    return(
        zero_membership_from - value
    ) / (
        zero_membership_from
        - full_membership_until
    )

def right_shoulder_membership(
        value: float,
        zero_membership_until: float,
        full_membership_from: float,
) -> float:
    """Função de pertinência do tipo ombro direito"""
    
    if value <= zero_membership_until:
        return 0.0
    
    if value >= full_membership_from:
        return 1.0
    
    return (
        value - zero_membership_until
    ) / (
        full_membership_from
        - zero_membership_until
    )

class FuzzyPositionController:
    """
    Controlador fuzzy de posição.

    Entradas:
        erro de posição;
        derivada do erro.

    Saída:
        Tensão aplicada ao motor.
    """
    
    LABELS = (
        "NG",
        "NP",
        "Z",
        "PP",
        "PG",
    )

    OUTPUT_SINGLETONS = {
        "NG": -1.0,
        "NP": -0.5,
        "Z": 0.0,
        "PP": 0.5,
        "PG": 1.0,
    }

    RULE_TABLE = {
        "NG": {
            "NG": "NG",
            "NP": "NG",
            "Z": "NG",
            "PP": "NP",
            "PG": "Z",
        },
        "NP": {
            "NG": "NG",
            "NP": "NG",
            "Z": "NP",
            "PP": "Z",
            "PG": "PP",
        },
        "Z": {
            "NG": "NG",
            "NP": "NP",
            "Z": "Z",
            "PP": "PP",
            "PG": "PG",
        },
        "PP": {
            "NG": "NP",
            "NP": "Z",
            "Z": "PP",
            "PP": "PG",
            "PG": "PG",
        },
        "PG": {
            "NG": "Z",
            "NP": "PP",
            "Z": "PG",
            "PP": "PG",
            "PG": "PG",
        },
    }

    def __init__(
            self,
            error_scale: float = 3.0,
            derivative_scale: float = 0.10,
            output_scale: float = 12.0,
            voltage_limit: float = 12.0,
        ) -> None:
            if voltage_limit <= 0:
                raise ValueError(
                    "O limite de tensão deve ser positivo."
                )
            
            self.error_scale = error_scale
            self.derivative_scale = derivative_scale
            self.output_scale = output_scale
            self.voltage_limit = voltage_limit
        
    @staticmethod
    def memberships(
        value: float,
    ) ->  dict[str, float]:
        """Calcula os graus de pertinência."""

        normalized = float(
            np.clip(
                value,
                -1.0,
                1.0,
            )
        )
        return{
            "NG": left_shoulder_membership(
                normalized,
                -1.0,
                -0.5,
            ),
            "NP": triangular_menbership(
                normalized,
                -1.0,
                -0.5,
                0.0,
            ),
            "Z": triangular_menbership(
                normalized,
                -0.5,
                0.0,
                0.5,
            ),
            "PP": triangular_menbership(
                normalized,
                0.0,
                0.5,
                1.0,
            ),
            "PG": right_shoulder_membership(
                normalized,
                0.5,
                1.0,
            ),
        }
    
    def compute(
            self,
            error: float,
            error_derivative: float,
    ) -> float:
        """Calcula a tensão de controle."""
        
        normalized_error = np.clip(
            self.error_scale * error,
            -1.0,
            1.0,
        )

        normalized_derivative = np.clip(
            (
            self.derivative_scale
            * error_derivative
            ),
            -1.0,
            1.0,
        )

        error_memberships = self.memberships(
            normalized_error
        )
        derivative_memberships = self.memberships(
            normalized_derivative
        )

        weighted_sum = 0.0
        activation_sum = 0.0

        for (
            error_label,
            error_degree,
        ) in error_memberships.items():
            for (
                derivative_label,
                derivative_degree,
            ) in derivative_memberships.items():
                
                activation = min(
                    error_degree,
                    derivative_degree,
                )

                if activation <= 0.0:
                    continue

                output_label = self.RULE_TABLE[
                    error_label
                ][derivative_label]

                weighted_sum += (
                    activation
                    * self.OUTPUT_SINGLETONS[
                        output_label
                    ]
                )

                activation_sum += activation
        
        if activation_sum > 0.0:
            normalized_output = (
                weighted_sum
                / activation_sum
            )
        else:
            normalized_output = 0.0

        voltage = (
            self.output_scale
            * normalized_output
        )

        return float(
            np. clip(
                voltage,
                -self.voltage_limit,
                self.voltage_limit,
            )
        )
