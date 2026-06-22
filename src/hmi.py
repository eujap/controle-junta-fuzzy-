from __future__ import annotations

from collections import deque
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.fuzzy_controller import FuzzyPositionController
from src.motor_model import MotorParameters, RobotJointPlant


class FuzzyControlHMI:
    """Interface supervisória do controle de posição da junta."""

    SAMPLE_TIME = 0.01
    UPDATE_INTERVAL_MS = 10

    VOLTAGE_LIMIT = 12.0

    DISTURBANCE_TORQUE = -0.08
    DISTURBANCE_DURATION = 0.20

    HISTORY_SECONDS = 8.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        self.root.title(
            "Controle Fuzzy de Junta Robótica"
        )

        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        motor_parameters = MotorParameters(
            inertia=0.01,
            viscous_friction=0.10,
            torque_constant=0.01,
            back_emf_constant=0.01,
            resistance=1.0,
        )

        self.plant = RobotJointPlant(
            parameters=motor_parameters,
            sample_time=self.SAMPLE_TIME,
        )

        self.controller = FuzzyPositionController(
            error_scale=4.0,
            derivative_scale=0.20,
            output_scale=12.0,
            voltage_limit=self.VOLTAGE_LIMIT,
        )

        self.running = False
        self.simulation_time = 0.0

        self.reference = 0.0
        self.previous_error = 0.0

        self.control_voltage = 0.0
        self.current_error = 0.0
        self.current_disturbance = 0.0

        self.disturbance_until = 0.0

        self.plot_update_counter = 0

        number_of_samples = int(
            self.HISTORY_SECONDS
            / self.SAMPLE_TIME
        )

        self.time_history: deque[float] = deque(
            maxlen=number_of_samples
        )

        self.reference_history: deque[float] = deque(
            maxlen=number_of_samples
        )

        self.position_history: deque[float] = deque(
            maxlen=number_of_samples
        )

        self.error_history: deque[float] = deque(
            maxlen=number_of_samples
        )

        self.voltage_history: deque[float] = deque(
            maxlen=number_of_samples
        )

        self.disturbance_history: deque[float] = deque(
            maxlen=number_of_samples
        )

        self.reference_entry_var = tk.StringVar(
            value="1.0"
        )

        self.position_var = tk.StringVar(
            value="0.0000 rad"
        )

        self.reference_display_var = tk.StringVar(
            value="0.0000 rad"
        )

        self.error_var = tk.StringVar(
            value="0.0000 rad"
        )

        self.velocity_var = tk.StringVar(
            value="0.0000 rad/s"
        )

        self.voltage_var = tk.StringVar(
            value="0.0000 V"
        )

        self.current_var = tk.StringVar(
            value="0.0000 A"
        )

        self.disturbance_var = tk.StringVar(
            value="0.0000 N.m"
        )

        self.saturation_var = tk.StringVar(
            value="Normal"
        )

        self.status_var = tk.StringVar(
            value="Simulação pausada"
        )

        self.time_var = tk.StringVar(
            value="0.00 s"
        )

        self.create_widgets()
        self.initialize_plot()
        self.append_initial_values()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application,
        )

        self.schedule_next_step()

    def create_widgets(self) -> None:
        """Cria os componentes da interface."""

        main_frame = ttk.Frame(
            self.root,
            padding=10,
        )

        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        main_frame.columnconfigure(
            1,
            weight=1,
        )

        main_frame.rowconfigure(
            0,
            weight=1,
        )

        left_panel = ttk.Frame(
            main_frame,
            padding=10,
        )

        left_panel.grid(
            row=0,
            column=0,
            sticky="ns",
            padx=(0, 10),
        )

        graph_panel = ttk.Frame(
            main_frame,
            padding=5,
        )

        graph_panel.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        self.graph_panel = graph_panel

        self.create_control_panel(left_panel)
        self.create_indicator_panel(left_panel)

    def create_control_panel(
        self,
        parent: ttk.Frame,
    ) -> None:
        """Cria os controles da simulação."""

        control_frame = ttk.LabelFrame(
            parent,
            text="Controles",
            padding=10,
        )

        control_frame.pack(
            fill=tk.X,
            pady=(0, 10),
        )

        ttk.Label(
            control_frame,
            text="Referência angular [rad]:",
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 5),
        )

        reference_entry = ttk.Entry(
            control_frame,
            textvariable=self.reference_entry_var,
            width=18,
        )

        reference_entry.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 8),
        )

        reference_entry.bind(
            "<Return>",
            lambda _event: self.apply_reference(),
        )

        apply_button = ttk.Button(
            control_frame,
            text="Aplicar referência",
            command=self.apply_reference,
        )

        apply_button.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=3,
        )

        self.start_pause_button = ttk.Button(
            control_frame,
            text="Iniciar",
            command=self.toggle_simulation,
        )

        self.start_pause_button.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=3,
        )

        disturbance_button = ttk.Button(
            control_frame,
            text="Aplicar distúrbio",
            command=self.apply_disturbance,
        )

        disturbance_button.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=3,
        )

        reset_button = ttk.Button(
            control_frame,
            text="Reiniciar",
            command=self.reset_simulation,
        )

        reset_button.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=3,
        )

        ttk.Separator(
            control_frame,
            orient=tk.HORIZONTAL,
        ).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=10,
        )

        ttk.Button(
            control_frame,
            text="Referência 1,0 rad",
            command=lambda: self.set_reference_value(
                1.0
            ),
        ).grid(
            row=7,
            column=0,
            sticky="ew",
            padx=(0, 2),
            pady=2,
        )

        ttk.Button(
            control_frame,
            text="Referência 0,5 rad",
            command=lambda: self.set_reference_value(
                0.5
            ),
        ).grid(
            row=7,
            column=1,
            sticky="ew",
            padx=(2, 0),
            pady=2,
        )

        ttk.Button(
            control_frame,
            text="Referência -0,5 rad",
            command=lambda: self.set_reference_value(
                -0.5
            ),
        ).grid(
            row=8,
            column=0,
            sticky="ew",
            padx=(0, 2),
            pady=2,
        )

        ttk.Button(
            control_frame,
            text="Referência zero",
            command=lambda: self.set_reference_value(
                0.0
            ),
        ).grid(
            row=8,
            column=1,
            sticky="ew",
            padx=(2, 0),
            pady=2,
        )

        control_frame.columnconfigure(
            0,
            weight=1,
        )

        control_frame.columnconfigure(
            1,
            weight=1,
        )

    def create_indicator_panel(
        self,
        parent: ttk.Frame,
    ) -> None:
        """Cria os indicadores numéricos."""

        indicator_frame = ttk.LabelFrame(
            parent,
            text="Estado do sistema",
            padding=10,
        )

        indicator_frame.pack(
            fill=tk.X,
        )

        indicators = [
            (
                "Tempo:",
                self.time_var,
            ),
            (
                "Referência:",
                self.reference_display_var,
            ),
            (
                "Posição:",
                self.position_var,
            ),
            (
                "Erro:",
                self.error_var,
            ),
            (
                "Velocidade:",
                self.velocity_var,
            ),
            (
                "Tensão:",
                self.voltage_var,
            ),
            (
                "Corrente:",
                self.current_var,
            ),
            (
                "Distúrbio:",
                self.disturbance_var,
            ),
            (
                "Saturação:",
                self.saturation_var,
            ),
            (
                "Estado:",
                self.status_var,
            ),
        ]

        for row, (
            label_text,
            variable,
        ) in enumerate(indicators):

            ttk.Label(
                indicator_frame,
                text=label_text,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=3,
            )

            ttk.Label(
                indicator_frame,
                textvariable=variable,
            ).grid(
                row=row,
                column=1,
                sticky="e",
                pady=3,
            )

        indicator_frame.columnconfigure(
            1,
            weight=1,
        )

    def initialize_plot(self) -> None:
        """Inicializa os gráficos da interface."""

        self.figure = Figure(
            figsize=(9, 7),
            dpi=100,
        )

        self.position_axis = (
            self.figure.add_subplot(311)
        )

        self.voltage_axis = (
            self.figure.add_subplot(312)
        )

        self.error_axis = (
            self.figure.add_subplot(313)
        )

        self.reference_line, = (
            self.position_axis.plot(
                [],
                [],
                "--",
                label="Referência",
            )
        )

        self.position_line, = (
            self.position_axis.plot(
                [],
                [],
                label="Posição",
            )
        )

        self.position_axis.set_ylabel(
            "Posição [rad]"
        )

        self.position_axis.set_title(
            "Controle de posição da junta"
        )

        self.position_axis.grid(True)
        self.position_axis.legend(
            loc="upper right"
        )

        self.voltage_line, = (
            self.voltage_axis.plot(
                [],
                [],
                label="Tensão",
            )
        )

        self.voltage_axis.axhline(
            self.VOLTAGE_LIMIT,
            linestyle="--",
            label="Limite superior",
        )

        self.voltage_axis.axhline(
            -self.VOLTAGE_LIMIT,
            linestyle="--",
            label="Limite inferior",
        )

        self.voltage_axis.set_ylabel(
            "Tensão [V]"
        )

        self.voltage_axis.set_ylim(
            -13.0,
            13.0,
        )

        self.voltage_axis.grid(True)
        self.voltage_axis.legend(
            loc="upper right"
        )

        self.error_line, = (
            self.error_axis.plot(
                [],
                [],
                label="Erro",
            )
        )

        self.error_axis.set_xlabel(
            "Tempo [s]"
        )

        self.error_axis.set_ylabel(
            "Erro [rad]"
        )

        self.error_axis.grid(True)
        self.error_axis.legend(
            loc="upper right"
        )

        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.graph_panel,
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )

    def append_initial_values(self) -> None:
        """Adiciona os valores iniciais ao histórico."""

        self.time_history.append(0.0)
        self.reference_history.append(0.0)
        self.position_history.append(0.0)
        self.error_history.append(0.0)
        self.voltage_history.append(0.0)
        self.disturbance_history.append(0.0)

    def set_reference_value(
        self,
        value: float,
    ) -> None:
        """Define e aplica uma referência rápida."""

        self.reference_entry_var.set(
            f"{value:.2f}"
        )

        self.apply_reference()

    def apply_reference(self) -> None:
        """Aplica a referência digitada pelo usuário."""

        try:
            new_reference = float(
                self.reference_entry_var.get()
                .replace(",", ".")
            )
        except ValueError:
            messagebox.showerror(
                "Valor inválido",
                "Digite uma referência numérica.",
            )
            return

        self.reference = new_reference

        # Evita um pico artificial na derivada
        # quando a referência é alterada.
        self.previous_error = (
            self.reference
            - self.plant.theta
        )

        self.reference_display_var.set(
            f"{self.reference:.4f} rad"
        )

    def toggle_simulation(self) -> None:
        """Inicia ou pausa a simulação."""

        self.running = not self.running

        if self.running:
            self.start_pause_button.configure(
                text="Pausar"
            )

            self.status_var.set(
                "Simulação em execução"
            )
        else:
            self.start_pause_button.configure(
                text="Continuar"
            )

            self.status_var.set(
                "Simulação pausada"
            )

    def apply_disturbance(self) -> None:
        """Aplica um torque externo durante 0,2 s."""

        if not self.running:
            messagebox.showinfo(
                "Simulação pausada",
                "Inicie a simulação antes de "
                "aplicar o distúrbio.",
            )
            return

        self.disturbance_until = (
            self.simulation_time
            + self.DISTURBANCE_DURATION
        )

    def reset_simulation(self) -> None:
        """Reinicia completamente a simulação."""

        self.running = False

        self.start_pause_button.configure(
            text="Iniciar"
        )

        self.status_var.set(
            "Simulação pausada"
        )

        self.plant.reset(
            initial_position=0.0,
            initial_velocity=0.0,
        )

        self.simulation_time = 0.0
        self.reference = 0.0
        self.previous_error = 0.0

        self.control_voltage = 0.0
        self.current_error = 0.0
        self.current_disturbance = 0.0

        self.disturbance_until = 0.0

        self.time_history.clear()
        self.reference_history.clear()
        self.position_history.clear()
        self.error_history.clear()
        self.voltage_history.clear()
        self.disturbance_history.clear()

        self.reference_entry_var.set("1.0")

        self.append_initial_values()
        self.update_indicators()
        self.update_plot()

    def simulation_step(self) -> None:
        """Executa uma iteração de controle de 10 ms."""

        error = (
            self.reference
            - self.plant.theta
        )

        error_derivative = (
            error
            - self.previous_error
        ) / self.SAMPLE_TIME

        voltage = self.controller.compute(
            error=error,
            error_derivative=error_derivative,
        )

        if (
            self.simulation_time
            < self.disturbance_until
        ):
            disturbance = (
                self.DISTURBANCE_TORQUE
            )
        else:
            disturbance = 0.0

        theta, _omega, _current = (
            self.plant.step(
                voltage=voltage,
                disturbance_torque=disturbance,
            )
        )

        self.simulation_time += (
            self.SAMPLE_TIME
        )

        self.control_voltage = voltage
        self.current_disturbance = disturbance

        self.current_error = (
            self.reference - theta
        )

        self.previous_error = error

        self.time_history.append(
            self.simulation_time
        )

        self.reference_history.append(
            self.reference
        )

        self.position_history.append(
            self.plant.theta
        )

        self.error_history.append(
            self.current_error
        )

        self.voltage_history.append(
            self.control_voltage
        )

        self.disturbance_history.append(
            self.current_disturbance
        )

    def update_indicators(self) -> None:
        """Atualiza os valores numéricos da tela."""

        self.time_var.set(
            f"{self.simulation_time:.2f} s"
        )

        self.reference_display_var.set(
            f"{self.reference:.4f} rad"
        )

        self.position_var.set(
            f"{self.plant.theta:.4f} rad"
        )

        self.error_var.set(
            f"{self.current_error:.4f} rad"
        )

        self.velocity_var.set(
            f"{self.plant.omega:.4f} rad/s"
        )

        self.voltage_var.set(
            f"{self.control_voltage:.4f} V"
        )

        self.current_var.set(
            f"{self.plant.current:.4f} A"
        )

        self.disturbance_var.set(
            f"{self.current_disturbance:.4f} N.m"
        )

        if abs(
            self.control_voltage
        ) >= self.VOLTAGE_LIMIT - 1e-6:
            self.saturation_var.set(
                "SATURADO"
            )
        else:
            self.saturation_var.set(
                "Normal"
            )

    def update_plot(self) -> None:
        """Atualiza os gráficos em tempo real."""

        time_values = list(
            self.time_history
        )

        if not time_values:
            return

        reference_values = list(
            self.reference_history
        )

        position_values = list(
            self.position_history
        )

        voltage_values = list(
            self.voltage_history
        )

        error_values = list(
            self.error_history
        )

        self.reference_line.set_data(
            time_values,
            reference_values,
        )

        self.position_line.set_data(
            time_values,
            position_values,
        )

        self.voltage_line.set_data(
            time_values,
            voltage_values,
        )

        self.error_line.set_data(
            time_values,
            error_values,
        )

        maximum_time = max(
            self.HISTORY_SECONDS,
            self.simulation_time,
        )

        minimum_time = max(
            0.0,
            maximum_time
            - self.HISTORY_SECONDS,
        )

        for axis in (
            self.position_axis,
            self.voltage_axis,
            self.error_axis,
        ):
            axis.set_xlim(
                minimum_time,
                maximum_time,
            )

        self.position_axis.relim()
        self.position_axis.autoscale_view(
            scalex=False,
            scaley=True,
        )

        self.error_axis.relim()
        self.error_axis.autoscale_view(
            scalex=False,
            scaley=True,
        )

        self.voltage_axis.set_ylim(
            -13.0,
            13.0,
        )

        self.canvas.draw_idle()

    def schedule_next_step(self) -> None:
        """Agenda continuamente as atualizações da HMI."""

        if self.running:
            self.simulation_step()

        self.update_indicators()

        self.plot_update_counter += 1

        # Atualiza o gráfico a cada 50 ms,
        # evitando sobrecarregar a interface.
        if self.plot_update_counter >= 5:
            self.update_plot()
            self.plot_update_counter = 0

        self.root.after(
            self.UPDATE_INTERVAL_MS,
            self.schedule_next_step,
        )

    def close_application(self) -> None:
        """Fecha a interface com segurança."""

        self.running = False
        self.root.destroy()


def main() -> None:
    """Inicia a interface supervisória."""

    root = tk.Tk()

    FuzzyControlHMI(
        root=root
    )

    root.mainloop()


if __name__ == "__main__":
    main()