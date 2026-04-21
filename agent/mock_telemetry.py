"""
Mock Telemetry Generator
========================

Generates fake PID telemetry and environment context for testing the LLM
supervisor without depending on the PID team's code.

Each mock function simulates a specific control scenario that the LLM
supervisor should respond to differently.

Usage:
    from agent.mock_telemetry import MockScenarios
    telemetry, env_ctx = MockScenarios.oscillating()
"""

from agent.interfaces import PIDTelemetry, EnvironmentContext


class MockScenarios:
    """Factory for mock (telemetry, environment) pairs covering the six
    test scenarios from the Milestone Report."""

    # ── S1: Steady-state tracking ──────────────────────────────────
    @staticmethod
    def steady_state() -> tuple[PIDTelemetry, EnvironmentContext]:
        """Everything is fine. LLM should output HOLD."""
        telemetry = PIDTelemetry(
            timestamp="2026-04-05T14:30:00",
            kp=2.0, ki=0.05, kd=0.8,
            indoor_temp=21.8, setpoint=22.0,
            tracking_error=0.2,
            max_overshoot=0.3,
            control_signal=150.0,
            cumulative_energy_kwh=1.2,
            oscillation_count=1,
            cost_J=3.1,
            w_energy=0.33, w_comfort=0.34, w_response=0.33,
        )
        env = EnvironmentContext(
            outdoor_temp=15.0,
            weather_condition="clear",
            solar_irradiance_w=400.0,
            electricity_price=0.08,
            tariff_tier="off_peak",
        )
        return telemetry, env

    # ── S2: Weather disturbance (10°C outdoor drop) ────────────────
    @staticmethod
    def weather_disturbance() -> tuple[PIDTelemetry, EnvironmentContext]:
        """Outdoor temp dropped 10°C. Indoor temp falling, large error.
        LLM should increase comfort weight or add feedforward compensation."""
        telemetry = PIDTelemetry(
            timestamp="2026-04-05T15:00:00",
            kp=2.0, ki=0.05, kd=0.8,
            indoor_temp=19.5, setpoint=22.0,
            tracking_error=2.5,
            max_overshoot=0.0,
            control_signal=2800.0,
            cumulative_energy_kwh=2.1,
            oscillation_count=0,
            cost_J=22.0,
            w_energy=0.33, w_comfort=0.34, w_response=0.33,
        )
        env = EnvironmentContext(
            outdoor_temp=2.0,
            weather_condition="snow",
            weather_forecast_3h="Temperature staying around 0-2°C for next 3 hours",
            solar_irradiance_w=50.0,
            electricity_price=0.10,
            tariff_tier="mid_peak",
        )
        return telemetry, env

    # ── S3: Cost optimization (peak tariff) ─────────────────────────
    @staticmethod
    def peak_tariff() -> tuple[PIDTelemetry, EnvironmentContext]:
        """Electricity price is 3x normal (peak). Room is comfortable.
        LLM should shift weights toward energy savings."""
        telemetry = PIDTelemetry(
            timestamp="2026-04-05T18:30:00",
            kp=2.0, ki=0.05, kd=0.8,
            indoor_temp=22.3, setpoint=22.0,
            tracking_error=-0.3,
            max_overshoot=0.4,
            control_signal=800.0,
            cumulative_energy_kwh=5.5,
            oscillation_count=2,
            cost_J=15.0,
            w_energy=0.33, w_comfort=0.34, w_response=0.33,
        )
        env = EnvironmentContext(
            outdoor_temp=10.0,
            weather_condition="clear",
            solar_irradiance_w=0.0,
            electricity_price=0.30,
            tariff_tier="peak",
        )
        return telemetry, env

    # ── S4: Occupancy adaptation (guests coming) ────────────────────
    @staticmethod
    def guests_arriving() -> tuple[PIDTelemetry, EnvironmentContext]:
        """Guests arriving in 1 hour, room currently empty and cool.
        LLM should proactively pre-heat via setpoint correction."""
        telemetry = PIDTelemetry(
            timestamp="2026-04-05T17:00:00",
            kp=2.0, ki=0.05, kd=0.8,
            indoor_temp=19.0, setpoint=20.0,
            tracking_error=1.0,
            max_overshoot=0.0,
            control_signal=500.0,
            cumulative_energy_kwh=3.0,
            oscillation_count=0,
            cost_J=8.0,
            w_energy=0.5, w_comfort=0.3, w_response=0.2,
        )
        env = EnvironmentContext(
            outdoor_temp=8.0,
            weather_condition="cloudy",
            solar_irradiance_w=100.0,
            electricity_price=0.12,
            tariff_tier="mid_peak",
            is_occupied=False,
            scheduled_events="Guests arriving at 18:00",
        )
        return telemetry, env

    # ── Oscillation problem ─────────────────────────────────────────
    @staticmethod
    def oscillating() -> tuple[PIDTelemetry, EnvironmentContext]:
        """Temperature oscillating around setpoint. High oscillation count.
        LLM should increase ω_response or suggest reducing Kp."""
        telemetry = PIDTelemetry(
            timestamp="2026-04-05T16:00:00",
            kp=5.0, ki=0.10, kd=0.5,
            indoor_temp=22.4, setpoint=22.0,
            tracking_error=-0.4,
            max_overshoot=1.8,
            control_signal=1200.0,
            cumulative_energy_kwh=4.0,
            oscillation_count=12,
            cost_J=18.5,
            w_energy=0.33, w_comfort=0.34, w_response=0.33,
        )
        env = EnvironmentContext(
            outdoor_temp=12.0,
            weather_condition="clear",
            solar_irradiance_w=300.0,
            electricity_price=0.10,
            tariff_tier="mid_peak",
        )
        return telemetry, env

    # ── High energy consumption ──────────────────────────────────────
    @staticmethod
    def high_energy() -> tuple[PIDTelemetry, EnvironmentContext]:
        """Temperature is on-target but energy consumption is excessive.
        LLM should increase ω_energy to signal optimizer to find
        more efficient gains."""
        telemetry = PIDTelemetry(
            timestamp="2026-04-05T20:00:00",
            kp=3.0, ki=0.08, kd=1.0,
            indoor_temp=22.1, setpoint=22.0,
            tracking_error=-0.1,
            max_overshoot=0.2,
            control_signal=2500.0,
            cumulative_energy_kwh=8.7,
            oscillation_count=2,
            cost_J=25.0,
            w_energy=0.33, w_comfort=0.34, w_response=0.33,
        )
        env = EnvironmentContext(
            outdoor_temp=5.0,
            weather_condition="clear",
            solar_irradiance_w=0.0,
            electricity_price=0.15,
            tariff_tier="mid_peak",
        )
        return telemetry, env

    # ── S5: Multi-disturbance stress (24h) ──────────────────────────
    @staticmethod
    def multi_disturbance() -> tuple[PIDTelemetry, EnvironmentContext]:
        """Everything going wrong at once: error is growing, oscillations
        are present, and tariff is peak."""
        telemetry = PIDTelemetry(
            timestamp="2026-04-06T08:00:00",
            kp=2.5, ki=0.06, kd=0.7,
            indoor_temp=20.0, setpoint=22.0,
            tracking_error=2.0,
            max_overshoot=2.5,
            control_signal=2900.0,
            cumulative_energy_kwh=15.0,
            oscillation_count=8,
            cost_J=35.0,
            w_energy=0.33, w_comfort=0.34, w_response=0.33,
        )
        env = EnvironmentContext(
            outdoor_temp=-2.0,
            weather_condition="snow",
            weather_forecast_3h="Continued snow, temp dropping to -5°C",
            solar_irradiance_w=20.0,
            electricity_price=0.28,
            tariff_tier="peak",
            scheduled_events="Work-from-home day, occupied all day",
        )
        return telemetry, env

    @classmethod
    def all_scenarios(cls) -> dict[str, tuple[PIDTelemetry, EnvironmentContext]]:
        """Return all mock scenarios as a dict for batch testing."""
        return {
            "steady_state": cls.steady_state(),
            "weather_disturbance": cls.weather_disturbance(),
            "peak_tariff": cls.peak_tariff(),
            "guests_arriving": cls.guests_arriving(),
            "oscillating": cls.oscillating(),
            "high_energy": cls.high_energy(),
            "multi_disturbance": cls.multi_disturbance(),
        }
