# Project Pipeline Overview

This project builds an HVAC control evaluation pipeline from experimental thermal data cleaning to multi-condition controller comparison.

## 1. Data Cleaning from Bulb Test Experiments

We started from the bulb test experimental dataset and selected the **`insulated-concrete`** sheet because it most closely resembles the thermal behavior of a real building envelope with insulation and thermal inertia. Missing or discontinuous values in this sheet were filled using **linear interpolation**, producing a clean and continuous dataset suitable for model fitting.

The purpose of this step was to convert raw experimental measurements into a reliable thermal response dataset that could support identification of a simplified building model.

## 2. 1R1C Thermal Model Identification

Using the cleaned bulb test data, we fitted a **1R1C thermal model** for the simulated room. This model represents the indoor thermal dynamics with:

- one thermal resistance \(R\)
- one thermal capacitance \(C\)

The fitted parameters were saved and reused in all later control experiments as the shared simulation environment. In the implementation, the control scripts read these fitted parameters from the RC parameter file and use the same building model across all conditions. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

To validate whether this simplified thermal model was adequate, we also generated fitting metrics to check how well the identified model matched the cleaned bulb test data before proceeding to controller evaluation.

## 3. Weather Data Generation

We then generated **two days of outdoor temperature data** for the control experiments.

- **Day 1** was used only to determine the initial PID parameters for the baseline controller.
- **Day 2** was used for the formal experiments and metric evaluation.

This separates parameter initialization from final testing and keeps the comparison across controller conditions fair. The C0 script explicitly defines Day 1 as the PID tuning day and Day 2 as the formal baseline run. :contentReference[oaicite:4]{index=4}

## 4. C0: Fixed PID Baseline

**C0** is the baseline condition.

Its workflow is:

1. Use the fitted 1R1C model as the thermal environment.
2. Use **Day 1** weather data to search for a good fixed PID controller.
3. Tune PID parameters \((K_p, K_i, K_d)\) using grid search and a normalized cost function.
4. Run the tuned fixed PID controller on **Day 2** with a predefined setpoint schedule.

C0 represents the conventional fixed-parameter control setting and serves as the reference for all later conditions. 

## 5. C1: LLM Setpoint-Only Control

**C1** keeps the same thermal model and reuses the tuned PID parameters from C0, so the controller itself is still fixed.

The main change in C1 is the **setpoint schedule**, which is generated from user-style natural language demands. These include time-specific commands such as temporary nighttime cooling, temporary evening warming, and sleep-time preference recovery. C1 also saves a user instruction log that records:

- the user’s natural-language request
- the converted machine command

This condition tests whether semantic setpoint scheduling alone can improve control behavior without changing the PID gains. 

## 6. C2: Reactive Online PID Supervision

**C2** uses the same setpoint schedule as C1, but no longer keeps PID parameters fixed.

Instead, the system performs **online supervision every 5 minutes** and adjusts PID gains reactively using recent control telemetry, including tracking error, power behavior, and short-window performance statistics. In this version:

- setpoints are **not** adjusted online
- weights are **not** adjusted online
- only PID parameters are updated during operation

C2 therefore represents **reactive closed-loop PID supervision**, where the controller is corrected in response to observed control performance. 

## 7. C3: Proactive Online PID Supervision

**C3** builds on C2 by adding a **full-action proactive supervisor**.

C3 starts from the same setpoint schedule as C1 and C2, but it can adjust the online setpoint around that baseline. Every 5 minutes it uses recent telemetry plus **future context**, including:

- the next **1 hour** of generated outdoor temperature data
- the upcoming synthetic electricity tariff window
- user habit signals such as:
- nighttime cooler preference
- evening higher-temperature demand
- sleep-time stability preference

A proactive adjustment may change PID gains, online setpoint, and/or cost weights before a disturbance, tariff change, or demand shift occurs, rather than only after error has already appeared.

## 8. Unified Evaluation Metrics

All four conditions are evaluated using the same performance metrics, organized into four dimensions:

### Thermal Comfort
- `MAD`
- `CVR`

### Energy Efficiency
- `CE`
- `EC`

### Responsiveness
- `RT`
- `ST`

### Stability
- `OC`
- `MO`

These metric categories follow the experimental design described in the project document. :contentReference[oaicite:9]{index=9}

## 9. Final Comparison Across C0–C3

After all four conditions are executed, the results are compared in two ways.

### 9.1 Per-Metric Comparison

A comparison script reads the metrics produced by C0, C1, C2, and C3 and generates **one bar chart per metric**, with:

- x-axis: `C0`, `C1`, `C2`, `C3`
- y-axis: metric value

This provides a direct horizontal comparison for each performance indicator.

### 9.2 Normalized Weighted Overall Score

To support an overall model ranking, we also added a normalized weighted scoring method.

Each metric is first converted into a **benefit score** using min-max normalization under the rule that **lower is better**. Then the metrics are grouped by dimension and averaged within each dimension:

- Comfort: `MAD_C`, `CVR_fraction`
- Energy: `CE_kWh`, `EC_USD`
- Responsiveness: `RT_s`, `ST_s`
- Stability: `OC_count`, `MO_C`

Finally, an overall weighted score is computed using the following weights:

- **Comfort = 4**
- **Energy = 3**
- **Responsiveness = 1**
- **Stability = 2**

The final score is:

\[
\text{Overall Score} =
\frac{4 \cdot \text{Comfort} + 3 \cdot \text{Energy} + 1 \cdot \text{Responsiveness} + 2 \cdot \text{Stability}}{10}
\]

This makes it possible to evaluate not only which model performs best on individual metrics, but also which model is best **overall** under the chosen project priorities.

## Summary

In summary, the full pipeline is:

1. Clean bulb test data from the `insulated-concrete` sheet using linear interpolation.
2. Fit a 1R1C thermal model from the cleaned data.
3. Validate the fitted thermal model with fitting metrics.
4. Generate two days of outdoor weather data.
5. Use Day 1 to initialize the baseline PID controller.
6. Run Day 2 experiments under four conditions:
   - **C0**: fixed PID baseline
   - **C1**: LLM setpoint-only
   - **C2**: reactive online PID supervision
   - **C3**: proactive online PID supervision
7. Compare all conditions using unified control metrics and a normalized weighted overall score.

This pipeline enables a systematic evaluation of how increasingly intelligent HVAC control strategies affect comfort, energy, responsiveness, and stability.
