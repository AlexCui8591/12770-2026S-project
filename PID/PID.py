import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("boptest_data_1.csv")

R = 0.2010101998055624
C = 48378.21542324912
Pmax = 1000.0

dt = df["dt"].iloc[0]
To_series = df["To"].values
time_hours = df["time"].values / 3600

Tset_series = np.where((time_hours % 24) < 8, 30.0, 35.0)

Ti0 = df["Ti"].iloc[0]

Kp = 100
Ki = 0.001
Kd = 10

Ti_pid = [Ti0]
P_heater_list = []

Ti_free = [Ti0]

integral = 0.0
prev_error = Tset_series[0] - Ti0

for k in range(len(df) - 1):
    Ti = Ti_pid[-1]
    To = To_series[k]
    Tset = Tset_series[k]

   
    error = Tset - Ti
    integral += error * dt
    derivative = (error - prev_error) / dt

    P_heater = Kp * error + Ki * integral + Kd * derivative
    P_heater = np.clip(P_heater, 0.0, Pmax)

    Ti_next = Ti + dt / C * ((To - Ti) / R + P_heater)

    Ti_pid.append(Ti_next)
    P_heater_list.append(P_heater)

    prev_error = error

    
    Ti_natural = Ti_free[-1] + dt / C * ((To - Ti_free[-1]) / R)
    Ti_free.append(Ti_natural)


result = df.iloc[:len(Ti_pid)].copy()
result["Ti_pid"] = Ti_pid
result["Ti_free"] = Ti_free
result["Tset"] = Tset_series[:len(Ti_pid)]
result["P_heater_W"] = [P_heater_list[0]] + P_heater_list


plt.figure(figsize=(12, 6))
plt.plot(result["time"]/3600, result["Ti_pid"], label="PID controlled Ti", linewidth=2)
plt.plot(result["time"]/3600, result["Ti_free"], label="1R1C natural Ti", linestyle="--")
plt.plot(result["time"]/3600, result["Tset"], label="Setpoint", linestyle=":")
plt.plot(result["time"]/3600, result["To"], label="Outdoor To", alpha=0.5)

plt.xlabel("Time [hour]")
plt.ylabel("Temperature [°C]")
plt.title("PID Response")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 4))
plt.plot(result["time"]/3600, result["P_heater_W"], label="Heater Power (W)")
plt.xlabel("Time [hour]")
plt.ylabel("Power [W]")
plt.legend()
plt.grid(True)
plt.show()