import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_DEVICES = 50               # Number of IoMT devices
SERVER_CPU_CAPACITY = 1e6       # Server computational capacity (in CPU cycles/sec)
SECURITY_OVERHEAD_FACTOR = 1.2  # 20% extra cost for secure communication
MAX_ITER = 100                  # Maximum iterations for convergence
PRICE_INIT = 0.001              # Initial price per computation unit
PRICE_STEP = 0.0001             # Step size for price update

# Device parameters (random initialization)
np.random.seed(42)  # For reproducibility
task_sizes = np.random.uniform(500, 2000, NUM_DEVICES)     # in CPU cycles
local_cpu_freqs = np.random.uniform(0.5e6, 2e6, NUM_DEVICES)  # in CPU cycles/sec
energy_per_cycle = np.random.uniform(1e-9, 3e-9, NUM_DEVICES)  # Energy consumption per CPU cycle (Joules)

# Constants
BANDWIDTH = 1e6           # Hz
TRANSMISSION_POWER = 0.5  # Watts
NOISE_POWER = 1e-9        # Watts

# Channel gains (simplified path loss model)
distances = np.random.uniform(10, 100, NUM_DEVICES)  # Distance to edge server (meters)
path_loss_exponent = 3
channel_gains = 1 / (distances ** path_loss_exponent)

# Function to calculate transmission rate
def calculate_rate(gain):
    return BANDWIDTH * np.log2(1 + (TRANSMISSION_POWER * gain) / NOISE_POWER)

# Function for device to decide local or offload
def device_decision(price, secure=True):
    decisions = []
    for i in range(NUM_DEVICES):
        # Local execution cost
        time_local = task_sizes[i] / local_cpu_freqs[i]
        energy_local = task_sizes[i] * energy_per_cycle[i]
        local_cost = energy_local + time_local

        # Offloading cost
        rate = calculate_rate(channel_gains[i])
        time_offload = task_sizes[i] / rate
        energy_offload = TRANSMISSION_POWER * time_offload
        transmission_cost = energy_offload + time_offload
        if secure:
            transmission_cost *= SECURITY_OVERHEAD_FACTOR

        offloading_cost = transmission_cost + price * task_sizes[i]

        if offloading_cost < local_cost:
            decisions.append(1)  # Offload
        else:
            decisions.append(0)  # Local execution
    return np.array(decisions)

# Main simulation
price = PRICE_INIT
history_price = []
history_revenue = []
history_offloading_ratio = []

for iteration in range(MAX_ITER):
    decisions = device_decision(price)
    num_offloading = np.sum(decisions)

    # Server revenue = price * total offloaded computation
    total_offloaded = np.sum(task_sizes * decisions)
    revenue = price * total_offloaded

    history_price.append(price)
    history_revenue.append(revenue)
    history_offloading_ratio.append(num_offloading / NUM_DEVICES)

    # Update price: if too many devices offloading, raise price; otherwise lower
    if num_offloading > 0.7 * NUM_DEVICES:
        price += PRICE_STEP
    elif num_offloading < 0.3 * NUM_DEVICES:
        price -= PRICE_STEP

    # Keep price positive
    price = max(price, 0.00001)

print("Final Price: ", price)
print("Final Offloading Ratio: ", history_offloading_ratio[-1])

# Plot results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history_price, label='Price')
plt.xlabel('Iterations')
plt.ylabel('Price per CPU Cycle')
plt.title('Price Evolution')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_offloading_ratio, label='Offloading Ratio')
plt.xlabel('Iterations')
plt.ylabel('Offloading Devices Ratio')
plt.title('Device Offloading Ratio Over Time')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Compare Energy Consumption (Secure vs Insecure)
secure_energy = []
insecure_energy = []

for i in range(NUM_DEVICES):
    rate = calculate_rate(channel_gains[i])
    time_offload = task_sizes[i] / rate
    energy_offload = TRANSMISSION_POWER * time_offload

    secure_total = (energy_offload + time_offload) * SECURITY_OVERHEAD_FACTOR
    insecure_total = energy_offload + time_offload

    secure_energy.append(secure_total)
    insecure_energy.append(insecure_total)

plt.figure(figsize=(8, 5))
plt.plot(secure_energy, label='Secure Offloading Cost', marker='o')
plt.plot(insecure_energy, label='Insecure Offloading Cost', marker='x')
plt.xlabel('Device Index')
plt.ylabel('Offloading Cost (Energy + Delay)')
plt.title('Secure vs Insecure Offloading Cost')
plt.grid()
plt.legend()
plt.show()