from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Simulation setup
start_scope()
defaultclock.dt = 0.1*ms

# Neuron parameters and model
rows, cols = 4, 4
num_exc = 32*32  # 32x32 grid for excitatory neurons
num_inh = 32*32  # 32x32 grid for inhibitory neurons
tau_M, tau_E, tau_I = 20*ms, 20*ms, 40*ms
orientation_preferences = linspace(0, pi, num_exc)  # Example orientation preferences
grid_size = (32, 32) # Grid size for neuron placement

W_standard = 0.75
K_L=1.1

# two sum and difference model parameters
W_plus = W_standard * (K_L-1)
W_FF = W_standard * (K_L+1)

# Linear model for orientation selectivity
linear_eqs = '''
dr/dt = (-r + dot(W, r) + I)/tau_M : 1
theta : radian  # Preferred orientation
I : 1  # External input
x : meter (constant) # Spatial x-coordinate
y : meter (constant) # Spatial y-coordinate
'''
linear_eqs_plus = '''
dr_plus/dt = (-(1 + W_plus) * r_plus + W_FF * r_minus)/tau_M : 1
dr_minus/dt = -r_minus/tau_M : 1
I : 1 (constant)
theta : radian  # Preferred orientation
x : meter (constant) # Spatial x-coordinate
y : meter (constant) # Spatial y-coordinate
'''
linear_eqs_minus = '''
dr_minus/dt = -r_minus/tau_M : 1
I : 1 (constant)
theta : radian  # Preferred orientation
x : meter (constant) # Spatial x-coordinate
y : meter (constant) # Spatial y-coordinate
'''

rate_eqs = '''
g_plus = (exp(-t/tau_M) - exp(-(1 + W_plus) * t/tau_M))/W_plus : 1
dr/dt = (-r + r_base * sqrt((W_FF * g_plus) ** 2 + exp(-t/tau_M) ** 2))*20/tau_M : Hz
r_base : Hz (constant)
I : 1 (constant)
W_plus : 1 (constant)
theta : radian  # Preferred orientation
x : meter (constant)  # Spatial x-coordinate
y : meter (constant) # Spatial y-coordinate
'''

# Input based on orientation difference
def orientation_input(theta_diff):
    return  4*exp(-theta_diff**2/(20 * degrees)**2)

linear_exc_neurons = NeuronGroup(num_exc, rate_eqs, method= 'euler')

linear_exc_neurons.r = 1*Hz + np.random.rand(num_exc)*Hz
linear_exc_neurons.theta = orientation_preferences
linear_exc_neurons.W_plus = W_plus  # Ensure internal consistency

linear_inh_neurons = NeuronGroup(num_inh, rate_eqs, method = 'euler')
linear_inh_neurons.r = 1*Hz + np.random.rand(num_exc)*Hz
linear_inh_neurons.theta = orientation_preferences
linear_inh_neurons.W_plus = W_plus  # Ensure internal consistency

# initialize the grid positions
grid_dist = 25*umeter
# Assuming cols, rows, and grid_dist are defined as before
# Example for setting x and y for excitatory neurons
linear_exc_neurons.run_regularly('''
    x = int(i % cols) * grid_dist - (cols / 2.0) * grid_dist
    y = int(i // cols) * grid_dist - (rows / 2.0) * grid_dist
''', dt=defaultclock.dt)

# Do the same for inhibitory neurons if necessary
linear_inh_neurons.run_regularly('''
    x = int(i % cols) * grid_dist - (cols / 2.0) * grid_dist
    y = int(i // cols) * grid_dist - (rows / 2.0) * grid_dist
''', dt=defaultclock.dt)

# Ensure these run_regularly commands are placed before the 'run(duration)' command

# External stimulus orientation
stimulus_orientation = 45 * pi/180  # Example stimulus orientation

# Random connections (no self-connections)
S_exc2exc = Synapses(linear_exc_neurons, linear_exc_neurons)
S_exc2exc.connect('i != j and abs(theta_pre - theta_post) < pi')

S_exc2inh = Synapses(linear_exc_neurons, linear_inh_neurons)
S_exc2inh.connect('i != j and abs(theta_pre - theta_post) < pi')

S_inh2exc = Synapses(linear_inh_neurons, linear_exc_neurons)
S_inh2exc.connect('i != j and abs(theta_pre - theta_post) < pi')

S_inh2inh = Synapses(linear_inh_neurons, linear_inh_neurons)
S_inh2inh.connect('i != j and abs(theta_pre - theta_post) < pi')

# Example connection with orientation-based input
@network_operation
def update_input(t):
    if t > defaultclock.dt:
        theta_stimulus = 90 * pi / 180  # Convert to radians for consistency
        theta_diff = abs(linear_exc_neurons.theta - theta_stimulus)
        linear_exc_neurons.I =  4 * exp(-theta_diff ** 2 / (20 * (pi/180)) ** 2) * W_FF
        linear_inh_neurons.I =  4 * exp(-theta_diff ** 2 / (20 * (pi / 180)) ** 2)

# Simulation time
duration = 80*ms

# Initialize monitors for the excitatory and inhibitory neurons
R_mon_exc = StateMonitor(linear_exc_neurons, 'r', record=True)
R_mon_inh = StateMonitor(linear_inh_neurons, 'r', record=True)

# Run the simulation
run(duration)

# Plot setup
colors = ['b', 'g', 'r', 'c', 'm']  # Colors for P1 to P5
time_points = R_mon_exc.t / ms  # Time points in ms

plt.figure(figsize=(10, 6))

# Plotting |r(t)| for each pattern
for pattern_idx in range(5):  # Assuming 5 different patterns simulated separately
    # Placeholder for actual |r(t)| values; use recorded data
    abs_r_exc = np.abs(R_mon_exc.r[pattern_idx])
    abs_r_inh = np.abs(R_mon_inh.r[pattern_idx])

    # Averaging or otherwise combining excitatory and inhibitory responses might be needed
    combined_response = (abs_r_exc + abs_r_inh) / 2.0  # Simplified example

    plt.plot(time_points, combined_response, label=f'Pattern P{pattern_idx + 1}', color=colors[pattern_idx])

plt.xlabel('Time (ms)')
plt.ylabel('|r(t)|')
plt.xlim([0, 8])  # Corresponding to 0 to 8 ms
plt.ylim([0, 15])  # As per the instructions
plt.title('Distribution Curve of activity vector |r(t)| Along Time Course')
plt.legend()
plt.grid(True)
plt.show()