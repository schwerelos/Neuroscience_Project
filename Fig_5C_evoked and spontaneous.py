from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_neurons_exc = 40000  # Number of excitatory neurons
num_neurons_inh = 10000  # Number of inhibitory neurons
num_frames = 40000  # Number of frames
theta_stim = 20.  # Stimulus orientation
t_stim = 80 * ms  # Stimulus duration

# Model parameters
tau_M, tau_E, tau_I = 20 * ms, 20 * ms, 40 * ms
C = 400 * pF
g_leak = 10 * nS
E_leak = -70 * mV
E_E = 0 * mV
E_I = -70 * mV
v_thresh = -54 * mV
v_reset = -60 * mV
t_refract = 1.75 * ms

# Define the spiking model equations
eqs_spiking_capacity = '''
dv/dt = (g_leak * (E_leak - v) + g_E * (E_E - v) + g_I * (E_I - v)) / C : volt (unless refractory)
g_E : siemens (constant)
g_I : siemens (constant)
'''
#capacity C /g = tau (time constant)
eqs_spiking = '''
dv/dt = g_leak * (E_leak - v)/C +  (E_E - v)/tau_E +  (E_I - v)/tau_I  : volt (unless refractory)
g_E : siemens (constant)
g_I : siemens (constant)
'''

# Create excitatory and inhibitory neuron groups for the 0° evoked map
exc_neurons_evoked = NeuronGroup(num_neurons_exc, eqs_spiking, threshold='v>v_thresh', reset='v=v_reset', refractory='t_refract', method='euler')

# Set initial conditions for the 0° evoked map
exc_neurons_evoked.v = E_leak

# Define the spatial and temporal filters
spatial_filter_width = 200  # μm
temporal_filter_gamma = 40  # Hz

# Define spatial filter
spatial_filter = np.exp(-(np.arange(-num_neurons_exc/2, num_neurons_exc/2)**2) / (spatial_filter_width**2))

# Define temporal filter
temporal_filter_num_steps = 1000
temporal_filter = (np.linspace(0, num_frames, temporal_filter_num_steps) ** 2) * np.exp(-temporal_filter_gamma * np.linspace(0, num_frames, temporal_filter_num_steps))

# Generate background input for the 0° evoked map
temporal_filter_reshaped = temporal_filter[:, np.newaxis]  # Reshape to column vector
background_input_evoked = 4 * nS * spatial_filter[:, np.newaxis] * temporal_filter_reshaped.T

# Add background input to excitatory neurons for the 0° evoked map
@network_operation(dt=defaultclock.dt)
def input_update_evoked():
    exc_neurons_evoked.g_E = background_input_evoked[:, int(t_stim / defaultclock.dt)]

# Create excitatory neuron group for the control map
exc_neurons_control = NeuronGroup(num_neurons_exc, eqs_spiking, threshold='v>v_thresh', reset='v=v_reset', refractory='t_refract', method='euler')

# Set initial conditions for the control map
exc_neurons_control.v = E_leak

# Chunk size for generating background input
chunk_size = 1000


# Function to compute background input control
def compute_background_input_control(num_neurons_exc, num_frames):
    # Generate random background input with the correct shape
    background_input_control = np.random.rand(num_neurons_exc, num_frames)
    return background_input_control

def input_update_control():
    # Call compute_background_input_control to get background input chunk
    background_input_control_chunk = compute_background_input_control(num_neurons_exc, num_frames)

    # Check if the shape is correct
    if background_input_control_chunk.shape != (num_neurons_exc, num_frames):
        raise ValueError("Shape of background input chunk is {}, expected ({}, {})".format(
            background_input_control_chunk.shape, num_neurons_exc, num_frames))

# Call the function to update the background input
input_update_control()


# Generate simulated responses for the 0° evoked map
spontaneous_frames_evoked = np.random.rand(num_frames)

# Calculate the correlation coefficient for the 0° evoked map
correlation_coefficients_evoked = np.zeros(num_frames)
expanded_exc_neurons_v_evoked = np.repeat(exc_neurons_evoked.v, num_frames // num_neurons_exc)
for i in range(num_frames):
    correlation_coefficients_evoked[i] = np.corrcoef(expanded_exc_neurons_v_evoked, spontaneous_frames_evoked)[0, 1]

# Run the simulation for the control map
run(num_frames * defaultclock.dt)

# Generate simulated responses for the control map
spontaneous_frames_control = np.random.rand(num_frames)

# Calculate the correlation coefficient for the control map
correlation_coefficients_control = np.zeros(num_frames)
expanded_exc_neurons_v_control = np.repeat(exc_neurons_control.v, num_frames // num_neurons_exc)
for i in range(num_frames):
    correlation_coefficients_control[i] = np.corrcoef(expanded_exc_neurons_v_control, spontaneous_frames_control)[0, 1]

# Bin the correlation coefficients and count the number of frames in each bin for the 0° evoked map
num_bins = 20
hist_evoked, bin_edges_evoked = np.histogram(correlation_coefficients_evoked, bins=num_bins)

# Bin the correlation coefficients and count the number of frames in each bin for the control map
hist_control, bin_edges_control = np.histogram(correlation_coefficients_control, bins=num_bins)

# Calculate the fraction of frames in each bin for the 0° evoked map
fraction_frames_evoked = hist_evoked / num_frames

# Calculate the fraction of frames in each bin for the control map
fraction_frames_control = hist_control / num_frames

# Plot the correlation coefficient distributions for both maps
plt.figure(figsize=(8, 6))
plt.plot(bin_edges_evoked[:-1], fraction_frames_evoked, marker='o', linestyle='-', label='0° Evoked Map')
plt.plot(bin_edges_control[:-1], fraction_frames_control, marker='o', linestyle='--', label='Control Map')
plt.xlim(-0.6, 0.6)
plt.ylim(0, 1)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Fraction of Frames')
plt.title('Correlation Coefficient Distribution Comparison')
plt.grid(True)
plt.legend()
plt.show()
