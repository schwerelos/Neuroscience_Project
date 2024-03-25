import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from numpy import cos
from brian2tools import *
#brian_plot(M) function to plot the map

t1 = time.time()
# Parameters
# Model Parameters
num_neurons = 32  # Number of neurons
M_plus_exc, M_plus_inh  = 32, 32 #size of each barrel(in neurons)
M_minus_exc, M_minus_inh = 32, 32
N_plus_exc, N_plus_inh, N_minus_exc, N_minus_inh = 4, 4, 4, 4 #neurons per barrel
barrelarraysize = 5 #choose 3 or 4 if memory error
Nbarrels = barrelarraysize**2
#Stimulation
stim_change_time = 2*ms
theta_stim = 20.
#Neuron parameters
tau_M, tau_E, tau_I = 20*ms, 20*ms, 40*ms
C = 400*pfarad
g_leak = 10*nS
r_E, r_I = 1*Hz, 0*Hz
E_leak, E_E, E_I = -70*mV, 0*mV, -70*mV
v_thresh, v_reset = -54*mV, -60*mV
t_refract = 1.75*mV

W_standard = 0.75
K_L = 1.1
W_plus = W_standard * (K_L-1)
W_FF = W_standard * (K_L+1)

#P_minus, difference modes model
eqs_p_minus = '''
#spiking model equation
dv/dt = [g_leak*(E_leak - v) + g_E*(E_E - v) + g_I*(E_I -v)] / C : volt (unless refractory)
g_E : nS
g_I : nS
is_active = abs((barrel_x + 0.5 - bar_x) * cos(direction) + (barrel_y + 0.5 - bar_y) * sin(direction)) < 0.5: boolean
barrel_x : integer # The x index of the barrel
barrel_y : integer # The y index of the barrel
# Stimulus parameters (same for all neurons)
bar_x = cos(direction)*(t - stim_start_time)/(5*ms) + stim_start_x : 1 (shared)
bar_y = sin(direction)*(t - stim_start_time)/(5*ms) + stim_start_y : 1 (shared)
direction : 1 (shared)# direction of the current stimulus
stim_start_time : second (shared)# start time of the current stimulus
stim_start_x : 1 (shared)# start position of the stimulus
stim_start_y : 1 (shared)# start position of the stimulus
'''

P_minus = NeuronGroup(num_neurons, eqs_p_minus, threshold='v>v_thresh', reset='v=v_reset', method='euler', name='p_minus_E')
#P_minus_inh = NeuronGroup(num_neurons, eqs_p_minus, threshold='v>v_thresh', reset='v=v_reset', method='euler', name='p_minus_I')
P_minus.barrel_x = '(i // N_minus_exc) % barrelarraysize + 0.5'
P_minus.barrel_y = 'i // (barrelarraysize*N_minus_exc) + 0.5'
#P_minus_inh.barrel_x = '(i // N_inh_exc) % barrelarraysize + 0.5'
#P_minus_inh.barrel_y = 'i // (barrelarraysize*N_inh_exc) + 0.5'

stimradius = (11+1)*.5

# Chose a new randomly oriented bar every 60ms
#direction = rand()*2*pi
runner_code = '''
direction = 0.5*pi
stim_start_x = barrelarraysize / 2.0 - cos(direction)*stimradius
stim_start_y = barrelarraysize / 2.0 - sin(direction)*stimradius
stim_start_time = t
'''
P_minus.run_regularly(runner_code, dt=100*ms, when='start')
#P_minus_inh.run_regularly(runner_code, dt=100*ms, when='start')

#P_plus, sum modes model
eqs_p_plus = '''
#linear rate model equation
dr/dt = -r/tau_E :Hz
barrel_idx : integer
x : 1  # in "barrel width" units
y : 1  # in "barrel width" units
'''

P_plus = NeuronGroup(num_neurons, eqs_p_plus, threshold='v>v_thresh', reset='v=v_reset', method='euler', name='p_plus_E')
#P_plus_inh = NeuronGroup(num_neurons, eqs_p_plus, threshold='v>v_thresh', reset='v=v_reset', method='euler', name='p_plus_I')
#P_plus_exc.r = r_E
#P_plus_inh.r = r_I

# Subgroups for excitatory and inhibitory neurons in P_plus_exc and P_plus_inh
P_plus_exc = P_plus[:Nbarrels*num_neurons]
P_plus_inh = P_plus[Nbarrels*num_neurons:]

P_minus_exc = P_minus[:Nbarrels*num_neurons]
P_minus_inh = P_minus[Nbarrels*num_neurons:]

#P_plus_exc and P_plus_inh excitatory
# The units for x and y are the width/height of a single barrel
P_plus_exc.x = '(i % (barrelarraysize*num_neurons)) * (1.0/num_neurons)'
P_plus_exc.y = '(i // (barrelarraysize*num_neurons)) * (1.0/num_neurons)'
P_plus_exc.barrel_idx = 'floor(x) + floor(y)*barrelarraysize'

P_plus_inh.x = '(i % (barrelarraysize*num_neurons)) * (1.0/num_neurons)'
P_plus_inh.y = '(i // (barrelarraysize*num_neurons)) * (1.0/num_neurons)'
P_plus_inh.barrel_idx = 'floor(x) + floor(y)*barrelarraysize'

P_minus_exc.x = '(i % (barrelarraysize*num_neurons)) * (1.0/num_neurons)'
P_minus_exc.y = '(i // (barrelarraysize*num_neurons)) * (1.0/num_neurons)'
P_minus_exc.barrel_idx = 'floor(x) + floor(y)*barrelarraysize'

P_minus_inh.x = '(i % (barrelarraysize*num_neurons)) * (1.0/num_neurons)'
P_minus_inh.y = '(i // (barrelarraysize*num_neurons)) * (1.0/num_neurons)'
P_minus_inh.barrel_idx = 'floor(x) + floor(y)*barrelarraysize'

print("Building synapses, please wait...")

# Feedforward connections (plastic)
feedforward = Synapses(P_minus, P_plus,
                       model='''#Stimulus rate equation
                       theta : radian (constant)
                       rate_evoked = 4 * exp(-theta**2/(theta_stim)**2) : 1''',
                       on_post='''P_plus.r += P_minus.r*W_FF''',
                       name='feedforward')
# Connect neurons in the same barrel with 50% probability
feedforward.connect('(barrel_x_pre + barrelarraysize*barrel_y_pre) == barrel_idx_post', p=0.5)
feedforward.w = W_FF

# Excitatory lateral connections
recurrent_exc = Synapses(P_plus_exc, P_plus, model='w:volt',
                         name='recurrent_exc')
recurrent_exc.connect(p='.15*exp(-.5*(((x_pre-x_post)/.4)**2+((y_pre-y_post)/.4)**2))')
recurrent_exc.w['j<Nbarrels*num_neurons'] = W_plus # excitatory->excitatory
recurrent_exc.w['j>=Nbarrels*num_neurons'] = W_FF # excitatory->inhibitory


# Inhibitory lateral connections
print('inhibitory lateral')
recurrent_inh = Synapses(P_plus_inh, P_plus_exc,
                         name='recurrent_inh')
recurrent_inh.connect(p='exp(-.5*(((x_pre-x_post)/.2)**2+((y_pre-y_post)/.2)**2))')

if get_device().__class__.__name__ == 'RuntimeDevice':
    print('Total number of connections')
    print('feedforward: %d' % len(feedforward))

    t2 = time.time()
    print("Construction time: %.1fs" % (t2 - t1))

run(100*ms, report='text')

# Calculate the preferred direction of each cell in P_plus by doing a
# vector average of the selectivity of the projecting P_minus cells, weighted
# by the synaptic weight.
_r = bincount(feedforward.j,
              weights=feedforward.w * cos(feedforward.selectivity_pre)/feedforward.N_incoming,
              minlength=len(P_plus_exc))
_i = bincount(feedforward.j,
              weights=feedforward.w * sin(feedforward.selectivity_pre)/feedforward.N_incoming,
              minlength=len(P_plus_exc))
selectivity_exc = (arctan2(_r, _i) % (2*pi))*180./pi

subplot(211)
scatter(P_minus.x[:Nbarrels*num_neurons], P_minus.y[:Nbarrels*num_neurons],
        c=selectivity_exc[:Nbarrels*num_neurons],
        edgecolors='none', marker='s', cmap='hsv')
vlines(np.arange(barrelarraysize), 0, barrelarraysize, 'k')
hlines(np.arange(barrelarraysize), 0, barrelarraysize, 'k')
clim(0, 360)
colorbar()
show()

subplot(212)
scatter(P_plus.x[:Nbarrels*num_neurons], P_plus.y[:Nbarrels*num_neurons],
        c=selectivity_exc[:Nbarrels*num_neurons],
        edgecolors='none', marker='s', cmap='hsv')
vlines(np.arange(barrelarraysize), 0, barrelarraysize, 'k')
hlines(np.arange(barrelarraysize), 0, barrelarraysize, 'k')
clim(0, 360)
colorbar()
show()
