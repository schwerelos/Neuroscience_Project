from brian2 import *
import matplotlib.pyplot as plt
import customized_function
import numpy as np


N_E = 1      # represents ecitatory and inhibitory group
N_I = 1

tau=20*ms

standard_rate_factor=1     # be served as rate 1 in the first figure

weight_standard = 4.28
K_L=1.1

# two sum and difference model parameters
W_plus = weight_standard * (K_L-1)
W_FF = weight_standard * (K_L+1)

eqs_E = '''
dr/dt = (-r + S_e2e + S_i2e + I_E) / tau  : Hz
S_e2e :Hz
S_i2e :Hz
I_E :Hz
'''

eqs_I = '''
dr/dt = (-r + S_e2i + S_i2i + I_I) / tau  : Hz
S_e2i :Hz
S_i2i :Hz
I_I :Hz
'''

# iteration equation using sum and difference basis
eqs_plus = '''
dr/dt = (-r + S_m2p + S_p2p) / tau : Hz
S_m2p :Hz
S_p2p :Hz
'''

eqs_minus = '''
dr/dt = -r / tau : Hz
'''


# normal model
G_exc = NeuronGroup(N_E, eqs_E,method='euler')
G_inh = NeuronGroup(N_I, eqs_I,method='euler')
# model with changing basis (plus: sum basis   minus: difference basis)
G_exc_none = NeuronGroup(N_E, eqs_E,method='euler')
G_inh_none = NeuronGroup(N_E, eqs_I,method='euler')


#initializing two parallel models
G_exc.r = '0*Hz'  # initialize firing rate, one unit rate
G_inh.r = '0*Hz'  # initialize firing rate
G_exc.I_E = 1 * Hz

G_exc_none.r = '0*Hz'  # initialize firing rate, one unit rate
G_inh_none.r = '0*Hz'  # initialize firing rate
G_exc_none.I_E = 1 * Hz


state_monitor_exc = StateMonitor(G_exc, 'r', record=True)
state_monitor_inh = StateMonitor(G_inh, 'r', record=True)

state_monitor_exc_none = StateMonitor(G_exc_none, 'r', record=True)
state_monitor_inh_none = StateMonitor(G_inh_none, 'r', record=True)

@network_operation(dt=defaultclock.dt)
# update virtual synaptic variables
def my_network_operation(t):
    if t > 15 * tau:
        G_exc.I_E = 0 * Hz
        G_exc_none.I_E = 0 * Hz

    G_exc.S_e2e = G_exc.r * weight_standard
    G_inh.S_e2i = G_exc.r * weight_standard
    G_exc.S_i2e = G_inh.r * weight_standard * (-K_L)
    G_inh.S_i2i = G_inh.r * weight_standard * (-K_L)

    G_exc_none.S_e2e = 0
    G_inh_none.S_e2i = 0
    G_exc_none.S_i2e = 0
    G_inh_none.S_i2i = 0


run(30*tau)







plt.plot(state_monitor_exc.t/ms, state_monitor_exc.r[0], label='r_E with recurrent',color='red')
plt.plot(state_monitor_exc_none.t/ms, state_monitor_exc_none.r[0], label='r_E without recurrent',color='blue')


plt.xlabel('Time (ms)')
plt.ylabel('Firing rate (Hz)')
plt.legend()


plt.tight_layout()
plt.show()
