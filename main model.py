from brian2 import *
import matplotlib.pyplot as plt
import customized_function
import numpy as np


N_E = 1      # represents ecitatory and inhibitory group
N_I = 1

tau=20*ms

standard_rate_factor=10000     # be served as rate 1 in the first figure

weight_standard = 4.28
K_L=1.1

# two sum and difference model parameters
W_plus = weight_standard * (K_L-1)
W_FF = weight_standard * (K_L+1)

eqs = '''
dr/dt = -r / tau : Hz
'''

# iteration equation using sum and difference basis
eqs_plus = '''
dr/dt = -(1+W_plus)*r / tau : Hz
'''

eqs_minus = '''
dr/dt = -r / tau : Hz
'''


# normal model
G_exc = NeuronGroup(N_E, eqs, threshold='rand() < r * dt',method='euler')
G_inh = NeuronGroup(N_I, eqs, threshold='rand() < r * dt',method='euler')
# model with changing basis (plus: sum basis   minus: difference basis)
G_plus = NeuronGroup(N_E, eqs, threshold='rand() < r * dt',method='euler')
G_minus = NeuronGroup(N_E, eqs, threshold='rand() < r * dt',method='euler')


#initializing two parallel models
G_exc.r = 'standard_rate_factor*Hz'  # initialize firing rate, one unit rate
G_inh.r = '0*Hz'  # initialize firing rate

G_plus.r = '0.5 * standard_rate_factor*Hz'  # initialize firing rate, one unit rate
G_minus.r = '0.5 * standard_rate_factor*Hz'  # initialize firing rate, one unit rate


# connections of the first model
synapse_e2e = Synapses(G_exc, G_exc, 'w : 1', on_pre='r_post += w / tau', method='euler')
# i is always the first element of butter,j the second
synapse_e2e.connect(i=0,j=0)
synapse_e2e.w = weight_standard    # weight standard * unit rate

synapse_e2i = Synapses(G_exc, G_inh, 'w : 1', on_pre='r_post += w / tau', method='euler')
# i is always the first element of butter,j the second
synapse_e2i.connect(i=0,j=0)
synapse_e2i.w = weight_standard

synapse_i2e = Synapses(G_inh, G_exc, 'w : 1', on_pre='r_post += w / tau', method='euler')
# i is always the first element of butter,j the second
synapse_i2e.connect(i=0,j=0)
synapse_i2e.w = -(weight_standard * K_L)

synapse_i2i = Synapses(G_inh, G_inh, 'w : 1', on_pre='r_post += w / tau', method='euler')
# i is always the first element of butter,j the second
synapse_i2i.connect(i=0,j=0)
synapse_i2i.w = -(weight_standard * K_L)

# connection of the second model
synapse_sum2sum = Synapses(G_plus, G_plus, 'w : 1', on_pre='r_post += w / tau', method='euler')
# i is always the first element of butter,j the second
synapse_sum2sum.connect(i=0,j=0)
synapse_sum2sum.w = -(W_plus)

synapse_diff2sum = Synapses(G_minus, G_plus, 'w : 1', on_pre='r_post += w / tau', method='euler')
# i is always the first element of butter,j the second
synapse_diff2sum.connect(i=0,j=0)
synapse_diff2sum.w = W_FF






state_monitor_exc = StateMonitor(G_exc, 'r', record=True)
state_monitor_inh = StateMonitor(G_inh, 'r', record=True)

state_monitor_sum = StateMonitor(G_plus, 'r', record=True)
state_monitor_diff = StateMonitor(G_minus, 'r', record=True)


run(6*tau)




plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.title('origional model')
plt.plot(state_monitor_exc.t/ms, state_monitor_exc.r[0], label='r_E',color='green')
plt.plot(state_monitor_inh.t/ms, state_monitor_inh.r[0], label='r_I',color='red')
plt.plot(state_monitor_inh.t/ms, (state_monitor_exc.r[0]+state_monitor_inh.r[0]), label='r_E+r_I',color='blue')
plt.plot(state_monitor_inh.t/ms, (state_monitor_exc.r[0]-state_monitor_inh.r[0]), label='r_E-r_I',color='black')

plt.xlabel('Time (ms)')
plt.ylabel('Firing rate (Hz)')
plt.legend()

plt.subplot(122)
plt.title('sum and difference model')
plt.plot(state_monitor_sum.t/ms, (state_monitor_sum.r[0]+state_monitor_diff.r[0]), label='r_E',color='green')
plt.plot(state_monitor_sum.t/ms, (state_monitor_sum.r[0]-state_monitor_diff.r[0]), label='r_I',color='red')
plt.plot(state_monitor_sum.t/ms, 2*state_monitor_sum.r[0], label='r_E+r_I',color='blue')
plt.plot(state_monitor_diff.t/ms, 2*state_monitor_diff.r[0], label='r_E-r_I',color='black')




plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.legend()

plt.tight_layout()
plt.show()
