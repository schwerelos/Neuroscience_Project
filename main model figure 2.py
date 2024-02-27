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

# define the equation with the external input I

eqs = '''
dr/dt = (I-r) / tau : Hz  
I: Hz 
'''

# define neuron group for two parallel models
# namely hebbian amplification and balanced model
# none means without connection  re means with connection
G_exc_hebb_none = NeuronGroup(N_E, eqs, threshold='rand() < r * dt',method='euler')
G_exc_hebb_re = NeuronGroup(N_E, eqs, threshold='rand() < r * dt',method='euler')

G_exc_bal = NeuronGroup(N_E, eqs, threshold='rand() < r * dt',method='euler')
G_inh_bal = NeuronGroup(N_I, eqs, threshold='rand() < r * dt',method='euler')

#initializing two parallel models
G_exc_hebb_none.r = '0*Hz'  # initialize firing rate, one unit rate
G_exc_hebb_none.I = '1000000*Hz' # to make a significant strong impulse (otherwise the unction won't be so smooth :P )

G_exc_hebb_re.r = '0*Hz'  # initialize firing rate, one unit rate
G_exc_hebb_re.I = '1000000*Hz' # to make a significant strong impulse (otherwise the unction won't be so smooth :P )

G_exc_bal.r = '0*Hz'  # initialize firing rate, one unit rate
G_inh_bal.r = '0*Hz'  # initialize firing rate, one unit rate


# connections of the first model
synapse_eh2eh_none = Synapses(G_exc_hebb_none, G_exc_hebb_none, 'w : 1', on_pre='r_post += w / tau', method='euler')
# i is always the first element of butter,j the second
synapse_eh2eh_none.connect(i=0,j=0)
synapse_eh2eh_none.w = 0

synapse_eh2eh_re = Synapses(G_exc_hebb_re, G_exc_hebb_re, 'w : 1', on_pre='r_post += w / tau', method='euler')
# i is always the first element of butter,j the second
synapse_eh2eh_re.connect(i=0,j=0)
synapse_eh2eh_re.w = 0.75



state_monitor_exc_hebb_none = StateMonitor(G_exc_hebb_none, 'r', record=True)
state_monitor_exc_hebb_re = StateMonitor(G_exc_hebb_re, 'r', record=True)

@network_operation(when='before_synapses')
def my_network_operation(t):
    if t > defaultclock.dt:
        G_exc_hebb_none.I= 0 * Hz  # 去除外界输入
        G_exc_hebb_re.I = 0 * Hz  # 去除外界输入

run(10*tau)

plt.plot(state_monitor_exc_hebb_none.t/ms, state_monitor_exc_hebb_none.r[0], label='amp=1',color='blue')
plt.plot(state_monitor_exc_hebb_re.t/ms, state_monitor_exc_hebb_re.r[0], label='amp=4',color='red')
plt.legend()
plt.show()

