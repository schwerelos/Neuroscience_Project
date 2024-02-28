
from brian2 import *
import matplotlib.pyplot as plt
import customized_function
import numpy as np

start_scope()

# define some value
N_E = 2500      #number of excitatory and inhibitory neuron
N_I = 1000

J=0.1*mV
g=8       #inhibitory connection factor
J_E=J
J_I=-(g*J)      #connection strength, fixed

V_th=20*mV      # firing threshold and reset voltage of LIF
V_r=10*mV
t_ref=2*ms      #refractory period

D=1.5*ms        #trasmission delay

Poisson_rate=15*kHz
enhenced_Poisson_rate=Poisson_rate*1.4      #increase 4.% as the first picture state

tau_m=10*ms

target_firing_rate = 8*Hz





# exc_group = NeuronGroup(10, eqs, threshold='v>0.8*mV', reset='v = 0*mV', method='exact')



# buffer = customized_function.assign_connection(10,10,5)
# synapse_e2e = Synapses(exc_group, exc_group, 'w : 1', on_pre='v_post += J*w')
# # i is always the first element of butter,j the second
# synapse_e2e.connect(i=[count[0] for count in buffer], j=[count[1] for count in buffer])
#
# synapse_e2e.w=1


# MS=StateMonitor(exc_group, 'v', record=buffer[1][1])

# MS=StateMonitor(exc_group, 'v', record=0)
# net=Network(exc_group,MS)
#
# net.run=100*ms
#
# plot(MS.t/ms, MS.v[0])
# xlabel('Time (ms)')
# ylabel('v');
# plt.show()




# tau = 10*ms
# eqs = '''
# dv/dt = (1*volt-v)/tau : volt
# '''

tau = 10*ms
eqs = '''
dv/dt = (0.04*volt-v)/tau : volt (unless refractory)
'''


# exc_group = NeuronGroup(1, eqs, threshold='v>0.8*volt', reset='v = 0*volt', method='exact')
exc_group = NeuronGroup(10, eqs, threshold='v>V_th', reset='v=V_r', refractory=0*ms, method='exact')
exc_group.v=np.arange(0,0.01,0.001)*volt

buffer = customized_function.assign_connection(10,10,5)
synapse_e2e = Synapses(exc_group, exc_group, 'w : 1', on_pre='v_post += 14*J*w')
# i is always the first element of butter,j the second
synapse_e2e.connect(i=[count[0] for count in buffer], j=[count[1] for count in buffer])
# synapse_e2e.connect(i=[0], j=[1])
synapse_e2e.w=1

print(synapse_e2e.w)

M = StateMonitor(exc_group, 'v', record=[buffer[1][1],buffer[2][1]])
# M = StateMonitor(exc_group, 'v', record=True)

net=Network(M,synapse_e2e,exc_group)
net.run(50*ms)

plot(M.t/ms, M.v[0], label='Neuron 0')
plot(M.t/ms, M.v[1], label='Neuron 1')
xlabel('Time (ms)')
ylabel('v')
legend();
plt.show()





