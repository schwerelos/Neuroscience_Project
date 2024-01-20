
from brian2 import *
import matplotlib.pyplot as plt
import customized_function
import numpy as np

start_scope()

# define some value
N_E = 10      #number of excitatory and inhibitory neuron
N_I = 100

J=0.1*mV
g=8       #inhibitory connection factor
J_E=J
J_I=-(g*J)      #connection strength, fixed

V_th=20*mV      # firing threshold and reset voltage of LIF
V_r=10*mV
t_ref=2*ms      #refractory period

D=1.5*ms        #trasmission delay

Poisson_rate=2*kHz
enhenced_Poisson_rate=Poisson_rate*1.4      #increase 4.% as the first picture state

tau_m=10*ms

target_firing_rate = 8*Hz



tau = 10*ms
eqs = '''
dv/dt = (0.04*volt-v)/tau : volt (unless refractory)
dfai/dt = -fai/tau :1
'''

# generate an exc group with poisson input
exc_group = NeuronGroup(N_E, eqs, threshold='v>V_th', reset='v=V_r; fai = fai+1', refractory=0*ms, method='exact')
poissoninput_group = PoissonGroup((N_E), rates=Poisson_rate)

# assign initial value to dismatch the spike time of each element
exc_group.v=np.arange(0,0.01,0.001)*volt


#random assign 5 connections
buffer = customized_function.assign_connection(10,10,5)
synapse_e2e = Synapses(exc_group, exc_group, 'w : 1', on_pre='v_post += 15*J*w')
# i is always the first element of butter,j the second
synapse_e2e.connect(i=[count[0] for count in buffer], j=[count[1] for count in buffer])
# synapse_e2e.connect(i=[0], j=[1])
synapse_e2e.w=1

# poisson input  connection is one to one
synapse_p2e = Synapses(poissoninput_group, exc_group, 'w : 1', on_pre='v_post += 10*J*w')
synapse_p2e.connect(i=list(range(0,N_E,1)), j=list(range(0,N_E,1)))    #i and j is the same, from 0 to N_E-1
synapse_p2e.w = 'rand()'

#check the first randomly connected source and target neuron, and check connections
first_random_source=buffer[0][0]     #should be a number such as 10
first_random_target=buffer[0][1]

print('the first randomly connected source neuron is ', first_random_source)
print('the first randomly connected target neuron is ', first_random_target)
customized_function.check_synapseconnection(synapse_e2e)

#some monitor
M = StateMonitor(exc_group, 'v', record=[first_random_target])
M_fai = StateMonitor(exc_group, 'fai', record=[0])
SP_poisson = SpikeMonitor(poissoninput_group)
SP_exc = SpikeMonitor(exc_group)

net=Network(exc_group,poissoninput_group,synapse_e2e,synapse_p2e,M_fai,M,SP_poisson,SP_exc)
net.run(10*ms)

# selected 2 elements
# first is the time when the poisson input of the first target spikes
# second is the first source neuron, which connects the first target spikes
selected_spikes = SP_poisson.t[SP_poisson.i == first_random_target]    #selected the poisson of the first target out
selected_spikes_sc = SP_exc.t[SP_exc.i == first_random_source]    #selected the time when the first source spike


# plot the v of the first target
# can see the effect of poisson input and pre spike
plot(M.t/ms, M.v[0], label='Neuron 0')
# plot(M.t/ms, M.v[1], label='Neuron 1')
xlabel('Time (ms)')
ylabel('v')
legend();

for count in selected_spikes:
    plt.axvline(count/ms, color='red', linestyle='--', linewidth=0.8)   #highlight the poisson input time
for count in selected_spikes_sc:
    plt.axvline(count/ms, color='blue', linestyle='--', linewidth=0.8)   #highlight the source neuron input time



# plot(M_fai.t/ms, M_fai.fai[0], label='Neuron 0')

#plot the figure of the poisson input
# can see the independence of the poisson input while the input of the first target is highlighted
plt.figure()     #plot the poisson input image, highlight the input of first_random_target
plot(SP_poisson.t/ms, SP_poisson.i,'.k')
for count in selected_spikes:
    plt.axvline(count/ms, color='red', linestyle='--', linewidth=0.8)
plt.axhline(first_random_target, color='green', linestyle='-', linewidth=1.2)
plt.show()


