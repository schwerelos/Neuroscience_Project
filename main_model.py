''''''
           #                    _ooOoo_
           #                   o8888888o
           #                   88" . "88
           #                   (| -_- |)
           #                   O\  =  /O
           #                ____/`---'\____
           #              .'  \\|     |//  `.
           #             /  \\|||  :  |||//  \
           #            /  _||||| -:- |||||-  \
           #            |   | \\\  -  /// |   |
           #            | \_|  ''\---/''  |   |
           #            \  .-\__  `-`  ___/-. /
           #          ___`. .'  /--.--\  `. . __
           #       ."" '<  `.___\_<|>_/___.'  >'"".
           #      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
           #      \  \ `-.   \_ __\ /__ _/   .-` /  /
           # ======`-.____`-.___\_____/___.-`____.-'======
           #                    `=---='
           # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           #            佛祖保佑        永无BUG
           #   佛曰:
           #          写字楼里写字间，写字间里程序员；
           #          程序人员写程序，又拿程序换酒钱。
           #          酒醒只在网上坐，酒醉还来网下眠；
           #          酒醉酒醒日复日，网上网下年复年。
           #          但愿老死电脑间，不愿鞠躬老板前；
           #          奔驰宝马贵者趣，公交自行程序员。
           #          别人笑我忒疯癫，我笑自己命太贱；
           #          不见满街漂亮妹，哪个归得程序员？
''''''


# IMPORTANT ！！！！
# don't forget the neuron number
# the current variable are all unitless
# pay attentiom to the current poisson amplitude, don't foorget to change it back to standard

from brian2 import *
import matplotlib.pyplot as plt
import customized_function
import numpy as np

start_scope()

# define some value
N_E = 1000      #number of excitatory and inhibitory neuron
N_I = 250


V_th=20e-3      # firing threshold and reset voltage of LIF
V_r=10e-3
t_ref=2e-3*second     #refractory period

D=1*ms        #trasmission delay

Poisson_rate=15*kHz
enhenced_Poisson_rate=Poisson_rate*1.4      #increase 4.% as the first picture state

target_firing_rate = 8

# time constant in axonal/dendritic equations
beta_a=0.4 * second
beta_d=0.4 * second

# time constant for main equation and calcium concentration
tau_m=20 * ms
tau_ca=1 * second
tau_standard= 1 * second   # standard time constant to match the unit

J=0.1e-3
g=8       #inhibitory connection factor
J_E=J
J_I=-(g*J)      #connection strength, fixed
J_poisson = 40 * J  # the effect of poisson amplitude is taum J as specified in the article

# some other constant
weight_standard=1   #standard weight for each synapse, only C change
defaultclock.dt = 0.05 * ms
sim_duration = 100 * ms
total_step=int(sim_duration/defaultclock.dt)
step_count=[0]




# build up neuron groups
# one excitatory group one inhibitory grou[
# assign each nuron a marker (1-3500 in which 1-2500 is excitatory) and divide into three parts based on marker
# input and readout group will be seperately established

# use simple function as an example
group_eqs = '''
dv/dt = (-v)/tau_m : 1 (unless refractory)
'''
group_eqs_exc='''
dv/dt = (-v)/tau_m : 1 (unless refractory)
dfai/dt=(-fai)/tau_ca :1   
drou_a_abs/dt= (target_firing_rate - fai)/beta_a :1
drou_d_abs/dt= (target_firing_rate - fai)/beta_d :1
'''
synapse_eqs_exc='''
dc/dt= (irri_term_first - (c * irri_term_second))/tau_standard :1  (clock-driven)
w:1
irri_term_first:1
irri_term_second:1
'''

exc_group = NeuronGroup(N_E, model=group_eqs_exc, threshold='v>V_th', reset='v=V_r', refractory=t_ref, method='euler')
inh_group = NeuronGroup(N_I, model=group_eqs, threshold='v>V_th', reset='v=V_r', refractory=t_ref, method='euler')
poissoninput_group = PoissonGroup((N_E+N_I), rates=Poisson_rate)
readout_group = NeuronGroup(1, model=group_eqs, threshold='v>V_th', reset='v=V_r', refractory=t_ref, method='euler')   #the first picture



#assign US C1 C2 group randomly
# special for picture 1
# the following three are 3 arries, each 1500 long, containing marker of neurons to determine their group
# in each maker, any element between 1-2499 is exc neuron, 2501-3499 is inh
# around 1166-1168 for each len(marker) expected
US_marker,C1_marker,C2_marker = customized_function.assign_group(N_E,N_I)



#build up synapse    #remember to fix the weight term
# i to i, i to e, e to i will be determined as mentioned, randomly select and connect
# e to e will be set as no connection temporarily as mentioned
# input group, one to one connection, 1-2499 connect exc group 2500-3499 connect inh group
# readout connect all US group (first picture)
synapse_i2i = Synapses(inh_group, inh_group, 'w : 1', on_pre='v_post += J_I', method='euler', delay=D)
# i is always the first element of butter,j the second
synapse_i2i.connect(i='k for k in sample(N_I, size=int(0.1*N_I))')
synapse_i2i.w = weight_standard

synapse_i2e = Synapses(inh_group, exc_group, 'w : 1', on_pre='v_post += J_I', method='euler',delay=D)
# i is always the first element of butter,j the second
synapse_i2e.connect(i='k for k in sample(N_I, size=int(0.1*N_I))')
synapse_i2e.w = weight_standard


synapse_e2i = Synapses(exc_group, inh_group, 'w : 1', on_pre='v_post += J_E', method='euler',delay=D)
# i is always the first element of butter,j the second
synapse_e2i.connect(i='k for k in sample(N_E, size=int(0.1*N_E))')
synapse_e2i.w = weight_standard

# bind poisson group 1-2500 to exc 2501-3500 to ihn, connection is one to one
synapse_p2e = Synapses(poissoninput_group, exc_group, 'w : 1', on_pre='v_post += J_poisson', method='euler')
synapse_p2e.connect(i=list(range(0,N_E,1)), j=list(range(0,N_E,1)))    #i and j is the same, from 0 to N_E-1
synapse_p2e.w = weight_standard

synapse_p2i = Synapses(poissoninput_group, inh_group, 'w : 1', on_pre='v_post += J_poisson', method='euler')
synapse_p2i.connect(i=list(range(N_E,N_E+N_I,1)), j=list(range(0,N_I,1)))   #2500-3499 of poisson connect to 0-1499 of inh
synapse_p2i.w = weight_standard

# bind all exc to readout group
synapse_e2r = Synapses(exc_group, readout_group, 'w : 1', on_pre='v_post += J_E', method='euler',delay=D)
synapse_e2r.connect(i=list(range(0,N_E,1)), j=[0])   # all exc (0-2499) connect to readout[0]
synapse_e2r.w = weight_standard

# bind all exc to exc but with zero initial weight
synapse_e2e = Synapses(exc_group, exc_group, model=synapse_eqs_exc, on_pre='v_post += J_E*c', method='euler',delay=D)
synapse_e2e.connect(condition='i!=j')   # all econnect to all exc
synapse_e2e.w = weight_standard

# connect all exc neurons to themselves to update fai
synapse_exc_selfcon = Synapses(exc_group, exc_group, model=synapse_eqs_exc, on_pre='fai_post += 1', method='euler')
synapse_exc_selfcon.connect(condition='i==j')   # all econnect to all exc


#customized_function.check_synapseconnection(synapse_e2e)

# network = Network(exc_group , inh_group , poissoninput_group , readout_group,
# synapse_i2i , synapse_i2e , synapse_e2i , synapse_p2i , synapse_p2e , synapse_e2r
# )

# zero matrix to initialize the parameters, just for the first irritation
C_matrix= np.ones((N_E, N_E))
K_in=np.sum(C_matrix, axis=1)   # array[i] means K_in_i  +1 helps to avoid the /0 error during the first run
K_out=np.sum(C_matrix,axis=0)   # array[i] means K_out_i

synapse_e2e.c=1   # give one connections for each in the initial step to prevent the /0 error

synapse_e2e.irri_term_first=0
synapse_e2e.irri_term_second=0




# some monitors
Vol_mon=StateMonitor(exc_group,'v',record=[1,2,3])
abs_mon=StateMonitor(exc_group,'rou_a_abs',record=[1,2,3])
fai_mon=StateMonitor(exc_group,'fai',record=[1,2,3])
c_mon=StateMonitor(synapse_e2e,'c',record=[1,2,3])
poi_mon=SpikeMonitor(poissoninput_group)



@network_operation(when='before_synapses')
def my_network_operation():
    # the main purpose of these steps are getting the irritation terms for synaptic variable dc/dt
    # and record the intermediate variables by the way xD

    # convert synaptic variable c into matrix form
    # don't need to be converted back because c will irritate in ODE independently
    # note that the physical meaning of C[i][j] is i from j, but not i to j
    C_matrix = customized_function.synapsevar_to_matrix(synapse_e2e, synapse_e2e.c)

    # calculate Kin and Kout
    K_in = np.sum(C_matrix, axis=1)  # array[i] means K_in_i
    K_out = np.sum(C_matrix, axis=0)  # array[i] means K_out_i


    # from abs to real free element number, abs should be prepared during the step in group variablees
    rou_a_plus = np.clip(exc_group.rou_a_abs, 0, np.inf)   # free axon surplus, larger than or equal to zero
    rou_a_minus = np.clip(exc_group.rou_a_abs, -np.inf, 0)   # free axon deficit, smaller than or equal to zero
    rou_d_plus = np.clip(exc_group.rou_d_abs, 0, np.inf)   # free dendritic surplus, larger than or equal to zero
    rou_d_minus = np.clip(exc_group.rou_d_abs, -np.inf, 0)  # free dendritic deficit, smaller than or equal to zero

    # prepare for correction
    rows,cols=C_matrix.shape
    buffer_a = np.zeros(rows)
    buffer_d = np.zeros(rows)

    for count in range(rows):
        buffer_a += (rou_d_minus[count]/K_in[count]) * (C_matrix[count][:]) # correction term  ad sequence is verified
        buffer_d += (rou_a_minus[count]/K_out[count]) * (C_matrix[:][count])

    # for the first round, Kin Kout is 0, induce nan error
    # repalce all na with 0 to prevent the initialization bug
    buffer_a[isnan(buffer_a)]=0
    buffer_d[isnan(buffer_d)]=0

    # generate correction term  these are array and final rou
    rou_a_plus_corrected = rou_a_plus + buffer_a
    rou_d_plus_corrected = rou_d_plus + buffer_d
    rou = max(sum(rou_a_plus_corrected),sum(rou_d_plus_corrected))   # should be a number

    # prepare irritation matrix for C
    irri_matrix_first = np.zeros(C_matrix.shape)   #irritation term in synaptic equation, to be converted
    irri_matrix_second = np.zeros(C_matrix.shape)   #second parameter in the synaptic ODE

    # fill the irritation matrix for C, only the elements on the synapse_e2e list are involved
    for i, j in zip(synapse_e2e.i, synapse_e2e.j):
        irri_matrix_first[i][j] = rou_d_plus[i]*rou_a_plus[j]/rou   #fill in the data based on the ODE
        irri_matrix_second[i][j] = (rou_d_minus[i] / K_in[i])+(rou_a_minus[j] / K_out[j])

    # for the first round, Kin Kout is 0, induce nan error
    # repalce all na with 0 to prevent the initialization bug
    irri_matrix_first[isnan(irri_matrix_first)]=0
    irri_matrix_second[isnan(irri_matrix_second)]=0

    # convert that into standard brain 2 form for dc/dt irritation
    # irripre and irripost should be in shape [N,1], in which N is the connection number with sequenced element
    # transpose inside the customized function make sure that [i][j] represents i from j after being transposed
    irri_term_first = customized_function.matrix_to_synapsevar(synapse_e2e, irri_matrix_first)
    irri_term_second = customized_function.matrix_to_synapsevar(synapse_e2e, irri_matrix_second)

    #print("progress bar" )
    step_count[0]+=1
    print(step_count, ' of ', total_step,' has been finished')



run(sim_duration)
plot(poi_mon.t / ms, poi_mon.i, '.k')
#plot(Vol_mon.t / ms, Vol_mon.v[0], 'k')
plt.show()
pass

#network.run=(sim_duration)


