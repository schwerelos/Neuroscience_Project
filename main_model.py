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

from brian2 import *
import matplotlib.pyplot as plt
import customized_function

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

target_firing_rate = 8*Hz



# build up neuron groups
# one excitatory group one inhibitory grou[
# assign each nuron a marker (1-3500 in which 1-2500 is excitatory) and divide into three parts based on marker
# input and readout group will be seperately established

# use simple function as an example
# don't forget unless refractory order
group_eqs = '''
dv/dt = (I-v)/tau : 1 (unless refractory)
I : 1
tau : second
'''
exc_group = NeuronGroup(N_E, group_eqs, threshold='v>V_th', reset='v=V_r', refractory=t_ref, method='exact')
inh_group = NeuronGroup(N_I, group_eqs, threshold='v>V_th', reset='v=V_r', refractory=t_ref, method='exact')
poissoninput_group = PoissonGroup((N_E+N_I), rates=Poisson_rate)
readout_group = NeuronGroup(1, group_eqs, threshold='v>V_th', reset='v=V_r', refractory=t_ref, method='exact')   #the first picture



#assign US C1 C2 group randomly
# special for picture 1
# the following three are 3 arries, each 1500 long, containing marker of neurons to determine their group
# in each maker, any element between 1-2499 is exc neuron, 2501-3499 is inh
# around 1166-1168 for each len(marker) expected
US_marker,C1_marker,C2_marker = customized_function.assign_group()



#build up synapse    #remember to fix the weight term
# i to i, i to e, e to i will be determined as mentioned, randomly select and connect
# e to e will be set as no connection temporarily as mentioned
# input group, one to one connection, 1-2499 connect exc group 2500-3499 connect inh group
# readout connect all US group (first picture)
buffer = customized_function.assign_connection(N_I,N_I,0.1*N_I)
synapse_i2i = Synapses(inh_group, inh_group, 'w : 1', on_pre='v_post += w')
# i is always the first element of butter,j the second
synapse_i2i.connect(i=[count[0] for count in buffer], j=[count[1] for count in buffer])
synapse_i2i.w = 'rand()'

buffer = customized_function.assign_connection(N_I,N_E,0.1*N_I)
synapse_i2e = Synapses(inh_group, exc_group, 'w : 1', on_pre='v_post += w')
# i is always the first element of butter,j the second
synapse_i2e.connect(i=[count[0] for count in buffer], j=[count[1] for count in buffer])
synapse_i2e.w = 'rand()'

buffer = customized_function.assign_connection(N_E,N_I,0.1*N_E)
synapse_e2i = Synapses(exc_group, inh_group, 'w : 1', on_pre='v_post += w')
# i is always the first element of butter,j the second
synapse_e2i.connect(i=[count[0] for count in buffer], j=[count[1] for count in buffer])
synapse_e2i.w = 'rand()'

# bind poisson group 1-2500 to exc 2501-3500 to ihn, connection is one to one
synapse_p2e = Synapses(poissoninput_group, exc_group, 'w : 1', on_pre='v_post += w')
synapse_p2e.connect(i=list(range(0,N_E,1)), j=list(range(0,N_E,1)))    #i and j is the same, from 0 to N_E-1
synapse_p2e.w = 'rand()'

synapse_p2i = Synapses(poissoninput_group, inh_group, 'w : 1', on_pre='v_post += w')
synapse_p2i.connect(i=list(range(N_E,N_E+N_I,1)), j=list(range(0,N_I,1)))   #2500-3499 of poisson connect to 0-1499 of inh
synapse_p2i.w = 'rand()'

#bind all exc to readout group
synapse_e2r = Synapses(exc_group, readout_group, 'w : 1', on_pre='v_post += w')
synapse_e2r.connect(i=list(range(0,N_E,1)), j=[0])   # all exc (0-2499) connect to readout[0]
synapse_e2r.w = 'rand()'

# customized_function.check_synapseconnection(synapse_i2e)

network = Network(exc_group , inh_group , poissoninput_group , readout_group,
synapse_i2i , synapse_i2e , synapse_e2i , synapse_p2i , synapse_p2e , synapse_e2r
)


