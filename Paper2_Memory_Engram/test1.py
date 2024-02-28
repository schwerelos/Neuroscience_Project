
from brian2 import *
from random import sample
import numpy as np
import customized_function

from brian2 import *

start_scope()

matrix_shape = (10, 1)

# 创建一个具有相同单位的零矩阵，单位为 volt
# a = np.zeros((10,1)) * 1 * volt

# M=(np.zeros(10,1))*1*volt

# 定义神经元模型和微分方程
tau = defaultclock.dt
eqs = '''
dv/dt = a / tau : volt (unless refractory)
a: volt
'''
eqs_synapse='''
w:1
c:1
'''
print(defaultclock.dt)
# 创建神经元群体
neurons = NeuronGroup(100, model=eqs, method='euler', threshold='v > 1000*volt', reset='v=0*volt')

# 设置初始化条件
neurons.v = 0 * volt
neurons.a = 0 * volt

synapse_e2e = Synapses(neurons, neurons, model=eqs_synapse, on_pre='v_post += w*c*volt')
synapse_e2e.connect(condition='i!=j')   # all econnect to all exc
synapse_e2e.w = 'rand()'
synapse_e2e.c = 'rand()'

# print(neurons.N)

# print(synapse_e2e.c.shape)
# print(synapse_e2e.w.shape)
# print(synapse_e2e.w[2, 6])
# print(synapse_e2e.c[2, 5])
# 在每个时钟步之前执行的代码块
print(synapse_e2e.c)
C = customized_function.synapsevar_to_matrix(synapse_e2e,synapse_e2e.c)
print(C)
synapse_e2e.c = customized_function.matrix_to_synapsevar(synapse_e2e,C)
print(synapse_e2e.c)
# print(synapse_e2e.source.N)
# print(synapse_e2e.target.N)
@network_operation(when='before_synapses')
def update_a_before_synapses():
    for i in range(len(neurons.a)):
         neurons.a[i] += rand()*volt
         neurons.a[i] += i*volt
         synapse_e2e.c[2, 6]+=1
        # print(synapse_e2e.c[2, 5])
        # neurons.a[i] = 2 * neurons.a[i]
    #neurons.a += 'i*volt'

M=StateMonitor(neurons,['v','a'],record=True)

# 运行模拟
sim_duration = 1 * ms
run(sim_duration)


plot(M.t/ms,M.v[3],'r')
plot(M.t/ms,M.a[3],'b')


# plt.show()