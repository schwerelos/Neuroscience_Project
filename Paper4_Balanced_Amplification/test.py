from brian2 import *

# 定义模拟参数
num_neurons = 1  # 只有一个神经元
tau = 10*ms      # 时间常数

# 定义模型方程
model_eqs = '''
dr/dt = -r / tau : Hz
'''

# 创建神经元群组
G = NeuronGroup(num_neurons, model_eqs, threshold='rand() < r * dt', method='exact')
G.r = '100*Hz'  # 初始firing rate

# 记录神经元发放
spike_monitor = SpikeMonitor(G)

# 记录firing rate变化
state_monitor = StateMonitor(G, 'r', record=True)

# 运行模拟
run(2*second)

# 绘制结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(state_monitor.t/ms, state_monitor.r[0], label='Firing rate')
plt.xlabel('Time (ms)')
plt.ylabel('Firing rate (Hz)')
plt.legend()

plt.subplot(122)
plt.plot(spike_monitor.t/ms, spike_monitor.i, 'k.', label='Spikes')
plt.xlabel('Time (ms)')
plt.xlim(0, 1000)
plt.ylabel('Neuron index')
plt.legend()

plt.tight_layout()
plt.show()
