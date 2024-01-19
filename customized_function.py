from random import sample
import numpy as np

def test_function():
    print("my little pony")

# this function can generate n connections from neuron group A and B randomly
# N_a and N_b are the number of neurons in each group, expected to be a integral, such as 30,50
# N_connection is the number of random connection you want to generate
# return a generated connection pair such as [(3,5),(6,9)]
def assign_connection (N_a, N_b, N_connection):
    N_connection=int(N_connection)    #prevent a bug that random can't recognize number of division as float
    random_connections = [(i, j) for i in range(1, N_a) for j in range(1, N_b)]
    random_connections = sample(random_connections, N_connection)
    return random_connections

# As figure 1, this function can generate the arraies of 3 groups randomly
# 3 random groups divided from 1-3500, the length of each is 1500
# cooperate with the main_program, in which 1-2499 is the marker of exc group, 2500-3499 is he marker of ihn group
def assign_group ():
    np.random.seed(42)   # a seed like minecraft
    numbers = np.arange(0, 3500)    #0-3499
    np.random.shuffle(numbers)      #randomly shuffle the array

    #divide the shuffled arrayy into 3 parts
    group_size = len(numbers) // 3

    group1 = numbers[:group_size]
    group2 = numbers[group_size:2 * group_size]
    group3 = numbers[2 * group_size:]

    return group1, group2, group3


# this function is used to check all the connections from synapse perspective
# input the connection variable you want to check
# use as: customized_function.check_synapseconnection(synapse_i2i)
# the supposed connection is described in the main model
def check_synapseconnection(input_connection):
    buffer=input_connection
    buffer_a=[]
    buffer_b=[]
    # trace total connection number
    print("Number of connections:", len(buffer))
    # trace connection details
    max_source = max(buffer.i)
    max_target = max(buffer.j)
    for count in range(0,3,1):
        buffer_a.append(buffer.i[count])
        buffer_b.append(buffer.j[count])
    print("Source neurons examples:", buffer_a,'the max is',max_source)
    print("Target neurons examples:", buffer_b,'the max is',max_target)
    # trace source and target groups, notice that 'Neurogroup_1' is a system bug
    print("Source group is:", buffer.source.name)
    print("target group is:", buffer.target.name)


