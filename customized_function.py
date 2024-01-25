from random import sample
import numpy as np
import warnings

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
def assign_group (N_exc,N_inh):
    np.random.seed(42)   # a seed like minecraft
    numbers = np.arange(0, N_exc+N_inh)    #0-3499
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



# this function is to build up storage matrix for synapse variables
# as we know, official synapse variables are stored in 1D but could be searched through 2D [i,j] (not [i][j])
# this function is to return the standard 2D synapse variable to prevent risk
# synapsecon is the synnapse name, synapsevar is the synapse variable name you want to convert
# source and target should be the source and target group of synapsecon (must to be consistent)
# i to j which is not connected will be marked as nan
# use as W = customized_function.synapsevar_matrix_generator(synapses,synapses.w,neurons,neurons)
def synapsevar_to_matrix (synapsename,synapsevar):
    buffer=synapsevar
    W_EE = np.full((synapsename.source.N,synapsename.target.N),np.nan)    # create a matrix with the size source * target [i,j] with 0.
    # buffer.i  buffer.j are like grid in matlab, which contains coordinates pairs of all the potential connection
    for i,j in zip(synapsename.i,synapsename.j):
        W_EE[i][j]=buffer[i,j]

    return W_EE



# this function is to convert the storage matrix to synapse variable
# as the inverse function of 'synapsevar_to_matrix'
# the matrix should be in forms of M*N, a_mn means the var number of connection i=m and j=n
# the result should be the update of synapsevar, consistent with the official synapsevar storage form
# you could convert a synapsevar to matrix and convert it back, it is the same
# use as: synapse_e2e.c = customized_function.matrix_to_synapsevar(synapse_e2e,C)
def matrix_to_synapsevar (synapsename,matrix):
    buffer=matrix
    output_array=[]    #reset the synapsevar
    for i,j in zip(synapsename.i,synapsename.j):    # go through all the connection
        output_array.append(buffer[i][j])   #for every connection in the list, give the matrix number to the connection

    non_nan_sum = np.sum(~np.isnan(buffer))   # all non-nan element number , to see if there is any miss
    if len(synapsename.i)!= non_nan_sum :
        warning_message=f'''data missing during convertion,
        {len(synapsename.i)} valid connections while {non_nan_sum} valid matrix elements   
        '''
        warnings.warn(warning_message, Warning)

    return output_array




# here is a storage, in which all vvariavles with units are recorded

#  define some value
# N_E = 2500      #number of excitatory and inhibitory neuron
# N_I = 1000
#
# J=0.1*mV
# g=8       #inhibitory connection factor
# J_E=J
# J_I=-(g*J)      #connection strength, fixed
#
# V_th=20*mV      # firing threshold and reset voltage of LIF
# V_r=10*mV
# t_ref=2*ms      #refractory period
#
# D=1.5*ms        #trasmission delay
#
# Poisson_rate=15*kHz
# enhenced_Poisson_rate=Poisson_rate*1.4      #increase 4.% as the first picture state
#
# target_firing_rate = 8*Hz
#
# # time constant in axonal/dendritic equations
# beta_a=0.4 * volt/second   # this unit can make sure the a and d are unitless number
# beta_d=0.4 * volt/second
#
# # time constant for main equation and calcium concentration
# tau_m=1 * second
# tau_ca=1 * second

