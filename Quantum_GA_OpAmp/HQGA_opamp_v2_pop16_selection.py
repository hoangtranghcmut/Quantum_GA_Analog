import math
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import datetime as dt
import os

#######################################################################
#                         ALGORITHM PARAMETERS                        #          
#######################################################################
popSize = 16                        # Define here the population size
genomeLength = 16                   # Define here the chromosome length
generation_max = int(sys.argv[1])   # Define here the maximum number of
                                    # generations/iterations
# define range for input
bounds = [[0.85, 4.0], [0.23, 0.4], [0.7, 1.0], [0.06, 0.4], [2, 2.8], [18.0, 23.0], [0.1, 1.0], [16.0, 22.0], [0.25, 0.5], [0.3, 1]] # w01, l01, w23, l23, w47, w5, l457, w6, l6, Cc

# Initialization global variables
theta          = 0
iteration      = 0
the_best_chrom = 0
generation     = 0

#######################################################################
#                         VARIABLES ALGORITHM                         #         
#######################################################################
top_bottom = 2
QuBitZero = np.array([[1], [0]])
QuBitOne = np.array([[0], [1]])
AlphaBeta = np.empty([top_bottom])
fitness = np.empty([popSize]) # different from np.array, BE CAREFUL 
probability = np.empty([popSize])
# qpv: quantum chromosome (or population vector, QPV), nqpv: new qpv
qpv = np.empty([popSize, genomeLength*len(bounds), top_bottom])         
nqpv = np.empty([popSize, genomeLength*len(bounds), top_bottom]) 
# chromosome: classical chromosome
chromosome = np.empty([popSize, genomeLength*len(bounds)],dtype=int) 
child1 = np.empty([popSize, genomeLength*len(bounds), top_bottom]) 
child2 = np.empty([popSize, genomeLength*len(bounds), top_bottom]) 
global_best_fitness = 0
global_best_chrom = np.empty([genomeLength*len(bounds)])


#######################################################################
#                  QUANTUM POPULATION INITIALIZATION                  #   
#######################################################################
def Init_population():
    # Hadamard gate
    r2 = math.sqrt(2.0)           
    h = np.array([[1/r2, 1/r2],[1/r2,-1/r2]])
    # Rotation Q-gate
    theta = 0
    rot = np.empty([2,2])
    # Initial population array (individual x chromosome)
    i = 0; j = 0
    for i in range(0,popSize):
        for j in range(0,genomeLength*len(bounds)):
            theta = np.random.uniform(0,1)*90   
            theta = math.radians(theta)
            rot[0,0] = math.cos(theta); rot[0,1]=-math.sin(theta)
            rot[1,0] = math.sin(theta); rot[1,1]=math.cos(theta)
            AlphaBeta[0] = rot[0,0]*(h[0][0]*QuBitZero[0]+h[0][1]*QuBitZero[1]) + rot[0,1]*(h[1][0]*QuBitZero[0]+h[1][1]*QuBitZero[1])
            AlphaBeta[1] = rot[1,0]*(h[0][0]*QuBitZero[0]+h[0][1]*QuBitZero[1]) + rot[1,1]*(h[1][0]*QuBitZero[0]+h[1][1]*QuBitZero[1])
            # alpha squared
            qpv[i,j,0] = np.around(1*pow(AlphaBeta[0],2),2) 
            # beta squared
            qpv[i,j,1] = np.around(1*pow(AlphaBeta[1],2),2) 


######################################################################
#                           MAKE A QUANTUM MEASUREMENT               #                  
######################################################################
# p_alpha: probability of finding qubit in alpha state    
def Measure(p_alpha):
    for i in range(0,popSize):     
        for j in range(0,genomeLength*len(bounds)):
            if p_alpha <= qpv[i, j, 0]:
                chromosome[i,j] = 0
            else:
                chromosome[i,j] = 1
            

######################################################################
#                         DECODE POPULATION                          #                  
######################################################################
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end] # end is exclusive
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        # store
        value_rounded = np.round(value, 2)
        decoded.append(value_rounded)
    return decoded


#########################################################
#                  FITNESS EVALUATION                   #                  
#########################################################
def Fitness_evaluation(params_list):
    u = 1e-6
    p = 1e-12

    fitness1 = np.zeros((popSize, 1), dtype='f')
    params_matrix = np.matrix(params_list)

    w01_1  = params_matrix[0, 0] * u
    w01_2  = params_matrix[1, 0] * u
    w01_3  = params_matrix[2, 0] * u
    w01_4  = params_matrix[3, 0] * u
    w01_5  = params_matrix[4, 0] * u
    w01_6  = params_matrix[5, 0] * u
    w01_7  = params_matrix[6, 0] * u
    w01_8  = params_matrix[7, 0] * u
    w01_9  = params_matrix[8, 0] * u
    w01_10 = params_matrix[9, 0] * u
    w01_11 = params_matrix[10, 0] * u
    w01_12 = params_matrix[11, 0] * u
    w01_13 = params_matrix[12, 0] * u
    w01_14 = params_matrix[13, 0] * u
    w01_15 = params_matrix[14, 0] * u
    w01_16 = params_matrix[15, 0] * u

    l01_1 = params_matrix[0, 1] * u
    l01_2 = params_matrix[1, 1] * u
    l01_3 = params_matrix[2, 1] * u
    l01_4 = params_matrix[3, 1] * u
    l01_5 = params_matrix[4, 1] * u
    l01_6 = params_matrix[5, 1] * u
    l01_7 = params_matrix[6, 1] * u
    l01_8 = params_matrix[7, 1] * u
    l01_9 = params_matrix[8, 1] * u
    l01_10 = params_matrix[9, 1] * u
    l01_11 = params_matrix[10, 1] * u
    l01_12 = params_matrix[11, 1] * u
    l01_13 = params_matrix[12, 1] * u
    l01_14 = params_matrix[13, 1] * u
    l01_15 = params_matrix[14, 1] * u
    l01_16 = params_matrix[15, 1] * u

    w23_1 = params_matrix[0, 2] * u
    w23_2 = params_matrix[1, 2] * u
    w23_3 = params_matrix[2, 2] * u
    w23_4 = params_matrix[3, 2] * u
    w23_5 = params_matrix[4, 2] * u
    w23_6 = params_matrix[5, 2] * u
    w23_7 = params_matrix[6, 2] * u
    w23_8 = params_matrix[7, 2] * u
    w23_9 = params_matrix[8, 2] * u
    w23_10 = params_matrix[9, 2] * u
    w23_11 = params_matrix[10, 2] * u
    w23_12 = params_matrix[11, 2] * u
    w23_13 = params_matrix[12, 2] * u
    w23_14 = params_matrix[13, 2] * u
    w23_15 = params_matrix[14, 2] * u
    w23_16 = params_matrix[15, 2] * u

    l23_1 = params_matrix[0, 3] * u
    l23_2 = params_matrix[1, 3] * u
    l23_3 = params_matrix[2, 3] * u
    l23_4 = params_matrix[3, 3] * u
    l23_5 = params_matrix[4, 3] * u
    l23_6 = params_matrix[5, 3] * u
    l23_7 = params_matrix[6, 3] * u
    l23_8 = params_matrix[7, 3] * u
    l23_9 = params_matrix[8, 3] * u
    l23_10 = params_matrix[9, 3] * u
    l23_11 = params_matrix[10, 3] * u
    l23_12 = params_matrix[11, 3] * u
    l23_13 = params_matrix[12, 3] * u
    l23_14 = params_matrix[13, 3] * u
    l23_15 = params_matrix[14, 3] * u
    l23_16 = params_matrix[15, 3] * u

    w47_1 = params_matrix[0, 4] * u
    w47_2 = params_matrix[1, 4] * u
    w47_3 = params_matrix[2, 4] * u
    w47_4 = params_matrix[3, 4] * u
    w47_5 = params_matrix[4, 4] * u
    w47_6 = params_matrix[5, 4] * u
    w47_7 = params_matrix[6, 4] * u
    w47_8 = params_matrix[7, 4] * u
    w47_9 = params_matrix[8, 4] * u
    w47_10 = params_matrix[9, 4] * u
    w47_11 = params_matrix[10, 4] * u
    w47_12 = params_matrix[11, 4] * u
    w47_13 = params_matrix[12, 4] * u
    w47_14 = params_matrix[13, 4] * u
    w47_15 = params_matrix[14, 4] * u
    w47_16 = params_matrix[15, 4] * u
    
    w5_1 = params_matrix[0, 5] * u
    w5_2 = params_matrix[1, 5] * u
    w5_3 = params_matrix[2, 5] * u
    w5_4 = params_matrix[3, 5] * u
    w5_5 = params_matrix[4, 5] * u
    w5_6 = params_matrix[5, 5] * u
    w5_7 = params_matrix[6, 5] * u
    w5_8 = params_matrix[7, 5] * u
    w5_9 = params_matrix[8, 5] * u
    w5_10 = params_matrix[9, 5] * u
    w5_11 = params_matrix[10, 5] * u
    w5_12 = params_matrix[11, 5] * u
    w5_13 = params_matrix[12, 5] * u
    w5_14 = params_matrix[13, 5] * u
    w5_15 = params_matrix[14, 5] * u
    w5_16 = params_matrix[15, 5] * u

    l457_1 = params_matrix[0, 6] * u
    l457_2 = params_matrix[1, 6] * u
    l457_3 = params_matrix[2, 6] * u
    l457_4 = params_matrix[3, 6] * u
    l457_5 = params_matrix[4, 6] * u
    l457_6 = params_matrix[5, 6] * u
    l457_7 = params_matrix[6, 6] * u
    l457_8 = params_matrix[7, 6] * u
    l457_9 = params_matrix[8, 6] * u
    l457_10 = params_matrix[9, 6] * u
    l457_11 = params_matrix[10, 6] * u
    l457_12 = params_matrix[11, 6] * u
    l457_13 = params_matrix[12, 6] * u
    l457_14 = params_matrix[13, 6] * u
    l457_15 = params_matrix[14, 6] * u
    l457_16 = params_matrix[15, 6] * u

    w6_1 = params_matrix[0, 7] * u
    w6_2 = params_matrix[1, 7] * u
    w6_3 = params_matrix[2, 7] * u
    w6_4 = params_matrix[3, 7] * u
    w6_5 = params_matrix[4, 7] * u
    w6_6 = params_matrix[5, 7] * u
    w6_7 = params_matrix[6, 7] * u
    w6_8 = params_matrix[7, 7] * u
    w6_9 = params_matrix[8, 7] * u
    w6_10 = params_matrix[9, 7] * u
    w6_11 = params_matrix[10, 7] * u
    w6_12 = params_matrix[11, 7] * u
    w6_13 = params_matrix[12, 7] * u
    w6_14 = params_matrix[13, 7] * u
    w6_15 = params_matrix[14, 7] * u
    w6_16 = params_matrix[15, 7] * u

    l6_1 = params_matrix[0, 8] * u
    l6_2 = params_matrix[1, 8] * u
    l6_3 = params_matrix[2, 8] * u
    l6_4 = params_matrix[3, 8] * u
    l6_5 = params_matrix[4, 8] * u
    l6_6 = params_matrix[5, 8] * u
    l6_7 = params_matrix[6, 8] * u
    l6_8 = params_matrix[7, 8] * u
    l6_9 = params_matrix[8, 8] * u
    l6_10 = params_matrix[9, 8] * u
    l6_11 = params_matrix[10, 8] * u
    l6_12 = params_matrix[11, 8] * u
    l6_13 = params_matrix[12, 8] * u
    l6_14 = params_matrix[13, 8] * u
    l6_15 = params_matrix[14, 8] * u
    l6_16 = params_matrix[15, 8] * u

    Cc_1 = params_matrix[0, 9] * p
    Cc_2 = params_matrix[1, 9] * p
    Cc_3 = params_matrix[2, 9] * p
    Cc_4 = params_matrix[3, 9] * p
    Cc_5 = params_matrix[4, 9] * p
    Cc_6 = params_matrix[5, 9] * p
    Cc_7 = params_matrix[6, 9] * p
    Cc_8 = params_matrix[7, 9] * p
    Cc_9 = params_matrix[8, 9] * p
    Cc_10 = params_matrix[9, 9] * p
    Cc_11 = params_matrix[10, 9] * p
    Cc_12 = params_matrix[11, 9] * p
    Cc_13 = params_matrix[12, 9] * p
    Cc_14 = params_matrix[13, 9] * p
    Cc_15 = params_matrix[14, 9] * p
    Cc_16 = params_matrix[15, 9] * p

    # Input parameters for Ocean
    params_file = "./HQGA_params_opamp.txt"
    fw = open(params_file, 'w')
    fw.write("%s" %w01_1)
    fw.write("\n%s" %w01_2)
    fw.write("\n%s" %w01_3)
    fw.write("\n%s" %w01_4)
    fw.write("\n%s" %w01_5)
    fw.write("\n%s" %w01_6)
    fw.write("\n%s" %w01_7)
    fw.write("\n%s" %w01_8)
    fw.write("\n%s" %w01_9)
    fw.write("\n%s" %w01_10)
    fw.write("\n%s" %w01_11)
    fw.write("\n%s" %w01_12)
    fw.write("\n%s" %w01_13)
    fw.write("\n%s" %w01_14)
    fw.write("\n%s" %w01_15)
    fw.write("\n%s" %w01_16)

    fw.write("\n%s" %l01_1)
    fw.write("\n%s" %l01_2)
    fw.write("\n%s" %l01_3)
    fw.write("\n%s" %l01_4)
    fw.write("\n%s" %l01_5)
    fw.write("\n%s" %l01_6)
    fw.write("\n%s" %l01_7)
    fw.write("\n%s" %l01_8)
    fw.write("\n%s" %l01_9)
    fw.write("\n%s" %l01_10)
    fw.write("\n%s" %l01_11)
    fw.write("\n%s" %l01_12)
    fw.write("\n%s" %l01_13)
    fw.write("\n%s" %l01_14)
    fw.write("\n%s" %l01_15)
    fw.write("\n%s" %l01_16)

    fw.write("\n%s" %w23_1)
    fw.write("\n%s" %w23_2)
    fw.write("\n%s" %w23_3)
    fw.write("\n%s" %w23_4)
    fw.write("\n%s" %w23_5)
    fw.write("\n%s" %w23_6)
    fw.write("\n%s" %w23_7)
    fw.write("\n%s" %w23_8)
    fw.write("\n%s" %w23_9)
    fw.write("\n%s" %w23_10)
    fw.write("\n%s" %w23_11)
    fw.write("\n%s" %w23_12)
    fw.write("\n%s" %w23_13)
    fw.write("\n%s" %w23_14)
    fw.write("\n%s" %w23_15)
    fw.write("\n%s" %w23_16)

    fw.write("\n%s" %l23_1)
    fw.write("\n%s" %l23_2)
    fw.write("\n%s" %l23_3)
    fw.write("\n%s" %l23_4)
    fw.write("\n%s" %l23_5)
    fw.write("\n%s" %l23_6)
    fw.write("\n%s" %l23_7)
    fw.write("\n%s" %l23_8)
    fw.write("\n%s" %l23_9)
    fw.write("\n%s" %l23_10)
    fw.write("\n%s" %l23_11)
    fw.write("\n%s" %l23_12)
    fw.write("\n%s" %l23_13)
    fw.write("\n%s" %l23_14)
    fw.write("\n%s" %l23_15)
    fw.write("\n%s" %l23_16)

    fw.write("\n%s" %w47_1)
    fw.write("\n%s" %w47_2)
    fw.write("\n%s" %w47_3)
    fw.write("\n%s" %w47_4)
    fw.write("\n%s" %w47_5)
    fw.write("\n%s" %w47_6)
    fw.write("\n%s" %w47_7)
    fw.write("\n%s" %w47_8)
    fw.write("\n%s" %w47_9)
    fw.write("\n%s" %w47_10)
    fw.write("\n%s" %w47_11)
    fw.write("\n%s" %w47_12)
    fw.write("\n%s" %w47_13)
    fw.write("\n%s" %w47_14)
    fw.write("\n%s" %w47_15)
    fw.write("\n%s" %w47_16)

    fw.write("\n%s" %w5_1)
    fw.write("\n%s" %w5_2)
    fw.write("\n%s" %w5_3)
    fw.write("\n%s" %w5_4)
    fw.write("\n%s" %w5_5)
    fw.write("\n%s" %w5_6)
    fw.write("\n%s" %w5_7)
    fw.write("\n%s" %w5_8)
    fw.write("\n%s" %w5_9)
    fw.write("\n%s" %w5_10)
    fw.write("\n%s" %w5_11)
    fw.write("\n%s" %w5_12)
    fw.write("\n%s" %w5_13)
    fw.write("\n%s" %w5_14)
    fw.write("\n%s" %w5_15)
    fw.write("\n%s" %w5_16)

    fw.write("\n%s" %l457_1)
    fw.write("\n%s" %l457_2)
    fw.write("\n%s" %l457_3)
    fw.write("\n%s" %l457_4)
    fw.write("\n%s" %l457_5)
    fw.write("\n%s" %l457_6)
    fw.write("\n%s" %l457_7)
    fw.write("\n%s" %l457_8)
    fw.write("\n%s" %l457_9)
    fw.write("\n%s" %l457_10)
    fw.write("\n%s" %l457_11)
    fw.write("\n%s" %l457_12)
    fw.write("\n%s" %l457_13)
    fw.write("\n%s" %l457_14)
    fw.write("\n%s" %l457_15)
    fw.write("\n%s" %l457_16)

    fw.write("\n%s" %w6_1)
    fw.write("\n%s" %w6_2)
    fw.write("\n%s" %w6_3)
    fw.write("\n%s" %w6_4)
    fw.write("\n%s" %w6_5)
    fw.write("\n%s" %w6_6)
    fw.write("\n%s" %w6_7)
    fw.write("\n%s" %w6_8)
    fw.write("\n%s" %w6_9)
    fw.write("\n%s" %w6_10)
    fw.write("\n%s" %w6_11)
    fw.write("\n%s" %w6_12)
    fw.write("\n%s" %w6_13)
    fw.write("\n%s" %w6_14)
    fw.write("\n%s" %w6_15)
    fw.write("\n%s" %w6_16)

    fw.write("\n%s" %l6_1)
    fw.write("\n%s" %l6_2)
    fw.write("\n%s" %l6_3)
    fw.write("\n%s" %l6_4)
    fw.write("\n%s" %l6_5)
    fw.write("\n%s" %l6_6)
    fw.write("\n%s" %l6_7)
    fw.write("\n%s" %l6_8)
    fw.write("\n%s" %l6_9)
    fw.write("\n%s" %l6_10)
    fw.write("\n%s" %l6_11)
    fw.write("\n%s" %l6_12)
    fw.write("\n%s" %l6_13)
    fw.write("\n%s" %l6_14)
    fw.write("\n%s" %l6_15)
    fw.write("\n%s" %l6_16)

    fw.write("\n%s" %Cc_1)
    fw.write("\n%s" %Cc_2)
    fw.write("\n%s" %Cc_3)
    fw.write("\n%s" %Cc_4)
    fw.write("\n%s" %Cc_5)
    fw.write("\n%s" %Cc_6)
    fw.write("\n%s" %Cc_7)
    fw.write("\n%s" %Cc_8)
    fw.write("\n%s" %Cc_9)
    fw.write("\n%s" %Cc_10)
    fw.write("\n%s" %Cc_11)
    fw.write("\n%s" %Cc_12)
    fw.write("\n%s" %Cc_13)
    fw.write("\n%s" %Cc_14)
    fw.write("\n%s" %Cc_15)
    fw.write("\n%s" %Cc_16)

    fw.close()

    os.system("ocean -restore ./HQGA_opamp_v1.ocn")
    fr = open("./HQGA_results_opamp.txt", "r")
    output = fr.readlines()
    output = np.array(output)
    results = output.astype(np.float64)
    results = results.reshape(popSize, 9)
    fr.close()
    # print(results)

    cond    = np.transpose(results[:,0]).reshape(-1,1)
    PM      = np.transpose(results[:,1]).reshape(-1,1)
    DC_gain = np.transpose(results[:,2]).reshape(-1,1)
    CMRR    = np.transpose(results[:,3]).reshape(-1,1)
    UGB     = np.transpose(results[:,4]).reshape(-1,1)
    Power   = np.transpose(results[:,5]).reshape(-1,1)
    PSRR_n  = np.transpose(results[:,6]).reshape(-1,1)
    PSRR_p  = np.transpose(results[:,7]).reshape(-1,1)
    SR      = np.transpose(results[:,8]).reshape(-1,1)

    tan_60 = math.tan(math.radians(60))

    for i in range(popSize):
        if cond[i,0] == 0:
            fitness1[i,0] = -1
        elif ((PM[i,0] < 60.0) | (DC_gain[i,0] < 50.0) | (UGB[i,0] < 50.0) | (CMRR[i,0] < 50.0) | 
              (Power[i,0] > 200.0) | (SR[i,0] < 50.0) | (PSRR_n[i,0] < 50.0) | (PSRR_p[i,0] < 120.0)):
            fitness1[i,0] = 0
        else: fitness1[i,0] = UGB[i,0]* 1e+6 * CL / IREF * (math.tan(math.radians(PM[i,0]))/tan_60)
     
    fitness1 = fitness1.flatten()
    fitness1 = fitness1.tolist()

    return fitness1, results


#########################################################
#                QUANTUM ROTATION GATE                  #
#########################################################    
def rotation():
    rot = np.empty([2,2])
    # Lookup table of the rotation angle
    for i in range(0,popSize):
       for j in range(0,genomeLength*len(bounds)):
           if fitness[i] < global_best_fitness:
               if chromosome[i,j] == 0 and global_best_chrom[j] == 1:
                   # Define the rotation angle: delta_theta (e.g. 0.0785398163) = pi/40
                   delta_theta = 0.025*np.pi 
                   rot[0,0]    = math.cos(delta_theta); rot[0,1] = -math.sin(delta_theta)
                   rot[1,0]    = math.sin(delta_theta); rot[1,1] = math.cos(delta_theta)
                   nqpv[i,j,0] = (rot[0,0]*qpv[i,j,0]) + (rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1] = (rot[1,0]*qpv[i,j,0]) + (rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]  = round(nqpv[i,j,0],2)
                   qpv[i,j,1]  = round(1-nqpv[i,j,0],2)
               if chromosome[i,j] == 1 and global_best_chrom[j] ==  0:
                   # Define the rotation angle: delta_theta (e.g. -0.0785398163) = -pi/40
                   delta_theta = -0.025*np.pi
                   rot[0,0]    = math.cos(delta_theta); rot[0,1] = -math.sin(delta_theta)
                   rot[1,0]    = math.sin(delta_theta); rot[1,1] = math.cos(delta_theta)
                   nqpv[i,j,0] = (rot[0,0]*qpv[i,j,0]) + (rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1] = (rot[1,0]*qpv[i,j,0]) + (rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]  = round(nqpv[i,j,0],2)
                   qpv[i,j,1]  = round(1-nqpv[i,j,0],2)


#########################################################
#           X-PAULI QUANTUM MUTATION GATE               #
#########################################################
# pop_mutation_rate: mutation rate in the population
# mutation_rate: probability of a mutation of a bit 
def mutation(mutation_rate):
    for i in range(0,popSize):
        for j in range(0,genomeLength*len(bounds)):
            mutate = np.random.randint(100)
            mutate = mutate/100
            if mutate <= mutation_rate:
                nqpv[i,j,0] = qpv[i,j,1]
                nqpv[i,j,1] = qpv[i,j,0]
            else:
                nqpv[i,j,0] = qpv[i,j,0]
                nqpv[i,j,1] = qpv[i,j,1]
    for i in range(0,popSize):
        for j in range(0,genomeLength*len(bounds)):
            qpv[i,j,0] = nqpv[i,j,0]
            qpv[i,j,1] = nqpv[i,j,1]

#########################################################
#           TOURNAMENT SELECTION OPERATOR               #
#########################################################
def select_p_tournament():
    # Generate two unique random numbers in the specified range
    random_numbers = np.random.choice(range(0, popSize), size=2, replace=False)
    # Extract the two random numbers
    random_number1, random_number2 = random_numbers
    if(random_number1 < random_number2):
        LB, UB = random_number1, random_number2
    else: LB, UB = random_number2, random_number1
    selected = np.random.randint(popSize)
    for i in range(LB, UB):
        # check if better (e.g. perform a tournament)
        if fitness[i] > fitness[selected]:
            selected = i
    return selected
    
#########################################################
#          ONE-POINT QUANTUM CROSSOVER OPERATOR         #
#########################################################  
def mating(crossover_rate):
    j = 0
    crossover_point = 0
    parent1 = select_p_tournament()
    parent2 = select_p_tournament()
    if random.random() <= crossover_rate:
        crossover_point = np.random.randint(genomeLength-2)
    j = 0 
    while (j <= genomeLength-2):
        if j <= crossover_point:
            child1[parent1,j,0] = round(qpv[parent1,j,0],2)
            child1[parent1,j,1] = round(qpv[parent1,j,1],2)
            child2[parent2,j,0] = round(qpv[parent2,j,0],2)
            child2[parent2,j,1] = round(qpv[parent2,j,1],2)
        else:
            child1[parent1,j,0] = round(qpv[parent2,j,0],2)
            child1[parent1,j,1] = round(qpv[parent2,j,1],2)
            child2[parent2,j,0] = round(qpv[parent1,j,0],2)
            child2[parent2,j,1] = round(qpv[parent1,j,1],2)
        j = j + 1
    j = 0
    for j in range(0, genomeLength):
        qpv[parent1,j,0] = child1[parent1,j,0]
        qpv[parent1,j,1] = child1[parent1,j,1]
        qpv[parent2,j,0] = child2[parent2,j,0]
        qpv[parent2,j,1] = child2[parent2,j,1]
       
def crossover(crossover_rate):
    c = 1
    # while (c<=popSize):
    while (c <= popSize/2):
       mating(crossover_rate)
       c = c + 1


#########################################################
#                     MAIN PROGRAM                      #
#########################################################
# test file
# test_file = "test_7.txt"
# test_file = open(test_file, 'w')

# define dataframe
column_names = ["Iteration", "Begin", "End", "Time (s)", "Fitness", "Condition", "w01 (um)", "l01 (um)", "w23 (um)", "l23 (um)", 
                "w47 (um)", "w5 (um)", "l457 (um)", "w6 (um)", "l6 (um)", "Cc (pF)", "DC Gain (dB)", "UGB (MHz)", "PM (degree)", 
                "CMRR (dB)", "Power (uW)", "PSRR- (dB)", "PSRR+ (dB)", "Slew Rate (V/us)"]
df = pd.DataFrame(columns=column_names)

# define units
u = 1e-6
p = 1e-12
f = 1e-15

# define value for load capacitance at output node
CL = 1*p
IREF = 20*u

# define probabilities
p_alpha = np.random.rand()
r_cross = 0.8
r_mut = 0.05

# perform HQGA
Init_population()
Measure(p_alpha)
decoded = [decode(bounds, genomeLength, ch) for ch in chromosome]
fitness, results_design_params = Fitness_evaluation(decoded)
global_best_fitness = np.amax(fitness)
global_best_chrom = chromosome[np.argmax(fitness)]
results_xlsx = results_design_params[np.argmax(fitness)]

while (generation<=generation_max-1):
    # print("generation = ", generation)
    # print("fitness = ", fitness)
    begin_time = dt.datetime.now().strftime("%H:%M:%S")

    Measure(p_alpha)
    decoded = [decode(bounds, genomeLength, ch) for ch in chromosome]
    fitness, results_design_params = Fitness_evaluation(decoded)

    for i in range(popSize):
        if fitness[i] > global_best_fitness:
            global_best_chrom, global_best_fitness, results_xlsx = chromosome[i], fitness[i], results_design_params[i]
            # test_file.write("New global best at i = %d\n" %i)
        else:
            global_best_chrom, global_best_fitness, results_xlsx = global_best_chrom, global_best_fitness, results_xlsx  
    global_best_chrom_params = decode(bounds, genomeLength, global_best_chrom)
    
    ##### debug test ######
    # test_file.write("\nGeneration = %d\n" %generation)
    # test_file.write("\nVariables\n")
    # for i in range(0, popSize):
    #     for j in range(0, len(bounds)):
    #         test_file.write("%f\n" %decoded[i][j])
    # test_file.write("\n\nDelay and Power\n")
    # for i in range(popSize):
    #     for j in range(0, 2):
    #         test_file.write("%f\n" %results_design_params[i][j])
    # test_file.write("\nGlobal best fitness = %f\n" %global_best_fitness)
    # test_file.write("Global best chrom is ")
    # for i in range(0, genomeLength*len(bounds)):
    #     test_file.write("%d" %global_best_chrom[i])
    # test_file.write("\nGlobal best W1 = %f" %global_best_chrom_params[0])
    # test_file.write("\nGlobal best W67 = %f" %global_best_chrom_params[1])
    # test_file.write("\nGlobal best W89 = %f" %global_best_chrom_params[2])
    # test_file.write("\nGlobal best C01 = %f" %global_best_chrom_params[3])
    # test_file.write("\nDelay of global best = %f" %results_xlsx[0])
    # test_file.write("\nPower of global best = %f\n" %results_xlsx[1])
    # test_file.write("###############################################################\n\n")


    new_row = [
        {
            "Iteration": generation,
            "Begin": begin_time,
            "End": dt.datetime.now().strftime("%H:%M:%S"),
            "Time (s)": dt.datetime.strptime(dt.datetime.now().strftime("%H:%M:%S"),
                                                '%H:%M:%S') - dt.datetime.strptime(begin_time, '%H:%M:%S'),
            "Fitness"         : global_best_fitness,
            "Condition"       : results_xlsx[0],
            "w01 (um)"        : global_best_chrom_params[0],
            "l01 (um)"        : global_best_chrom_params[1],
            "w23 (um)"        : global_best_chrom_params[2],
            "l23 (um)"        : global_best_chrom_params[3],
            "w47 (um)"        : global_best_chrom_params[4],
            "w5 (um)"         : global_best_chrom_params[5],
            "l457 (um)"       : global_best_chrom_params[6],
            "w6 (um)"         : global_best_chrom_params[7],
            "l6 (um)"         : global_best_chrom_params[8],
            "Cc (pF)"         : global_best_chrom_params[9],
            "DC Gain (dB)"    : results_xlsx[2],
            "UGB (MHz)"       : results_xlsx[4],
            "PM (degree)"     : results_xlsx[1],
            "CMRR (dB)"       : results_xlsx[3],
            "Power (uW)"      : results_xlsx[5],
            "PSRR- (dB)"      : results_xlsx[6], 
            "PSRR+ (dB)"      : results_xlsx[7], 
            "Slew Rate (V/us)": results_xlsx[8],
        }]
    df = df.append(new_row, ignore_index=True)
    print(df)
    
    rotation()
    crossover(r_cross)
    mutation(r_mut)

    generation=generation+1
    
#output results to excel file
df.to_excel("../result_excel_recorrect_amax_argmax/iter150/HQGA_opamp_pop16_theta0.025pi_iter150_rcross0.8_rmut0.05_UGB50_P200_SR50.xlsx", sheet_name = "HQGA_OPAMP", index = False)
