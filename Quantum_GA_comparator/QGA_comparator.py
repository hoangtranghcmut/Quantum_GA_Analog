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
popSize=16                           # Define here the population size
genomeLength=16                     # Define here the chromosome length
generation_max= int(sys.argv[1])    # Define here the maximum number of
                                    # generations/iterations
# define range for input
bounds = [[0.12, 2.0], [0.12, 2.0], [0.12, 2.0], [0.1, 0.8]] # w_m1, w_m67, w_m89, c01

# Initialization global variables
theta=0
iteration=0
the_best_chrom=0
generation=0


#######################################################################
#                         VARIABLES ALGORITHM                         #         
#######################################################################
top_bottom = 2
QuBitZero = np.array([[1], [0]])
QuBitOne = np.array([[0], [1]])
AlphaBeta = np.empty([top_bottom])
fitness = np.array([popSize]) 
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
    r2=math.sqrt(2.0)           
    h=np.array([[1/r2, 1/r2],[1/r2,-1/r2]])
    # Rotation Q-gate
    theta=0
    rot=np.empty([2,2])
    # Initial population array (individual x chromosome)
    i=0; j=0
    for i in range(0,popSize):
        for j in range(0,genomeLength*len(bounds)):
            theta=np.random.uniform(0,1)*90   
            theta=math.radians(theta)
            rot[0,0]=math.cos(theta); rot[0,1]=-math.sin(theta)
            rot[1,0]=math.sin(theta); rot[1,1]=math.cos(theta)
            AlphaBeta[0]=rot[0,0]*(h[0][0]*QuBitZero[0]+h[0][1]*QuBitZero[1])+rot[0,1]*(h[1][0]*QuBitZero[0]+h[1][1]*QuBitZero[1])
            AlphaBeta[1]=rot[1,0]*(h[0][0]*QuBitZero[0]+h[0][1]*QuBitZero[1])+rot[1,1]*(h[1][0]*QuBitZero[0]+h[1][1]*QuBitZero[1])
            # alpha squared
            qpv[i,j,0]=np.around(1*pow(AlphaBeta[0],2),2) 
            # beta squared
            qpv[i,j,1]=np.around(1*pow(AlphaBeta[1],2),2) 


######################################################################
#                           MAKE A MEASUREMENT                       #                  
######################################################################
# p_alpha: probability of finding qubit in alpha state    
def Measure(p_alpha):
    for i in range(0,popSize):     
        for j in range(0,genomeLength*len(bounds)):
            if p_alpha<=qpv[i, j, 0]:
                chromosome[i,j]=0
            else:
                chromosome[i,j]=1
            

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
    n = 1e-9
    f = 1e-15

    fitness1 = np.zeros((popSize,1), dtype='f')
    params_matrix = np.matrix(params_list)

    w_m1_1 = params_matrix[0, 0] * u
    w_m1_2 = params_matrix[1, 0] * u
    w_m1_3 = params_matrix[2, 0] * u
    w_m1_4 = params_matrix[3, 0] * u
    w_m1_5 = params_matrix[4, 0] * u
    w_m1_6 = params_matrix[5, 0] * u
    w_m1_7 = params_matrix[6, 0] * u
    w_m1_8 = params_matrix[7, 0] * u
    w_m1_9 = params_matrix[8, 0] * u
    w_m1_10 = params_matrix[9, 0] * u
    w_m1_11 = params_matrix[10, 0] * u
    w_m1_12 = params_matrix[11, 0] * u
    w_m1_13 = params_matrix[12, 0] * u
    w_m1_14 = params_matrix[13, 0] * u
    w_m1_15 = params_matrix[14, 0] * u
    w_m1_16 = params_matrix[15, 0] * u

    w_m67_1 = params_matrix[0, 1] * u
    w_m67_2 = params_matrix[1, 1] * u
    w_m67_3 = params_matrix[2, 1] * u
    w_m67_4 = params_matrix[3, 1] * u
    w_m67_5 = params_matrix[4, 1] * u
    w_m67_6 = params_matrix[5, 1] * u
    w_m67_7 = params_matrix[6, 1] * u
    w_m67_8 = params_matrix[7, 1] * u
    w_m67_9 = params_matrix[8, 1] * u
    w_m67_10 = params_matrix[9, 1] * u
    w_m67_11 = params_matrix[10, 1] * u
    w_m67_12 = params_matrix[11, 1] * u
    w_m67_13 = params_matrix[12, 1] * u
    w_m67_14 = params_matrix[13, 1] * u
    w_m67_15 = params_matrix[14, 1] * u
    w_m67_16 = params_matrix[15, 1] * u

    w_m89_1 = params_matrix[0, 2] * u
    w_m89_2 = params_matrix[1, 2] * u
    w_m89_3 = params_matrix[2, 2] * u
    w_m89_4 = params_matrix[3, 2] * u
    w_m89_5 = params_matrix[4, 2] * u
    w_m89_6 = params_matrix[5, 2] * u
    w_m89_7 = params_matrix[6, 2] * u
    w_m89_8 = params_matrix[7, 2] * u
    w_m89_9 = params_matrix[8, 2] * u
    w_m89_10 = params_matrix[9, 2] * u
    w_m89_11 = params_matrix[10, 2] * u
    w_m89_12 = params_matrix[11, 2] * u
    w_m89_13 = params_matrix[12, 2] * u
    w_m89_14 = params_matrix[13, 2] * u
    w_m89_15 = params_matrix[14, 2] * u
    w_m89_16 = params_matrix[15, 2] * u

    c01_1 = params_matrix[0, 3] * f
    c01_2 = params_matrix[1, 3] * f
    c01_3 = params_matrix[2, 3] * f
    c01_4 = params_matrix[3, 3] * f
    c01_5 = params_matrix[4, 3] * f
    c01_6 = params_matrix[5, 3] * f
    c01_7 = params_matrix[6, 3] * f
    c01_8 = params_matrix[7, 3] * f
    c01_9 = params_matrix[8, 3] * f
    c01_10 = params_matrix[9, 3] * f
    c01_11 = params_matrix[10, 3] * f
    c01_12 = params_matrix[11, 3] * f
    c01_13 = params_matrix[12, 3] * f
    c01_14 = params_matrix[13, 3] * f
    c01_15 = params_matrix[14, 3] * f
    c01_16 = params_matrix[15, 3] * f

    # Input parameters for Ocean
    params_file = "HQGA_params.txt"
    fw = open(params_file, 'w')
    fw.write("\n%s" %w_m1_1)
    fw.write("\n%s" %w_m1_2)
    fw.write("\n%s" %w_m1_3)
    fw.write("\n%s" %w_m1_4)
    fw.write("\n%s" %w_m1_5)
    fw.write("\n%s" %w_m1_6)
    fw.write("\n%s" %w_m1_7)
    fw.write("\n%s" %w_m1_8)
    fw.write("\n%s" %w_m1_9)
    fw.write("\n%s" %w_m1_10)
    fw.write("\n%s" %w_m1_11)
    fw.write("\n%s" %w_m1_12)
    fw.write("\n%s" %w_m1_13)
    fw.write("\n%s" %w_m1_14)
    fw.write("\n%s" %w_m1_15)
    fw.write("\n%s" %w_m1_16)


    fw.write("\n%s" %w_m67_1)
    fw.write("\n%s" %w_m67_2)
    fw.write("\n%s" %w_m67_3)
    fw.write("\n%s" %w_m67_4)
    fw.write("\n%s" %w_m67_5)
    fw.write("\n%s" %w_m67_6)
    fw.write("\n%s" %w_m67_7)
    fw.write("\n%s" %w_m67_8)
    fw.write("\n%s" %w_m67_9)
    fw.write("\n%s" %w_m67_10)
    fw.write("\n%s" %w_m67_11)
    fw.write("\n%s" %w_m67_12)
    fw.write("\n%s" %w_m67_13)
    fw.write("\n%s" %w_m67_14)
    fw.write("\n%s" %w_m67_15)
    fw.write("\n%s" %w_m67_16)

    fw.write("\n%s" %w_m89_1)
    fw.write("\n%s" %w_m89_2)
    fw.write("\n%s" %w_m89_3)
    fw.write("\n%s" %w_m89_4)
    fw.write("\n%s" %w_m89_5)
    fw.write("\n%s" %w_m89_6)
    fw.write("\n%s" %w_m89_7)
    fw.write("\n%s" %w_m89_8)
    fw.write("\n%s" %w_m89_9)
    fw.write("\n%s" %w_m89_10)
    fw.write("\n%s" %w_m89_11)
    fw.write("\n%s" %w_m89_12)
    fw.write("\n%s" %w_m89_13)
    fw.write("\n%s" %w_m89_14)
    fw.write("\n%s" %w_m89_15)
    fw.write("\n%s" %w_m89_16)


    fw.write("\n%s" %c01_1)
    fw.write("\n%s" %c01_2)
    fw.write("\n%s" %c01_3)
    fw.write("\n%s" %c01_4)
    fw.write("\n%s" %c01_5)
    fw.write("\n%s" %c01_6)
    fw.write("\n%s" %c01_7)
    fw.write("\n%s" %c01_8)
    fw.write("\n%s" %c01_9)
    fw.write("\n%s" %c01_10)
    fw.write("\n%s" %c01_11)
    fw.write("\n%s" %c01_12)
    fw.write("\n%s" %c01_13)
    fw.write("\n%s" %c01_14)
    fw.write("\n%s" %c01_15)
    fw.write("\n%s" %c01_16)
    fw.close()

    os.system("ocean -restore comparator_v2_HQGA_pop16.ocn")
    fr = open("HQGA_results.txt", "r")
    output = fr.readlines()
    output = np.array(output)
    results = output.astype(np.float64)
    results = results.reshape(popSize, 2)
    fr.close()
    # print(results)

    Delay = np.transpose(results[:,0]).reshape(-1,1)
    Power = np.transpose(results[:,1]).reshape(-1,1)

    fitness1[:, ] = Delay*Power*1e-3
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
           if fitness[i]>global_best_fitness:
               if chromosome[i,j]==0 and global_best_chrom[j]==1:
                   # Define the rotation angle: delta_theta (e.g. 0.0785398163) = pi/40
                   delta_theta=0.05*np.pi 
                   rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta)
                   rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta)
                   nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]=round(nqpv[i,j,0],2)
                   qpv[i,j,1]=round(1-nqpv[i,j,0],2)
               if chromosome[i,j]==1 and global_best_chrom[j]==0:
                   # Define the rotation angle: delta_theta (e.g. -0.0785398163) = -pi/40
                   delta_theta=-0.05*np.pi
                   rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta)
                   rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta)
                   nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]=round(nqpv[i,j,0],2)
                   qpv[i,j,1]=round(1-nqpv[i,j,0],2)


#########################################################
#           X-PAULI QUANTUM MUTATION GATE               #
#########################################################
# pop_mutation_rate: mutation rate in the population
# mutation_rate: probability of a mutation of a bit 
def mutation(mutation_rate):
    for i in range(0,popSize):
        for j in range(0,genomeLength*len(bounds)):
            mutate=np.random.randint(100)
            mutate=mutate/100
            if mutate<=mutation_rate:
                nqpv[i,j,0]=qpv[i,j,1]
                nqpv[i,j,1]=qpv[i,j,0]
            else:
                nqpv[i,j,0]=qpv[i,j,0]
                nqpv[i,j,1]=qpv[i,j,1]
    for i in range(0,popSize):
        for j in range(0,genomeLength*len(bounds)):
            qpv[i,j,0]=nqpv[i,j,0]
            qpv[i,j,1]=nqpv[i,j,1]

#########################################################
#           TOURNAMENT SELECTION OPERATOR               #
#########################################################
def select_p_tournament():
    selected = np.random.randint(popSize)
    for i in range(0, np.random.randint(0, popSize)):
        # check if better (e.g. perform a tournament)
        if fitness[i] < fitness[selected]:
            selected = i
    return selected
    
#########################################################
#            ONE-POINT CROSSOVER OPERATOR               #
#########################################################  
def mating(crossover_rate):
    j=0
    crossover_point=0
    parent1=select_p_tournament()
    parent2=select_p_tournament()
    if random.random()<=crossover_rate:
        crossover_point=np.random.randint(genomeLength-2)
    j=0 
    while (j<=genomeLength-2):
        if j<=crossover_point:
            child1[parent1,j,0]=round(qpv[parent1,j,0],2)
            child1[parent1,j,1]=round(qpv[parent1,j,1],2)
            child2[parent2,j,0]=round(qpv[parent2,j,0],2)
            child2[parent2,j,1]=round(qpv[parent2,j,1],2)
        else:
            child1[parent1,j,0]=round(qpv[parent2,j,0],2)
            child1[parent1,j,1]=round(qpv[parent2,j,1],2)
            child2[parent2,j,0]=round(qpv[parent1,j,0],2)
            child2[parent2,j,1]=round(qpv[parent1,j,1],2)
        j=j+1
    j=0
    for j in range(0,genomeLength):
        qpv[parent1,j,0]=child1[parent1,j,0]
        qpv[parent1,j,1]=child1[parent1,j,1]
        qpv[parent2,j,0]=child2[parent2,j,0]
        qpv[parent2,j,1]=child2[parent2,j,1]
       
def crossover(crossover_rate):
    c=1
    # while (c<=popSize):
    while (c<=popSize/2):
       mating(crossover_rate)
       c=c+1


#########################################################
#                     MAIN PROGRAM                      #
#########################################################
# test file
test_file = "test_7.txt"
test_file = open(test_file, 'w')

# define dataframe
column_names = ["Iteration", "Begin", "End", "Time (s)", "Fitness", "W1 (um)", "W6 = W7 (um)", "W8 = W9 (um)", "C0 = C1 (fF)", "Delay (ps)", "Power (uW)"]
df = pd.DataFrame(columns=column_names)

# define units
u = 1e-6
p = 1e-12
f = 1e-15

# define probabilities
p_alpha = np.random.rand()
r_cross = 0.8
r_mut = 0.05

# perform HQGA
Init_population()
Measure(p_alpha)
decoded = [decode(bounds, genomeLength, ch) for ch in chromosome]
fitness, results_delay_power = Fitness_evaluation(decoded)
global_best_fitness = np.amin(fitness)
global_best_chrom = chromosome[np.argmin(fitness)]
results_xlsx = results_delay_power[np.argmin(fitness)]

while (generation<=generation_max-1):
    # print("generation = ", generation)
    # print("fitness = ", fitness)
    begin_time = dt.datetime.now().strftime("%H:%M:%S")

    Measure(p_alpha)
    decoded = [decode(bounds, genomeLength, ch) for ch in chromosome]
    fitness, results_delay_power = Fitness_evaluation(decoded)

    for i in range(popSize):
        if fitness[i] < global_best_fitness:
            global_best_chrom, global_best_fitness, results_xlsx = chromosome[i], fitness[i], results_delay_power[i]
            test_file.write("New global best at i = %d\n" %i)
        else:
            global_best_chrom, global_best_fitness, results_xlsx = global_best_chrom, global_best_fitness, results_xlsx  
    global_best_chrom_params = decode(bounds, genomeLength, global_best_chrom)
    
    #rotation()
    #crossover(r_cross)
    #mutation(r_mut)

    # print("output to excel = ", results_xlsx)


    ##### debug test ######
    test_file.write("\nGeneration = %d\n" %generation)
    test_file.write("\nVariables\n")
    for i in range(0, popSize):
        for j in range(0, len(bounds)):
            test_file.write("%f\n" %decoded[i][j])
    test_file.write("\n\nDelay and Power\n")
    for i in range(popSize):
        for j in range(0, 2):
            test_file.write("%f\n" %results_delay_power[i][j])
    test_file.write("\nGlobal best fitness = %f\n" %global_best_fitness)
    test_file.write("Global best chrom is ")
    for i in range(0, genomeLength*len(bounds)):
        test_file.write("%d" %global_best_chrom[i])
    test_file.write("\nGlobal best W1 = %f" %global_best_chrom_params[0])
    test_file.write("\nGlobal best W67 = %f" %global_best_chrom_params[1])
    test_file.write("\nGlobal best W89 = %f" %global_best_chrom_params[2])
    test_file.write("\nGlobal best C01 = %f" %global_best_chrom_params[3])
    test_file.write("\nDelay of global best = %f" %results_xlsx[0])
    test_file.write("\nPower of global best = %f\n" %results_xlsx[1])
    test_file.write("###############################################################\n\n")



    new_row = [
        {
            "Iteration": generation,
            "Begin": begin_time,
            "End": dt.datetime.now().strftime("%H:%M:%S"),
            "Time (s)": dt.datetime.strptime(dt.datetime.now().strftime("%H:%M:%S"),
                                                '%H:%M:%S') - dt.datetime.strptime(begin_time, '%H:%M:%S'),
            "Fitness": global_best_fitness,
            "W1 (um)": global_best_chrom_params[0],
            "W6 = W7 (um)": global_best_chrom_params[1],
            "W8 = W9 (um)": global_best_chrom_params[2],
            "C0 = C1 (fF)": global_best_chrom_params[3],
            "Delay (ps)": results_xlsx[0],
            "Power (uW)": results_xlsx[1],
        }]
    df = df.append(new_row, ignore_index=True)
    print(df)
    
    rotation()
    crossover(r_cross)
    mutation(r_mut)

    generation=generation+1
    
#output results to excel file
df.to_excel("TEST_HQGA_pop16_theta0.05pi_iter20_PDP_rcross0.8_rmut0.05.xlsx", sheet_name = "HQGA", index = False)
