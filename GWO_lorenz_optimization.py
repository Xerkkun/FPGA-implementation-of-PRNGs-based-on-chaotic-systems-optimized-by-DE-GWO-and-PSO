# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 21:52:14 2023

@author: Joeli
"""

import subprocess
import numpy as np
import pandas as pd
import time
#import random
from scipy.fft import fft
from scipy.signal import find_peaks

#%%_________________________________________
# FUNCTIONS
#___________________________________________


#______________________________________________________________________________
# EIGEN VALUES: Calculates de eigenvalues for the system from the Jacobian 
# Matrix and returns them in an array
#______________________________________________________________________________


def eigenvalues(vals):
    
    a = vals[0]
    b = vals[1]
    c = vals[2]
    
    x = np.sqrt(c*(b-1)) 
    y = np.sqrt(c*(b-1)) 
    z = b - 1 
    J = np.array([ [-a, a, 0], [b - z, -1, -x], [y, x, -c] ])                   # Jacobian matrix for the Lorenz atractor system
    eig_val , eig_vec = np.linalg.eig(J)                                        # We extract and return the eigenvalues
    return eig_val #, eig_vec


#______________________________________________________________________________
# LORENTZ ATRACTOR GENERATOR: Estimates de time-step and No. of Iterations for
# the Lorenz System from its eigenvalues. It, then, uses de Forwar-Euler Method
# to produce the timeseries X,Y and Z that will be evaluated later by TISEAN
#______________________________________________________________________________
    

def lorentz_atractor_generator(a,b,c,scaling_factor):
    # Eigenvalores
    step_info = 1/np.abs(eigenvalues([a,b,c]))
    
    # Valor de paso
    h = np.min(step_info)/50
    
    # No. Iteraciones
    t_max = np.max(step_info * scaling_factor)
    iteraciones = int(np.round(t_max / h))

    # Initial conditions:                                                       (Vecinity to equilibrium point)
    x = np.sqrt(c*(b-1)) + 0.01 
    y = np.sqrt(c*(b-1)) + 0.01
    z = b - 1 + 0.01
    
    # Initialization of position vectors: 
    x_pos = []
    y_pos = []
    z_pos = []

    x_pos.append(x)
    y_pos.append(y)
    z_pos.append(z)

    for i in range(iteraciones):
        # Forward - Euler Method: 
        x = x + h * (a * (y - x))
        y = y + h * (x * (b - z) - y)
        z = z + h * (x * y - c * z)
        
        # Save the computed values: 
        x_pos.append(x)
        y_pos.append(y)
        z_pos.append(z)
    
    # X, Y and Z vector for analysis (without transient state)
    x_vec = x_pos[int(len(x_pos)*.70):].copy()
    y_vec = y_pos[int(len(x_pos)*.70):].copy()
    z_vec = z_pos[int(len(x_pos)*.70):].copy()
    
    #__________________________________________________________________________
    # FAST FOURIER TRANSFORM: Allows us to detect periodic, non - chaotic 
    # signals to be discarded
    #__________________________________________________________________________
    
    # X TIME SERIES:
    
    fourier_x = np.abs(fft(x_vec))                                               # FFT in the real numbers domain
    fourier_x = fourier_x[1:int(np.floor(len(fourier_x)/2))].copy()
    fourier_x = fourier_x/max(fourier_x)                                         # Normalizing our FFT vector
    peaks_x, _ = find_peaks(fourier_x, height = 0.01)                            # Detecting the peaks in the FFT 
    
    # Y TIME SERIES:
    
    fourier_y = np.abs(fft(y_vec))                                               # FFT in the real numbers domain
    fourier_y = fourier_y[1:int(np.floor(len(fourier_y)/2))].copy()
    fourier_y = fourier_y/max(fourier_y)                                         # Normalizing our FFT vector
    peaks_y, _ = find_peaks(fourier_y, height = 0.01)                            # Detecting the peaks in the FFT 
    
    # Z TIME SERIES:
    
    fourier_z = np.abs(fft(z_vec))                                               # FFT in the real numbers domain
    fourier_z = fourier_z[1:int(np.floor(len(fourier_z)/2))].copy()
    fourier_z = fourier_z/max(fourier_z)                                         # Normalizing our FFT vector
    peaks_z, _ = find_peaks(fourier_z, height = 0.01)                            # Detecting the peaks in the FFT 
    
    n_peaks = min(len(peaks_x),len(peaks_y),len(peaks_z))                        # Determining the minimun number of peaks detected for [X,Y,Z] in the Lorenz atractor
    
    
    superiorPeaks_x, _ = find_peaks(fourier_x, height = 0.01)
    
    # We save the array and return the number of peaks:
    x_vector = x_pos.copy()
    y_vector = y_pos.copy()
    z_vector = z_pos.copy()
    M = np.array([np.array(x_vector),np.array(y_vector),np.array(z_vector)]).T
    
    if n_peaks <= 40 :                          # or (np.abs(x_mean) < 0.05 and np.abs(y_mean) < 0.05):
        return False
    else: 
        np.savetxt('xyz.txt', M, delimiter=' ')
        return True

# Kaplan - Yorke Dimension Estimation:
def Dky_estimator():
    df = pd.read_csv('xyz.txt')
    lines = str(len(df) - 20000)
    # Run lyap_spec command and capture the output
    cmd = ['lyap_spec','xyz.txt','-x'+lines,'-c1,2,3','-m3,1','-k500']            # We remove the transitory section and use 50 as the neighbourhood size
    output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL) 
    # stderr allows us to hide the output of the .exe file to save the time of printing its output

    # Parse the output to extract the Kaplan-Yorke dimension
    lines = output.split('\n')
    for line in reversed(lines):
        if line.startswith('#estimated KY-Dimension='):
            dimension = np.float64(line.split('=')[1].strip())
            return dimension

    # Return None if the dimension couldn't be found
    return 0
    
# GWO:
def gwo_algorithm(n_iter,n_wolves, n_dim):
    """Implementation of the Grey Wolf Optimization (GWO) algorithm"""
    
    # Start time of the GWO Algorithm: 
    total_time_start = time.time()
    # Initialize iteration durations llist: 
    iter_times = []
    # Initialize values for best solution, fitness values and particle progress:
    fitness_history =[]
    
    # Initialize upper and lower bounds for a,b, and c: 
    a_ub = 60
    a_lb = 0.001
    b_ub = 180
    b_lb = 1                                                # To respect the formula for the non-origin equilibrium point
    c_ub = 20 
    c_lb = 0.001
    
    # Initialize particles and their solutions:
    a_vals = np.random.uniform(a_lb,a_ub,(n_wolves,1))
    b_vals = np.random.uniform(b_lb,b_ub,(n_wolves,1))
    c_vals = np.random.uniform(c_lb,c_ub,(n_wolves,1))
    
    # Initialize search agents and their solutions:
    wolves_positions = np.array([a_vals,b_vals,c_vals]).squeeze().T
    
    # Initialize alpha, beta and delta positions:
    alpha_pos       = np.zeros(n_dim)
    alpha_fitness   = 0
    
    beta_pos        = np.zeros(n_dim)
    beta_fitness    = 0
    
    delta_pos       = np.zeros(n_dim)
    delta_fitness   = 0
        
    # Initializa alpha, beta and delta coordinates: 
    alpha_positions = []
    beta_positions  = []
    delta_positions = []
    # Initialize fitness for each positions:
    wolves_fitness  = np.zeros(n_wolves)
    # Initialize positions vector: 
    positions = []
    
    # Some pre-known values to start the solutions vector: 
    #wolves_positions[0] = [4.3100, 142.7631, 0.3887]
    #wolves_positions[1] = [7.5449, 180, 0.7149]
    #wolves_positions[2] = [3.0713, 162.6945, 0.3861]
    #wolves_positions[3] = [2.2126, 88.5874, 0.2716]
    #wolves_positions[4] = [4.0230, 129.8195, 0.3760]
    #wolves_positions[5] = [3.2845, 137.6160, 0.4885]
    #wolves_positions[6] = [4.2591, 136.9205, 0.3981]
    
    # Main loop
    for epoch in range(n_iter):
        iter_time_start = time.time()
        positions.append(wolves_positions.copy())
        for i in range(n_wolves):
            eig = eigenvalues(wolves_positions[i])
            if (np.real(eig[1]) > 5e-2 and isinstance(eig[1], complex)):                                   # and np.abs(eig[1]) > 9.4):                      # Real positive parts and empirical threshold to avoid periodic oscilators
                # Second criterion: FFT analysis with small amount of peaks 
                if lorentz_atractor_generator(wolves_positions[i][0],wolves_positions[i][1],wolves_positions[i][2], 1000):
                    fitness = Dky_estimator()
                    
                    # CONFIRMANDO FITNESS:
                    # Dado que ciertos valores pueden ser sobre-estimados por TISEAN, re-calculamos con un nuevo factor de escalamiento para los casos:  
                    # D_KY > 2.20
                    # D_KY > 2.22
                    # D_KY > 2.24 
                    # D_KY > 2.26
                    # Todo D_Ky > 2.3 o D_KY < 0 es descartado.
                    
                    if (fitness > 2.20):   
                        print('Possible over-estimation! Fitness:', fitness, '\nRe-calculating...')
                        if (lorentz_atractor_generator(wolves_positions[i][0],wolves_positions[i][1],wolves_positions[i][2], 1100)):
                            alternate_fitness = Dky_estimator()
                            print('Alternate fitness: ', alternate_fitness, '\n')
                            fitness = np.min([fitness, alternate_fitness])
                        else:
                            fitness = 0
                            print('Periodic! Fitness: ', fitness)

                    if (fitness > 2.22):
                        print('Possible over-estimation! Fitness:', fitness, '\nRe-calculating...')
                        if (lorentz_atractor_generator(wolves_positions[i][0],wolves_positions[i][1],wolves_positions[i][2], 1200)):
                            alternate_fitness = Dky_estimator()
                            print('Alternate fitness: ', alternate_fitness, '\n')
                            fitness = np.min([fitness, alternate_fitness])
                        else:
                            fitness = 0
                            print('Periodic! Fitness: ', fitness)
                            
                    if (fitness > 2.24):
                        print('Possible over-estimation! Fitness:', fitness, '\nRe-calculating...')
                        if (lorentz_atractor_generator(wolves_positions[i][0],wolves_positions[i][1],wolves_positions[i][2], 1300)):
                            alternate_fitness = Dky_estimator()
                            print('Alternate fitness: ', alternate_fitness, '\n')
                            fitness = np.min([fitness, alternate_fitness])
                        else:
                            fitness = 0
                            print('Periodic! Fitness: ', fitness)
                    
                    if (fitness > 2.26):
                        print('Possible over-estimation! Fitness:', fitness, '\nRe-calculating...')
                        if (lorentz_atractor_generator(wolves_positions[i][0],wolves_positions[i][1],wolves_positions[i][2], 1400)):
                            alternate_fitness = Dky_estimator()
                            print('Alternate fitness: ', alternate_fitness, '\n')
                            fitness = np.min([fitness, alternate_fitness])
                        else:
                            fitness = 0
                            print('Periodic! Fitness: ', fitness)
                        
                    if (fitness > 2.3 or fitness < 2.0):
                            print('Kaplan-Yorke Dimension Over-Estimation Error! \n')
                            fitness = 0
                            
                    print('epoch: ',epoch + 1, 'particle: ', i + 1, '>>> OF INTEREST!\n Fitness: ', fitness, '\n')                    
                else: 
                    fitness = 0
                    print('epoch: ',epoch + 1, 'particle: ', i + 1, '>>> NOT OF INTEREST! Periodic!\n Fitness: ', fitness, '\n')
            else:
                fitness = 0
                print('epoch: ',epoch + 1, 'particle: ', i + 1, '>>> NOT OF INTEREST! Convergent!\n Fitness: ', fitness, '\n')

            
            wolves_fitness[i] = fitness
           
            if fitness > alpha_fitness:
                delta_fitness   = beta_fitness      # Update delta
                delta_pos       = beta_pos.copy()
                
                beta_fitness    = alpha_fitness     # Update delta
                beta_pos        = alpha_pos.copy()
                
                alpha_fitness   = fitness           # Update alpha
                alpha_pos       = wolves_positions[i].copy()
            if fitness < alpha_fitness and fitness > beta_fitness:
                delta_fitness   = beta_fitness      # Update delta
                delta_pos       = beta_pos.copy()
                
                beta_fitness    = fitness           # Update beta
                beta_pos        = wolves_positions[i].copy()
                
            if fitness < alpha_fitness and fitness < beta_fitness and fitness > delta_fitness:
                delta_fitness   = fitness           # Update delta
                delta_pos       = wolves_positions[i].copy()
        
        fitness_history.append(wolves_fitness.copy())
        
        alpha_positions.append(alpha_pos.copy())
        beta_positions.append(beta_pos.copy())
        delta_positions.append(delta_pos.copy())
        
        # a decreases linearly (2 -> 0)
        a = 2 - epoch * ((2) / n_iter)
        
        # Update position of each search agent: 
        for i in range(n_wolves):
            r1  =   np.random.uniform(0, 1,(1,n_dim))
            r2  =   np.random.uniform(0, 1,(1,n_dim))
            
            A1 = 2 * a * r1 - a     # eq. (3.3)
            C1 = 2 * r2             # eq. (3.4)
            
            D_alpha = abs(C1 * alpha_pos - wolves_positions[i])          # eq. (3.5 pt. 1)
            X1 = alpha_pos - A1 * D_alpha                                # eq. (3.6 pt. 1)                                
            
            r1  =   np.random.uniform(0, 1,(1,n_dim))
            r2  =   np.random.uniform(0, 1,(1,n_dim))
            
            A2 = 2 * a * r1 - a         # eq. (3.3)
            C2 = 2 * r2                 # eq. (3.4)
            
            D_beta = abs(C2 * beta_pos - wolves_positions[i])          # eq. (3.5 pt. 2)
            X2 = beta_pos - A2 * D_beta                                  # eq. (3.6 pt. 2) 
            
            r1  =   np.random.uniform(0, 1,(1,n_dim))
            r2  =   np.random.uniform(0, 1,(1,n_dim))
            
            A3 = 2 * a * r1 - a     # eq. (3.3)
            C3 = 2 * r2             # eq. (3.4)
            
            D_delta = abs(C3 * delta_pos - wolves_positions[i])        # eq. (3.5 pt. 3)
            X3 = delta_pos - A3 * D_delta                                # eq. (3.6 pt. 3) 
            
            new_solution = (X1 + X2 + X3) / 3
            
            # Bound the solution within the search space
            new_solution[0][0] = np.clip(new_solution[0][0],a_lb,a_ub) 
            new_solution[0][1] = np.clip(new_solution[0][1],b_lb,b_ub)
            new_solution[0][2] = np.clip(new_solution[0][2],c_lb,c_ub)
            
            wolves_positions[i] = new_solution
        
        #wolves_positions = np.clip(wolves_positions, lb, ub) 
        
        best_solution   =   alpha_pos
        best_fitness    =   alpha_fitness
        
        print('\n ALPHA POSITION', best_solution, 'Fitnesss: ', alpha_fitness)
        print('\n BETA POSITION', beta_pos, 'Fitness: ', beta_fitness)
        print('\n DELTA POSITION', delta_pos, 'Fitness: ', delta_fitness, '\n')
        
        # ITERATION DURATIONS (s):
        iter_time_end = time.time()
        iter_duration = iter_time_end - iter_time_start
        iter_times.append(iter_duration)
    
    # PRINTING THE TOTAL TIMES: 
        
    # Iterations Average Execution: 
    iter_avrg_time = sum(iter_times)/len(iter_times)
    print('ITERATION AVERAGE TIME(s)', str(iter_avrg_time))
    
    # Total Execution Time:    
    total_time_end = time.time()
    print('TOTAL TIME (s):', str(total_time_end - total_time_start))
                
    return best_solution, best_fitness, positions, [alpha_positions, beta_positions, delta_positions], fitness_history

#%% Example usage
n_iterations = 10                     # Number of iterations        # 25 - 100
n_particles  = 20                     # Number of particles          # 50 
n_dimensions = 3                      # Number of dimensions

best_solution, best_fitness, particle_trajectories, leader_positions, trajectories_fitness = gwo_algorithm(n_iterations,n_particles, n_dimensions)
print('\nALGORITHM FINISHED!')
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

#np.save('GWO_vStep_25_07_2023v2_nIter_25_nWolves_20_bestSolution',best_solution)
#np.save('GWO_vStep_25_07_2023v2_nIter_25_nWolves_20_bestFitness',best_fitness)
#np.save('GWO_vStep_25_07_2023v2_nIter_25_nWolves_20_fitnessHistory',trajectories_fitness)
#np.save('GWO_vStep_25_07_2023v2_nIter_25_nWolves_20_Trajectories',particle_trajectories)
#np.save('GWO_vStep_25_07_2023v2_nIter_25_nWolves_20_leaderTrajectories',leader_positions)
    
#%% Resulting Lorenz System

import matplotlib.pyplot as plt

# Parámetros
a = best_solution[0] # particle_trajectories[0][9][0] # leader_positions[0][14][0] #
b = best_solution[1] # particle_trajectories[0][9][1] # leader_positions[0][14][1] #
c = best_solution[2] # particle_trajectories[0][9][2] # leader_positions[0][14][2] #
# Eigenvalores
step_info = 1/np.abs(eigenvalues([a,b,c]))

# Valor de paso
h = np.min(step_info)/50

# No. Iteraciones
t_max = np.max(step_info) * 125
trans = int(np.round(t_max / h))
iteraciones = 100000             #int(np.round(t_max / h))

# Condiciones iniciales:
x = np.sqrt(c*(b-1)) + 0.01
y = np.sqrt(c*(b-1)) + 0.01
z = b - 1 + 0.01

# Inicialización de Vectores de Posición
x_pos = []
y_pos = []
z_pos = []

x_pos.append(x)
y_pos.append(y)
z_pos.append(z)

for i in range(iteraciones):
    # Método numérico
    x = x + h * (a * (y - x))
    y = y + h * (x * (b - z) - y)
    z = z + h * (x * y - c * z)
    
    # Actualizamos
    x_pos.append(x)
    y_pos.append(y)
    z_pos.append(z)

M = np.array([np.array(x_pos),np.array(y_pos),np.array(z_pos)]).T
np.savetxt('xyz.txt', M, delimiter=' ')

x_pos = x_pos[trans:]
y_pos = y_pos[trans:]
z_pos = z_pos[trans:]

plt.close('all')

ax = plt.figure().add_subplot()
ax.plot(x_pos,linewidth='0.5')
#ax.set_title('Atractor de Lorentz (Trayectoria en X)')
ax.set_ylabel('x')
ax.set_xlabel('Iterations')

#ax = plt.figure().add_subplot()
#ax.plot(x_pos,y_pos,linewidth='0.5')
#ax.set_title('Atractor de Lorentz x-y')

ax = plt.figure().add_subplot()
ax.plot(x_pos,z_pos,linewidth='0.5')
#ax.set_title('Atractor de Lorentz x-z')
ax.set_xlabel('x')
ax.set_ylabel('z')

#ax = plt.figure().add_subplot()
#ax.plot(y_pos,z_pos,linewidth='0.5')
#ax.set_title('Atractor de Lorentz y-z')

ax = plt.figure().add_subplot(projection = '3d')
ax.plot(x_pos,y_pos,z_pos,linewidth='0.5')
ax.set_title('Atractor de Lorentz')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# Ploteando eigenvalores 

ax = plt.figure().add_subplot()

eig = eigenvalues([a,b,c])
ax.scatter(np.real(eig),np.imag(eig))
#ax.set_title('Eigenvalues')
ax.set_xlabel('Re')
ax.set_ylabel('Im')
#ax.set_aspect('equal')
ax.grid(True, which = 'both')
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')

# Transformada de Fourier

x_vec = x_pos.copy()

fourier_x = np.abs(np.fft.fft(x_vec))                                           # FFT in the real numbers domain
fourier_x = fourier_x/len(x_pos)                                                # Normalizing our FFT vector
freq = np.fft.fftfreq(len(fourier_x), d = h)

half_point = int(len(fourier_x)/2)

fourier_x = np.concatenate((fourier_x[-half_point:],fourier_x[:half_point+1]))
#fourier_x = fourier_x[int(len(fourier_x)):int(len(fourier_x))]
freq = np.concatenate((freq[-half_point:],freq[:half_point+1]))
#freq = freq[int(len(freq)):int(len(freq))]

ax = plt.figure().add_subplot()
ax.plot(fourier_x,linewidth='0.5')
ax.set_ylabel('abs(X)')
ax.set_xlabel('Frequency (hz)')

# Subplots

# samples = 50000
# x_readjusted = np.linspace(0,len(x_pos[-samples:])/1000,len(x_pos[-samples:]))


# fig, axs = plt.subplots(2, 2, figsize=(14,16))
# axs[0, 0].plot(x_pos[-samples:],z_pos[-samples:],linewidth='0.5', label = np.round(best_fitness, 5))
# #axs[0, 0].set_title('\u03C3 = ' + str(round(a,4))+' \u03C1 = '+ str(round(b,4)) + ' \u03B2 = ' + str(round(c,4)))
# axs[0, 0].set_xlabel('x')
# axs[0, 0].set_ylabel('z')
# axs[0, 0].legend()
# axs[0, 0].patch.set_edgecolor('black')  
# axs[0, 0].patch.set_linewidth('1.5') 
# eig = eigenvalues([a,b,c])
# axs[0, 1].scatter(np.real(eig),np.imag(eig), label = np.round(eig,2))
# axs[0, 1].legend()
# axs[0, 1].set_xlabel('Re')
# axs[0, 1].set_ylabel('Im')
# axs[0, 1].grid(True, which = 'both')
# axs[0, 1].axhline(y = 0, color = 'k')
# axs[0, 1].axvline(x = 0, color = 'k')
# axs[0, 1].patch.set_edgecolor('black')  
# axs[0, 1].patch.set_linewidth('1.5') 
# axs[1, 0].plot(x_readjusted,x_pos[-samples:],linewidth='0.5',label = 'h = ' + str(round(h,5)))
# axs[1, 0].legend()
# axs[1, 0].set_ylabel('x(t)')
# axs[1, 0].set_xlabel('Iterations')
# axs[1, 0].patch.set_edgecolor('black')  
# axs[1, 0].patch.set_linewidth('1.5') 
# axs[1, 1].plot(freq,fourier_x,linewidth='0.5')#
# axs[1, 1].set_ylabel('X')
# axs[1, 1].set_xlabel('Frequency (hz)')
# axs[1, 1].patch.set_edgecolor('black')  
# axs[1, 1].patch.set_linewidth('1.5') 
# axs[1, 1].set_xlim((-10,10))
#axs[1, 1].set_title('Fourier Spectrum')

# %% Finding best results

#fitness_values = np.array(trajectories_fitness).flatten()
#fitness_values.sort()
#best_values = fitness_values[-10:]    
# np.where(trajectories_fitness == best_values[-1])
