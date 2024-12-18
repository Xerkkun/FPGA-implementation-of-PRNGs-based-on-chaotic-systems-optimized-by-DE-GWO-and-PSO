"""Date: 7/12/2023
Runs with random seeds
Polaco algorithm"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import subprocess
from pathlib import Path
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks
import time
import random

start_time = time.time()
# =============================================================
#Método de Runge Kutta de 4to orden
def runge_kutta4(X, h, sigma, rho, beta):
    k1 = lorenz_equations(X, sigma, rho, beta)
    k2 = lorenz_equations(X + h*0.5*np.array(k1), sigma, rho, beta)
    k3 = lorenz_equations(X + h*0.5*np.array(k2), sigma, rho, beta)
    k4 = lorenz_equations(X + h*np.array(k3), sigma, rho, beta)
    X = X + (h/6.)*(np.array(k1) + 2*np.array(k2) + 2*np.array(k3) + np.array(k4))
    return X
# =============================================================
# Ecuaciones del sistema de Lorenz
def lorenz_equations(X, sigma, rho, beta):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]
# =============================================================
# Definir la función objetivo a optimizar (Dimension de Kaplan-Yorke)
def lorenz_objective(X):
    #global global_i
    
    #Si son muchos valores, decimar para ingresar a tisean
    # Parámetros del sistema de Lorenz
    sigma = X[0]  #0,60
    rho = X[1]    #0.001,180
    beta = X[2]   #0.001,30
# =============================================================
    # Ajustar la h a partir de los valores propios
    x_eq, y_eq, z_eq = np.sqrt(np.abs(beta*(rho-1))),np.sqrt(np.abs(beta*(rho-1))),rho-1
    
    J = np.array([ [-sigma, sigma, 0], [rho - z_eq, -1, -x_eq], [y_eq, x_eq, -beta]   ])
    eigenvalues, _ = np.linalg.eig(J)

    vp = np.array( [eigenvalues.real,eigenvalues.imag], dtype=float )
    vp_non_zero = vp != 0 
    vp_inverse = 1/np.abs(vp[vp_non_zero])
    vp_min = np.amin(vp_inverse)
    vp_max = np.amax(vp_inverse)
    #print(vp[vp_non_zero],vp_inverse)
    #t_step=0.001
    t_step=np.round(vp_min/10,5)
    
    transient=int(vp_max*5/t_step)
    steady_state = int(1E4)
    num_steps=transient+steady_state
    n = int(1E4)
          
    # print('Ancho de paso: ', t_step)
    # print('Número de pasos: ', num_steps)
    # print('Transitorio:, ', transient)
# =============================================================
    #Condiciones iniciales (cercanas a los puntos de equilibrio)
    x0, y0, z0 = -x_eq, y_eq, z_eq
    #sigma = 10
    #rho = 28
    #beta = 8/3

    #sigma = 4.0
    #rho = 45.92
    #beta = 0.16
# =============================================================
    # Tiempo de integración
    #t_start = 0
    #t_end = 50
    #t_step = 0.01
    #num_steps = int((t_end - t_start) / t_step)

    sol = np.zeros((num_steps+1, 3))
    sol[0] = [x0, y0, z0]

    # Resolver el sistema de Lorenz usando Forward Euler
    # for i in range(num_steps):
    #     dx, dy, dz = lorenz_equations(sol[i], sigma, rho, beta)
    #     sol[i+1] = sol[i] + t_step * np.array([dx, dy, dz])

    # Resolver el sistema de Lorenz usando Runge-Kutta 4th
    for i in range(num_steps):
        sol[i+1] = runge_kutta4(sol[i],t_step,sigma,rho,beta)
        
        # Condición de paro por desbordamiento
        if (sol[i+1,0]>1E3 or sol[i+1,0]<-1E3):
            sumax = 1
            break
# =============================================================
    # Transformada de Fourier
    # Quitar el transitorio
    sol2 = sol[transient:,:]
    # t2 = np.linspace(t_step*transient, t_step*num_steps, num_steps+1-transient)

    # dt2 = t2[1] - t2[0]

    Y2 = fft(sol2[:,0]) / (num_steps+1-transient)  # Transformada normalizada
    # Y3 = fft(sol2[:,1]) / (num_steps+1-transient)  # Transformada normalizada
    Y4 = fft(sol2[:,2]) / (num_steps+1-transient)  # Transformada normalizada
    # frq2 = fftfreq(num_steps+1-transient, dt2) 
    sumax = sum(abs(Y2))    
    # sumay = sum(abs(Y3))    
    # sumaz = sum(abs(Y4))    
    
    # Encuentra los picos
    peaksx, _ = find_peaks(np.abs(Y2[0:len(sol2[:,0])//2]))
    # peaksy, _ = find_peaks(np.abs(Y3[0:len(sol2[:,1])//2]))
    peaksz, _ = find_peaks(np.abs(Y4[0:len(sol2[:,2])//2]))
    num_peaksx = len(peaksx)
    # num_peaksy = len(peaksy)
    # num_peaksz = len(peaksz)

    # Encuentra picos a partir de un umbral
    # Calcular la magnitud de la transformada de Fourier
    # magnitudex = np.abs(Y2)
    # magnitudey = np.abs(Y3)
    magnitudez = np.abs(Y4)

    # Encontrar los índices de los picos que sobrepasan un valor específico
    threshold = 6
    # peaks_thrx, _ = find_peaks(magnitudex, height=threshold)
    # peaks_thry, _ = find_peaks(magnitudey, height=threshold)
    peaks_thrz, _ = find_peaks(magnitudez, height=threshold)
    
    # num_peaks_thrx = len(peaks_thrx)
    # num_peaks_thry = len(peaks_thry)
    num_peaks_thrz = len(peaks_thrz)
    
   #Calcular el valor más alto que alcanzan los picos
    # max_peak_value_x = np.max(magnitudex[peaksx])
    # max_peak_value_y = np.max(magnitudey[peaksy])
    # 

    # print("El valor más alto que alcanzan los picos para x es", max_peak_value_x)
    # print("El valor más alto que alcanzan los picos para y es", max_peak_value_y)
    # print("El valor más alto que alcanzan los picos para z es", max_peak_value_z)

# =============================================================
# Condición para descartar evoluciones temporales periódicas       
    # Valores propios conjugados con parte real positiva
    # Potencia de la transformada de Fourier mayor a un determinado valor
    
    if(sumax > 150 and num_peaksx > 200):
    # WOLF
        #lyap_specPATH = './a.out 10 0 30 ' + str(X[0]) + ' ' + str(X[1]) + ' ' + str(X[2]) + ' 0.001 1 2000 0.01'
        lyap_specPATH = './a.out ' + str(-x_eq) + ' ' + str(y_eq) + ' ' + str(z_eq)+ ' ' + str(X[0]) + ' ' + str(X[1]) + ' ' + str(X[2]) + ' ' + str(t_step) + ' ' + str(transient) + ' 2000 0.01'
        
        # Ejecuta el comando y captura la salida estándar
        result = subprocess.run(lyap_specPATH, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # Procesa la salida
        output = result.stdout.decode('utf-8').strip()
        parts = output.split()

        # Inicializa lyapunov_array
        lyapunov_array = np.array([0.0, 0.0, 0.0])

        # Extrae los exponentes de Lyapunov
        lyapunov_exponents = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
        lyapunov_array = np.array(lyapunov_exponents)
        
        # Ordena los exponentes de Lyapunov de mayor a menor
        lyapunov_sorted = np.sort(lyapunov_array)[::-1]

        # Encuentra el valor de j
        j = 0
        sum_lyapunov = 0
        for exponent in lyapunov_sorted:
            if sum_lyapunov + exponent > 0:
                sum_lyapunov += exponent
                j += 1
            else:
                break

        # Calcula la dimensión de Kaplan-Yorke
        if j < len(lyapunov_sorted):
            DKY = j + sum_lyapunov / abs(lyapunov_sorted[j])
        else:
            DKY = j  # En caso de que todos los exponentes sean positivos
        #print(lyapunov_array, DKY)
    
        # max_peak_value_z = np.max(magnitudez[peaksz])
        
        # if(max_peak_value_z > 10.5 or num_peaks_thrz > 4):
        #     DKY = -1.0
            
    else: DKY = -1.0 
                

    # print("Variables",X)
    # print("Potencia", suma)
    # print("Valores propios", eigenvalues)
    # print("Dimension Kaplan-Yorke",-DKY_mean)
    
    #Graficar 
    # label_potx = 'pot=' + str(np.round(sumax,2)) + ',peaks=' + str(num_peaksx) + ',' + str(num_peaks_thrx)
    # label_poty = 'pot=' + str(np.round(sumay,2)) + ',peaks=' + str(num_peaksy) + ',' + str(num_peaks_thry)
    # label_potz = 'pot=' + str(np.round(sumaz,2)) + ',peaks=' + str(num_peaksz) + ',' + str(num_peaks_thrz)
    # label_eig = str(np.round(eigenvalues.real, 2) + np.round(eigenvalues.imag, 2) * 1j)
    # label_dky = str(np.round(-DKY_mean,4))
    # label_h = str(t_step)
    # title = str(np.round(X,2))
    # #file_name = str(j) + ".png"
    
    # fig = plt.figure(figsize=(10, 8))

    # ax1 = fig.add_subplot(321)
    # ax1.plot(sol[:,0], sol[:,2])
    # plt.legend([label_dky])
    # plt.xlabel('Eje x')
    # plt.ylabel('Eje z')
    # plt.title(title)
    # plt.grid(True)

    # ax2 = fig.add_subplot(322)
    # ax2.scatter(eigenvalues.real, eigenvalues.imag, marker="o")
    # plt.legend([label_eig])
    # plt.xlabel('Re')
    # plt.ylabel('Im')
    # plt.axhline(0, color="black")
    # plt.axvline(0, color="black")

    # ax1 = fig.add_subplot(323)
    # ax1.plot(t2, sol2[:,0])
    # plt.legend([label_h])
    # ax1.set_xlabel('Tiempo (s)')
    # ax1.set_ylabel('$x(t)$')

    # ax2 = fig.add_subplot(324)
    # ax2.vlines(frq2, 0, np.abs(Y2.imag))
    # plt.legend([label_potx])
    # plt.xlim(-10, 10)
    # plt.xlabel('Frecuencia (Hz)')
    # plt.ylabel('Im($Y_x$)')
    
    # ax2 = fig.add_subplot(325)
    # ax2.vlines(frq2, 0, np.abs(Y3.imag))
    # plt.legend([label_poty])
    # plt.xlim(-10, 10)
    # plt.xlabel('Frecuencia (Hz)')
    # plt.ylabel('Im($Y_y$)')
    
    # ax2 = fig.add_subplot(326)
    # ax2.vlines(frq2, 0, np.abs(Y4.imag))
    # plt.legend([label_potz])
    # plt.xlim(-10, 10)
    # plt.xlabel('Frecuencia (Hz)')
    # plt.ylabel('Im($Y_z$)')
    
    # fig.tight_layout()
    # plt.show()
    
    # DKY es la funcion objetivo que se busca maximizar
    return -DKY

# Definir el problema de optimización
class LorenzProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=0, n_eq_constr=0, xl=[0.001, 0.001, 0.001], xu=[60, 180, 30])
        
        # Espacios de busqueda
        # sigma = X[0]  #0.001,60
        # rho = X[1]    #0.001,180
        # beta = X[2]   #0.001,30

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([lorenz_objective(x) for x in X])
        
        
# Guarda la salida estándar original
original_stdout = sys.stdout

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

n_gen = 20
problem = LorenzProblem()
algorithm = DE(pop_size=20,
                initial_velocity='zero')
termination = get_termination("n_gen", n_gen)

random_seed = np.random.randint(1,20,size=10)

for run in range(0,10):
    f = open("progress_de_polaco"+ str(run) + ".txt", 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    res = minimize(problem,
                algorithm,
                termination,
                seed=random_seed[run],
                save_history=True,
                verbose=True)

    # Obtener las soluciones óptimas
    solutions = res.X[0],res.X[1],res.X[2]
    objectives = res.F

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} segundos")

    # Restaurar stdout
    sys.stdout = original
    f.close()

    last_generation = res.history[-1]
    solutions_last_gen = last_generation.pop.get('X')

    salida = np.concatenate((solutions_last_gen, last_generation.pop.get("F")), axis=1)
    output_file = "last_gen_de_polaco" + str(run) + ".dat"
    np.savetxt(output_file, salida, delimiter="  ")