# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:05:01 2025

@author: Patrick
"""

#%% Libraries
import numpy as np
from numpy.polynomial import chebyshev
import scipy.optimize
import matplotlib.pyplot as plt

path = "C:/Users/pf122/Desktop/Uni/Frankfurt/2023-24/Machine Learning/Merton Model Num Sol/"

#%% Define parameters as globals
rho     = 0.05
gamma   = 2
r       = 0.02
mu      = 0.06
sigma   = 0.2
eta     = (mu-r)/sigma #Sharpe Ratio

#%% Utility Functions
def u(c):
    if gamma == 1:
        return np.log(c)
    else:
        return c**(1-gamma)/(1-gamma)
    
def u_prime_inverse(x):
    if gamma == 1:
        return 1/x
    else:
        return x**(-1/gamma)

#%% Grids
#---------------------------- Asset Grid --------------------------------------
a_min = 0.8    #Minimum Wealth Level  
a_max = 10      #Maximum Wealth Level
N_a = 30       #Number of Gridpoints
#---- Chebyshev Grid of assets


def asset_grid(N_a,a_min,a_max):
    cheby_nodes = []
    for k in range(0,N_a):
        cheby_nodes.append(np.cos((2*k + 1) / (2*N_a) * np.pi))
    cheby_nodes = np.sort(cheby_nodes)
    
    a_grid = 0.5*(a_max+a_min) + 0.5*(a_max-a_min)*np.array(cheby_nodes)
    return cheby_nodes, a_grid

cheby_nodes, a_grid = asset_grid(N_a,a_min,a_max)

#---------------------------- Time Grid ---------------------------------------
T=1                                 #Terminal Time
N_t = 50                            #Number of points in the time grid
time_grid = np.linspace(T,0,N_t+1)  #Linearly spaced time Grid
dt = time_grid[0] - time_grid[1]    #Time Increment

#%%  Terminal Condition at t=T
v_terminal = u(a_grid)

def cheby_terminal_root(theta):
    V_cheby = chebyshev.Chebyshev(theta)
    return v_terminal - V_cheby(cheby_nodes)

theta_0 = np.ones(N_a)*0.1
root = scipy.optimize.root(cheby_terminal_root, theta_0)
theta_0 = root.x


#Plot Results
plot_cheby_nodes, plot_a_grid = asset_grid(200,a_grid[0],a_grid[-1])

V_cheby_terminal = chebyshev.Chebyshev(theta_0)
plt.plot(plot_a_grid, V_cheby_terminal(plot_cheby_nodes), label='Chebyshev', 
         linestyle='dashed', linewidth=3)
plt.plot(plot_a_grid, u(plot_a_grid), label='V_T(a)')
plt.legend(fontsize=16)
plt.title("True V(T,a) vs. Spline Approximation")
plt.savefig(path + "Figures/Terminal_Value_Comparison_Cheby.pdf", dpi=600)
plt.show()
del plot_cheby_nodes, plot_a_grid
"""
For a_min and a_max as in the cubic spline implementation the chebyshev polynomial
was no longer monotone. This is bad because consumption vitally hinges on a
positive derivative of the Value Function.
"""

#%% PDE Loss Functions

def loss_root_implicit(theta_n1,theta_n):
    
    #Value Function V^{n+1}
    V_cheby = chebyshev.Chebyshev(theta_n1)
    dV_cheby = V_cheby.deriv()*2/(a_max-a_min)
    ddV_cheby = dV_cheby.deriv()*(2/(a_max-a_min))**2

    #Value Function V^n
    V_cheby_n = chebyshev.Chebyshev(theta_n)
    
    #Compute optimal policies as function of V^n at collocation points
    c_n1 = u_prime_inverse(dV_cheby(cheby_nodes))
    pi_n1 = -eta*dV_cheby(cheby_nodes)/(sigma*a_grid*ddV_cheby(cheby_nodes))
    
    #Evaluate PDE Loss
    loss = ((V_cheby_n(cheby_nodes)-V_cheby(cheby_nodes))/dt + u(c_n1) 
            + (r*a_grid + pi_n1*eta*sigma*a_grid - c_n1)*dV_cheby(cheby_nodes) 
            + 0.5*(pi_n1*sigma*a_grid)**2*ddV_cheby(cheby_nodes) 
            - rho*V_cheby(cheby_nodes))
    
    return loss 

def loss_root_semi_implicit(theta_n1,theta_n):
    
    #Value Function V^{n+1}
    V_cheby = chebyshev.Chebyshev(theta_n1)
    dV_cheby = V_cheby.deriv()*2/(a_max-a_min)
    ddV_cheby = dV_cheby.deriv()*(2/(a_max-a_min))**2

    #Value Function V^n
    V_cheby_n = chebyshev.Chebyshev(theta_n)
    dV_cheby_n = V_cheby_n.deriv()*2/(a_max-a_min)
    ddV_cheby_n = dV_cheby_n.deriv()*(2/(a_max-a_min))**2
    
    #Compute optimal policies as function of V^n at collocation points
    c_n = u_prime_inverse(dV_cheby_n(cheby_nodes))
    pi_n = -eta*dV_cheby_n(cheby_nodes)/(sigma*a_grid*ddV_cheby_n(cheby_nodes))
    
    #Evaluate PDE Loss
    loss = ((V_cheby_n(cheby_nodes)-V_cheby(cheby_nodes))/dt + u(c_n) 
            + (r*a_grid + pi_n*eta*sigma*a_grid - c_n)*dV_cheby(cheby_nodes) 
            + 0.5*(pi_n*sigma*a_grid)**2*ddV_cheby(cheby_nodes) 
            - rho*V_cheby(cheby_nodes))
    
    return loss 

#%% Solve the Value Functions numerically with Chebyshev Polynomials

"""
Pick the root finding routine here:
    semi-implicit or
    fully-implicit method
"""
#Pick your routine (semi-implicit or fully implicit)
routine = loss_root_semi_implicit #Change to loss_root_implicit for fully implicit method

#Define Lists to store the resolts
Value_Functions = [chebyshev.chebval(cheby_nodes, theta_0)]
Thetas = [theta_0]

#------ Solve Value Functions iteratively 
print(str(routine))
for n in range(0,N_t):
    print("---------------------")
    print("Iteration: " + str(n) + ", t = " + str(time_grid[n+1]))
    
    #Initliase V^{n+1} as V^n
    theta_n = Thetas[n]
    V_cheby = Value_Functions[n]

    #Solve the Root-finding Problem at the collocation points 
    root = scipy.optimize.root(routine, theta_n, args=(theta_n))
   
    #Store Results
    theta_n1 = root.x
    Value_Functions.append(chebyshev.Chebyshev(theta_n1))
    Thetas.append(theta_n1)
    
    #Print Error
    print("Average L2 Error: " + str(np.linalg.norm(root.fun)/len(root.fun)))
    print("Supremum Error: " + str(np.max(np.abs(root.fun))))

#%% Compute the analytical solution

def analytical_solution(t,a_grid):
    r_tilde = (rho-(1-gamma)*r - 0.5* (1-gamma)/gamma * eta**2)/gamma
    f_t = 1/r_tilde *(1-np.exp(-r_tilde*(T-t))) + np.exp(-r_tilde*(T-t))
    
    V_analytical = u(a_grid)*f_t**gamma
    V_prime_analytical = a_grid**(-gamma)*f_t**gamma
    V_double_prime_analytical = -gamma  * a_grid**(-gamma-1)*f_t**gamma
    
    c_analytical = a_grid/f_t
    pi_analytical = np.array([1/gamma * eta/sigma]*len(a_grid))
    
    return V_analytical, V_prime_analytical,V_double_prime_analytical, c_analytical, pi_analytical

#%% Plot Value Function and compare with analytical solution

#Select time point to plot
t=0
t_index = int(np.abs(time_grid - t).argmin()) #extract time_index to select V^n

#------------------------- Numerical Solution ---------------------------------
#Extract numerically obtained V(t,a)
V_cheby = Value_Functions[t_index]

#Extract Derivatives
V_prime = V_cheby.deriv()*2/(a_max-a_min)
V_double_prime = V_prime.deriv()*(2/(a_max-a_min))**2

#------------------------- Analytical Solution ---------------------------------

#Create Fine Grid for Plotting
plot_cheby_nodes, plot_a_grid = asset_grid(200,a_grid[0],a_grid[-1])

V_analytical,  V_prime_analytical,V_double_prime_analytical, _ , _ = analytical_solution(t,plot_a_grid)


#------------------------------ Plot Comparison -------------------------------

# Create a figure and a 1x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fontsize_title = 20
fontsize_legend = 16

# Plot Value Function Comparison
axes[0].plot(plot_a_grid, V_cheby(plot_cheby_nodes), label='Numerical', linestyle='dashed', linewidth=3)
axes[0].plot(plot_a_grid, V_analytical, label='True')
axes[0].legend(fontsize=fontsize_legend)
axes[0].set_title("True vs. Numerical Value Function, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 1st Derivative Comparison
axes[1].plot(plot_a_grid, V_prime(plot_cheby_nodes), label='1st Derivative Numerical', linestyle='dashed', linewidth=3)
axes[1].plot(plot_a_grid, V_prime_analytical, label='1st Derivative Analytical')
axes[1].legend(fontsize=fontsize_legend)
axes[1].set_title("True vs. Numerical 1st Derivative, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 2nd Derivative Comparison
axes[2].plot(plot_a_grid, V_double_prime(plot_cheby_nodes), label='2nd Derivative Numerical', linestyle='dashed', linewidth=3)
axes[2].plot(plot_a_grid, V_double_prime_analytical, label='2nd Derivative Analytical')
axes[2].legend(fontsize=fontsize_legend)
axes[2].set_title("True vs. Numerical 2nd Derivative, t = " + str(t), fontsize=fontsize_title)

# Adjust layout
plt.tight_layout()
plt.savefig(path + "Figures/VF_Comparison_Cheby.pdf",dpi = 600, bbox_inches='tight')
plt.show()

#%% Plot Policies and compare with analytical solution

#Select time point to plot
t=0
t_index = int(np.abs(time_grid - t).argmin())

#------------------------- Numerical Solution ---------------------------------

#Create Fine Grid for Plotting
plot_cheby_nodes, plot_a_grid = asset_grid(200,a_grid[0],a_grid[-1])

#Extract numerically obtained V(t,a)
V_cheby = Value_Functions[t_index]

#Extract numerically obtained policies
V_prime = V_cheby.deriv()*2/(a_max-a_min)
V_double_prime = V_prime.deriv()*(2/(a_max-a_min))**2

c_cheby = (V_prime(plot_cheby_nodes)**(-1 / gamma))  # Optimal consumption
pi_cheby = -eta / (sigma*plot_a_grid) * V_prime(plot_cheby_nodes)/V_double_prime(plot_cheby_nodes) #Optimal pi


#------------------------- Analytical Solution ---------------------------------

_, _, _, c_analytical, pi_analytical = analytical_solution(t, plot_a_grid)

# Create a figure and a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot Value Function Comparison
axes[0].plot(plot_a_grid, c_cheby, label='Numerical', linestyle='dashed', linewidth=3)
axes[0].plot(plot_a_grid, c_analytical, label='Analytical')
axes[0].legend(fontsize=fontsize_legend)
axes[0].set_title("True vs. Numerical consumption, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 1st Derivative Comparison
axes[1].plot(plot_a_grid, pi_cheby, label='Numerical', linestyle='dashed', linewidth=3)
axes[1].plot(plot_a_grid, pi_analytical, label='Analytical')
axes[1].legend(fontsize=fontsize_legend)
axes[1].set_title("True vs. Numerical Value pi, t = " + str(t), fontsize=fontsize_title)
plt.savefig(path + "Figures/Policy_Comparison_Cheby.pdf",dpi = 600, bbox_inches='tight')
plt.show()

#%% Plot the Evolution of the Value Function

#Create Fine Grid for Plotting
plot_cheby_nodes, plot_a_grid = asset_grid(200,a_grid[0],a_grid[-1])

for VF in Value_Functions[1:]:
    V_cheby = VF
    plt.plot(plot_a_grid, V_cheby(plot_cheby_nodes))
plt.title('Evolution Numerical Value Function')
plt.savefig(path + "Figures/Value_Function_Evolution_Cheby.pdf")
plt.show()