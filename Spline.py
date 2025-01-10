# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:33:55 2025

@author: Patrick
"""
#%% Libraries
import numpy as np
import scipy.interpolate as spi
import scipy.optimize
from scipy.optimize import minimize
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
a_min = 0.05    #Minimum Wealth Level
a_max = 15      #Maximum Wealth Level
N_a = 150       #Number of Gridpoints
#---- Chebyshev Grid of assets
cheby_nodes = []
for k in range(0,N_a):
    cheby_nodes.append(np.cos((2*k + 1) / (2*N_a) * np.pi))
a_grid = np.sort(0.5*(a_max+a_min) + 0.5*(a_max-a_min)*np.array(cheby_nodes))
"""
Here you find two alternatives apart from the Chebyshev Grid

#---- 1) Exponentially spaced grids for assets
a_grid = np.logspace(np.log10(a_min), np.log10(a_max), N_a) 
#-------------------------------------------------

#----- 2) Linearly Spaced Grid
a_grid = np.linspace(a_min,a_max,N_a)
#-------------------------------------------------
"""

#---------------------------- Time Grid ---------------------------------------
T=1                                 #Terminal Time
N_t = 50                            #Number of points in the time grid
time_grid = np.linspace(T,0,N_t+1)  #Linearly spaced time Grid
dt = time_grid[0] - time_grid[1]    #Time Increment

#%%  Terminal Condition at t=T
v_terminal  = u(a_grid) #y-values of Spline at t=T
V_spline_terminal = spi.CubicSpline(a_grid,v_terminal) #Spline approximation of terminal condition

#Plot the Fit of the Spline
plt.plot(np.linspace(a_min,a_max,N_a*10), V_spline_terminal(np.linspace(a_min,a_max,N_a*10)), 
         label = 'Spline Approximation', linestyle='dashed', linewidth=3)
plt.plot(np.linspace(a_min,a_max,N_a*10),u(np.linspace(a_min,a_max,N_a*10)), label='V_T(a)')
plt.legend(fontsize=16)
plt.title("True V(T,a) vs. Spline Approximation")
plt.savefig(path + "Figures/Terminal_Value_Comparison.pdf", dpi=600)
plt.show()

#%% PDE Loss Functions
def loss_root_semi_implicit(y,V_spline_n):
    
    #Derivative of future Value Function
    V_prime_n = V_spline_n.derivative(1)
    V_double_prime_n = V_spline_n.derivative(2)
    
    # Optimal controls as a function of the future Value Function
    c_n = (V_prime_n(a_grid)**(-1 / gamma))  # Optimal consumption
    pi_n = -eta / (sigma*a_grid) * V_prime_n(a_grid)/V_double_prime_n(a_grid)  # Optimal portfolio

    #Current Value Function Spline and its derivatives
    V_spline = spi.CubicSpline(a_grid, y) #= V^{n+1}_{s}
    V_prime = V_spline.derivative(1)
    V_double_prime = V_spline.derivative(2)

    #HJB Loss
    loss = ((V_spline_n(a_grid)-V_spline(a_grid))/dt + u(c_n) 
            + (r*a_grid + pi_n*eta*sigma*a_grid - c_n)*V_prime(a_grid) 
            + 0.5*(pi_n*sigma*a_grid)**2*V_double_prime(a_grid) 
            - rho*V_spline(a_grid)
            )
    return loss

def loss_root_implicit(y,V_spline_n):
    
    #Current Value Function Spline and its derivatives
    V_spline = spi.CubicSpline(a_grid, y) #= V^{n+1}_{s}
    V_prime = V_spline.derivative(1)
    V_double_prime = V_spline.derivative(2)
    
    # Optimal controls as a function of the future Value Function
    c_n1 = (V_prime(a_grid)**(-1 / gamma))  # Optimal consumption
    pi_n1 = -eta / (sigma*a_grid) * V_prime(a_grid)/V_double_prime(a_grid)  # Optimal portfolio

    #HJB Loss
    loss = ((V_spline_n(a_grid)-V_spline(a_grid))/dt + u(c_n1) 
            + (r*a_grid + pi_n1*eta*sigma*a_grid - c_n1)*V_prime(a_grid) 
            + 0.5*(pi_n1*sigma*a_grid)**2*V_double_prime(a_grid) 
            - rho*V_spline(a_grid)
            )
    return loss

#%% Solve the Value Functions numerically with Cubic Splines

#Pick your routine (semi-implicit or fully implicit)
routine = loss_root_semi_implicit #Change to loss_root_implicit for fully implicit method

#Define Lists to store the resolts
Value_Functions = [V_spline_terminal]
roots = [v_terminal]

#------Do the 1st iteration
print("Iteration: n = 0" + ", t = " + str(time_grid[1]))
#Initialise V^1
y0 = roots[0]
V_spline_n = Value_Functions[0]

#Solve the Root-finding Problem at the collocation points a_grid
root = scipy.optimize.root(routine,y0,args=(V_spline_n))

#Store Results
y = root.x
Value_Functions.append(spi.CubicSpline(a_grid, y)) #V^1
roots.append(root)

#Print Error
print("Average L2 Error: " + str(np.linalg.norm(root.fun)/len(root.fun)))
print("Supremum Error: " + str(np.max(np.abs(root.fun))))
"""
It is generally advised to also print the Supremum error because this can reveal
that a convergence problem only exists with one specific gridpoint.
"""

#------Do all remaining iterations
for n in range(1,N_t):
    print("---------------------")
    print("Iteration: n = " + str(n) + ", t = " + str(time_grid[n+1]))
    
    #Initliase V^{n+1}
    yn = roots[n].x
    V_spline_n = Value_Functions[n]

    #Solve the Root-finding Problem at the collocation points a_grid
    root = scipy.optimize.root(routine,yn,args=(V_spline_n))
   
    #Store Results
    y = root.x
    Value_Functions.append(spi.CubicSpline(a_grid, y)) #V^{n+1}
    roots.append(root)
    
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
V_spline = Value_Functions[t_index]

#Extract numerically obtained policies
V_prime = V_spline.derivative(1)
V_double_prime = V_spline.derivative(2)

#------------------------- Analytical Solution ---------------------------------
V_analytical,  V_prime_analytical,V_double_prime_analytical, _ , _ = analytical_solution(t,a_grid)


#------------------------------ Plot Comparison -------------------------------
# Create a figure and a 1x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fontsize_title = 20
fontsize_legend = 16

# Plot Value Function Comparison
axes[0].plot(a_grid, V_spline(a_grid), label='Numerical', linestyle='dashed', linewidth=3)
axes[0].plot(a_grid, V_analytical, label='True')
axes[0].legend(fontsize=fontsize_legend)
axes[0].set_title("True vs. Numerical Value Function, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 1st Derivative Comparison
axes[1].plot(a_grid, V_prime(a_grid), label='1st Derivative Numerical', linestyle='dashed', linewidth=3)
axes[1].plot(a_grid, V_prime_analytical, label='1st Derivative Analytical')
axes[1].legend(fontsize=fontsize_legend)
axes[1].set_title("True vs. Numerical 1st Derivative, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 2nd Derivative Comparison
axes[2].plot(a_grid, V_double_prime(a_grid), label='2nd Derivative Numerical', linestyle='dashed', linewidth=3)
axes[2].plot(a_grid, V_double_prime_analytical, label='2nd Derivative Analytical')
axes[2].legend(fontsize=fontsize_legend)
axes[2].set_title("True vs. Numerical 2nd Derivative, t = " + str(t), fontsize=fontsize_title)

# Adjust layout
plt.tight_layout()
plt.savefig(path + "Figures/VF_Comparison.pdf",dpi = 600, bbox_inches='tight')
plt.show()


#--------------------- Plot Comparison Solution larger a_min ------------------
truncation = 0.5
a_grid_truncated = a_grid[a_grid>truncation]
V_analytical,  V_prime_analytical,V_double_prime_analytical, _, _ = analytical_solution(t, a_grid_truncated)

# Create a figure and a 1x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Plot Value Function Comparison
axes[0].plot(a_grid_truncated, V_spline(a_grid_truncated), label='Numerical', linestyle='dashed', linewidth=3)
axes[0].plot(a_grid_truncated, V_analytical, label='True')
axes[0].legend(fontsize=fontsize_legend)
axes[0].set_title("True vs. Numerical Value Function, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 1st Derivative Comparison
axes[1].plot(a_grid_truncated, V_prime(a_grid_truncated), label='1st Derivative Numerical', linestyle='dashed', linewidth=3)
axes[1].plot(a_grid_truncated, V_prime_analytical, label='1st Derivative Analytical')
axes[1].legend(fontsize=fontsize_legend)
axes[1].set_title("True vs. Numerical 1st Derivative, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 2nd Derivative Comparison
axes[2].plot(a_grid_truncated, V_double_prime(a_grid_truncated), label='2nd Derivative Numerical', linestyle='dashed', linewidth=3)
axes[2].plot(a_grid_truncated, V_double_prime_analytical, label='2nd Derivative Analytical')
axes[2].legend(fontsize=fontsize_legend)
axes[2].set_title("True vs. Numerical 2nd Derivative, t = " + str(t), fontsize=fontsize_title)

# Adjust layout
plt.tight_layout()
plt.savefig(path + "Figures/VF_Comparison_Truncated.pdf",dpi = 600, bbox_inches='tight')
plt.show()

#%% Plot Policies and compare with analytical solution

#Select time point to plot
t=0
t_index = int(np.abs(time_grid - t).argmin())

#------------------------- Numerical Solution ---------------------------------

#Extract numerically obtained V(t,a)
V_spline = Value_Functions[t_index]

#Extract numerically obtained policies
V_prime = V_spline.derivative(1)
V_double_prime = V_spline.derivative(2)

c_spline = (V_prime(a_grid)**(-1 / gamma))  # Optimal consumption
pi_spline = -eta / (sigma*a_grid) * V_prime(a_grid)/V_double_prime(a_grid) #Optimal pi

#------------------------- Analytical Solution ---------------------------------
_, _, _, c_analytical, pi_analytical = analytical_solution(t, a_grid)

# Create a figure and a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot Value Function Comparison
axes[0].plot(a_grid, c_spline, label='Numerical', linestyle='dashed', linewidth=3)
axes[0].plot(a_grid, c_analytical, label='Analytical')
axes[0].legend(fontsize=fontsize_legend)
axes[0].set_title("True vs. Numerical consumption, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 1st Derivative Comparison
axes[1].plot(a_grid, pi_spline, label='Numerical', linestyle='dashed', linewidth=3)
axes[1].plot(a_grid, pi_analytical, label='Analytical')
axes[1].legend(fontsize=fontsize_legend)
axes[1].set_title("True vs. Numerical Value pi, t = " + str(t), fontsize=fontsize_title)
plt.savefig(path + "Figures/Policy_Comparison.pdf",dpi = 600, bbox_inches='tight')
plt.show()

#--------------------- Plot Comparison Solution truncated a_grid---------------
truncation_min = 0.5
truncation_max = 10.5
a_grid_truncated = a_grid[a_grid>truncation_min]
a_grid_truncated = a_grid_truncated[a_grid_truncated<truncation_max]

c_spline = (V_prime(a_grid_truncated)**(-1 / gamma))  # Optimal consumption
pi_spline = -eta / (sigma*a_grid_truncated) * V_prime(a_grid_truncated)/V_double_prime(a_grid_truncated) #Optimal pi

_, _, _, c_analytical, pi_analytical = analytical_solution(t, a_grid_truncated)

# Create a figure and a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot Value Function Comparison
axes[0].plot(a_grid_truncated, c_spline, label='Numerical', linestyle='dashed', linewidth=3)
axes[0].plot(a_grid_truncated, c_analytical, label='Analytical')
axes[0].legend(fontsize=fontsize_legend)
axes[0].set_title("True vs. Numerical consumption, t = " + str(t), fontsize=fontsize_title)

# Plot Value Function 1st Derivative Comparison
axes[1].plot(a_grid_truncated, pi_spline, label='Numerical', linestyle='dashed', linewidth=3)
axes[1].plot(a_grid_truncated, pi_analytical, label='Analytical')
axes[1].legend(fontsize=fontsize_legend)
axes[1].set_title("True vs. Numerical Value pi, t = " + str(t), fontsize=fontsize_title)
plt.savefig(path + "Figures/Policy_Comparison_Truncated.pdf",dpi = 600, bbox_inches='tight')
plt.show()

#%% Plot the Evolution of the Value Function

for VF in Value_Functions:
    V_spline = VF
    threshold = a_min
    plt.plot(a_grid[a_grid > threshold], V_spline(a_grid[a_grid > threshold]))
plt.title('Evolution Numerical Value Function')
plt.savefig(path + "Figures/Value_Function_Evolution.pdf")
plt.show()

#%% Compute Jacobian of loss function with finite differences
"""
This shows that when using splines the Jacobian of the loss function is sparse,
i.e. the effects of changing the spline at one point are locally bounded. This
makes it very flexible.
"""
def compute_jacobian(y, V_spline_n, epsilon=1e-6):
    # Size of y
    n = len(y)  
    # Compute the loss at y
    loss_0 = routine(y, V_spline_n)  
    # Initialize Jacobian matrix
    jacobian = np.zeros((n, n))  
    
    for j in range(n):
        y_perturbed = y.copy()
        
        # Perturb the j-th element of y
        y_perturbed[j] += epsilon  
        
        # Compute loss with perturbed y
        loss_perturbed = routine(y_perturbed, V_spline_n)  
        
        # Approximate derivative. j-th column is change of y_j
        jacobian[:, j] = (loss_perturbed - loss_0) / epsilon  
    
    return jacobian

Jac = compute_jacobian(y, Value_Functions[-1], epsilon=1e-6)
