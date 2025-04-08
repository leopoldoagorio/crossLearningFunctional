#%%
import cvxpy as cp
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
#import os
import matplotlib.pyplot as plt
from lib.models import *
from lib.configs import datasets, argfull#, countries, estimator_vals,logg_every_e

from scipy.integrate import odeint
from scipy.optimize import curve_fit, minimize, least_squares

from tqdm import tqdm
import datetime
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

def crossLearningProject(x,u, epsilon):
    z = {country: cp.Variable(2) for country in x.keys()}
    #zg = cp.Variable(2)

    # Objective function
    objective = cp.Minimize(
        sum(cp.norm(x[country] + u[country] - z[country], 2)**2 for country in x.keys())
    )

    # Constraints
    constraints = [cp.norm(z[country] - z['GLOBAL'], 2) <= epsilon for country in x.keys()]

    # Add non-negativity constraints
    for country in x.keys():
        constraints.append(z[country][0] >= 0)
        constraints.append(z[country][1] >= 0)
    constraints.append(z['GLOBAL'][0] >= 0)
    constraints.append(z['GLOBAL'][1] >= 0)

    # Problem definition and solving
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return {country: z[country].value for country in x.keys()}

def cutdata(I_data,proportion=0.6):
    # Cut data to include, Imax, and up until I = 0.9*Imax
    Imax = np.max(I_data)
    tmax = np.argmax(I_data)
    t2 = np.where(I_data > proportion * Imax)[0][-1]
    I_data = I_data[:t2]
    return I_data,t2

class SIRModel:
    def __init__(self, y0,I_data, country='Fakistan'):
        self.y0 = y0  # Initial conditions (S0, I0, R0)
        self.I_data = I_data
        self.t_data = np.arange(len(I_data))  # Time data
        self.norm2Idata = np.sum((I_data)**2)
        self.norm1Idata = np.sum(np.abs(I_data))
        self.country = country
        self.beta = None  # To store fitted beta
        self.gamma = None  # To store fitted gamma

    # SIR model equations
    def sir_model(self, y, t, beta, gamma):
        S, I, R = y
        N = S + I + R  # Total population
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return dS, dI, dR

    # Function to compute the SIR rollout (S, I, R over time)
    def rollout_sir(self, t, beta, gamma):
        #beta, gamma = params
        ret = odeint(self.sir_model, self.y0, t, args=(beta, gamma))
        S, I, R = ret.T
        return S, I, R

    # Objective function for fitting: sum of squared differences between observed and model data
    def Iloss(self, x,norm='L2'):
        beta, gamma = x
        _, I_model, _ = self.rollout_sir(self.t_data,beta, gamma)
        #N = len(self.I_data)
        if norm == 'L2':
            error = np.sum((I_model - self.I_data) ** 2)/self.norm2Idata
        elif norm == 'L1':
            error = np.sum(np.abs(I_model - self.I_data))/self.norm1Idata
        else:
            raise ValueError("Unknown norm type. Use 'L1' or 'L2'.")
        return error

    def Iresidual(self, x):
        beta, gamma = x
        _, I_model, _ = self.rollout_sir(self.t_data, beta, gamma)
        return (I_model - self.I_data)#/np.sqrt(self.norm2Idata)

    # Objective function for fitting: sum of squared differences between observed and model data
    def regularizedIloss(self, x,z,u,lda):
        error = self.Iloss(x,norm='L2')+0.5*np.sum((x-z+u)**2)/lda
        return error

    def regularizedIresidual(self, x,z,u,lda):
        beta, gamma = x
        Ires = self.Iresidual(x)
        return np.append(Ires/np.sqrt(self.norm2Idata), (x-z+u)/np.sqrt(lda))

    def minimize_reg_loss_least_squares(self, z,u ,initial_guess=(0.3, 0.1),lda=1.0):
        result = least_squares(self.regularizedIresidual, initial_guess, args=(z,u,lda),bounds=([0,0],[np.inf,np.inf]))
        self.beta, self.gamma = result.x  # Store fitted beta and gamma
        return self.beta, self.gamma

    # Fit function to minimize the error using all three compartments (S, I, R)
    def minimize_reg_loss(self, z,u ,initial_guess=(0.3, 0.1),lda=1.0):
        result = minimize(self.regularizedIloss, initial_guess, args=(z,u,lda))
        self.beta, self.gamma = result.x  # Store fitted beta and gamma
        return self.beta, self.gamma

    def peak_lag_error(self, beta=None, gamma=None):
        if beta is not None and gamma is not None:
            self.beta = beta
            self.gamma = gamma
        # Find peak of observed data
        Imax = np.max(self.I_data)
        tmax = np.argmax(self.I_data)
        # Find peak of model data
        ts_rollout = np.arange(10*len(self.I_data))
        _, I_model, _ = self.rollout_sir(ts_rollout, self.beta, self.gamma)
        Imax_model = np.max(I_model)
        tmax_model = np.argmax(I_model)

        if tmax_model == len(ts_rollout)-1:
            print("Warning: tmax_model is at the end of the data. Consider increasing the data length.")

        # Compute peak error
        peak_error = np.abs((Imax - Imax_model))/Imax
        lag_error = np.abs(tmax - tmax_model)
        return peak_error, lag_error

    def plot_sir(self):
        t_fit = np.linspace(0, len(self.I_data), 1000)  # Time for plotting
        _, I_fit, _ = self.rollout_sir(t_fit, self.beta, self.gamma)

        # Plot Susceptible, Infected, and Recovered data and model fits

        #Plot Infected
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_data, self.I_data, 'o', label='Observed Infected')
        plt.plot(t_fit, I_fit, '-', label='Fitted Infected Curve')
        plt.xlabel('Time')
        plt.ylabel('Infected Population')
        plt.title('SIR Model Fitting: Infected Population for ' + self.country + ' (beta=%.4f, gamma=%.4f)' % (self.beta, self.gamma))
        plt.legend()
        plt.show()

def plot_params(models,z,y,w,epsilon):
    # plot parameters in parameter space
    plt.figure(figsize=(6,6))
    betas = [models[country].beta for country in countries]
    gammas = [models[country].gamma for country in countries]

    pglobal = np.asarray(z['GLOBAL'])
    plt.scatter(betas, gammas)
    plt.scatter(pglobal[0], pglobal[1], color='black', marker='x', label='Global')
    for country in countries:
        plt.text(models[country].beta, models[country].gamma, country)
    plt.text(pglobal[0], pglobal[1], 'Global')

    #circle of radius epsilon around global
    circle = plt.Circle(pglobal, epsilon, color='gray', fill=False, linestyle='dashed', alpha=0.5)

    # also plot zs for each country
    for country in countries:
        plt.scatter(z[country][0], z[country][1], color='red', marker='x', label=country + ' z')
        plt.text(z[country][0], z[country][1], country + ' z')
    plt.gca().add_patch(circle)

    #arrow from z to y
    #arrow from x (models) to w
    for country in countries:
        if country == 'URY' or country == 'PER':
            plt.arrow(z[country][0], z[country][1], y[country][0]-z[country][0], y[country][1]-z[country][1], head_width=0.005, head_length=0.005, fc='red', ec='red',linewidth=0.5)
            plt.text(y[country][0],y[country][1], country + ' y')
            plt.arrow(models[country].beta, models[country].gamma, w[country][0]-models[country].beta, w[country][1]-models[country].gamma, head_width=0.005, head_length=0.005, fc='black', ec='black',linewidth=0.5)
            plt.text(w[country][0], w[country][1], country + ' w')


    plt.grid()
    plt.xlabel('Beta')
    plt.ylabel('Gamma')
    plt.title('Parameter space')
    plt.show()

def global_residuals(params, models):
    residuals = np.array([])
    for country in models.keys():
        residuals = np.append(residuals, models[country].Iresidual(params))
    return np.array(residuals)

def minimize_global_loss(models):
    result = least_squares(global_residuals, initial_guess, args=(models,),bounds=([0,0],[np.inf,np.inf]))
    print(result)
    return result.x

#%%
# Load dataset
df = pd.read_csv('./data/owid-covid-data-old.csv')
np.random.seed(42)

#print(datasets)
# Extract data (assuming datasets format)
countries = [ 'USA', 'ESP', 'FRA','BRA','MEX','PRY', 'ITA', 'ARG', 'COL', 'CHL', 'PER', 'DEU', 'GBR', 'IND', 'JPN']
models = {}
doplots = False
for country in countries:
    y0 = datasets[country]['train']['array'][:, 0]  # Initial conditions (S0, I0, R0)
    I_data, _ = cutdata(datasets[country]['train']['array'][1, :],proportion=0.8)  # Cut data to include only up to 0.6*Imax
    models[country] = SIRModel(y0, I_data, country=country)

z = {}
u = {}
x = {}
for country in countries:
    z[country] = np.array([0.0, 0.0])
    u[country] = np.array([0.0, 0.0])
z['GLOBAL'] = np.array([0.0, 0.0])
u['GLOBAL'] = np.array([0.0, 0.0])

lda = 1.
#epsilon = 1.0#0.1
epsilons = np.linspace(0.0, 1.0, 101)
initial_guess = (2.5*1.205, 2.5*1.15)
Niter_short =50
Niter_long = 4*Niter_short # 200 iterations for long epsilon sweep

global_params = minimize_global_loss(models)

y0 = argfull['ARG']['train']['array'][:, 0]
I_data, _ = cutdata(argfull['ARG']['train']['array'][1, :],proportion=0.8)
modelargfull= SIRModel(y0, I_data, country='ARG')

modelargfull.beta = global_params[0]
modelargfull.gamma = global_params[1]
if True:
    peak_error_cent, lag_error_cent = modelargfull.peak_lag_error()
    print(f"Peak error cent: {peak_error_cent}, Lag error cent: {lag_error_cent}")
    #modelargfull.plot_sir()

#print("Global parameters: beta=%.4f, gamma=%.4f" % (global_params[0], global_params[1]))
#%%
converged = []
peak_errors = []
lag_errors = []

def admm_optimize(models_admm, epsilon_admm, Niter_admm, initial_guess_admm, lda_admm):
    z = {}
    u = {}
    x = {}
    for country in models_admm.keys():
        z[country] = np.array([0.0, 0.0])
        u[country] = np.array([0.0, 0.0])
    z['GLOBAL'] = np.array([0.0, 0.0])
    u['GLOBAL'] = np.array([0.0, 0.0])

    for k in range(Niter_admm):
        if k % 10 == 0:
            lda_admm = 0.9*lda_admm if k > 0 else lda_admm

        for country in models_admm.keys():
            x[country] = np.array(models_admm[country].minimize_reg_loss_least_squares(z[country], u[country], initial_guess_admm, lda_admm))
        y = {country: z[country] - u[country] for country in models_admm.keys()}
        w = {country: x[country] + u[country] for country in models_admm.keys()}
        x['GLOBAL'] = z['GLOBAL'] - u['GLOBAL']
        z = crossLearningProject(x, u, epsilon_admm)
        for country in countries:
            u[country] += x[country] - z[country]

    return x, z, y, w


for eps in tqdm(epsilons, desc="Epsilon Progress"):
    # if eps <0.1:
    #     Niter = Niter_long
    # else:
    #     Niter = Niter_short
    Niter = Niter_long
    x, z, y, w = admm_optimize(models, eps, Niter, initial_guess, lda)

    max_norm = max(np.linalg.norm(z[k] - x[k]) for k in z)
    print(f"Max norm: {max_norm}")
    mean_norm = np.mean([np.linalg.norm(z[k] - x[k]) for k in z])
    print(f"Mean norm: {mean_norm}")
    if max_norm > 0.05:#max(eps/10, 1e-4):
        print(f"Warning: max norm {max_norm} is greater than 0.05")
        converged_ = False
    else:
        converged_ = True
    converged.append(converged_)

    if eps==-1:
        plot_params(models,z,y,w,eps+1e-6)

        if True:
            y0 = argfull['ARG']['train']['array'][:, 0]
            I_data, _ = cutdata(argfull['ARG']['train']['array'][1, :],proportion=0.8)
            modelargfull= SIRModel(y0, I_data, country='ARG')
            modelargfull.beta = models['ARG'].beta
            modelargfull.gamma = models['ARG'].gamma
            peak_error, lag_error = modelargfull.peak_lag_error()
            print(f"Peak error: {peak_error}, Lag error: {lag_error}")
            modelargfull.plot_sir()

    modelargfull.beta = models['ARG'].beta#*0 + z['ARG'][0]
    modelargfull.gamma = models['ARG'].gamma# * 0 + z['ARG'][1]
    peak_error_, lag_error_ = modelargfull.peak_lag_error()
    peak_errors.append(peak_error_)
    lag_errors.append(lag_error_)
    print(f"Epsilon: {eps}, Peak error: {peak_error_}, Lag error: {lag_error_}")
    #plot_params(models,z,y,w,eps+1e-6)
    #reset everything
    # for country in countries:
    #     z[country] = np.array([0.0, 0.0])
    #     u[country] = np.array([0.0, 0.0])
    # z['GLOBAL'] = np.array([0.0, 0.0])
    # u['GLOBAL'] = np.array([0.0, 0.0])


epsilons = np.array(epsilons)+1e-12 # to avoid log(0)
peak_errors = np.array(peak_errors) + 1e-12 # to avoid log(0)
lag_errors = np.array(lag_errors) + 1e-12 # to avoid log(0)
converged = np.array(converged)

#converged[0] = True
#peak_errors[0] = peak_error_cent
#lag_errors[0] = lag_error_cent

# Filter only converged points
epsilons_converged = epsilons[converged]
peak_errors_converged = peak_errors[converged]
lag_errors_converged = lag_errors[converged]

# Compute max value among converged points
max_peak_error = np.max(peak_errors_converged)
max_lag_error = np.max(lag_errors_converged)

# Identify non-converged regions
epsilons_not_converged = epsilons[~converged]

# Peak error plot
plt.figure(figsize=(10, 6))
plt.plot(0, peak_error_cent, marker='o')
plt.plot(epsilons_converged, peak_errors_converged, marker='o', linestyle='-', color='blue', label='Peak error')

# Draw red horizontal segments only in non-converged regions
for epsilon in epsilons_not_converged:
    plt.hlines(y=max_peak_error, xmin=epsilon - 0.05, xmax=epsilon + 0.05, colors='red', linestyles='-')

plt.xlabel('Epsilon')
plt.ylabel('Error')
plt.title('Peak error as a function of epsilon')
plt.legend()
plt.savefig('peak_error'+f'{timestamp}.png') # Save the figure with timestamp
plt.show()

# Lag error plot
plt.figure(figsize=(10, 6))
plt.plot(0, lag_error_cent, marker='o')
plt.plot(epsilons_converged, lag_errors_converged, marker='o', linestyle='-', color='blue', label='Lag error')

# Draw red horizontal segments only in non-converged regions
for epsilon in epsilons_not_converged:
    plt.hlines(y=max_lag_error, xmin=epsilon - 0.05, xmax=epsilon + 0.05, colors='red', linestyles='-')

plt.xlabel('Epsilon')
plt.ylabel('Error')
plt.title('Lag error as a function of epsilon')
plt.legend()
plt.savefig('lag_error'+f'{timestamp}.png') # Save the figure with timestamp
plt.show()
#%%
# Same plots with log
# Peak error plot
plt.figure(figsize=(10, 6))
plt.plot(0, np.log(peak_error_cent), marker='o')
plt.plot(epsilons_converged, np.log(peak_errors_converged), marker='o', linestyle='-', color='blue', label='Peak error')

# Draw red horizontal segments only in non-converged regions
for epsilon in epsilons_not_converged:
    plt.hlines(y=np.log(max_peak_error), xmin=epsilon - 0.05, xmax=epsilon + 0.05, colors='red', linestyles='-')

plt.xlabel('Epsilon')
plt.ylabel('Error (log)')
plt.title('Peak error as a function of epsilon (log)')

plt.legend()
plt.savefig('peak_error_log'+f'{timestamp}.png') # Save the figure with timestamp
plt.show()

# Lag error plot
plt.figure(figsize=(10, 6))
plt.plot(0, np.log(lag_error_cent), marker='o')
plt.plot(epsilons_converged, np.log(lag_errors_converged), marker='o', linestyle='-', color='blue', label='Lag error')

# Draw red horizontal segments only in non-converged regions
for epsilon in epsilons_not_converged:
    plt.hlines(y=np.log(max_lag_error), xmin=epsilon - 0.05, xmax=epsilon + 0.05, colors='red', linestyles='-')

plt.xlabel('Epsilon')
plt.ylabel('Error (log)')
plt.title('Lag error as a function of epsilon (log)')
plt.legend()
plt.savefig('lag_error_log'+f'{timestamp}.png') # Save the figure with timestamp
plt.show()

#%%
# save to csv
converged[0] = True
peak_errors[0] = peak_error_cent
lag_errors[0] = lag_error_cent
# Generate a timestamp

# Save the results to a CSV file with the timestamp in the filename
df = pd.DataFrame({'epsilon': epsilons, 'converged': converged, 'peak_error': peak_errors, 'lag_error': lag_errors, 'peak_errors_log': np.log(peak_errors), 'lag_errors_log': np.log(lag_errors)})
df.to_csv(f'results_{timestamp}.csv', index=False)



if False:
    y0 = argfull['ARG']['train']['array'][:, 0]
    I_data, _ = cutdata(argfull['ARG']['train']['array'][1, :],proportion=0.8)
    modelargfull= SIRModel(y0, I_data, country='ARG')
    modelargfull.beta = models['ARG'].beta
    modelargfull.gamma = models['ARG'].gamma
    peak_error, lag_error = modelargfull.peak_lag_error()
    print(f"Peak error: {peak_error}, Lag error: {lag_error}")
    modelargfull.plot_sir()