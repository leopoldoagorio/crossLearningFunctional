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
# def crossLearningProject(barz, barzg, epsilon):
#     N = len(barz)
#     z = [cp.Variable(2) for _ in range(N)]
#     zg = cp.Variable(2)

#     # Objective function
#     objective = cp.Minimize(sum(cp.norm(barz[i] - z[i], 2) for i in range(N)) + cp.norm(barzg - zg, 2))

#     # Constraints
#     constraints = [cp.norm(z[i] - zg, 2) <= epsilon for i in range(N)]

#     # Problem definition and solving
#     prob = cp.Problem(objective, constraints)
#     prob.solve()

#     return [zi.value for zi in z], zg.value

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
    #print(I_data>0.6*Imax)
    #print("Imax: ", Imax, "tmax: ", tmax, "t2: ", t2)
    #S_data = S_data[:t2]
    I_data = I_data[:t2]
    #R_data = R_data[:t2]
    return I_data,t2#S_data, I_data, R_data, t2

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

    def plot_sir(self, N=None,filename=None):
        t_fit = np.linspace(0, len(self.I_data), 1000)  # Time for plotting
        _, I_fit, _ = self.rollout_sir(t_fit, self.beta, self.gamma)

        # Plot Susceptible, Infected, and Recovered data and model fits

        # Plot Infected
        plt.figure(figsize=(10, 6))
        if N is not None:
            plt.plot(self.t_data[:N], self.I_data[:N], 'co')
            plt.plot(self.t_data[N:], self.I_data[N:], 'co', markerfacecolor='none')
        else:
            plt.plot(self.t_data, self.I_data, 'o')
        plt.plot(t_fit, I_fit, 'r-', label='Fitted Infected Curve')
        plt.xlabel('Time')
        plt.ylabel('Infected Population')
        plt.title('SIR Model Fitting: Infected Population for ' + self.country + ' (beta=%.4f, gamma=%.4f)' % (self.beta, self.gamma))
        #plt.legend()
        plt.show()
        if filename is not None:
            # Assume t_fit and I_fit are of the same length
            max_length = len(t_fit)

            # Pad t_data and I_data to match the length of t_fit
            t_data = np.pad(self.t_data, (0, max_length - len(self.t_data)))
            I_data = np.pad(self.I_data, (0, max_length - len(self.I_data)))

            # Save data to CSV
            df = pd.DataFrame({
                't_fit': t_fit,
                'I_fit': I_fit,
                't_data': t_data,
                'I_data': I_data,
            })
            df.to_csv(filename, index=False)

def plot_params(models,z,y,w,epsilon,filename = None):
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
    if filename is not None:
        df = pd.DataFrame({
            'Beta': betas,
            'Gamma': gammas,
            'Global Beta': [pglobal[0]]*len(countries),
            'Global Gamma': [pglobal[1]]*len(countries),
            'country': countries
        })
        df.to_csv(filename, index=False)
    plt.show()

# Load dataset
df = pd.read_csv('./data/owid-covid-data-old.csv')
np.random.seed(42)

#print(datasets)
# Extract data (assuming datasets format)
countries = ['USA', 'ESP', 'FRA','BRA','MEX','PRY', 'ITA', 'ARG', 'COL', 'CHL', 'PER', 'DEU', 'GBR', 'IND', 'JPN']
models = {}
doplots = False
for country in countries:
    # Extract initial conditions and data for each country
    y0 = datasets[country]['train']['array'][:, 0]  # Initial conditions (S0, I0, R0)
    #S_data = datasets[country]['train']['array'][0, :]  # Susceptible data
    #I_data = datasets[country]['train']['array'][1, :]  # Infected data
    I_data, _ = cutdata(datasets[country]['train']['array'][1, :],proportion=0.8)  # Cut data to include only up to 0.6*Imax
    #R_data = datasets[country]['train']['array'][2, :]  # Recovered/Removed data
    models[country] = SIRModel(y0, I_data, country=country)

# country = 'ESP'
# y0_ESP = datasets[country]['train']['array'][:, 0]  # Initial conditions (S0, I0, R0)
# S_data_ESP = datasets[country]['train']['array'][0, :]  # Susceptible data
# I_data_ESP = datasets[country]['train']['array'][1, :]  # Infected data
# R_data_ESP = datasets[country]['train']['array'][2, :]  # Recovered/Removed data

# country = 'ITA'
# y0_ITA = datasets[country]['train']['array'][:, 0]  # Initial conditions (S0, I0, R0)
# S_data_ITA = datasets[country]['train']['array'][0, :]  # Susceptible data
# I_data_ITA = datasets[country]['train']['array'][1, :]  # Infected data
# R_data_ITA = datasets[country]['train']['array'][2, :]  # Recovered/Removed data

# country = 'FRA'
# y0_FRA = datasets[country]['train']['array'][:, 0]  # Initial conditions (S0, I0, R0)
# S_data_FRA = datasets[country]['train']['array'][0, :]  # Susceptible data
# I_data_FRA = datasets[country]['train']['array'][1, :]  # Infected data
# R_data_FRA = datasets[country]['train']['array'][2, :]  # Recovered/Removed data

# deduct 58 millions from S_data
#S_data = S_data- 38000000
#y0[0] = y0[0] - 38000000



#S_data, I_data, R_data, _ = cutdata(S_data, I_data, R_data)

#mean diff (S_data+R_data+I_data)
#print("Mean diff: ", np.mean(np.diff(S_data+R_data+I_data)), "Min diff", np.min(np.diff(S_data+R_data+I_data)), "Max diff", np.max(np.diff(S_data+R_data+I_data)))

# save S, I, R data to csv
#df = pd.DataFrame({'S': S_data, 'I': I_data, 'R': R_data})
#df.to_csv('crosslearning/data/SIR_data+'+country+'.csv', index=False)
z = {}
u = {}
x = {}
for country in countries:
    z[country] = np.array([0.0, 0.0])
    u[country] = np.array([0.0, 0.0])
z['GLOBAL'] = np.array([0.0, 0.0])
u['GLOBAL'] = np.array([0.0, 0.0])

lda = 1.
epsilon = 0.01
initial_guess = (2.5*1.205, 2.5*1.15)
Niter =600

def global_residuals(params, models):
    residuals = np.array([])
    for country in models.keys():
        residuals = np.append(residuals, models[country].Iresidual(params))
    return np.array(residuals)

def minimize_global_loss(models):
    result = least_squares(global_residuals, initial_guess, args=(models,),bounds=([0,0],[np.inf,np.inf]))
    print(result)
    return result.x

global_params = minimize_global_loss(models)

if True: #argentina flasheado con centralizado
    y0 = argfull['ARG']['train']['array'][:, 0]
    I_data, _ = cutdata(argfull['ARG']['train']['array'][1, :],proportion=0.8)
    modelargfull= SIRModel(y0, I_data, country='ARG')
    #t = np.arange(0,100,1)
    #_, Isim, _ = modelargfull.rollout_sir(t, 0.1481, 0.0)
    #plt.plot(t, Isim, label='Simulated Infected')
    modelargfull.beta = global_params[0]
    modelargfull.gamma = global_params[1]
    peak_error, lag_error = modelargfull.peak_lag_error()
    print(f"Peak error: {peak_error}, Lag error: {lag_error}")
    modelargfull.plot_sir(N=10)#,filename='SIR_fitted_ARG_10_0_'+timestamp+'.csv')
    # TEST PLOT WITH GAMMA = 0
    # modelargfull.beta = 0.1481
    # modelargfull.gamma = 0.0
    # modelargfull.I_data = [1]*100
    # modelargfull.t_data = np.arange(100)
    # modelargfull.plot_sir()


#print("Global parameters: beta=%.4f, gamma=%.4f" % (global_params[0], global_params[1]))
#%%
for k in tqdm(range(Niter), desc="Optimization Progress"):
    if k%10 == 0:
        lda = lda*0.9
    for country in countries:
        x[country] = np.array(models[country].minimize_reg_loss_least_squares(z[country], u[country], initial_guess, lda)) #xk+1 = argmin f(x) + 0.5*||x-zk+uk||^2
    y = {country: z[country] - u[country] for country in countries} #zk-uk
    x['GLOBAL'] = z['GLOBAL'] - u['GLOBAL']
    z = crossLearningProject(x, u, epsilon) #zk+1
    w = {country: x[country] + u[country] for country in countries} #xk+1+uk


    for country in countries:
        u[country] += x[country] - z[country]
        print(f"u[{country}]: {u[country]}")

    if k % 50 == 0:
        print(f"Max norm: {max(np.linalg.norm(z[k] - x[k]) for k in z)}")
        print(f"Lambda: {lda}")
        plot_params(models, z,y,w, epsilon)
        pass
plot_params(models,z,y,w,epsilon, filename='params_N10_eps_01_'+timestamp+'.csv') # save the final parameter plot
print("Fitted parameters")
for country in countries:
    print(f"{country}: beta={models[country].beta:.4f}, gamma={models[country].gamma:.4f}")
    if doplots:
        models[country].plot_sir()





if True:
    N=10
    # define the filename with N, epsilon and a timestamp
    filename = f'SIR_fitted_ARG_{N}_{epsilon}_{timestamp}.csv'
    y0 = argfull['ARG']['train']['array'][:, 0]
    I_data, _ = cutdata(argfull['ARG']['train']['array'][1, :],proportion=0.8)
    modelargfull= SIRModel(y0, I_data, country='ARG')
    #t = np.arange(0,100,1)
    #_, Isim, _ = modelargfull.rollout_sir(t, 0.1481, 0.0)
    #plt.plot(t, Isim, label='Simulated Infected')
    modelargfull.beta = models['ARG'].beta
    modelargfull.gamma = models['ARG'].gamma
    peak_error, lag_error = modelargfull.peak_lag_error()
    print(f"Peak error: {peak_error}, Lag error: {lag_error}")
    modelargfull.plot_sir(N, filename=filename)