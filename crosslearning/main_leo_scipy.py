import numpy as np
import pandas as pd
#import os
import matplotlib.pyplot as plt
from lib.models import *
from lib.configs import datasets, countries, estimator_vals,logg_every_e

from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.optimize import minimize

class SIRModel:
    def __init__(self, y0):
        self.y0 = y0  # Initial conditions (S0, I0, R0)
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
        ret = odeint(self.sir_model, self.y0, t, args=(beta, gamma))
        S, I, R = ret.T
        return S, I, R

    # Objective function for fitting: sum of squared differences between observed and model data
    def objective(self, params, t_data, S_data, I_data, R_data):
        beta, gamma = params
        S_model, I_model, R_model = self.rollout_sir(t_data, beta, gamma)
        normSdata = np.sum((S_data)**2)
        normIdata = np.sum((I_data)**2)
        normRdata = np.sum((R_data)**2)
        error = np.sum((S_model - S_data) ** 2 + (I_model - I_data) ** 2 + (R_model - R_data) ** 2)
        #print("NormSdata: ", normSdata/normSdata, "NormIdata: ", normIdata/normSdata, "NormRdata: ", normRdata/normSdata)
        #error = np.sum((S_model - S_data) ** 2/normSdata + 80000*(I_model - I_data) ** 2/normIdata + (R_model - R_data) ** 2/normRdata)
        #print("Error S: ", np.sum((S_model - S_data) ** 2)/np.sum((S_model - S_data) ** 2), "Error I: ", np.sum((I_model - I_data) ** 2)/np.sum((S_model - S_data) ** 2), "Error R: ", np.sum((R_model - R_data) ** 2)/np.sum((S_model - S_data) ** 2))
        return error

    # Fit function to minimize the error using all three compartments (S, I, R)
    def fit_objective(self, t_data, S_data, I_data, R_data, initial_guess=(0.3, 0.1)):
        result = minimize(self.objective, initial_guess, args=(t_data, S_data, I_data, R_data))#, bounds=[(0, 1), (0, 1)])
        self.beta, self.gamma = result.x  # Store fitted beta and gamma
        return self.beta, self.gamma

    def fit_onlyI(self, t_data, S_data, I_data, R_data, initial_guess=(0.3, 0.1)):
        def rollout_sir_wrap(t, beta, gamma):
            _, I, _ = self.rollout_sir(t, beta, gamma)
            return I

        params, _ = curve_fit(rollout_sir_wrap, t_data, I_data, p0=initial_guess)
        self.beta, self.gamma = params  # Store the fitted beta and gamma
        return self.beta, self.gamma

    def fit_concat(self, t_data, S_data, I_data, R_data, initial_guess=(0.3, 0.1)):
        def rollout_sir_concat(t, beta, gamma):
            S, I, R = self.rollout_sir(t, beta, gamma)
            return np.concatenate((S,I,R))

        params, _ = curve_fit(rollout_sir_concat, t_data, np.concatenate((S_data,I_data,R_data)), p0=initial_guess)
        self.beta, self.gamma = params  # Store the fitted beta and gamma
        return self.beta, self.gamma

    def beta_gamma_contour_plots(self, t_data, S_data, I_data, R_data):
        # Generate beta and gamma values for the contour plot
            # z_values = np.linspace(0.0, 1.0, 100)
            # w_values = np.linspace(0.0, 1.0, 100)
            # z_mesh, w_mesh = np.meshgrid(z_values, w_values)
            # h = 7.0
            # A=0.01
            # B=0.77
            # P=0.11
            # C=2*h-0.09#0.01
            # D=0.81
            # Q=0.14

            # beta_values = A*z_values + B*w_values + P
            # gamma_values = C*z_values + D*w_values + Q


        #beta_values = (w_values - z_values)/2
        #gamma_values = (w_values + z_values)/2

        beta_values = np.linspace(0.01, 1.0, 100)
        gamma_values = np.linspace(0.01, 1.0, 100)
        beta_mesh, gamma_mesh = np.meshgrid(beta_values, gamma_values)

        # Compute the objective function for each pair of beta and gamma values
        error_mesh = np.array([[self.objective([beta, gamma], t_data, S_data, I_data, R_data) for beta in beta_values] for gamma in gamma_values])

        # Plot the contour plot
        plt.figure(figsize=(10, 6))
        plt.contourf(beta_mesh, gamma_mesh, np.log10(error_mesh), levels=40, cmap='jet')
        # plt.contourf(z_mesh, w_mesh, np.log10(error_mesh), levels=40, cmap='jet')
        plt.colorbar()
        #plt.xlabel('z')
        # plt.ylabel('w')
        plt.xlabel('Beta')
        plt.ylabel('Gamma')
        plt.title('Objective Function Contour Plot')
        plt.show()


# Load dataset
df = pd.read_csv('crosslearning/data/owid-covid-data-old.csv')

# Extract data (assuming datasets format)
country = 'ITA'
y0 = datasets[country]['train']['array'][:, 0]  # Initial conditions (S0, I0, R0)
S_data = datasets[country]['train']['array'][0, :]  # Susceptible data
I_data = datasets[country]['train']['array'][1, :]  # Infected data
R_data = datasets[country]['train']['array'][2, :]  # Recovered/Removed data

# save S, I, R data to csv
df = pd.DataFrame({'S': S_data, 'I': I_data, 'R': R_data})
df.to_csv('crosslearning/data/SIR_data+'+country+'.csv', index=False)


t_data = np.arange(len(I_data))  # Time data
initial_guess = (0.3, 0.1)  # Initial guess for beta and gamma

# Create SIR model and fit the data using all three compartments
sir_model = SIRModel(y0)
params = sir_model.fit_concat(t_data, S_data, I_data, R_data)
print(f"Fitted parameters: beta={params[0]:.4f}, gamma={params[1]:.4f}")

# Overwrite beta=0.2026, gamma=0.1850
params = np.array([0.29, 0.265])
params = np.array([0.7995, 0.8005])

# Generate fitted curve for plotting
t_fit = np.linspace(0, len(I_data), 1000)  # Time for plotting
S_fit, I_fit, R_fit = sir_model.rollout_sir(t_fit, *params)

# Plot Susceptible, Infected, and Recovered data and model fits

# Plot Susceptible
# plt.figure(figsize=(10, 6))
# plt.plot(t_data, S_data, 'o', label='Observed Susceptible')
# plt.xlabel('Time')
# plt.ylabel('Susceptible Population')
# plt.title('SIR Model Fitting: Susceptible Population')
# plt.legend()
# plt.show()

#Plot Infected
plt.figure(figsize=(10, 6))
plt.plot(t_data, I_data, 'o', label='Observed Infected')
plt.plot(t_fit, I_fit, '-', label='Fitted Infected Curve')
plt.xlabel('Time')
plt.ylabel('Infected Population')
plt.title('SIR Model Fitting: Infected Population')
plt.legend()
plt.show()

# Contour plot of beta and gamma
sir_model.beta_gamma_contour_plots(t_data, S_data, I_data, R_data)


# # Plot Recovered/Removed
# plt.figure(figsize=(10, 6))
# plt.plot(t_data, R_data, 'o', label='Observed Recovered/Removed')
# plt.plot(t_fit, R_fit, '-', label='Fitted Recovered/Removed Curve')
# plt.xlabel('Time')
# plt.ylabel('Recovered/Removed Population')
# plt.title('SIR Model Fitting: Recovered/Removed Population')
# plt.legend()
# plt.show()

# # Plot Recovered/Removed
# plt.figure(figsize=(10, 6))
# plt.plot(t_data, S_data+R_data+I_data, 'o', label='Observed Recovered/Removed')
# #plt.plot(t_fit, R_fit, '-', label='Fitted Recovered/Removed Curve')
# plt.xlabel('Time')
# plt.ylabel('SUM Population')
# plt.title('SUM Population')
# plt.legend()
# plt.show()

# # Plot all three compartments together
# plt.figure(figsize=(10, 6))
# plt.plot(t_fit, S_fit, '-', label='Fitted Susceptible Curve')
# plt.plot(t_fit, I_fit, '-', label='Fitted Infected Curve')
# plt.plot(t_fit, R_fit, '-', label='Fitted Recovered/Removed Curve')
# plt.xlabel('Time')
# plt.ylabel('Population')
# plt.title('SIR Model Fitting: Susceptible, Infected, and Recovered Populations')
# plt.legend()
# plt.show()