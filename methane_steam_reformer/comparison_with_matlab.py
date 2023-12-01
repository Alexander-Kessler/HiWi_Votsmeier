"""
Compare the solution of the python code with the matlab project.
"""
# Import libraries
import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load results from python and numpy
python_z = np.load("python_z.npy")
python_x_CH4 = np.load("python_x_CH4.npy")
python_x_H20 = np.load("python_x_H20.npy")
python_x_H2 = np.load("python_x_H2.npy")
python_x_CO = np.load("python_x_CO.npy")
python_x_CO2 = np.load("python_x_CO2.npy")
python_x_N2 = np.load("python_x_N2.npy")
python_X_CH4 = np.load("python_X_CH4.npy")
python_Y_CO2 = np.load("python_Y_CO2.npy")
python_T_avg = np.load("python_T_avg.npy")
python_results = np.column_stack((python_x_CH4, python_x_H20, python_x_H2, python_x_CO, \
                                  python_x_CO2, python_x_N2, python_X_CH4, python_Y_CO2, \
                                      python_T_avg))

df = pd.read_csv("results_matlab.txt", sep=",")
matlab_z = df["z_2D"].to_numpy()
matlab_x_CH4 = df["x_CH4_2D"].to_numpy()
matlab_x_H20 = df["x_H2O_2D"].to_numpy()
matlab_x_H2 = df["x_H2_2D"].to_numpy()
matlab_x_CO = df["x_CO_2D"].to_numpy()
matlab_x_CO2 = df["x_CO2_2D"].to_numpy()
matlab_x_N2 = df["x_N2_2D"].to_numpy()
matlab_X_CH4 = df["X_CH4_2D"].to_numpy()
matlab_Y_CO2 = df["Y_CO2_2D"].to_numpy()
matlab_T_avg = df["T_avg_2D"].to_numpy()
matlab_results = np.column_stack((matlab_x_CH4, matlab_x_H20, matlab_x_H2, matlab_x_CO, \
                                  matlab_x_CO2, matlab_x_N2, matlab_X_CH4, matlab_Y_CO2, \
                                      matlab_T_avg))
data_top = list(df.columns)
    
# Fit the data to a polynamial function and calculate the MSD
def polynomial(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
msd = []
for i in range(1,9):
    if i<8:
        continue
    
    # fit the data to a polynomial function
    python_params, covariance = curve_fit(polynomial, python_z, python_results[:,i], p0=initial_guess)
    matlab_params, covariance = curve_fit(polynomial, matlab_z, matlab_results[:,i], p0=initial_guess)
    
    # generate data with fit parameter
    a, b, c, d, e, f = python_params
    fit_z_python = np.linspace(np.amin(python_z), np.amax(python_z), 100)
    fit_y_python = polynomial(fit_z_python, a, b, c, d, e, f)
    a, b, c, d, e, f = matlab_params
    fit_z_matlab = np.linspace(np.amin(matlab_z), np.amax(matlab_z), 100)
    fit_y_matlab = polynomial(fit_z_matlab, a, b, c, d, e, f)
    
    
    # Plot der Originaldaten und der gefitteten Funktion
    plt.scatter(python_z, python_results[:,i], label='Messdaten Python', color='b')
    plt.scatter(matlab_z, matlab_results[:,i], label='Messdaten Matlab', color='r')
    #plt.plot(fit_z_python, fit_y_python, label='Polynomial Python', color='b')
    #plt.plot(fit_z_matlab, fit_y_matlab, label='Polynomial Matlab', color='r')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel(data_top[i+1])
    plt.show()
    
    msd.append(np.mean((fit_y_python - fit_y_matlab)**2))
