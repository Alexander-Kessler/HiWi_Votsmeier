"""
Calculation of an industrial steam reformer considering the steam reforming 
reaction, water gas shift reaction and the direct reforming reaction. The 
reactor reactor model corresponds to the final project of the Matlab course 
and is a pseudo-homogeneous fixed-bed reactor that is solved in two dimensions. 
In the heat balance, the heat transfer is resolved radially instead of a heat 
transfer coefficient U. The radial discretisation is carried out using the 
finite volume method with dynamic volume element sizes. A pressure gradient 
occurs in the reactor in the radial direction, which is why the volume element 
sizes are varied according to Darcy's law. The variables of species and 
temperature are solved in each volume element with its own temperature. The 
balancing takes place from the centre of the reactor outwards.  First, the 
differentials are solved with the ODE solver of scipy and considered as an 
analytical solution. Then a PINN is trained, which should come as close as 
possible to the analytical solution.
   
Code written by Alexander Keßler on 25.11.23
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.constants import R
import math
import os
import time

# set default tensor
torch.set_default_dtype(torch.double)

# set matplotlib style
plt.style.use(r'thesis.mplstyle')

class generate_data():
    def __init__(self, inlet_mole_fractions, bound_conds, reactor_conds):
        """Constructor:
        Args:
            inlet_mole_fractions (list): Inlet mole fractions of CH4, H20, H2, 
                                         CO, CO2 and N2 [-]
            bound_conds (list): Pressure [bar], flow velocity [m s-1], inlet 
                                temperature [K], temperature of the steam 
                                reformer wall [K]
            reactor_conds (list): efficiency factor of the catalyst [-],
                                  number of volume elements [-]
            
        Params:
            x_CH4_0 (float): Inlet mole fraction of methane [-]
            x_H20_0 (float): Inlet mole fraction of water [-]
            x_H2_0 (float): Inlet mole fraction of hydrogen [-]
            x_CO_0 (float): Inlet mole fraction of carbon monoxide [-]
            x_CO2_0 (float): Inlet mole fraction of carbon dioxide [-]
            x_N2_0 (float): Inlet mole fraction of nitrogen [-]
            
            p (float): Pressure [bar]
            u (float): flow velocity [m s-1]
            T0 (int): inlet temperature [K]
            T_wall (int): temperature of the steam reformer wall [K]
            
            eta (float): efficiency factor of the catalyst [-]
            n_elements (float): number of volume elements [-]
            
            k0 (1D-array): preexponential factor/ arrhenius parameter [kmol Pa0.5 kgcat-1 h-1]
            E_A (1D-array): activation energy/ arrhenius parameter [J/mol]
            K_ads_0 (1D-array): adsorption constant [Pa-1] for K_ads_0[0:3]
            G_R_0 (1D-array): adsorption enthalpy [J mol-1]
            R (float): gas constant [J K-1 mol-1]
            d_pi (float): inner catalyst ring diameter [m]
            d_in (float): diameter inner tube [m]
            d_out (float): diameter outer tube [m]
            A (float): cross section tube [m2]
            V_dot (float): volumetric flow rate [m^3 h-1]
            em (float): emmissivity for heat transfer [-]
            epsilon (float): void fraction cat.-bet [-]
            lambda_s (float): radial thermal conductivity fixed bed [W m-1 K-1]
            rho_s (float): density of solid material in fixed bed [kg m-3]
            rho_b (float): density fixed bed [kg m-3]
            
            cp_coef (2D-array): coefficients of the NASA polynomials [j mol-1 K-1]
            H_0 (1D-array): enthalpies of formation [J mol-1]
            S_0 (1D-array): entropy of formation [J mol-1 K-1]
            MW (1D-array): molecular weight [kg mol-1]
        """
        
        self.x_CH4_0 = inlet_mole_fractions[0]
        self.x_H2O_0 = inlet_mole_fractions[1]
        self.x_H2_0 = inlet_mole_fractions[2]
        self.x_CO_0 = inlet_mole_fractions[3]
        self.x_CO2_0 = inlet_mole_fractions[4]
        self.x_N2_0 = inlet_mole_fractions[5]
        self.inlet_mole_fractions = np.array(inlet_mole_fractions[0:6])
        
        self.p = bound_conds[0]
        self.u0 = bound_conds[1]
        self.T0 = bound_conds[2]
        self.T_wall = bound_conds[3]
        
        self.eta = reactor_conds[0]
        self.n_elements = reactor_conds[1]
        
        self.k0 = np.array([4.225*1e15*math.sqrt(1e5),1.955*1e6*1e-5,1.020*1e15*math.sqrt(1e5)])
        self.E_A = np.array([240.1*1e3,67.13*1e3,243.9*1e3])
        self.K_ads_0 = np.array([8.23*1e-5*1e-5,6.12*1e-9*1e-5,6.65*1e-4*1e-5,1.77*1e5])
        self.G_R_ads = np.array([-70.61*1e3,-82.90*1e3,-38.28*1e3,88.68*1e3])
        self.R = R
        self.d_pi = 0.0084
        self.d_in = 0.1016
        self.d_out = 0.1322
        self.A = math.pi * (self.d_in/2)**2
        self.V_dot_0 = self.u0 * 3600 * self.A
        self.em = 0.8
        self.epsilon = 0.4 + 0.05 * self.d_pi/self.d_in + 0.412 * \
            self.d_pi**2/self.d_in**2
        self.lambda_s = 0.3489
        self.rho_s = 2355
        self.rho_b = self.rho_s * (1-self.epsilon)
        
        self.cp_coef = np.array([[19.238,52.09*1e-3,11.966*1e-6,-11.309*1e-9], \
                                 [32.22,1.9225*1e-3,10.548*1e-6,-3.594*1e-9], \
                                 [27.124,9.267*1e-3,-13.799*1e-6,7.64*1e-9], \
                                 [30.848,-12.84*1e-3,27.87*1e-6,-12.71*1e-9], \
                                 [19.78,73.39*1e-3,-55.98*1e-6,17.14*1e-9], \
                                 [31.128, -13.556*1e-3, 26.777*1e-6, -11.673*1e-9]])
        self.H_0 = np.array([-74850, -241820, 0, -110540, -393500])
        self.S_0 = np.array([-80.5467, -44.3736, 0,	89.6529, 2.9515])
        self.MW = np.array([16.043, 18.02, 2.016, 28.01, 44.01, 28.01]) * 1e-3

        self.reactor_radii = np.array([[0.],[0.01606437],[0.02271845],[0.02782431],[0.03212874],[0.03592102],[0.03934951],[0.04250233],[0.0454369 ],[0.04819311],[0.0508    ]])
        self.area_elements = self.reactor_radii[1:]**2 * math.pi - self.reactor_radii[:-1]**2 * math.pi

        
    def calc_inlet_ammount_of_substances(self):
        """
        Calculate the ammount of substances at the inlet of the PFR.
        
        New Params:
            total_concentration (float): Concentration of the gas [kmol m-3]
            concentrations (1D-array): Concentrations of all species [kmol m-3]
            ammount_of_substances (1D-array): Ammount of substances of all species [kmol h-1]
        """
 
        total_concentration = (self.p * 1e5 * 1e-3) / (self.R*self.T0)
        concentrations = total_concentration * np.array([self.x_CH4_0, self.x_H2O_0, \
                                                         self.x_H2_0, self.x_CO_0, \
                                                         self.x_CO2_0, self.x_N2_0])
        ammount_of_substances = concentrations * self.V_dot_0
        
        n_CH4_0 = ammount_of_substances[0]
        n_H2O_0 = ammount_of_substances[1]
        n_H2_0 = ammount_of_substances[2]
        n_CO_0 = ammount_of_substances[3]
        n_CO2_0 = ammount_of_substances[4]
        n_N2_0 = ammount_of_substances[5]
        self.n_N2_0 = ammount_of_substances[5]
        
        return np.array([n_CH4_0, n_H2O_0, n_H2_0, n_CO_0, n_CO2_0, n_N2_0])
           
    def calc_mole_fractions(self, n_vector):
        """
        Calculate the mole fractions of all species.
        
        Args:
            n_vector (1D-array): Ammount of substances of all species of the 
                                 volume elements [kmol h-1]
        """
        
        mole_fractions = np.zeros([6*self.n_elements])
        for i in range(self.n_elements):
            n_element = n_vector[i::self.n_elements]
            mole_fraction_element = n_element/np.sum(n_element)
            mole_fractions[i::self.n_elements] = mole_fraction_element
        
        return mole_fractions
    
    def calc_mu_gas(self, T, x_i):
        """
        Calculates the dynamic viscosities of the single components mu_i and then
        calculates the gas mixture dynamic viscosity.

        Args:
            T (float): temperature of the volume element [K]
            x_i (1D-array): ammount of substance of all species of the volume 
                            element [-]
        
        New Params:
            mu_mix (float): gas mixture dynamic viscosity [Pa s]
        """
        
        # PHYSICAL PROPERTIES OF LIQUIDS AND GASES, 
        # https://booksite.elsevier.com/9780750683661/Appendix_C.pdf
        # NASA Polynomial in CGS-Unit µP (Poise)
        mu_CH4 = 3.844 + 4.0112*1e-1*T + -1.4303*1e-4*T**2
        mu_H2O = -36.826 + 4.29*1e-1*T + -1.62*1e-5*T**2
        mu_H2 = 27.758 + 2.12*1e-1*T + -3.28*1e-5*T**2
        mu_CO = 23.811 + 5.3944*1e-1*T + -1.5411*1e-4*T**2
        mu_CO2 = 11.811 + 4.9838*1e-1*T + -1.0851*1e-4*T**2
        mu_N2 = 42.606 + 4.75*1e-1*T + -9.88*1e-5*T**2
        
        mu_gas = np.array([mu_CH4, mu_H2O, mu_H2, mu_CO, mu_CO2, mu_N2]) * 1e-6 # µP -> P
        
        # Method of Wilke: 
        # Chapter 9.5 THE PROPERTIES OF GASES AND LIQUIDS Bruce E. Poling / A 
        # Simple and Accurate Method for Calculatlng Viscosity of Gaseous Mixtures
        # by Thomas A. Davidson
        mu_mix = 0
        phi = np.zeros([len(x_i),len(x_i)])
        for i in range(len(x_i)):
            for j in range(len(x_i)):
                phi[i,j] = (1 + (mu_gas[i]/mu_gas[j])**(0.5) * (self.MW[j]/self.MW[i])**(0.25))**2/ \
                    (8*(1 + (self.MW[i]/self.MW[j]) ))**(0.5)
            mu_mix = mu_mix + x_i[i]*mu_gas[i]/(np.dot(phi[i,:],x_i))
            
        mu_mix = mu_mix * 0.1 # Convert CGS-Unit Poise to SI-Unit Pa*s 
        return mu_mix
    
    def calc_area_radii(self, temperature, mole_fractions): 
        """
        Calculation of the areas and radii of the volume elements as a function 
        of temperature. In the radial direction, a pressure gradient is created 
        by the temperature gradient, which is taken into account using Darcy's law.
        
        Args:
            T (1D-array): temperature of the volume elements [K]
            x_i (1D-array): ammount of substance of all species of the volume 
                            elements [-]
        
        New Params:
            mu_gas (float): dynamic viscosity of gas mixture [Pa s]
            rho_gas (float): density of gas mixture [kg m-3]
            area_elements (1D-array): area of the volume elements [m]
            radii_elements (1D-array): radii of the volume elements starting 
                                       from the centre of the reactor [m]
        """
        
        def calc_rho_gas(T, x_i):
            # Calculate density of gas mixture via ideal gas (mixing rule = Dalton)
            rho_gas = np.sum(np.multiply((x_i*self.p*1e5),self.MW))/(self.R * T)
            return rho_gas
        
        # Calculate area of the volume elements
        mu_gas = np.zeros([self.n_elements,1])
        rho_gas = np.zeros([self.n_elements,1])
        for i in range(self.n_elements):
            mu_gas[i] = self.calc_mu_gas(temperature[i], mole_fractions[i::self.n_elements])
            rho_gas[i] = calc_rho_gas(temperature[i], mole_fractions[i::self.n_elements])
        
        area_elements_devided_by_total_area = np.divide(mu_gas,rho_gas)/\
            np.sum(np.divide(mu_gas,rho_gas))
        area_elements = area_elements_devided_by_total_area * self.A
        
        # Calculate radii of the volume elements
        radii_elements = np.zeros([self.n_elements+1,1])
        for i in range(self.n_elements):
            radii_elements[i+1] = (area_elements[i]/math.pi + radii_elements[i]**2)**0.5
        
        return (area_elements, radii_elements)
    
    def calc_flow_velocity(self, mole_fractions, area_elements, temperature):
        """
        Calculation of the flow velocity as a function of temperature and gas 
        composition.

        Args:
            mole_fractions (1D-array): ammount of substance of all species of 
                                       all volume elements [-]
            area_elements (1D-array): area of the volume elements [m]
            temperature (1D-array): temperature of all volume elements [K]
                                           
        New Params:
            molar_mass (float): molecular weight [kg mol-1]
            flow_velocity_average (float): Flow velocity averaged over all 
                                           volume elements [m/s]
        """
        
        # Calculation of the averaged quantities over the elements
        mole_fractions_average = np.zeros([6,1])
        for i in range(self.n_elements):
            for j in range(6):
                mole_fractions_average[j] = mole_fractions_average[j] + \
                    mole_fractions[i+(j*self.n_elements)] * area_elements[i]/np.sum(area_elements)
        temperature_average = np.sum(np.multiply(temperature, np.squeeze(area_elements)))/np.sum(area_elements)
        molar_mass_0 = np.dot(self.inlet_mole_fractions, self.MW)
        molar_mass_average = np.dot(np.squeeze(mole_fractions_average), self.MW)
        
        # Calculate averaged flow velocity 
        flow_velocity_average = self.u0 * (temperature_average*molar_mass_0)/ \
            (self.T0*molar_mass_average)
        
        return flow_velocity_average
        
        
    def calculate_thermo_properties(self, T):
        """
        Calculation of the reaction enthalpies and the equilibrium constants Kp.
        The temperature dependence of the heat capacity is considered with NASA 
        polynomials in order to consider the temperature dependence of the 
        enthalpies of formation and the entropies of formation. 
        
        Args:
            T (float): temperature [K]
        
        New Params:
            H_i (1D-array): enthalpies of formation [J mol-1]
            S_i (1D-array): entropies of formation [J mol-1 K-1]
            H_R (1D-array): reaction enthalpies [J mol-1]
            S_R (1D-array): reaction entropies [J mol-1 K-1]
            G_R (1D-array): reaction gibbs energies [J mol-1]
            Kp (1D-array): equilibrium constant, Kp[1,3]=[Pa], Kp[2]=[-]
        """
        
        # Calculation of the standard formation enthalpy at given temperature
        # with Kirchhoffsches law
        H_i = self.H_0.reshape(-1,1) + np.matmul(self.cp_coef[:5,:], \
                np.array([[T], [(1/2)*T**2], [(1/3)*T**3], \
                [(1/4)*(T)**4]])) - np.matmul(self.cp_coef[:5,:], np.array( \
                [[298.15], [(1/2)*298.15**2], [(1/3)*298.15**3], [(1/4)*298.15**4]]))
        
        # Calculation of the standard formation entropy at given temperature
        S_i = self.S_0.reshape(-1,1) + np.matmul(self.cp_coef[:5,:], \
                np.array([[math.log(T)], [T], [(1/2)*T**2], \
                [(1/3)*T**3]])) - np.matmul(self.cp_coef[:5,:], \
                np.array([[math.log(298.15)], [298.15], [(1/2)*298.15**2], \
                          [(1/3)*298.15**3]]))
        
        # Calculation of standard reaction enthalpies with Satz von Hess at 
        # standard conditions (T = 298.15 K)
        H_R = np.zeros(3)
        H_R[0] = -H_i[0].item() - H_i[1].item() + H_i[3].item() + 3*H_i[2].item()
        H_R[1] = -H_i[3].item() - H_i[1].item() + H_i[4].item() + H_i[2].item()
        H_R[2] = -H_i[0].item() - 2*H_i[1].item() + H_i[4].item() + 4*H_i[2].item()
        
        # Calculation of standard reaction entropies with Satz von Hess at 
        # standard conditions (T = 298.15 K)
        S_R = np.zeros(3)
        S_R[0] = -S_i[0].item() - S_i[1].item() + S_i[3].item() + 3*S_i[2].item()
        S_R[1] = -S_i[3].item() - S_i[1].item() + S_i[4].item() + S_i[2].item()
        S_R[2] = -S_i[0].item() - 2*S_i[1].item() + S_i[4].item() + 4*S_i[2].item()
        
        # Calculation of the free reaction enthalpy with the Gibbs Helmoltz equation
        G_R = H_R - T * S_R
        
        # Calculation of the rate constants
        Kp = np.exp(-G_R / (self.R*T)) * np.array([1e10,1,1e10])
        
        return [Kp, H_R]
                                                   

    def xu_froment(self, temperature, partial_pressures, area_elements):
        """
        Calculation of the differentials from the mass balance with the kinetic 
        approach of Xu, Froment for each volume element.
        
        Args:
            temperature (1D-array): temperature of the volume elements [K]
            partial_pressures (1D-array): partial pressures of all species of 
                                          the volume elements [Pa]
            area_elements (1D-array): area of the volume elements [m]
            
        New Params:
            k (list): velocity constant, k[1,3]=[kmol Pa0.5 kgcat-1 h-1]
                                         k[2]=[kmol Pa-1 kgcat-1 h-1]
            K_ads (1D-array): adsorption constants, K_ads[1-3]=[Pa-1]
            r_total (1D-array): reaction rates [kmol kgcat-1 h-1]
            dn_dz (1D-array): derivatives of the amounts of substances of the 
                              species in dependence of the reactor length [kmol m-1 h-1]
        """
        
        # Calculate derivative of the mass balance of each element
        dn_dz = np.zeros([6*self.n_elements,1])
        r_total = np.zeros([self.n_elements,3])
        for i in range(self.n_elements):
            # Extract partial presses from the element
            partial_pressures_element = partial_pressures[i::self.n_elements]
            
            # Calculate reaction rates with a Langmuir-Hinshelwood-Houghen-Watson approach
            k = self.k0 * np.exp(-self.E_A/(self.R*temperature[i]))
            K_ads = self.K_ads_0 * np.exp(-self.G_R_ads/(self.R*temperature[i]))
            Kp, H_R = generate_data.calculate_thermo_properties(self, temperature[i])
            
            DEN = 1 + partial_pressures_element[3]*K_ads[0] + partial_pressures_element[2]*K_ads[1] + \
                partial_pressures_element[0]*K_ads[2] + (partial_pressures_element[1]*K_ads[3]) / partial_pressures_element[2]
            r_total_element = np.zeros(3)
            r_total_element[0] = (k[0] / (partial_pressures_element[2]**2.5)) * (partial_pressures_element[0] * \
                        partial_pressures_element[1] - (((partial_pressures_element[2]**3) * partial_pressures_element[3]) / \
                        Kp[0])) / (DEN**2)
            r_total_element[1] = (k[1] / partial_pressures_element[2]) * (partial_pressures_element[3] * \
                        partial_pressures_element[1] - ((partial_pressures_element[2] * partial_pressures_element[4]) / \
                        Kp[1])) / (DEN**2)
            r_total_element[2] = (k[2] / (partial_pressures_element[2]**3.5)) * (partial_pressures_element[0] * \
                        (partial_pressures_element[1]**2) - (((partial_pressures_element[2]**4) * partial_pressures_element[4]) / \
                        Kp[2])) / (DEN**2)
            
            # Calculate derivatives of the mass balance
            dn_dz_element = np.zeros(6)
            dn_dz_element[0] = self.eta * area_elements[i].item() * (-r_total_element[0] - r_total_element[2]) * self.rho_b
            dn_dz_element[1] = self.eta * area_elements[i].item() * (-r_total_element[0] - r_total_element[1] - 2*r_total_element[2]) * self.rho_b
            dn_dz_element[2] = self.eta * area_elements[i].item() * (3*r_total_element[0] + r_total_element[1] + 4*r_total_element[2]) * self.rho_b
            dn_dz_element[3] = self.eta * area_elements[i].item() * (r_total_element[0] - r_total_element[1]) * self.rho_b
            dn_dz_element[4] = self.eta * area_elements[i].item() * (r_total_element[1] + r_total_element[2]) * self.rho_b
            dn_dz_element[5] = 0
            
            dn_dz[i::self.n_elements] = np.expand_dims(dn_dz_element, axis=1)
            r_total[i,:] = self.eta * r_total_element

        return [dn_dz, r_total]
    
    def heat_balance(self, mole_fractions, temperature, r_total, flow_velocity, radii_elements):
        """
        Calculation of the differentials of the heat balance for each volume 
        element.

        Args:
            mole_fractions (1D-array): ammount of substance of all species of 
                                       all volume elements [-]
            temperature (1D-array): temperature of all volume elements [K]
            r_total (1D-array): reaction rates [kmol kgcat-1 h-1]
            flow_velocity (float): flow velocity of the gas [m s-1]
            radii_elements (1D-array): radii of the volume elements [m]
            
        New Params:
            heat_flow_rate (1D-array): heat flow rate through the volume elements
                                       [kJ m-2 h-1]
            mult_density_gas_cp (1D-array): product of density_gas and cp_gas [J m-3 K-1]
            density_gas (float): density of the gas [kg m-3]
            cp_i (1D-array): heat capacities of each component [J mol-1 K-1]
            alpha_w_int (float): Heat transfer coefficient between fixed bed and 
                                 wall [kJ m-2 h-1 K-1]
            lambda_rad (float): Thermal conductivity in the fixed bed 
                                [kJ m-1 h-1 K-1]
            cp_gas (float): heat capacity of the gas mixture [J kg-1 K-1]
            dTdz_cond (1D-array): heat exchange with environment [K m-1]
            dTdz_react (1D-array): heat production through the reaction [K m-1]
            dTdz (1D-array): derivatives of the temperature in dependence of the 
                             reactor length of each volume element [K m-1]
        """
        
        def calc_lambda_gas(T, x_i):
            """
            Calculates the conductivities of the single components lambda_i and then
            the gas mixture conductivity.
            
            Args:
                T (float): temperature of the volume element [K]
                x_i (1D-array): ammount of substance of all species of the volume 
                                element [-]
            """
            
            # PHYSICAL PROPERTIES OF LIQUIDS AND GASES, 
            # https://booksite.elsevier.com/9780750683661/Appendix_C.pdf
            lambda_CH4 = -0.00935 + 1.4028*1e-4*T + 3.318*1e-8*T**2
            lambda_H2O = 0.00053 + 4.7093*1e-5*T + 4.9551*1e-8*T**2
            lambda_H2 = 0.03951 + 4.5918*1e-4*T + -6.4933*1e-8*T**2
            lambda_CO = 0.00158 + 8.2511*1e-5*T + -1.9081*1e-8*T**2
            lambda_CO2 = -0.012 + 1.0208*1e-4*T + -2.2403*1e-8*T**2
            lambda_N2 = 0.00309 + 7.593*1e-5*T + -1.1014*1e-8*T**2

            lmbd_gas = np.array([lambda_CH4, lambda_H2O, lambda_H2, lambda_CO, lambda_CO2, lambda_N2])
            
            # Method of Wassiljewa:
            # Chapter 10.6 THE PROPERTIES OF GASES AND LIQUIDS Bruce E. Poling / A 
            # Slmple and Accurate Method for Calculatlng Viscosity of Gaseous Mixtures
            # by Thomas A. Davidson
            # Calculation A(i,j) by Mason, Saxena: Mason EA, Saxena SC. Approximate 
            # formula for the thermal conductivity of gas mixtures. 
            # Phys Fluids 1958;1:361e9
            lmbd_mix = 0
            A = np.zeros([len(x_i), len(x_i)])
            for i in range(len(x_i)):
                for j in range(len(x_i)):
                    A[i,j] = (1 + (lmbd_gas[i]/lmbd_gas[j])**(0.5) * (self.MW[i]/self.MW[j])**(0.25))**2/ \
                        (8*(1 + (self.MW[i]/self.MW[j]) ))**(0.5)
                     
                lmbd_mix = lmbd_mix + x_i[i]*lmbd_gas[i]/np.dot(A[i,:], x_i)
            
            return lmbd_mix
                    
        def calc_heat_transfer(T, x_i, rho_g, cp_i, u_gas):
            """
            Calculation of the heat transfer in the fixed bed (lmbd_er) and 
            between the fixed bed and the reactor wall (a_w). It is assumed 
            that there is no temperature gradient in the reactor wall. 
            Accordingly, the temperature on the outer reactor wall is 
            equal to the temperature in the inner reactor wall.

            Args:
                T (float): temperature of the volume element [K]
                x_i (1D-array): ammount of substance of all species of the volume 
                                element [-]
                rho_g (float): density of the gas mixture [kg m-3]
                cp_i (1D-array): heat capacities of each species [J mol-1 K-1]
                u_gas (float): flow velocity of the gas mixture [m s-1]
            
            New Params:
                a_w: Heat transfer coefficient between fixed bed and wall 
                     [kJ m-2 h-1 K-1]
                lmbd_er: Thermal conductivity in the fixed bed [kJ m-1 h-1 K-1]
            """
            
            mu_g = self.calc_mu_gas(T, x_i)
            lmbd_g = calc_lambda_gas(T, x_i)
            cps_g_mix = np.dot(x_i, (cp_i * (1/self.MW)))
            l_c = self.d_pi
            N_Re = rho_g * u_gas * l_c / mu_g
            N_Pr = cps_g_mix * mu_g / lmbd_g;

            a_w = (1 - 1.5 * (self.d_in/self.d_pi)**(-1.5)) * \
                   (lmbd_g * 3.6)/self.d_pi * N_Re**(0.59) * N_Pr**(1/3);
            a_rs = 0.8171*(self.em/(2-self.em))*(T/100)**(3);
            a_ru = (0.8171*(T/100)**(3))/(1+self.epsilon/2 * (1-self.epsilon) * \
                    (1-self.em)/self.em);

            lmbd_er_0 = self.epsilon*(lmbd_g*3.6 + 0.95*a_ru*self.d_pi) + \
                        (0.95 * (1 - self.epsilon))/(2/(3*self.lambda_s*3.6) + \
                        1/(10 * lmbd_g*3.6 + a_rs * self.d_pi))
            
            lmbd_er = lmbd_er_0 + 0.111 * lmbd_g*3.6 * (N_Re * N_Pr**(1/3))/(1 + 46 * \
                      (self.d_pi/self.d_out)**(2))
            
            return (a_w, lmbd_er)

        # Calculate heat flow rate
        heat_flow_rate = np.zeros([self.n_elements+1,1])
        mult_density_gas_cp = np.zeros([self.n_elements,1])
        for i in range(self.n_elements):
            density_gas = np.sum(self.p * 1e5 * mole_fractions[i::self.n_elements] * self.MW) / \
                                (self.R * temperature[i])
            cp_i = self.cp_coef[:,0] + self.cp_coef[:,1] * temperature[i] + self.cp_coef[:,2] * \
                temperature[i]**2 + self.cp_coef[:,3] * temperature[i]**3
            alpha_w_int, lambda_rad = calc_heat_transfer(temperature[i],\
                mole_fractions[i::self.n_elements], density_gas, cp_i, flow_velocity)
            
            cp_gas = np.dot(mole_fractions[i::self.n_elements],cp_i/self.MW)
            mult_density_gas_cp[i] = cp_gas*density_gas*1e3
            
            if i == 0:
                # first boundary condition
                heat_flow_rate[i] = 0
            elif i == self.n_elements-1:
                heat_flow_rate[i] = lambda_rad*(temperature[i-1]-temperature[i])/(radii_elements[i+1]-radii_elements[i])
                # second boundary condition
                heat_flow_rate[-1] = (alpha_w_int*lambda_rad/(0.5*(radii_elements[i+1]-radii_elements[i])))/ \
                    (alpha_w_int+(lambda_rad/(0.5*(radii_elements[i+1]-radii_elements[i]))))* (temperature[-1]-self.T_wall)
            else:
        	    heat_flow_rate[i] = lambda_rad*(temperature[i-1]-temperature[i])/(radii_elements[i+1]-radii_elements[i])
        
        # Calculate derivative of the heat balance
        dTdz = np.zeros([self.n_elements,1])
        for i in range(self.n_elements):
            dr = radii_elements[i+1]-radii_elements[i]
            # Calculation of the source term for external heat exchange
            dTdz_cond = (radii_elements[i]*heat_flow_rate[i]*1e3 - radii_elements[i+1]*heat_flow_rate[i+1]*1e3)/ \
                (dr *(radii_elements[i]+dr*0.5)*mult_density_gas_cp[i]*flow_velocity*3.600)
            # Calculation of the source term for the reaction
            Kp, H_R = generate_data.calculate_thermo_properties(self, temperature[i])
            s_H = -np.sum((H_R*1e3)*(r_total[i,:])*self.rho_b)
            dTdz_react = s_H/(mult_density_gas_cp[i]*flow_velocity*3.600)
            dTdz[i] = dTdz_cond+dTdz_react           
            
        return dTdz
        
    def calc_results(self, y):
        """
        Calculate the 
        
        Args:
            y (2D-array): Analytical solution of the reactor model containing the 
                          ammount of substances [kmol h-1] and temperatures [K] 
                          of each element as a function of the reactor length.
        
        New Params:
            X_CH4 (1D-array): Conversion of methane averaged over all volume 
                              elements as a function of reactor length [-]
            Y_CO2 (1D-array): Yield of carbon dioxide averaged over all volume 
                              elements as a function of reactor length [-]
            T_avg (1D-array): Reactor temperature averaged over all volume 
                              elements as a function of reactor length [K]
            radii_2D (2D-array): Radii of the volume elements as a function of 
                                 the reactor length to create the 2D plot of the 
                                 reactor temperature [m]
        """
        
        # Calculate mole fractions
        self.x_CH4 = np.sum(y[:,0*self.n_elements:1*self.n_elements],axis=1)/np.sum(y[:,:6*self.n_elements], axis=1)
        self.x_H2O = np.sum(y[:,1*self.n_elements:2*self.n_elements],axis=1)/np.sum(y[:,:6*self.n_elements], axis=1)
        self.x_H2 = np.sum(y[:,2*self.n_elements:3*self.n_elements],axis=1)/np.sum(y[:,:6*self.n_elements], axis=1)
        self.x_CO = np.sum(y[:,3*self.n_elements:4*self.n_elements],axis=1)/np.sum(y[:,:6*self.n_elements], axis=1)
        self.x_CO2 = np.sum(y[:,4*self.n_elements:5*self.n_elements],axis=1)/np.sum(y[:,:6*self.n_elements], axis=1)
        self.x_N2 = np.sum(y[:,5*self.n_elements:6*self.n_elements],axis=1)/np.sum(y[:,:6*self.n_elements], axis=1)
        
        # Calculate conversion of CH4
        self.X_CH4 = (np.sum(y[0,:self.n_elements])-np.sum(y[:,:self.n_elements],axis=1)) / \
            np.sum(y[0,:self.n_elements])
        
        # Calculate yield of CO2
        self.Y_CO2 = (np.sum(y[:,4*self.n_elements:5*self.n_elements],axis=1)-np.sum(y[0,4*self.n_elements:5*self.n_elements])) / \
            np.sum(y[0,:self.n_elements])
            
        # Calculate temperature
        self.T_avg = np.sum(y[:,6*self.n_elements:],axis=1)/self.n_elements
        
        # Calculate radii for 2D temperature plot
        radii_2D = np.zeros([self.n_elements+1,len(y)])
        for i in range(len(y)):
            #area_elements, radii_elements = self.calc_area_radii(y[i,6*self.n_elements:], y[i,:6*self.n_elements])
            radii_elements = self.reactor_radii
            radii_2D[:,i] = np.squeeze(radii_elements)
        self.radii_2D = radii_2D
        
    
    def ODEs(y,z,self):
        """
        Function for the ODE solver.

        Params:
            ammount_of_substances (1D-array): ammount of substances of all species 
                                              of all volume elements [kmol h-1]
            temperature (1D-array): reactor temperature of all volume elements [K]
            mole_fractions (1D-array): mole fractions of all species of all volume
                                       elements [-]
            partial_pressures (1D-array): partial pressures of all species of all
                                          volume elements [Pa]
            area_elements (1D-array): area of the volume elements [m]
            radii_elements (1D-array): radii of the volume elements starting 
                                       from the centre of the reactor [m]
            flow_velocity (float): flow velocity of the gas [m s-1]
        """
        
        # Extract the values
        ammount_of_substances = np.array(y[:6*self.n_elements])
        temperature = y[6*self.n_elements:7*self.n_elements]
        
        # Calculate mole fractions
        mole_fractions = generate_data.calc_mole_fractions(self,ammount_of_substances)
        
        # Calculate inlet partial pressures
        partial_pressures = self.p * 1e5 * mole_fractions
        
        # Calculate the element area and radii with darcy's law
        #area_elements, radii_elements = generate_data.calc_area_radii(self, temperature, mole_fractions)
        area_elements = self.area_elements
        radii_elements = self.reactor_radii

        # Calculation of the flow velocity as a function of the gas composition 
        # and temperature
        flow_velocity = generate_data.calc_flow_velocity(self, mole_fractions, area_elements, temperature)
        
        ## Mass balance
        # Calculate reaction rates with a Langmuir-Hinshelwood-Houghen-Watson approach
        dn_dz, r_total = generate_data.xu_froment(self, temperature, partial_pressures, area_elements)
        
        ## Heat balance
        dTdz = generate_data.heat_balance(self, mole_fractions, temperature, r_total, flow_velocity, radii_elements)
        
        # Combine derivatives
        dydz = np.append(dn_dz, dTdz)

        return dydz
        
    def solve_ode(self, reactor_lengths, plot):
        """
        Solution of the ODEs for the mass balance and heat balance with the 
        scipy solver for the steam reformer. The results can then be plotted.
        
        Args:
            reactor_lengths (1D-array): reactor lengths for the ODE-Solver [m]
            plot (bool): boolean indicates whether the results should be plotted
            
        Params:
            y (2D-array): Analytical solution of the reactor model containing the 
                          ammount of substances [kmol h-1] and temperatures [K] 
                          of each element as a function of the reactor length.
        """
        
        # Calculate inlet ammount of substances
        ammount_of_substance_inlet = generate_data.calc_inlet_ammount_of_substances(self)
        
        # Prepare initial conditions
        y_0 = np.zeros([7*self.n_elements])
        for i in range(6):
            y_0[i*self.n_elements:(i+1)*self.n_elements] = ammount_of_substance_inlet[i]/self.n_elements
        y_0[6*self.n_elements:7*self.n_elements] = self.T0
        
        # Solve ODE for isotherm, adiabatic or polytrop reactor
        y = odeint(generate_data.ODEs, y_0, reactor_lengths, args=(self,))
        
        # Calculate mole fraction, conversion and yield
        generate_data.calc_results(self, y)
        
        # Plotting results from the ODE-Solver
        if plot:
            generate_data.plot(self, reactor_lengths, y[:,6*self.n_elements:])
        
        return y
    
    def plot(self, reactor_lengths, T_2D):
        """
        Plotting results from the ODE-Solver. The first plot contains the mole 
        fraction, temperature, conversion of CH4 and yield of CO2 as a function 
        of reactor length. The second plot shows the cross-section of the reactor 
        in which the temperatures are plotted. 
        """
        
        ## Mole fractions plot
        fig1 = plt.figure(figsize=(10,7))
        ax1 = fig1.add_subplot(311)
        ax1.plot(reactor_lengths, self.x_CH4, '-', label=r'$x_{\rm{CH_{4}}}$')
        ax1.plot(reactor_lengths, self.x_H2O, '-', label=r'$x_{\rm{H_{2}O}}$')
        ax1.plot(reactor_lengths, self.x_H2, '-', label=r'$x_{\rm{H_{2}}}$')
        ax1.plot(reactor_lengths, self.x_CO, '-', label=r'$x_{\rm{CO}}$')
        ax1.plot(reactor_lengths, self.x_CO2, '-', label=r'$x_{\rm{CO_{2}}}$')
        #ax1.set_xlabel(r'$\rm{reactor}\:\rm{length}\:/\:\rm{m}$')
        ax1.set_ylabel(r'$\rm{mole}\:\:\rm{fraction}$')
        ax1.set_ylim(0,0.8)
        ax1.set_xlim(reactor_lengths[0],reactor_lengths[-1])
        ax1.legend(loc='upper right', ncol=5)
        
        ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                        bottom=True, top=True, left=True, right=True, direction='in')
        
        ## Temperature plot
        ax2 = fig1.add_subplot(312, sharex = ax1)
        ax2.plot(reactor_lengths, self.T_avg, 'r-', label='temperature')
        #ax2.set_xlabel(r'$\rm{reactor}\:\rm{length}\:/\:\rm{m}$')
        ax2.set_ylabel(r'$\rm{temperature}\:/\:\rm{K}$')
        ax2.set_xlim(reactor_lengths[0],reactor_lengths[-1])
        
        ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                        bottom=True, top=True, left=True, right=True, direction='in')
        
        plt.tight_layout()
        
        ## Conversion and yield plot
        ax3 = fig1.add_subplot(313, sharex = ax2)
        ax3.plot(reactor_lengths, self.X_CH4, 'r-', label=r'$X_{\rm{CH}_{4}}$')
        ax3.plot(reactor_lengths, self.Y_CO2, 'b-', label=r'$Y_{\rm{CO}_{2}}$')
        ax3.set_xlabel(r'$\rm{reactor}\:\rm{length}\:/\:\rm{m}$')
        ax3.set_ylabel(r'$\rm{conversion},\:\rm{yield}$')
        ax3.set_xlim(reactor_lengths[0],reactor_lengths[-1])
        ax3.legend(loc='center right')
        
        ax3.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                        bottom=True, top=True, left=True, right=True, direction='in')
        
        plt.tight_layout()
        
        ## 2D temperature plot
        fig2 = plt.figure(figsize=(10,7))
        ax4 = fig2.add_subplot(111, projection='3d')
        T_2D = T_2D.transpose()
        T_2D = np.concatenate((np.flip(T_2D, axis=0),T_2D[0,:].reshape((1, int(T_2D.size/self.n_elements))),T_2D), axis=0)
        radii_2D = self.radii_2D
        radii_2D = np.concatenate((np.flip(radii_2D, axis=0),-1*radii_2D[1:,:]), axis=0)
        
        surf = ax4.plot_surface(reactor_lengths, radii_2D*1e3, T_2D, cmap='viridis', edgecolors='k', linewidth=0.5, antialiased=True)
        ax4.view_init(elev=90, azim=-90)
        
        ax4.set_xlim(reactor_lengths[0],reactor_lengths[-1])
        ax4.set_zticks([])
        
        cbar = fig2.colorbar(surf, shrink=0.7)
        cbar.ax.set_ylabel(r'$\rm{temperature}\:/\:\rm{K}$')
        
        ax4.set_xlabel(r'$\rm{reactor}\:\rm{length}\:/\:\rm{m}$')
        ax4.set_ylabel(r'$\rm{radius}\:/\:10^{-3}\rm{m}$')
        
        plt.title('Temperature profile 2D reactor')
        
        # Make panes transparent
        ax4.xaxis.pane.fill = False
        ax4.yaxis.pane.fill = False
        ax4.zaxis.pane.fill = False
        
        # Remove grid lines
        ax4.grid(False)
        
        # Transparent spines
        ax4.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Transparent panes
        ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        plt.tight_layout()

    
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size_NN, hidden_size_NN, output_size_NN, num_layers_NN, T0):
        # Inheritance of the methods and attributes of the parent class 
        super(NeuralNetwork, self).__init__()
        
        # Defining the hyperparameters of the neural network
        self.input_size = input_size_NN
        self.output_size = output_size_NN
        self.hidden_size  = hidden_size_NN
        self.num_layers = num_layers_NN
        self.layers = torch.nn.ModuleList()
        self.T0 = T0
        
        # Add the first, hidden and last layer to the neural network
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))
    
    # Define the forward pass
    def forward(self, x):     
        # All layers of the neural network are passed through
        for layer in self.layers[:-1]:
            x = layer(x)
            # The activation function tanh is used to introduce non-linearity 
            # into the neural network to solve non-linear problems.
            x = torch.tanh(x)
        x = self.layers[-1](x)
        # The exponential function is used to ensure that the quantities 
        # are always positive. The multiplication of T0 provides a scaling of 
        # the high temperatures to a value close to 1 -> T/T0.
        x[:,0] = torch.exp(x[:,0])
        x[:,1] = torch.exp(x[:,1])
        x[:,2] = torch.exp(x[:,2])
        x[:,3] = torch.exp(x[:,3])
        x[:,4] = torch.exp(x[:,4])
        x[:,5] = torch.exp(x[:,5]) * self.T0
        
        # Limit the values of the tensors so that no NaN are generated
        x = torch.clamp(x, min=1e-10, max=1e10)
        
        return x

class PINN_loss(torch.nn.Module):
    def __init__(self, weight_factors, epsilon, inlet_mole_fractions, \
                 bound_conds, reactor_conds):
        """
            New Args:
                weight_factors (list): weighting factors [-]
                epsilon (float): causality parameter [-]

            New Params:
                w_n (float): weighting factor of all loss functions which 
                              depends on all species [-]
                w_T (float): weighting factor of all loss functions which 
                             depends on the reactor temperature [-]
                w_GE_n (float): weighting factor from the loss function of 
                                the governing equation which depends on all species [-]
                w_GE_T (float): weighting factor from the loss function of 
                                the governing equation which depends on the 
                                reactor temperature [-]
                w_IC_n (float): weighting factor from the loss function of 
                                the initial condition which depends on all species [-]
                w_IC_T (float): weighting factor from the loss function of 
                                the initial condition which depends on the 
                                reactor temperature [-]
                               
            For the other arguments and parameters, please look in the
            class generate_data().
        """
        super(PINN_loss, self).__init__()
        
        # New parameter
        self.w_n, self.w_T, self.w_GE_n, self.w_GE_T, self.w_IC_n, \
            self.w_IC_T = weight_factors
        self.causality_parameter = torch.tensor(epsilon)
        
        # Parameter known from the class generate_data()
        self.inlet_mole_fractions = torch.tensor(inlet_mole_fractions[0:6])
        
        self.p = torch.tensor(bound_conds[0])
        self.u0 = torch.tensor(bound_conds[1])
        self.T0 = torch.tensor(bound_conds[2])
        self.T_wall = torch.tensor(bound_conds[3])
        
        self.eta = torch.tensor(reactor_conds[0])
        self.n_elements = reactor_conds[1]
        
        self.k0 = torch.tensor([4.225*1e15*math.sqrt(1e5),1.955*1e6*1e-5,1.020*1e15*math.sqrt(1e5)])
        self.E_A = torch.tensor([240.1*1e3,67.13*1e3,243.9*1e3])
        self.K_ads_0 = torch.tensor([8.23*1e-5*1e-5,6.12*1e-9*1e-5,6.65*1e-4*1e-5,1.77*1e5])
        self.G_R_ads = torch.tensor([-70.61*1e3,-82.90*1e3,-38.28*1e3,88.68*1e3])
        self.R = R
        self.d_pi = 0.0084
        self.d_in = 0.1016
        self.d_out = 0.1322
        self.A = math.pi * (self.d_in/2)**2
        self.V_dot_0 = self.u0 * 3600 * self.A
        self.em = 0.8
        self.epsilon = 0.4 + 0.05 * self.d_pi/self.d_in + 0.412 * \
            self.d_pi**2/self.d_in**2
        self.lambda_s = 0.3489
        self.rho_s = 2355
        self.rho_b = self.rho_s * (1-self.epsilon)
        
        self.cp_coef = torch.tensor([[19.238,52.09*1e-3,11.966*1e-6,-11.309*1e-9], \
                                     [32.22,1.9225*1e-3,10.548*1e-6,-3.594*1e-9], \
                                     [27.124,9.267*1e-3,-13.799*1e-6,7.64*1e-9], \
                                     [30.848,-12.84*1e-3,27.87*1e-6,-12.71*1e-9], \
                                     [19.78,73.39*1e-3,-55.98*1e-6,17.14*1e-9], \
                                     [31.128, -13.556*1e-3, 26.777*1e-6, -11.673*1e-9]])
        self.H_0 = torch.tensor([-74850, -241820, 0, -110540, -393500])
        self.S_0 = torch.tensor([-80.5467, -44.3736, 0,	89.6529, 2.9515])
        self.MW = torch.tensor([16.043, 18.02, 2.016, 28.01, 44.01, 28.01]) * 1e-3
        self.U_perV = torch.tensor(2.3e+05)
    
    def calc_inlet_ammount_of_substances(self):
        """
        Calculate the ammount of substances at the inlet of the PFR.
        
        New Params:
            total_concentration (tensor): Concentration of the gas [kmol m-3]
            concentrations (tensor): Concentrations of all species [kmol m-3]
            ammount_of_substances (tensor): Ammount of substances of all species [kmol h-1]
        """
        total_concentration = (self.p * 1e5 * 1e-3) / (self.R*self.T0)
        concentrations = total_concentration * self.inlet_mole_fractions
        ammount_of_substances = concentrations * self.V_dot_0
        
        self.n_CH4_0 = ammount_of_substances[0]
        self.n_H2O_0 = ammount_of_substances[1]
        self.n_H2_0 = ammount_of_substances[2]
        self.n_CO_0 = ammount_of_substances[3]
        self.n_CO2_0 = ammount_of_substances[4]
        self.n_N2_0 = ammount_of_substances[5]
    
    def calc_mu_gas(self, T, x):
        """
        Calculates the dynamic viscosities of the single components mu_i and then
        calculates the gas mixture dynamic viscosity.

        Args:
            T (float): temperature of the volume element [K]
            x (1D-array): ammount of substance of all species of the volume 
                            element [-]
        
        New Params:
            mu_mix (float): gas mixture dynamic viscosity [Pa s]
        """
        
        # PHYSICAL PROPERTIES OF LIQUIDS AND GASES, 
        # https://booksite.elsevier.com/9780750683661/Appendix_C.pdf
        # NASA Polynomial in CGS-Unit µP (Poise)
        mu_CH4 = 3.844 + 4.0112*1e-1*T + -1.4303*1e-4*T**2
        mu_H2O = -36.826 + 4.29*1e-1*T + -1.62*1e-5*T**2
        mu_H2 = 27.758 + 2.12*1e-1*T + -3.28*1e-5*T**2
        mu_CO = 23.811 + 5.3944*1e-1*T + -1.5411*1e-4*T**2
        mu_CO2 = 11.811 + 4.9838*1e-1*T + -1.0851*1e-4*T**2
        mu_N2 = 42.606 + 4.75*1e-1*T + -9.88*1e-5*T**2
        
        mu_gas = torch.cat([mu_CH4, mu_H2O, mu_H2, mu_CO, mu_CO2, mu_N2],dim=1) * 1e-6 # µP -> P
        
        # Method of Wilke: 
        # Chapter 9.5 THE PROPERTIES OF GASES AND LIQUIDS Bruce E. Poling / A 
        # Simple and Accurate Method for Calculatlng Viscosity of Gaseous Mixtures
        # by Thomas A. Davidson
        num_components = 6
        mu_mix = torch.zeros([len(x),num_components])
        for i in range(num_components):
            phi_row = (1 + (mu_gas[:,i].view(-1, 1) / mu_gas) ** (0.5) * (self.MW / self.MW[i]) ** (0.25)) ** 2 / \
                      (8 * (1 + (self.MW[i] / self.MW))) ** (0.5)

            mu_mix[:,i] = x[:,i]*mu_gas[:,i]/(torch.sum(phi_row * x, dim=1))
            
        mu_mix_sum = torch.sum(mu_mix,dim=1).view(-1, 1) * 0.1 # Convert CGS-Unit Poise to SI-Unit Pa*s 
        
        return mu_mix_sum
    
    def calc_area_radii(self, mole_fractions): 
        """
        Calculation of the areas and radii of the volume elements as a function 
        of temperature. In the radial direction, a pressure gradient is created 
        by the temperature gradient, which is taken into account using Darcy's law.
        
        Args:
            T (1D-array): temperature of the volume elements [K]
            x_i (1D-array): ammount of substance of all species of the volume 
                            elements [-]
        
        New Params:
            mu_gas (float): dynamic viscosity of gas mixture [Pa s]
            rho_gas (float): density of gas mixture [kg m-3]
            area_elements (1D-array): area of the volume elements [m]
            radii_elements (1D-array): radii of the volume elements starting 
                                       from the centre of the reactor [m]
        """
        
        # Calculate mu_gas and rho_gas
        mu_gas = self.calc_mu_gas(self.temperature, mole_fractions)
        rho_gas = torch.sum(torch.mul((mole_fractions * self.p * 1e5), self.MW), dim=1).view(-1, 1) / (self.R * self.temperature)
    
        # Calculate the area of the volume elements
        area_elements_devided_by_total_area = torch.div(mu_gas, rho_gas)/\
            torch.div(mu_gas,rho_gas).view(-1, self.n_elements).sum(dim=1).repeat(\
                    self.n_elements).view(-1, 1)
        area_elements = area_elements_devided_by_total_area * self.A
        
        # Calculate radii of the volume elements
        reshaped_area_elements = area_elements.view(int(len(mole_fractions)/self.n_elements),self.n_elements)
        radii_elements_columns  = [torch.zeros([int(len(mole_fractions)/self.n_elements),1])]
        for i in range(self.n_elements):
            radii_elements_column = torch.sqrt((reshaped_area_elements[:,i].unsqueeze(1)/math.pi)+radii_elements_columns[i]**2)
            radii_elements_columns.append(radii_elements_column)
        radii_elements = torch.cat(radii_elements_columns, dim=1)
        radii_elements_new = radii_elements.view(-1, 1)

        return (area_elements, radii_elements_new)
    
    def calc_flow_velocity(self, mole_fractions, area_elements):
        """
        Calculation of the flow velocity as a function of temperature and gas 
        composition.

        Args:
            mole_fractions (1D-array): ammount of substance of all species of 
                                       all volume elements [-]
            area_elements (1D-array): area of the volume elements [m]
            temperature (1D-array): temperature of all volume elements [K]
                                           
        New Params:
            molar_mass (float): molecular weight [kg mol-1]
            flow_velocity_average (float): Flow velocity averaged over all 
                                           volume elements [m/s]
        """
        
        # Calculation of the averaged mole fraction
        reshaped_mole_fractions = mole_fractions.view(-1, self.n_elements, mole_fractions.size(1))
        reshaped_area_elements = area_elements.view(-1, self.n_elements, 1)
        mole_fractions_weighted = reshaped_mole_fractions * reshaped_area_elements
        mole_fractions_average = mole_fractions_weighted.sum(dim=1) / reshaped_area_elements.sum(dim=1)
        
        # Calculation of the averaged temperature
        reshaped_temperature = self.temperature.view(-1, self.n_elements, 1)
        temperature_weighted = reshaped_temperature * reshaped_area_elements
        temperature_average = temperature_weighted.sum(dim=1) / reshaped_area_elements.sum(dim=1)

        # Calculation of the averaged molar mass
        molar_mass_0 = torch.dot(self.inlet_mole_fractions, self.MW).expand(mole_fractions.size(0) // self.n_elements, 1)
        molar_mass_average = torch.mm(mole_fractions_average, self.MW.view(-1, 1))

        # Calculate averaged flow velocity 
        flow_velocity_average = self.u0 * (temperature_average * molar_mass_0) / (self.T0 * molar_mass_average)  
        flow_velocity_average = flow_velocity_average.repeat(self.n_elements, 1)
        
        return flow_velocity_average 
    
    def calculate_thermo_properties(self, T):
        """
        Calculation of the reaction enthalpies and the equilibrium constants Kp.
        The temperature dependence of the heat capacity is considered with NASA 
        polynomials in order to consider the temperature dependence of the 
        enthalpies of formation and the entropies of formation. 
        
        New Params:
            H_i (tensor): enthalpies of formation [J mol-1]
            S_i (tensor): entropies of formation [J mol-1 K-1]
            H_R (tensor): reaction enthalpies [J mol-1]
            S_R (tensor): reaction entropies [J mol-1 K-1]
            G_R (tensor): reaction gibbs energies [J mol-1]
            Kp (tensor): equilibrium constant, Kp[1,3]=[Pa], Kp[2]=[-]
        """
        
        # Calculation of the standard formation enthalpy at given temperature
        # with Kirchhoffsches law
        H_i = torch.unsqueeze(self.H_0, 1).repeat(1, T.size(0)) + \
                torch.matmul(self.cp_coef[:5,:], torch.cat([torch.transpose(t, 0, 1) for t in \
                [T, (1/2)*T**2, (1/3)*T**3, (1/4)*(T)**4]], dim=0)) - \
                torch.matmul(self.cp_coef[:5,:], torch.cat([torch.transpose(t, 0, 1) for t in \
                [torch.full((T.size(0), 1), 298.15), (1/2)*torch.full((T.size(0), 1), 298.15)**2, \
                (1/3)*torch.full((T.size(0), 1), 298.15)**3, (1/4)*torch.full((T.size(0), 1), 298.15)**4]], dim=0))
            
        # Calculation of the standard formation entropy at given temperature
        S_i = torch.unsqueeze(self.S_0, 1).repeat(1, T.size(0)) + \
                torch.matmul(self.cp_coef[:5,:], torch.cat([torch.transpose(t, 0, 1) for t in \
                [torch.log(T), T, (1/2)*T**2, (1/3)*T**3]], dim=0)) - \
                torch.matmul(self.cp_coef[:5,:], torch.cat([torch.transpose(t, 0, 1) for t in \
                [torch.log(torch.full((T.size(0), 1), 298.15)), torch.full((T.size(0), 1), 298.15), \
                (1/2)*torch.full((T.size(0), 1), 298.15)**2, (1/3)*torch.full((T.size(0), 1), 298.15)**3]], dim=0))
                
        # Calculation of standard reaction enthalpies with Satz von Hess at 
        # standard conditions (T = 298.15 K)
        H_R = torch.zeros(3, len(T))
        H_R[0,:] = -H_i[0,:] - H_i[1,:] + H_i[3,:] + 3*H_i[2,:]
        H_R[1,:] = -H_i[3,:] - H_i[1,:] + H_i[4,:] + H_i[2,:]
        H_R[2,:] = -H_i[0,:] - 2*H_i[1,:] + H_i[4,:] + 4*H_i[2,:]
        
        # Calculation of standard reaction entropies with Satz von Hess at 
        # standard conditions (T = 298.15 K)
        S_R = torch.zeros(3, len(T))
        S_R[0,:] = -S_i[0,:] - S_i[1,:] + S_i[3,:] + 3*S_i[2,:]
        S_R[1,:] = -S_i[3,:] - S_i[1,:] + S_i[4,:] + S_i[2,:]
        S_R[2,:] = -S_i[0,:] - 2*S_i[1,:] + S_i[4,:] + 4*S_i[2,:]
        
        # Calculation of the free reaction enthalpy with the Gibbs Helmoltz equation
        G_R = H_R - T.t() * S_R
        
        # Calculation of the rate constants
        Kp = torch.exp(-G_R / (self.R*T.t())) * torch.cat((torch.full((1, len(T)), 1e10), \
                torch.full((1, len(T)), 1), torch.full((1, len(T)), 1e10)), dim=0)
        
        return [Kp.t(), H_R.t()]
                                                   

    def xu_froment(self, partial_pressures, area_elements):
        """
        Calculation of the differentials from the mass balance with the kinetic 
        approach of Xu, Froment.
        
        Args:
            partial_pressures (tensor): partial pressures [Pa]
            
        New Params:
            k (tensor): velocity constant, k[1,3]=[kmol Pa0.5 kgcat-1 h-1]
                                           k[2]=[kmol Pa-1 kgcat-1 h-1]
            K_ads (tensor): adsorption constants, K_ads[1-3]=[Pa-1]
            r_total (tensor): reaction rates [kmol kgcat-1 h-1]
            dn_dz (tensor): derivatives of the amounts of substances of the 
                            species in dependence of the reactor length [kmol m-1 h-1]
        """
        
        # Calculate derivative of the mass balance of each element
        dn_dz = torch.zeros([len(partial_pressures),6])
        r_total = torch.zeros([len(partial_pressures),3])
        
        # Calculate reaction rates with a Langmuir-Hinshelwood-Houghen-Watson approach
        k = self.k0 * torch.exp(-self.E_A/(self.R*self.temperature))
        K_ads = self.K_ads_0 * torch.exp(-self.G_R_ads/(self.R*self.temperature))
        Kp, H_R = PINN_loss.calculate_thermo_properties(self, self.temperature)
        
        DEN = 1 + partial_pressures[:,3]*K_ads[:,0] + partial_pressures[:,2]*K_ads[:,1] + \
            partial_pressures[:,0]*K_ads[:,2] + (partial_pressures[:,1]*K_ads[:,3]) / partial_pressures[:,2]
        r_total = torch.zeros(len(partial_pressures),3)
        r_total[:,0] = (k[:,0] / (partial_pressures[:,2]**2.5)) * (partial_pressures[:,0] * \
                    partial_pressures[:,1] - (((partial_pressures[:,2]**3) * partial_pressures[:,3]) / \
                    Kp[:,0])) / (DEN**2)
        r_total[:,1] = (k[:,1] / partial_pressures[:,2]) * (partial_pressures[:,3] * \
                    partial_pressures[:,1] - ((partial_pressures[:,2] * partial_pressures[:,4]) / \
                    Kp[:,1])) / (DEN**2)
        r_total[:,2] = (k[:,2] / (partial_pressures[:,2]**3.5)) * (partial_pressures[:,0] * \
                    (partial_pressures[:,1]**2) - (((partial_pressures[:,2]**4) * partial_pressures[:,4]) / \
                    Kp[:,2])) / (DEN**2)
        r_total = self.eta * r_total
            
        # Calculate derivatives of the mass balance
        dn_dz = torch.zeros([len(partial_pressures),5])
        dn_dz[:,0] = area_elements.squeeze() * (-r_total[:,0] - r_total[:,2]) * self.rho_b
        dn_dz[:,1] = area_elements.squeeze() * (-r_total[:,0] - r_total[:,1] - 2*r_total[:,2]) * self.rho_b
        dn_dz[:,2] = area_elements.squeeze() * (3*r_total[:,0] + r_total[:,1] + 4*r_total[:,2]) * self.rho_b
        dn_dz[:,3] = area_elements.squeeze() * (r_total[:,0] - r_total[:,1]) * self.rho_b
        dn_dz[:,4] = area_elements.squeeze() * (r_total[:,1] + r_total[:,2]) * self.rho_b
        
        return [dn_dz, r_total]
    
    def heat_balance(self, mole_fractions, r_total, flow_velocity, radii_elements):
        """
        Calculation of the differentials of the heat balance for each volume 
        element.

        Args:
            mole_fractions (1D-array): ammount of substance of all species of 
                                       all volume elements [-]
            temperature (1D-array): temperature of all volume elements [K]
            r_total (1D-array): reaction rates [kmol kgcat-1 h-1]
            flow_velocity (float): flow velocity of the gas [m s-1]
            radii_elements (1D-array): radii of the volume elements [m]
            
        New Params:
            heat_flow_rate (1D-array): heat flow rate through the volume elements
                                       [kJ m-2 h-1]
            mult_density_gas_cp (1D-array): product of density_gas and cp_gas [J m-3 K-1]
            density_gas (float): density of the gas [kg m-3]
            cp_i (1D-array): heat capacities of each component [J mol-1 K-1]
            alpha_w_int (float): Heat transfer coefficient between fixed bed and 
                                 wall [kJ m-2 h-1 K-1]
            lambda_rad (float): Thermal conductivity in the fixed bed 
                                [kJ m-1 h-1 K-1]
            cp_gas (float): heat capacity of the gas mixture [J kg-1 K-1]
            dTdz_cond (1D-array): heat exchange with environment [K m-1]
            dTdz_react (1D-array): heat production through the reaction [K m-1]
            dTdz (1D-array): derivatives of the temperature in dependence of the 
                             reactor length of each volume element [K m-1]
        """

        def calc_lambda_gas(T, x):
            """
            Calculates the conductivities of the single components lambda_i and then
            the gas mixture conductivity.
            
            Args:
                T (float): temperature of the volume element [K]
                x_i (1D-array): ammount of substance of all species of the volume 
                                element [-]
            """
            
            # PHYSICAL PROPERTIES OF LIQUIDS AND GASES, 
            # https://booksite.elsevier.com/9780750683661/Appendix_C.pdf
            lambda_CH4 = -0.00935 + 1.4028*1e-4*T + 3.318*1e-8*T**2
            lambda_H2O = 0.00053 + 4.7093*1e-5*T + 4.9551*1e-8*T**2
            lambda_H2 = 0.03951 + 4.5918*1e-4*T + -6.4933*1e-8*T**2
            lambda_CO = 0.00158 + 8.2511*1e-5*T + -1.9081*1e-8*T**2
            lambda_CO2 = -0.012 + 1.0208*1e-4*T + -2.2403*1e-8*T**2
            lambda_N2 = 0.00309 + 7.593*1e-5*T + -1.1014*1e-8*T**2

            lmbd_gas = torch.cat([lambda_CH4, lambda_H2O, lambda_H2, lambda_CO, lambda_CO2, lambda_N2],dim=1)
            
            # Method of Wassiljewa:
            # Chapter 10.6 THE PROPERTIES OF GASES AND LIQUIDS Bruce E. Poling / A 
            # Slmple and Accurate Method for Calculatlng Viscosity of Gaseous Mixtures
            # by Thomas A. Davidson
            # Calculation A(i,j) by Mason, Saxena: Mason EA, Saxena SC. Approximate 
            # formula for the thermal conductivity of gas mixtures. 
            # Phys Fluids 1958;1:361e9
            num_components = 6
            lmbd_mix = torch.zeros([len(x),num_components])
            for i in range(num_components):
                A_row = (1 + (lmbd_gas[:,i].view(-1, 1) / lmbd_gas) ** (0.5) * (self.MW[i] / self.MW) ** (0.25)) ** 2 / \
                        (8 * (1 + (self.MW[i] / self.MW))) ** (0.5)
                        
                lmbd_mix[:,i] = x[:,i] * lmbd_gas[:,i] / (torch.sum(A_row * x, dim=1))
                
            lmbd_mix_sum = torch.sum(lmbd_mix,dim=1).view(-1, 1)
            
            return lmbd_mix_sum
                    
        def calc_heat_transfer(T, x, rho_g, cp, u_gas):
            """
            Calculation of the heat transfer in the fixed bed (lmbd_er) and 
            between the fixed bed and the reactor wall (a_w). It is assumed 
            that there is no temperature gradient in the reactor wall. 
            Accordingly, the temperature on the outer reactor wall is 
            equal to the temperature in the inner reactor wall.

            Args:
                T (float): temperature of the volume element [K]
                x_i (1D-array): ammount of substance of all species of the volume 
                                element [-]
                rho_g (float): density of the gas mixture [kg m-3]
                cp_i (1D-array): heat capacities of each species [J mol-1 K-1]
                u_gas (float): flow velocity of the gas mixture [m s-1]
            
            New Params:
                a_w: Heat transfer coefficient between fixed bed and wall 
                     [kJ m-2 h-1 K-1]
                lmbd_er: Thermal conductivity in the fixed bed [kJ m-1 h-1 K-1]
            """
            
            mu_g = self.calc_mu_gas(T, x)
            lmbd_g = calc_lambda_gas(T, x)
            cps_g_mix = torch.sum(x * (cp * (1/self.MW)), dim=1).view(-1, 1)
            l_c = self.d_pi
            N_Re = rho_g * u_gas * l_c / mu_g
            N_Pr = cps_g_mix * mu_g / lmbd_g

            a_w = (1 - 1.5 * (self.d_in/self.d_pi)**(-1.5)) * \
                   (lmbd_g * 3.6)/self.d_pi * N_Re**(0.59) * N_Pr**(1/3)
            a_rs = 0.8171*(self.em/(2-self.em))*(T/100)**(3)
            a_ru = (0.8171*(T/100)**(3))/(1+self.epsilon/2 * (1-self.epsilon) * \
                    (1-self.em)/self.em)

            lmbd_er_0 = self.epsilon*(lmbd_g*3.6 + 0.95*a_ru*self.d_pi) + \
                        (0.95 * (1 - self.epsilon))/(2/(3*self.lambda_s*3.6) + \
                        1/(10 * lmbd_g*3.6 + a_rs * self.d_pi))
            
            lmbd_er = lmbd_er_0 + 0.111 * lmbd_g*3.6 * (N_Re * N_Pr**(1/3))/(1 + 46 * \
                      (self.d_pi/self.d_out)**(2))
            
            return (a_w, lmbd_er)
        
        # Calculate quantities for heat transfer
        density_gas = torch.sum(self.p * 1e5 * mole_fractions * self.MW, dim=1, keepdim=True) / \
                            (self.R * self.temperature)
        cp_i = self.cp_coef[:,0] + self.cp_coef[:,1] * self.temperature + self.cp_coef[:,2] * \
            self.temperature**2 + self.cp_coef[:,3] * self.temperature**3
        cp_gas = torch.mul(mole_fractions, cp_i/self.MW).sum(dim=1, keepdim=True)
        mult_density_gas_cp = cp_gas*density_gas*1e3
        
        # Calculate heat transfer
        alpha_w_int, lambda_rad = calc_heat_transfer(self.temperature,\
                mole_fractions, density_gas, cp_i, flow_velocity)
            
        """
        #!!!! später entfernen!!!
        alpha_w_int = torch.full((1000,1), 4737.8884769821625)
        lambda_rad = torch.full((1000,1), 122.78596899483958)
        #!!!!
        """
        
        alpha_w_int = alpha_w_int.view(int(len(alpha_w_int)/self.n_elements), self.n_elements).t()
        lambda_rad = lambda_rad.view(int(len(lambda_rad)/self.n_elements), self.n_elements).t()
        temperature = self.temperature.view(int(len(self.temperature)/self.n_elements), self.n_elements).t()
        radii_elements = radii_elements.view(int(len(self.temperature)/self.n_elements), self.n_elements+1).t()
        
        heat_flow_rate = torch.zeros([self.n_elements+1, int(len(self.temperature)/self.n_elements)])
        for i in range(self.n_elements):   
            if i == 0:
                # first boundary condition
                heat_flow_rate[i,:] = 0
            elif i == self.n_elements-1:
                heat_flow_rate[i,:] = lambda_rad[i,:]*(temperature[i-1,:]-temperature[i,:])/(radii_elements[i+1,:]-radii_elements[i,:])
                # second boundary condition
                heat_flow_rate[i+1,:] = (alpha_w_int[i,:]*lambda_rad[i,:]/(0.5*(radii_elements[i+1,:]-radii_elements[i,:])))/ \
                    (alpha_w_int[i,:]+(lambda_rad[i,:]/(0.5*(radii_elements[i+1,:]-radii_elements[i,:]))))* (temperature[-1,:]-self.T_wall)
            else:
        	    heat_flow_rate[i,:] = lambda_rad[i,:]*(temperature[i-1,:]-temperature[i,:])/(radii_elements[i+1,:]-radii_elements[i,:])
        self.heat_flow_rate = heat_flow_rate

        # Calculate derivative of the heat balance
        Kp, H_R = self.calculate_thermo_properties(self.temperature)
        s_H = -torch.sum((H_R*1e3)*(r_total)*self.rho_b, dim=1, keepdim=True)
        s_H = s_H.view(int(len(s_H)/self.n_elements), self.n_elements).t()
        flow_velocity = flow_velocity.view(int(len(flow_velocity)/self.n_elements), self.n_elements).t()
        
        dTdz = torch.zeros([self.n_elements, int(len(self.temperature)/self.n_elements)])
        for i in range(self.n_elements):
            dr = radii_elements[i+1,:]-radii_elements[i,:]
            # Calculation of the source term for external heat exchange
            dTdz_cond = (radii_elements[i,:]*heat_flow_rate[i,:]*1e3 - radii_elements[i+1,:]*heat_flow_rate[i+1,:]*1e3)/ \
                (dr *(radii_elements[i,:]+dr*0.5)*mult_density_gas_cp[i,:]*flow_velocity[i,:]*3.600)
            # Calculation of the source term for the reaction
            dTdz_react = s_H[i,:]/(mult_density_gas_cp[i,:]*flow_velocity[i,:]*3.600)
            dTdz[i,:] = dTdz_cond+dTdz_react           
        
        dTdz = torch.reshape(dTdz.t(), (dTdz.shape[1]*self.n_elements, 1))
        return dTdz
    
    def calc_IC_loss(self, y_pred, x):
        """
        Calculate the loss function of the initial condition.
        
        Args:
            y_pred (tensor): Predicted output. Here ammount of substances from all
                             species [kmol h-1] and temperature [K].
            x (tensor): Input values. Here reactor length [m].
        """
        
        # Calculation of the mean square displacement between the predicted and 
        # original initial conditions
        loss_IC_CH4 = (torch.sum(y_pred[0:self.n_elements, 0]) - self.n_CH4_0)**2
        loss_IC_H2O = (torch.sum(y_pred[0:self.n_elements, 1]) - self.n_H2O_0)**2
        loss_IC_H2 = (torch.sum(y_pred[0:self.n_elements, 2]) - self.n_H2_0)**2
        loss_IC_CO = (torch.sum(y_pred[0:self.n_elements, 3]) - self.n_CO_0)**2
        loss_IC_CO2 = (torch.sum(y_pred[0:self.n_elements, 4]) - self.n_CO2_0)**2
        loss_IC_T = (torch.mean(y_pred[0:self.n_elements, 5]) - self.T0)**2
        
        return [loss_IC_CH4, loss_IC_H2O, loss_IC_H2, loss_IC_CO, loss_IC_CO2, loss_IC_T]
    
    def calc_GE_loss(self, y_pred, x):
        """
        Calculate the loss function of the governing equation.
        
        Args:
            y_pred (tensor): Predicted output. Here ammount of substances from all
                             species [kmol h-1] and temperature [K].
            x (tensor): Input values. Here reactor length [m].
        """
        
        ## Calculate the gradients of tensor values
        dn_dz_CH4 = torch.autograd.grad(outputs=y_pred[:, 0], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 0]),
                                retain_graph=True, create_graph=True)[0]
        dn_dz_H2O = torch.autograd.grad(outputs=y_pred[:, 1], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 1]),
                                retain_graph=True, create_graph=True)[0]     
        dn_dz_H2 = torch.autograd.grad(outputs=y_pred[:, 2], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 2]),
                                retain_graph=True, create_graph=True)[0]  
        dn_dz_CO = torch.autograd.grad(outputs=y_pred[:, 3], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 3]),
                                retain_graph=True, create_graph=True)[0]  
        dn_dz_CO2 = torch.autograd.grad(outputs=y_pred[:, 4], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 4]),
                                retain_graph=True, create_graph=True)[0]  
        dT_dz = torch.autograd.grad(outputs=y_pred[:, 5], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 5]),
                                retain_graph=True, create_graph=True)[0]
        
        ## Calculate the differentials
        # Calculate the mole fractions     
        mole_fractions = torch.cat([y_pred[:,:5], (self.n_N2_0/self.n_elements)*torch.ones(len(y_pred), 1)], dim=1)/ \
            torch.sum(torch.cat([y_pred[:,:5], (self.n_N2_0/self.n_elements)*torch.ones(len(y_pred), 1)], dim=1), dim=1).view(-1, 1)
        
        """
        #!!!!!!!!Diesen Bereich später löschen!!!!!!!!
        mole_fractions[:,0] = 0.2128
        mole_fractions[:,1] = 0.714
        mole_fractions[:,2] = 0.0259
        mole_fractions[:,3] = 0.0004
        mole_fractions[:,4] = 0.0119
        mole_fractions[:,5] = 0.035
        self.temperature[:] = 793
        #!!!!!!!!
        """
        
        # Calculate partial pressures
        partial_pressures = self.p * 1e5 * mole_fractions
        
        # Calculate the element area and radii with darcy's law
        area_elements, radii_elements = PINN_loss.calc_area_radii(self, mole_fractions)

        # Consider dependence of temperature and gas composition of the flow velocity
        flow_velocity = PINN_loss.calc_flow_velocity(self, mole_fractions, area_elements)
        
        # Calculate the differentials from the mass balance
        dn_dz_pred, r_total_pred = PINN_loss.xu_froment(self, partial_pressures, area_elements)
        
        # Calculate the differential from the heat balance
        dT_dz_pred = PINN_loss.heat_balance(self, mole_fractions, r_total_pred, flow_velocity, radii_elements)
        
        ## Calculation of the mean square displacement between the gradients of 
        ## autograd and differentials of the mass/ heat balance
        loss_GE_CH4 = (dn_dz_CH4 - dn_dz_pred[:,0].unsqueeze(1))**2
        loss_GE_H2O = (dn_dz_H2O - dn_dz_pred[:,1].unsqueeze(1))**2
        loss_GE_H2 = (dn_dz_H2 - dn_dz_pred[:,2].unsqueeze(1))**2
        loss_GE_CO = (dn_dz_CO - dn_dz_pred[:,3].unsqueeze(1))**2
        loss_GE_CO2 = (dn_dz_CO2 - dn_dz_pred[:,4].unsqueeze(1))**2
        loss_GE_T = (dT_dz - dT_dz_pred)**2
        
        return [loss_GE_CH4, loss_GE_H2O, loss_GE_H2, loss_GE_CO, loss_GE_CO2, loss_GE_T]
    
    def calc_BC_loss(self, y_pred, x):
        # Use the heat flow rate from the calculation of dTdz from GE
        dq_dt_center = torch.autograd.grad(outputs=self.heat_flow_rate[0,:], inputs=x,
                                grad_outputs=torch.ones_like(self.heat_flow_rate[0,:]),
                                retain_graph=True, create_graph=True)[0]
        dq_dt_wall = torch.autograd.grad(outputs=self.heat_flow_rate[-1,:], inputs=x,
                                grad_outputs=torch.ones_like(self.heat_flow_rate[-1,:]),
                                retain_graph=True, create_graph=True)[0]  

        # Calculation of losses
        loss_BC_center = (dq_dt_center[::self.n_elements] - self.heat_flow_rate[0,:].unsqueeze(1))**2
        loss_BC_wall = (dq_dt_wall[::self.n_elements] - self.heat_flow_rate[-1,:].unsqueeze(1))**2
        
        return [loss_BC_center, loss_BC_wall]
        
        
    def causal_training(self, x, loss_IC_CH4, loss_IC_H2O, loss_IC_H2, \
                        loss_IC_CO, loss_IC_CO2, loss_IC_T, loss_GE_CH4, \
                        loss_GE_H2O, loss_GE_H2, loss_GE_CO, loss_GE_CO2, \
                        loss_GE_T):
        """
        Previously, we used torch.mean() to average all the losses and the 
        neural network tried to reduce the gradient of all the points equally. 
        With this function, we implement a new approach in which the points are 
        optimised one after the other by considering weighting factors for the 
        individual points. The idea behind this is that the points are related 
        to each other and the error in the points at the beginning has an 
        influence on the error in the points later and thus accumulates. For 
        this reason, the prediction by the neural network has so far been very 
        good at the beginning and very bad at the end of the points. 
        """
        
        # Form the sum of the losses
        losses = torch.zeros_like(x)
        losses[0:self.n_elements] = loss_IC_CH4 + loss_IC_H2O + loss_IC_H2 + loss_IC_CO + \
            loss_IC_CO2 + loss_IC_T + loss_GE_CH4[0:self.n_elements] + loss_GE_H2O[0:self.n_elements] + \
                loss_GE_H2[0:self.n_elements] + loss_GE_CO[0:self.n_elements] + loss_GE_CO2[0:self.n_elements] + loss_GE_T[0:self.n_elements]
        losses[self.n_elements:] = loss_GE_CH4[self.n_elements:] + loss_GE_H2O[self.n_elements:] + loss_GE_H2[self.n_elements:] + \
            loss_GE_CO[self.n_elements:] + loss_GE_CO2[self.n_elements:] + loss_GE_T[self.n_elements:]
        
        # Calculate the weighting factors of the losses
        weight_factors = torch.zeros([int(losses.size(0)/self.n_elements)])
        for i in range(weight_factors.size(0)):
            w_i = torch.exp(-self.causality_parameter*torch.sum(losses[0:(i+1)*self.n_elements]))
            weight_factors[i] = w_i
        weight_factors = weight_factors.repeat_interleave(self.n_elements)
        self.weight_factors = weight_factors

        # Consider the weighting factors in the losses
        total_loss = torch.mean(weight_factors * losses)
            
        return total_loss
         
    def forward(self, x, y, y_pred):
        """
        Calculation of the total loss.
        
        Args:
            x (tensor): Input values. Here reactor length [m].
            y (tensor): Training data. Here ammount of substances from all species [kmol h-1]
                        and the temperature in the reactor [K].
            y_pred (tensor): Predicted output. Here ammount of substances from all 
                             species [kmol h-1] and the temperature in the reactor [K].
        """
        
        # Extract temperature
        self.temperature = y_pred[:, 5].reshape(-1, 1)
        
        # Calculation of the total loss
        loss_IC_CH4, loss_IC_H2O, loss_IC_H2, loss_IC_CO, \
            loss_IC_CO2, loss_IC_T = self.calc_IC_loss(y_pred, x)
        loss_GE_CH4, loss_GE_H2O, loss_GE_H2, loss_GE_CO, \
            loss_GE_CO2, loss_GE_T = self.calc_GE_loss(y_pred, x)
        loss_BC_center, loss_BC_wall = self.calc_BC_loss(y_pred, x)
    
        # Store losses before weighting
        losses_before_weighting = np.array([np.mean(loss_IC_CH4.detach().numpy()), \
                                   np.mean(loss_IC_H2O.detach().numpy()), \
                                   np.mean(loss_IC_H2.detach().numpy()), \
                                   np.mean(loss_IC_CO.detach().numpy()), \
                                   np.mean(loss_IC_CO2.detach().numpy()), \
                                   np.mean(loss_IC_T.detach().numpy()), \
                                   np.mean(loss_GE_CH4.detach().numpy()), \
                                   np.mean(loss_GE_H2O.detach().numpy()), \
                                   np.mean(loss_GE_H2.detach().numpy()), \
                                   np.mean(loss_GE_CO.detach().numpy()), \
                                   np.mean(loss_GE_CO2.detach().numpy()), \
                                   np.mean(loss_GE_T.detach().numpy())])

        # Consider weighting factors of the loss functions
        loss_IC_CH4 = self.w_IC_n * self.w_n * loss_IC_CH4
        loss_IC_H2O = self.w_IC_n * self.w_n * loss_IC_H2O
        loss_IC_H2 = self.w_IC_n * self.w_n * loss_IC_H2
        loss_IC_CO = self.w_IC_n * self.w_n * loss_IC_CO
        loss_IC_CO2 = self.w_IC_n * self.w_n * loss_IC_CO2
        loss_IC_T = self.w_IC_T * self.w_T * loss_IC_T
        loss_GE_CH4 = self.w_GE_n * self.w_n * loss_GE_CH4
        loss_GE_H2O = self.w_GE_n * self.w_n * loss_GE_H2O
        loss_GE_H2 = self.w_GE_n * self.w_n * loss_GE_H2
        loss_GE_CO = self.w_GE_n * self.w_n * loss_GE_CO
        loss_GE_CO2 = self.w_GE_n * self.w_n * loss_GE_CO2
        loss_GE_T = self.w_GE_T * self.w_T * loss_GE_T
        
        # Store losses after weighting
        losses_after_weighting = np.array([np.mean(loss_IC_CH4.detach().numpy()), \
                                   np.mean(loss_IC_H2O.detach().numpy()), \
                                   np.mean(loss_IC_H2.detach().numpy()), \
                                   np.mean(loss_IC_CO.detach().numpy()), \
                                   np.mean(loss_IC_CO2.detach().numpy()), \
                                   np.mean(loss_IC_T.detach().numpy()), \
                                   np.mean(loss_GE_CH4.detach().numpy()), \
                                   np.mean(loss_GE_H2O.detach().numpy()), \
                                   np.mean(loss_GE_H2.detach().numpy()), \
                                   np.mean(loss_GE_CO.detach().numpy()), \
                                   np.mean(loss_GE_CO2.detach().numpy()), \
                                   np.mean(loss_GE_T.detach().numpy())])
        
        # Calculate the total loss
        total_loss = PINN_loss.causal_training(self, x, loss_IC_CH4, loss_IC_H2O, \
                        loss_IC_H2, loss_IC_CO, loss_IC_CO2, loss_IC_T, loss_GE_CH4, \
                            loss_GE_H2O, loss_GE_H2, loss_GE_CO, loss_GE_CO2, loss_GE_T)
        
        return [total_loss, losses_before_weighting, losses_after_weighting]

def train(x, y, network, calc_loss, optimizer, num_epochs, plot_interval, analytical_solution, n_N2_0, n_elements, reactor_lengths):    
    def closure():
        """
        The Closure function is required for the L-BFGS optimiser.
        """
        # Set gradient to zero
        if torch.is_grad_enabled():
            optimizer.zero_grad()
            
        # Forward pass
        ypred = network(x)
        
        # Compute loss
        total_loss, losses_before_weighting, losses_after_weighting = \
            calc_loss(x, y, ypred)
        
        # Backward pass
        if total_loss.requires_grad:
            total_loss.backward()

        return total_loss
    
    def prepare_plots_folder():
        """
        This function is executed at the beginning of the training to prepare 
        the folder with the plots. It checks whether the folder for the plots 
        is existing. If it does not exist, the folder is created. If it is 
        exists, all files in the folder are deleted.
        """
        
        # Path to the folder with the Python script
        script_folder = os.path.dirname(os.path.abspath(__file__))
        folder_name = "plots"
        folder_path = os.path.join(script_folder, folder_name)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            # Create folder if it does not exist
            os.makedirs(folder_path)
        else:
            # Delete files in the folder if it exists
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Fehler beim Löschen der Datei {file_path}: {e}")
        
        print(f"Der Ordner '{folder_name}' wurde überprüft und gegebenenfalls erstellt bzw. geleert.")
    
    def calc_results(y_pred, n_elements):
        """
        Calculate the 
        
        Args:
            y (2D-array): Prediction of the reactor model containing the 
                          ammount of substances [kmol h-1] and temperatures [K] 
                          of each element as a function of the reactor length.
        
        New Params:
            X_CH4 (1D-array): Conversion of methane averaged over all volume 
                              elements as a function of reactor length [-]
            Y_CO2 (1D-array): Yield of carbon dioxide averaged over all volume 
                              elements as a function of reactor length [-]
            T_avg (1D-array): Reactor temperature averaged over all volume 
                              elements as a function of reactor length [K]
        """
        
        # Calculate mole fractions
        x_CH4 = np.sum(y_pred[:,0].reshape(-1, n_elements), axis=1)/np.sum(np.sum(y_pred[:,:6], axis=1).reshape(-1, n_elements), axis=1)
        x_H2O = np.sum(y_pred[:,1].reshape(-1, n_elements), axis=1)/np.sum(np.sum(y_pred[:,:6], axis=1).reshape(-1, n_elements), axis=1)
        x_H2 = np.sum(y_pred[:,2].reshape(-1, n_elements), axis=1)/np.sum(np.sum(y_pred[:,:6], axis=1).reshape(-1, n_elements), axis=1)
        x_CO = np.sum(y_pred[:,3].reshape(-1, n_elements), axis=1)/np.sum(np.sum(y_pred[:,:6], axis=1).reshape(-1, n_elements), axis=1)
        x_CO2 = np.sum(y_pred[:,4].reshape(-1, n_elements), axis=1)/np.sum(np.sum(y_pred[:,:6], axis=1).reshape(-1, n_elements), axis=1)
        x_N2 = np.sum(y_pred[:,5].reshape(-1, n_elements), axis=1)/np.sum(np.sum(y_pred[:,:6], axis=1).reshape(-1, n_elements), axis=1)
        
        # Calculate conversion of CH4
        X_CH4 = (np.sum(y_pred[:n_elements,0])-np.sum(y_pred[:,0].reshape(-1, n_elements), axis=1)) / \
            np.sum(y_pred[:n_elements,0])
        
        # Calculate yield of CO2
        Y_CO2 = (np.sum(y_pred[:,4].reshape(-1, n_elements), axis=1)-np.sum(y_pred[:n_elements,4])) / \
            np.sum(y_pred[:n_elements,0])
            
        # Calculate temperature
        T_avg = np.sum(y_pred[:,6].reshape(-1, n_elements),axis=1)/n_elements
        
        return np.column_stack((x_CH4, x_H2O, x_H2, x_CO, x_CO2, x_N2, X_CH4, Y_CO2, T_avg))
        
    
    # Prepare folders for the plots 
    prepare_plots_folder()
    
    # Calculate the ammount of substances at the inlet of the reactor
    calc_loss.calc_inlet_ammount_of_substances()
    
    # Training loop
    loss_values = np.zeros((num_epochs,25))
    for epoch in range(num_epochs):
        # Count the time per epoch
        start_time = time.time()
        # Updating the network parameters with calculated losses and gradients 
        # from the closure-function
        optimizer.step(closure)

        # Calculation of the current total loss after updating the network parameters 
        # in order to print them in the console afterwards and plot them.
        total_loss, losses_before_weighting, losses_after_weighting = \
            calc_loss(x, y, network(x))
        loss_values[epoch,:] = np.hstack((np.array(total_loss.detach().numpy()),\
                                          losses_before_weighting,losses_after_weighting))
            
        end_time = time.time()
        execution_time = end_time - start_time
        print('Epoch: ', epoch+1, 'Loss: ', total_loss.item(), 'causal weights sum: ', \
              np.round(np.sum(calc_loss.weight_factors.detach().numpy()),decimals=4), \
                  'Time_per_epoch: ', execution_time)

        # Create plots in the given plot_interval
        if (epoch+1)%plot_interval == 0 or epoch == 0 or epoch == num_epochs-1:
            # Predict ammount of substances an temperature with the neural network
            y_pred = network(x)
            y_pred = y_pred.detach().numpy()
            y_pred = np.insert(y_pred, 5, n_N2_0, axis=1)
            
            predicted_solution = calc_results(y_pred, n_elements)
            
            # Plot ammount of substances, temperature and weight factors
            plots(reactor_lengths, analytical_solution, predicted_solution, n_elements, save_plot=True, \
                        plt_num=epoch+1, weight_factors=calc_loss.weight_factors)
    
    ## Create final plots
    # Predict ammount of substances an temperature with the neural network
    y_pred = network(x)
    y_pred = y_pred.detach().numpy()
    y_pred = np.insert(y_pred, 5, n_N2_0, axis=1)
    
    predicted_solution = calc_results(y_pred, n_elements)
    
    # Plot ammount of substances, temperature and weight factors
    plots(reactor_lengths, analytical_solution, predicted_solution, n_elements, save_plot = True, \
                plt_num = f'{num_epochs}_final', weight_factors = calc_loss.weight_factors, loss_values=loss_values)
       

def plots(reactor_lengths, analytical_solution, predicted_solution, n_elements,\
      save_plot=False, plt_num=None, weight_factors=None, loss_values=None, msd_NN=None):
    """
    This function is used to save the plots during the training and after the 
    training. 
    
    Plots during training: mole fractions, temperature, weight factors 
    Plots after training: mole fractions, temperature, losses
    """
    # Path to the folder with the plots
    script_folder = os.path.dirname(os.path.abspath(__file__))
    folder_name = "plots"
    folder_path = os.path.join(script_folder, folder_name)
    
    # Mole fractions plot
    plt.figure()
    plt.plot(reactor_lengths, analytical_solution[:,0], 'g-', label=r'$x_{\rm{CH_{4}}}$')
    plt.scatter(reactor_lengths, predicted_solution[:,0], color = 'g', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{CH_{4},pred}}$')
    plt.plot(reactor_lengths, analytical_solution[:,1], 'r-', label=r'$x_{\rm{H_{2}O}}$')
    plt.scatter(reactor_lengths, predicted_solution[:,1], color = 'r', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{H_{2}O,pred}}$')
    plt.plot(reactor_lengths, analytical_solution[:,2], 'm-', label=r'$x_{\rm{H_{2}}}$')
    plt.scatter(reactor_lengths, predicted_solution[:,2], color = 'm', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{H_{2},pred}}$')
    plt.plot(reactor_lengths, analytical_solution[:,3], 'c-', label=r'$x_{\rm{CO}}$')
    plt.scatter(reactor_lengths, predicted_solution[:,3], color = 'c', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{CO,pred}}$')
    plt.plot(reactor_lengths, analytical_solution[:,4], 'y-', label=r'$x_{\rm{CO_{2}}}$')
    plt.scatter(reactor_lengths, predicted_solution[:,4], color = 'y', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{CO_{2},pred}}$')
    plt.plot(reactor_lengths, analytical_solution[:,5], 'b-', label=r'$x_{\rm{N_{2}}}$')
    plt.scatter(reactor_lengths, predicted_solution[:,5], color = 'b', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{N_{2},pred}}$')
    plt.xlabel(r'$reactor\:length\:/\:\rm{m}$')
    plt.ylabel(r'$mole\:fractions$')
    plt.ylim(0,0.75)
    plt.xlim(reactor_lengths[0],reactor_lengths[-1])
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        
    if save_plot:
        plt.savefig(f'{folder_path}/mole_fraction_{plt_num}.png', dpi=200)
        plt.close()
        
    # Temperature plot
    plt.figure()
    plt.plot(reactor_lengths, analytical_solution[:,8], 'r-', label=r'$T$')
    plt.scatter(reactor_lengths, predicted_solution[:,8], color = 'g', s=12, alpha=0.8, \
                edgecolors='b', label=r'$T_{\rm{pred}}$')
    plt.xlabel(r'$reactor\:length\:/\:\rm{m}$')
    plt.ylabel(r'$temperature\:/\:\rm{K}$')
    #plt.ylim(analytical_solution[0,8],analytical_solution[-1,8])
    plt.xlim(reactor_lengths[0],reactor_lengths[-1])
    plt.legend(loc='center right')
    
    if save_plot:
        plt.savefig(f'{folder_path}/temperature_{plt_num}.png', dpi=200)
        plt.close()
       
    # Plot weighting factors from causal training
    if weight_factors is not None:
        weight_factors_plot = np.mean(weight_factors.detach().numpy().reshape(-1, n_elements), axis=1)
        
        plt.figure()
        plt.plot(reactor_lengths, weight_factors_plot, 'r-', \
                 label=f'$sum:\:{np.round(np.sum(weight_factors.detach().numpy()),decimals=4)}$')
        plt.xlabel(r'$reactor\:length\:/\:\rm{m}$')
        plt.ylabel(r'$causal\:weighting\:factor$')
        plt.ylim(0,1)
        plt.xlim(reactor_lengths[0],reactor_lengths[-1])
        plt.legend(loc='upper right')
        
        if save_plot:
            plt.savefig(f'{folder_path}/causal_weights_{plt_num}.png', dpi=200)
            plt.close()

    # Plot losses without considering weighting factors
    if loss_values is not None:
        plt.figure()
        epochs = list(range(1, len(loss_values)+1))
        plt.plot(epochs, loss_values[:,0], '-', color='black', label=r'$L_{\rm{total}}$')
        plt.plot(epochs, loss_values[:,1], '-', color='tab:blue', label=r'$L_{\rm{IC,CH_{4}}}$')
        plt.plot(epochs, loss_values[:,2], '-', color='tab:orange', label=r'$L_{\rm{IC,H_{2}O}}$')
        plt.plot(epochs, loss_values[:,3], '-', color='tab:green', label=r'$L_{\rm{IC,H_{2}}}$')
        plt.plot(epochs, loss_values[:,4], '-', color='tab:red', label=r'$L_{\rm{IC,CO}}$')
        plt.plot(epochs, loss_values[:,5], '-', color='tab:purple', label=r'$L_{\rm{IC,CO_{2}}}$')
        plt.plot(epochs, loss_values[:,6], '-', color='tab:olive', label=r'$L_{\rm{IC,T}}$')
        plt.plot(epochs, loss_values[:,7], '--', color='tab:blue', label=r'$L_{\rm{GE,CH_{4}}}$')
        plt.plot(epochs, loss_values[:,8], '--', color='tab:orange', label=r'$L_{\rm{GE,H_{2}O}}$')
        plt.plot(epochs, loss_values[:,9], '--', color='tab:green', label=r'$L_{\rm{GE,H_{2}}}$')
        plt.plot(epochs, loss_values[:,10], '--', color='tab:red', label=r'$L_{\rm{GE,CO}}$')
        plt.plot(epochs, loss_values[:,11], '--', color='tab:purple', label=r'$L_{\rm{GE,CO_{2}}}$')
        plt.plot(epochs, loss_values[:,12], '--', color='tab:olive', label=r'$L_{\rm{GE,T}}$')
        
        plt.xlabel(r'$Number\:of\:epochs$')
        plt.ylabel(r'$losses$')
        plt.yscale('log')
        plt.xlim(0,epochs[-1])
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        
        if save_plot:
            plt.savefig(f'{folder_path}/loss_values_before_{plt_num}.png', dpi=200)
            plt.close()
        
        plt.figure()
        epochs = list(range(1, len(loss_values)+1))
        plt.plot(epochs, loss_values[:,0], '-', color='black', label=r'$L_{\rm{total}}$')
        plt.plot(epochs, loss_values[:,13], '-', color='tab:blue', label=r'$L_{\rm{IC,CH_{4}}}$')
        plt.plot(epochs, loss_values[:,14], '-', color='tab:orange', label=r'$L_{\rm{IC,H_{2}O}}$')
        plt.plot(epochs, loss_values[:,15], '-', color='tab:green', label=r'$L_{\rm{IC,H_{2}}}$')
        plt.plot(epochs, loss_values[:,16], '-', color='tab:red', label=r'$L_{\rm{IC,CO}}$')
        plt.plot(epochs, loss_values[:,17], '-', color='tab:purple', label=r'$L_{\rm{IC,CO_{2}}}$')
        plt.plot(epochs, loss_values[:,18], '-', color='tab:olive', label=r'$L_{\rm{IC,T}}$')
        plt.plot(epochs, loss_values[:,19], '--', color='tab:blue', label=r'$L_{\rm{GE,CH_{4}}}$')
        plt.plot(epochs, loss_values[:,20], '--', color='tab:orange', label=r'$L_{\rm{GE,H_{2}O}}$')
        plt.plot(epochs, loss_values[:,21], '--', color='tab:green', label=r'$L_{\rm{GE,H_{2}}}$')
        plt.plot(epochs, loss_values[:,22], '--', color='tab:red', label=r'$L_{\rm{GE,CO}}$')
        plt.plot(epochs, loss_values[:,23], '--', color='tab:purple', label=r'$L_{\rm{GE,CO_{2}}}$')
        plt.plot(epochs, loss_values[:,24], '--', color='tab:olive', label=r'$L_{\rm{GE,T}}$')
        
        plt.xlabel(r'$Number\:of\:epochs$')
        plt.ylabel(r'$losses$')
        plt.yscale('log')
        plt.xlim(0,epochs[-1])
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        
        if save_plot:
            plt.savefig(f'{folder_path}/loss_values_after_{plt_num}.png', dpi=200)
            plt.close()

        
if __name__ == "__main__":
    # Define parameters for the model
    reactor_lengths = np.linspace(0,12,num=100)
    inlet_mole_fractions = [0.2128,0.714,0.0259,0.0004,0.0119,0.035] #CH4,H20,H2,CO,CO2,N2
    reactor_radii = np.array([[0.],[0.01606437],[0.02271845],[0.02782431],[0.03212874],[0.03592102],[0.03934951],[0.04250233],[0.0454369 ],[0.04819311],[0.0508    ]])
    bound_conds = [25.7,2.14,793,1100] #p,u_in,T_in,T_wall
    reactor_conds = [0.007, 10] #eta, n_elements
    
    plot_analytical_solution = True #True,False
    
    input_size_NN = 1
    hidden_size_NN = 32
    output_size_NN = 6
    num_layers_NN = 3
    num_epochs = 50
    weight_factors = [1,1,1,1,1,1] #w_n,w_T,w_GE_n,w_GE_T,w_IC_n,w_IC_T
    epsilon = 0 #1e-7 #epsilon=0: old model, epsilon!=0: new model
    plot_interval = 10 # Plotting during NN-training
    
    # Calculation of the analytical curves
    model = generate_data(inlet_mole_fractions, bound_conds, reactor_conds)
    y = model.solve_ode(reactor_lengths, plot=plot_analytical_solution)
    
    analytical_solution = np.column_stack((model.x_CH4, model.x_H2O, model.x_H2, model.x_CO, \
                           model.x_CO2, model.x_N2, model.X_CH4, model.Y_CO2, \
                               model.T_avg))
    
    # Set up the neural network
    network = NeuralNetwork(input_size_NN=input_size_NN, hidden_size_NN=hidden_size_NN,\
                            output_size_NN=output_size_NN, num_layers_NN=num_layers_NN,\
                            T0 = bound_conds[2])
    optimizer = torch.optim.LBFGS(network.parameters(), lr=1, line_search_fn= \
                                  "strong_wolfe", max_eval=None, tolerance_grad \
                                      =1e-50, tolerance_change=1e-50)

    #x = torch.tensor(np.array([[i,j] for i in reactor_lengths for j in reactor_radii]), requires_grad=True)
    x = torch.tensor(np.repeat(reactor_lengths, reactor_conds[1]).reshape(-1, 1), requires_grad=True)
    #y = None
    
    # Train the neural network
    calc_loss = PINN_loss(weight_factors, epsilon, inlet_mole_fractions, \
                          bound_conds, reactor_conds) 
    train(x, y, network, calc_loss, optimizer, num_epochs, plot_interval, \
          analytical_solution, model.n_N2_0, reactor_conds[1], reactor_lengths)
    
    # Save model parameters
    #torch.save(network.state_dict(), 'PINN_state_03.10.23.pth')
