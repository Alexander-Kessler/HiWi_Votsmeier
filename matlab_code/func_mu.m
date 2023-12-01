function [mu_mix, mu_gas] = func_mu(T,xi,data)
% Function func_mu
% 
% --- Description ---
% Calculates the dynamic viscosities of the single components mu_i and then
% the gas mixture dynamic viscosity.
% 
% --- Dependencies ---
% none
% 
% --- Input ---
% Float Temperature : T in K
% Vector amount of substance : xi 
% 
% --- Output ---
% Float mu_mix in Pa*s

xi = reshape(xi, [6, 1]);

% PHYSICAL PROPERTIES OF LIQUIDS AND GASES, 
% https://booksite.elsevier.com/9780750683661/Appendix_C.pdf
% NASA Polynomial in CGS-Unit µP (Poise)
mu_CH4 = 3.844 + 4.0112*1e-1*T + -1.4303*1e-4*T^2;
mu_H2O = -36.826 + 4.29*1e-1*T + -1.62*1e-5*T^2;
mu_H2 = 27.758 + 2.12*1e-1*T + -3.28*1e-5*T^2;
mu_CO = 23.811 + 5.3944*1e-1*T + -1.5411*1e-4*T^2;
mu_CO2 = 11.811 + 4.9838*1e-1*T + -1.0851*1e-4*T^2;
mu_N2 = 42.606 + 4.75*1e-1*T + -9.88*1e-5*T^2;

mu_gas = [mu_CH4, mu_H2O, mu_H2, mu_CO, mu_CO2, mu_N2] * 1e-6; %µP -> P

% Method of Wilke: 
% Chapter 9.5 THE PROPERTIES OF GASES AND LIQUIDS Bruce E. Poling / A 
% Slmple and Accurate Method for Calculatlng Viscosity of Gaseous Mixtures
% by Thomas A. Davidson
mu_mix = 0;
phi = zeros(numel(xi),numel(xi));
for i = 1:numel(xi)
    for j = 1:numel(xi)
        phi(i,j) = (1 + (mu_gas(i)/mu_gas(j))^(0.5) * (data.MW(j)/data.MW(i))^(0.25))^2/ ...
         (8*(1 + (data.MW(i)/data.MW(j)) ))^(0.5);
    end
    mu_mix = mu_mix + xi(i)*mu_gas(i)/(phi(i,:) * xi);
end
mu_mix = mu_mix * 0.1; % Convert CGS-Unit Poise to SI-Unit Pa*s 
end

