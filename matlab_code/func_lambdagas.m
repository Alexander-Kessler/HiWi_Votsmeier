function [lmbd_mix] = func_lambdagas(T,xi,data)
% Function func_lambdagas
% 
% --- Description ---
% Calculates the conductivities of the single components lambda_i and then
% the gas mixture conductivity.
% 
% --- Dependencies ---
% none
% 
% --- Input ---
% Float Temperature : T in K
% Vector amount of substance : xi 
% 
% --- Output ---
% Float lmbd_mix

xi = reshape(xi, [6, 1]);

% PHYSICAL PROPERTIES OF LIQUIDS AND GASES, 
% https://booksite.elsevier.com/9780750683661/Appendix_C.pdf
lambda_CH4 = -0.00935 + 1.4028*1e-4*T + 3.318*1e-8*T^2;
lambda_H2O = 0.00053 + 4.7093*1e-5*T + 4.9551*1e-8*T^2;
lambda_H2 = 0.03951 + 4.5918*1e-4*T + -6.4933*1e-8*T^2;
lambda_CO = 0.00158 + 8.2511*1e-5*T + -1.9081*1e-8*T^2;
lambda_CO2 = -0.012 + 1.0208*1e-4*T + -2.2403*1e-8*T^2;
lambda_N2 = 0.00309 + 7.593*1e-5*T + -1.1014*1e-8*T^2;
% Unit lambda_new : W/(m K)
lmbd_gas = [lambda_CH4, lambda_H2O, lambda_H2, lambda_CO, lambda_CO2, lambda_N2];

% Method of Wassiljewa:
% Chapter 10.6 THE PROPERTIES OF GASES AND LIQUIDS Bruce E. Poling / A 
% Slmple and Accurate Method for Calculatlng Viscosity of Gaseous Mixtures
% by Thomas A. Davidson
% Calculation A(i,j) by Mason, Saxena: Mason EA, Saxena SC. Approximate 
% formula for the thermal conductivity of gas mixtures. 
% Phys Fluids 1958;1:361e9
lmbd_mix = 0;
A = zeros(numel(xi), numel(xi));
for i = 1:numel(xi)
    for j = 1:numel(xi)
        A(i,j) = (1 + (lmbd_gas(i)/lmbd_gas(j))^(0.5) * (data.MW(i)/data.MW(j))^(0.25))^2/ ...
         (8*(1 + (data.MW(i)/data.MW(j)) ))^(0.5);
    end
    lmbd_mix = lmbd_mix + xi(i)*lmbd_gas(i)/(A(i,:) * xi);
end
end

