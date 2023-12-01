function [a_w, lmbd_er] = calc_heat_transfer(T, x_i, config,rho_g,cp_g,u_T)
% Function rho_gas
% 
% --- Description ---
% Calculates a_w and lambda_er 
%
% --- Dependencies ---
% none
% 
% --- Input ---
% Float Temperature : T [K]
% Vector mole fractions: x_i
% Struct config:
% Float rho_g: Gas density [kg/m^3]
% Vector cp (cp_i): UNIT J/(mol K)
% Float u_T: Velocity [m/s]
% 
% --- Output ---
% Float a_w [kJ/m^2hK]
% Float lmbd_er [kJ/mhK]

mu_g = func_mu(T, x_i, config.data);
lmbd_g = func_lambdagas(T, x_i, config.data);
cps_g_mix = x_i * (cp_g .* 1./config.data.MW');
l_c = config.react.d_pi;
N_Re = rho_g * u_T * l_c / mu_g;
N_Pr = cps_g_mix * mu_g / lmbd_g;

a_w = (1 - 1.5 * (config.react.d_in/config.react.d_pi)^(-1.5)) * ...
       (lmbd_g * 3.6)/config.react.d_pi * N_Re^(0.59) * N_Pr^(1/3);
a_rs = 0.8171*(config.react.em/(2-config.react.em))*(T/100)^(3);
a_ru = (0.8171*(T/100)^(3))/(1+config.react.epsilon/2 * (1-config.react.epsilon) * ...
        (1-config.react.em)/config.react.em);

lmbd_er_0 = config.react.epsilon*(lmbd_g*3.6 + 0.95*a_ru*config.react.d_pi) + ...
            (0.95 * (1 - config.react.epsilon))/(2/(3*config.react.lambda_s*3.6) + ...
            1/(10 * lmbd_g*3.6 + a_rs * config.react.d_pi));

lmbd_er = lmbd_er_0 + 0.111 * lmbd_g*3.6 * (N_Re * N_Pr^(1/3))/(1 + 46 * ...
          (config.react.d_pi/config.react.d_out)^(2));
end



