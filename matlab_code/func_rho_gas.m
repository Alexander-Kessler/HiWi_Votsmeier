function [rho_gas] = func_rho_gas(T, ptot, x, const, data)
% Function rho_gas
% 
% --- Description ---
% Calculates rho of gas mixture via ideal gas
% mixing rule (Dalton)
% 
% --- Dependencies ---
% none
% 
% --- Input ---
% Float Temperature : T [K]
% Float Total pressure : ptot [Pa]
% Float amount of subst : x [-]
% Struct Data : data.MW (array, laying)
% Struct const : const.R
% 
% --- Output ---
% Float rho_gas [kg/m^3]
x = reshape(x, [6, 1]);
rho_gas = (sum((ptot * x) .* data.MW'))/(const.R * T);
end

