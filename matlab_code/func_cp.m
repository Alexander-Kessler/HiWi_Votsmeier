function [cp] = func_cp(coef, T)
% Function func_cp for cp_m
% 
% --- Description ---
% Gives molar heat capacity cp_mi of compounds i
% Uses NASA Polynomials
% Based on Parameters from Baerns
% 
% --- Dependencies ---
% none
% 
% --- Input ---
% Structure Data : data.cp_coef (array)
% Float Temperature : T
% 
% --- Output ---
% Vector cp (cp_i): UNIT J/(mol K)

    cp = coef(:,1)+coef(:,2)*T+coef(:,3)*T.^2+coef(:,4)*T.^3;
end

