function [Kp,delta_H_R] = func_Kp_HR(config,T)
    % Calculation of the standard formation enthalpy at given temperature
    % with Kirchhoffsches law
    H_CH4 = config.data.HB0(1) + (func_int_cp_H(config.data.cp_coef(1,:),T)-...
        func_int_cp_H(config.data.cp_coef(1,:),config.const.T0)); % J/mol
    H_H2O = config.data.HB0(2) + (func_int_cp_H(config.data.cp_coef(2,:),T)-...
        func_int_cp_H(config.data.cp_coef(2,:),config.const.T0));% J/mol
    H_H2 = config.data.HB0(3) + (func_int_cp_H(config.data.cp_coef(3,:),T)-...
        func_int_cp_H(config.data.cp_coef(3,:),config.const.T0)); % J/mol
    H_CO = config.data.HB0(4) + (func_int_cp_H(config.data.cp_coef(4,:),T)-...
        func_int_cp_H(config.data.cp_coef(4,:),config.const.T0)); % J/mol
    H_CO2 = config.data.HB0(5) + (func_int_cp_H(config.data.cp_coef(5,:),T)-...
        func_int_cp_H(config.data.cp_coef(5,:),config.const.T0)); % J/mol
    
    % Calculation of the standard formation entropy at given temperature
    S_CH4 = config.data.SB0(1) + (func_int_cp_S(config.data.cp_coef(1,:),T)-...
        func_int_cp_S(config.data.cp_coef(1,:),config.const.T0)); % J/mol*K
    S_H2O = config.data.SB0(2) + (func_int_cp_S(config.data.cp_coef(2,:),T)-...
        func_int_cp_S(config.data.cp_coef(2,:),config.const.T0)); % J/mol*K
    S_H2 = config.data.SB0(3) + (func_int_cp_S(config.data.cp_coef(3,:),T)-...
        func_int_cp_S(config.data.cp_coef(3,:),config.const.T0)); % J/mol*K
    S_CO = config.data.SB0(4) + (func_int_cp_S(config.data.cp_coef(4,:),T)-...
        func_int_cp_S(config.data.cp_coef(4,:),config.const.T0)); % J/mol*K
    S_CO2 = config.data.SB0(5) + (func_int_cp_S(config.data.cp_coef(5,:),T)-...
        func_int_cp_S(config.data.cp_coef(5,:),config.const.T0)); % J/mol*K

    % Calculation of standard reaction enthalpies with Satz von Hess at 
    % standard conditions (T = 298.15 K) for three reactions
    % 1. CH4 + H2O <=> CO + 3 H2
    % 2. CO + H2O <=> CO2 + H2
    % 3. CH4 + 2 H2O <=> CO2 + 4 H2
    delta_HR_std_1 = -H_CH4-H_H2O+H_CO+3*H_H2; % J/mol
    delta_HR_std_2 = -H_CO-H_H2O+H_CO2+H_H2; % J/mol
    delta_HR_std_3 = -H_CH4-2*H_H2O+H_CO2+4*H_H2; % J/mol
    
    % Calculation of standard reaction entropies with Satz von Hess at 
    % standard conditions (T = 298.15 K) for three reactions
    delta_SR_std_1 = -S_CH4-S_H2O+S_CO+3*S_H2; % J/mol*K
    delta_SR_std_2 = -S_CO-S_H2O+S_CO2+S_H2; % J/mol*K
    delta_SR_std_3 = -S_CH4-2*S_H2O+S_CO2+4*S_H2; % J/mol*K
       
    % Calculation of the free reaction enthalpy with the Gibbs Helmoltz equation
    delta_GR_std_1 = delta_HR_std_1-T*delta_SR_std_1; % J/mol
    delta_GR_std_2 = delta_HR_std_2-T*delta_SR_std_2; % J/mol
    delta_GR_std_3 = delta_HR_std_3-T*delta_SR_std_3; % J/mol
    
    % Calculation of the rate constants
    Kp_1 = exp(-delta_GR_std_1/(config.const.R*T)); % bar^2
    Kp_2 = exp(-delta_GR_std_2/(config.const.R*T));
    Kp_3 = exp(-delta_GR_std_3/(config.const.R*T)); % bar^2

    Kp = [Kp_1*1e10 Kp_2 Kp_3*1e10]; % Kp_1, Kp_3: Pa^2
    delta_H_R = [delta_HR_std_1 delta_HR_std_2 delta_HR_std_3]; % J/mol
end

