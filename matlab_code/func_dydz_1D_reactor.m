function [dydz] = func_dydz_1D_reactor(~,y,config,u_in,p_ges,T_in,T_wall,x_in,eta)
    % constants
    k_0_hin = [4.225*1e15*sqrt(1e5) 1.955*1e6*1e-5 1.020*1e15*sqrt(1e5)]; % k(1),k(3): kmol*Pa^0.5/kgcat*h | k2: kmol/Pa*kgcat*h
    E_A_hin = [240.1 67.13 243.9]*1e3; % J/mol
    K_ads_std = [8.23*1e-5*1e-5 6.12*1e-9*1e-5 6.65*1e-4*1e-5 1.77*1e5]; % K_ads(1-3): Pa^-1
    delta_GR_ads = [-70.61 -82.90 -38.28 88.68]*1e3; % J/mol
    
    % Unpack quantities from the integration
    n_i = y(1:6); % kmol/h
    T = y(7); % K
    
    % Consider the temperature dependence of the flow velocity with the ideal gas equation
    x_i = n_i./sum(n_i);
    M_in = dot(x_in',config.data.MW');
    M = dot(x_i,config.data.MW');
    u_gas = u_in * (T*M_in)/(T_in*M);   
    
    %% Solving the mass balance in axial direction
    % Calculate partial pressures with Dalton's law
    p_i = x_i.*p_ges; % Pa = kg/m*s^2
    p_CH4 = p_i(1); p_H20 = p_i(2); p_H2 = p_i(3); % Pa
    p_CO = p_i(4); p_CO2 = p_i(5); % Pa

    % Calculate the reaction rate with a Langmuir-Hinshelwood-Houghen-Watson approach
    k_hin = k_0_hin .* exp(-E_A_hin./(config.const.R*T)); % k(1),k(3): kmol*Pa^0.5/kgcat*h | k2: kmol/Pa*kgcat*h
    K_ads = K_ads_std .* exp(-delta_GR_ads./(config.const.R*T)); % K_ads(1-3): Pa^-1
    [K_p_TD,delta_H_R] = func_Kp_HR(config,T); % Kp(1), Kp(3): Pa^2 | delta_HR: % J/mol

    DEN = 1+p_CO*K_ads(1)+p_H2*K_ads(2)+p_CH4*K_ads(3)+(p_H20*K_ads(4))/p_H2;
    r_ges_1 = (k_hin(1)/(p_H2^2.5))*(p_CH4*p_H20-(((p_H2^3)*p_CO)/K_p_TD(1)))/(DEN^2); % kmol/kgcat*h
    r_ges_2 = (k_hin(2)/p_H2)*(p_CO*p_H20-((p_H2*p_CO2)/K_p_TD(2)))/(DEN^2); % kmol/kgcat*h
    r_ges_3 = (k_hin(3)/(p_H2^3.5))*(p_CH4*(p_H20^2)-(((p_H2^4)*p_CO2)/K_p_TD(3)))/(DEN^2); % kmol/kgcat*h
    
    % Define the differential equations
    dnCH4dz = eta*config.react.A*(-r_ges_1-r_ges_3)*config.react.rho_b; % kmol/m*h
    dnH20dz = eta*config.react.A*(-r_ges_1-r_ges_2-2*r_ges_3)*config.react.rho_b; % kmol/m*h
    dnH2dz = eta*config.react.A*(3*r_ges_1+r_ges_2+4*r_ges_3)*config.react.rho_b; % kmol/m*h
    dnCOdz = eta*config.react.A*(r_ges_1-r_ges_2)*config.react.rho_b; % kmol/m*h
    dnCO2dz = eta*config.react.A*(r_ges_2+r_ges_3)*config.react.rho_b; % kmol/m*h
    dnN2dz = 0; % kmol/m*h
    
    %% Solving the heat balance in axial direction
    % Calculation of the source term for the reaction
    r_ges = eta*[r_ges_1 r_ges_2 r_ges_3]; % kmol/kgcat*h
    s_H = -(delta_H_R.*1e3).*(r_ges).*config.react.rho_b; % J/m^3*h
    
    % Calculation of the source term for external heat exchange
    rho_gas = func_rho_gas(T,p_ges,x_i,config.const,config.data); % kg/m^3
    cp_i = func_cp(config.data.cp_coef,T); % J/mol*K
    [alpha_w_int, lambda_eff] = calc_heat_transfer(T,x_i',config,rho_gas,cp_i,u_gas); % a_w: kJ/m^2*h*K | lmbd_er: kJ/m*h*K

    Bi = (alpha_w_int*config.react.d_out)/(2*lambda_eff);
    Nu = 6*(Bi+4)/(Bi+3);
    alpha_w_bed = Nu*lambda_eff/config.react.d_out; % kJ/m^2*h*K
    U_h_bed = 1/((1/alpha_w_int)+(1/alpha_w_bed)); % kJ/m^2*h*K
    U_perV = (4/config.react.d_in)*U_h_bed; % kJ/m^3*h*K
    s_H_ext = -U_perV*1e3*(T-T_wall); % J/m^3*h
    
    % Define the differential equations
    cp_gas = cp_i ./ config.data.MW'; % J/kg*K
    sum_ci_cpi = dot(x_i,cp_gas)*rho_gas*1e3; % J/m^3*K
    dTdz = (sum(s_H(:))+s_H_ext)/(u_gas*3.6*sum_ci_cpi); % K/m

    dydz = [dnCH4dz;dnH20dz;dnH2dz;dnCOdz;dnCO2dz;dnN2dz;dTdz];
end

