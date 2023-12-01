function [dydz] = func_dydz_2D_reactor(~,y,config,N_elements,u_in,p_tot,T_1,T_wall,x_1,eta)
    % constants
    k_0_hin = [4.225*1e15*sqrt(1e5) 1.955*1e6*1e-5 1.020*1e15*sqrt(1e5)]; % k(1),k(3): kmol*Pa^0.5/kgcat*h | k2: kmol/Pa*kgcat*h
    E_A_hin = [240.1 67.13 243.9]*1e3; % J/mol
    K_ads_std = [8.23*1e-5*1e-5 6.12*1e-9*1e-5 6.65*1e-4*1e-5 1.77*1e5]; % K_ads(1-3): Pa^-1
    delta_GR_ads = [-70.61 -82.90 -38.28 88.68]*1e3; % J/mol
    
    % Unpack quantities from the integration
    n_i = zeros([6*N_elements,1]); % kmol/h
    T_i = zeros([N_elements,1]); % K
    for i = 1:N_elements
        n_i(6*(i-1)+1:6*i) = y(7*(i-1)+1:7*i-1);
        T_i(i) = y(7*i);
    end
    
    % Calculate partial pressures with Dalton's law
    x_i = zeros([6*N_elements,1]);
    for i = 1:N_elements
        n_i_element = n_i(6*(i-1)+1:6*i);
        x_i(6*(i-1)+1:6*i) = n_i_element./sum(n_i_element);
    end
    p_i = x_i.*p_tot; % Pa = kg/m*s^2

    %% Determination of the element area and radii with Darcy's law
    mu_g = zeros([N_elements, 1]); % Pa*s (dynamische Viskosität)
    rho_g = zeros([N_elements, 1]); % kg/m^3
    for i = 1:N_elements
        mu_g(i) = func_mu(T_i(i), x_i((i-1)*6+1:i*6)', config.data); % Pa*s
        rho_g(i) =  func_rho_gas(T_i(i), p_tot, x_i((i-1)*6+1:i*6)', config.const, config.data); % kg/m^3
    end
    Ai_Ages = (mu_g./rho_g)/sum(mu_g./rho_g);
    A_elements = Ai_Ages * config.react.A; % m^2
    
    % A = pi * (r_2^2-r_1^2)
    r_elements = zeros([N_elements+1,1]); % m
    for i = 1:N_elements
        r_elements(i+1) = sqrt(A_elements(i)/pi + r_elements(i)^2);
    end
    
    %% Calculate u(x,T) with averaged x and T
    x_i_avg = zeros([6,1]);
    for i = 1:N_elements
        for j = 1:6
            x_i_avg(j) = x_i_avg(j) + x_i(6*(i-1)+j) * A_elements(i)/sum(A_elements);
        end
    end
    T_avg = sum(T_i .* A_elements)/sum(A_elements); % K

    M_1 = dot(x_1(1:6)',config.data.MW'); % kg/mol (scalar product)
    M_avg = dot(x_i_avg,config.data.MW'); % kg/mol
    u_avg = u_in * (T_avg*M_1)/(T_1*M_avg); % m/s
    
    %% Calculate quantities and derivative of the mass balance of each element
    dydz = zeros([7*N_elements,1]);
    mult_rho_cp = zeros([N_elements, 1]); % J/m^3*K
    q = zeros([N_elements+1, 1]); % kJ/m^2*h
    s_H = zeros([N_elements, 1]); % J/m^3*h
    for i = 1:N_elements
        % Extract partial pressures
        p_CH4 = p_i(6*(i-1)+1); p_H20 = p_i(6*(i-1)+2); p_H2 = p_i(6*(i-1)+3); % Pa
        p_CO = p_i(6*(i-1)+4); p_CO2 = p_i(6*(i-1)+5); % Pa

        % Calculate the reaction rate with a Langmuir-Hinshelwood-Houghen-Watson approach
        k_hin = k_0_hin .* exp(-E_A_hin./(config.const.R*T_i(i))); % k(1),k(3): kmol*Pa^0.5/kgcat*h | k2: kmol/Pa*kgcat*h
        K_ads = K_ads_std .* exp(-delta_GR_ads./(config.const.R*T_i(i))); % K_ads(1-3): Pa^-1
        [K_p_TD,delta_H_R] = func_Kp_HR(config,T_i(i)); % Kp(1), Kp(3): Pa^2 | delta_HR: % J/mol
    
        DEN = 1+p_CO*K_ads(1)+p_H2*K_ads(2)+p_CH4*K_ads(3)+(p_H20*K_ads(4))/p_H2;
        r_ges_1 = (k_hin(1)/(p_H2^2.5))*(p_CH4*p_H20-(((p_H2^3)*p_CO)/K_p_TD(1)))/(DEN^2); % kmol/kgcat*h
        r_ges_2 = (k_hin(2)/p_H2)*(p_CO*p_H20-((p_H2*p_CO2)/K_p_TD(2)))/(DEN^2); % kmol/kgcat*h
        r_ges_3 = (k_hin(3)/(p_H2^3.5))*(p_CH4*(p_H20^2)-(((p_H2^4)*p_CO2)/K_p_TD(3)))/(DEN^2); % kmol/kgcat*h
        
        % Define the differential equations
        dnCH4dz = eta*A_elements(i)*(-r_ges_1-r_ges_3)*config.react.rho_b; % kmol/m*h
        dnH20dz = eta*A_elements(i)*(-r_ges_1-r_ges_2-2*r_ges_3)*config.react.rho_b; % kmol/m*h
        dnH2dz = eta*A_elements(i)*(3*r_ges_1+r_ges_2+4*r_ges_3)*config.react.rho_b; % kmol/m*h
        dnCOdz = eta*A_elements(i)*(r_ges_1-r_ges_2)*config.react.rho_b; % kmol/m*h
        dnCO2dz = eta*A_elements(i)*(r_ges_2+r_ges_3)*config.react.rho_b; % kmol/m*h
        dnN2dz = 0; % kmol/m*h

        dydz(7*(i-1)+1:7*(i-1)+6) = [dnCH4dz dnH20dz dnH2dz dnCOdz dnCO2dz dnN2dz];

        % Calculate properties for the heat balance
        rho_i = func_rho_gas(T_i(i),p_tot,x_i(6*(i-1)+1:6*(i-1)+6),config.const,config.data); % kg/m^3
        cp_i = func_cp(config.data.cp_coef,T_i(i)); % J/mol*K
        [alpha_w_int, lambda_rad] = calc_heat_transfer(T_i(i),x_i(6*(i-1)+1:6*(i-1)+6)',config,rho_i,cp_i,u_avg); % a_w: kJ/m^2*h*K | lambda_rad: kJ/m*h*K
        
        cp_g = dot(x_i(6*(i-1)+1:6*(i-1)+6),(cp_i ./ config.data.MW')); % J/kg*K
        mult_rho_cp(i) = cp_g*rho_i*1e3; % J/m^3*K
        
        if i == 1
            % first boundary condition
            q(i) = 0;
        elseif i == N_elements
            q(i) = lambda_rad*(T_i(i-1)-T_i(i))/(r_elements(i+1)-r_elements(i)); % kJ/m^2*h
            % second boundary condition
            q(end) = (alpha_w_int*lambda_rad/(0.5*(r_elements(i+1)-r_elements(i))))/...
                (alpha_w_int+(lambda_rad/(0.5*(r_elements(i+1)-r_elements(i)))))* (T_i(end)-T_wall); % kJ/m^2*h
        else
    	    q(i) = lambda_rad*(T_i(i-1)-T_i(i))/(r_elements(i+1)-r_elements(i)); % kJ/m^2*h
        end

        r_ges = eta*[r_ges_1 r_ges_2 r_ges_3]; % kmol/kgcat*h
        s_H(i) = -sum((delta_H_R.*1e3).*(r_ges).*config.react.rho_b); % J/m^3*h
    end
    
    %% Calculate derivative of the heat balance
    for i = 1:N_elements
        dr = r_elements(i+1)-r_elements(i); % m
        dTdz_cond = (r_elements(i)*q(i)*1e3 - r_elements(i+1)*q(i+1)*1e3)/(dr *(r_elements(i)+dr*0.5)*mult_rho_cp(i)*u_avg*3.600); % K/m
        dTdz_react = s_H(i)/(mult_rho_cp(i)*u_avg*3.600); % K/m
        dTdz = dTdz_cond+dTdz_react; % K/m
        dydz(7*(i-1)+7) = dTdz;
    end
end

