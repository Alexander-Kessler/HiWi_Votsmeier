function [A_nve,r_nve] = calc_nve(z,y,config,N_elements,p_tot)
    A_nve = zeros([N_elements,length(z)]);
    r_nve = zeros([N_elements+1,length(z)]);
    for j = 1:length(z)
        % Sort quantities from the ode15
        n_i = zeros([6*N_elements,1]); % kmol/h
        T_i = zeros([N_elements,1]); % K
        for i = 1:N_elements
            n_i(6*(i-1)+1:6*i) = y(7*(i-1)+1:7*i-1,j);
            T_i(i) = y(7*i,j);
        end
        
        % Calculate mass fractions
        x_i = zeros([6*N_elements,1]);
        for i = 1:N_elements
            n_i_element = n_i(6*(i-1)+1:6*i);
            x_i(6*(i-1)+1:6*i) = n_i_element./sum(n_i_element);
        end
    
        % Determination of the element area and radii with Darcy's law
        mu_g = zeros([N_elements, 1]); % Pa*s
        rho_g = zeros([N_elements, 1]); % kg/m^3
        for i = 1:N_elements
            mu_g(i) = func_mu(T_i(i), x_i((i-1)*6+1:i*6)', config.data);
            rho_g(i) =  func_rho_gas(T_i(i), p_tot, x_i((i-1)*6+1:i*6)', config.const, config.data);
        end
        Ai_Ages = (mu_g./rho_g)/sum(mu_g./rho_g);
        A_elements = Ai_Ages * config.react.A; % m^2
    
        r_elements = zeros([N_elements+1,1]); % m
        for i = 1:N_elements
            r_elements(i+1) = sqrt(A_elements(i)/pi + r_elements(i)^2);
        end
        
        A_nve(:,j) = A_elements;
        r_nve(:,j) = r_elements;
    end
end

