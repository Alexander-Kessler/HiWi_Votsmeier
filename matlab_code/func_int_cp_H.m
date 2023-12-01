function [cp_T] = func_int_cp_H(cp_coef,T)
    % Calculate the integral of cp
    cp_T = cp_coef(1)*T + 0.5*cp_coef(2)*T^2+(1/3)*cp_coef(3)*T^3 + ...
           (1/4)*cp_coef(4)*T^4;
end

