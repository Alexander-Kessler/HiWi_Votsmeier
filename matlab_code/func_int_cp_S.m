function [cp_T] = func_int_cp_S(cp_coef,T)
    % Calculate the integral of cp/T
    cp_T = cp_coef(1)*log(T) + cp_coef(2)*T+(1/2)*cp_coef(3)*T^2 + ...
           (1/3)*cp_coef(4)*T^3;
end

