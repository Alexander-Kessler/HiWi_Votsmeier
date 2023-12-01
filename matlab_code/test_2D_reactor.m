clearvars
% Variables from the app
x_in_CH4 = 0.2128;
x_in_H20 = 0.714;
x_in_H2 = 0.0259;
x_in_CO = 0.0004;
x_in_CO2 = 0.0119;
x_in_N2 = 0.035;
x_1 = [x_in_CH4 x_in_H20 x_in_H2 x_in_CO x_in_CO2 x_in_N2];
% 1:CH4; 2:H2O; 3:H2; 4:CO; 5:CO2; 6:N2; globally consistent

p_tot = 25.7*1e5; % Pa (Assumption: no pressure drop)
u_in = 2.14; % m/s
T_1 = 793; % K
eta = 0.007; % Efficiency factor for catalyst (Assumption: In each catalyst unit the efficiency factor is the same)
N_elements = 10;

% Run the function init_config to get all constants
run("init_config.m")

% Calculate amount of substance ni from mole fraction xi with the ideal gas equation
c_ges = (p_tot*1e-3)/(config.const.R*T_1); % kmol/m^3
c_1 = c_ges .* x_1; % kmol/m^3
V_dot = u_in*3600*config.react.A; % m^3/h
n_1 = c_1 * V_dot; % kmol/h

% Prepare y0 for ode15
y0 = zeros([7*N_elements,1]);
for i = 1:N_elements
    y0(7*(i-1)+1:7*i) = [n_1/N_elements T_1];
end

% Integration of the 2D-reactor
[z_2D,y_2D] = ode15s(@func_dydz_2D_reactor,[0 config.react.L],y0, [],config,N_elements,u_in,p_tot,T_1,x_1,eta);

% Calculation of the conversion and yield 
X_CH4_2D = (sum(y_2D(1,1:7:7*N_elements))-sum(y_2D(:,1:7:7*N_elements)'))/...
    sum(y_2D(1,1:7:7*N_elements));
Y_CO2_2D = (sum(y_2D(:,5:7:7*N_elements)')-sum(y_2D(1,5:7:7*N_elements)))/...
    sum(y_2D(1,1:7:7*N_elements));

disp(X_CH4_2D(end))
disp(Y_CO2_2D(end))

% Calculation of the amounts of substances
n_2D = zeros([length(z_2D), 5]);

for i = 1:length(z_2D)
    for j = 1:5
        n_2D(i,j) = sum(y_2D(i,j:7:7*N_elements))/N_elements;
    end
end

X_test = (n_2D(1,1)-n_2D(:,1))/n_2D(1,1);

