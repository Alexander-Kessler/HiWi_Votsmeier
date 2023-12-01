clearvars
% Variables from the app
x_in_CH4 = 0.2128;
x_in_H20 = 0.714;
x_in_H2 = 0.0259;
x_in_CO = 0.0004;
x_in_CO2 = 0.0119;
x_in_N2 = 0.035;
x_i = [x_in_CH4 x_in_H20 x_in_H2 x_in_CO x_in_CO2 x_in_N2];
% 1:CH4; 2:H2O; 3:H2; 4:CO; 5:CO2; 6:N2; globally consistent

p_ges = 25.7*1e5; % Pa (Assumption: no pressure drop)
u_in = 2.14; % m/s
T_in = 793; % K
T_wall = 1100; % K
eta = 0.007; % Efficiency factor for catalyst (Assumption: In each catalyst unit the efficiency factor is the same)

% Run the function init_config to get all constants
run("init_config.m")

% Calculate amount of substance ni from mole fraction xi with the ideal gas equation
c_ges = (p_ges*1e-3)/(config.const.R*T_in); % kmol/m^3
c_i = c_ges .* x_i; % kmol/m^3
V_dot = u_in*3600*config.react.A; % m^3/h
n_i = c_i * V_dot; % kmol/h

% Integration of the 1D-reactor
[z,y] = ode15s(@func_dydz_1D_reactor,[0 config.react.L],[n_i T_in], [],config,u_in,p_ges,T_in,x_i,eta);

% Calculation of the conversion and yield 
X_CH4 = (y(1,1)-y(:,1))/y(1,1);
Y_CO2 = (y(:,5)-y(1,5))/y(1,1);

disp(X_CH4(end))
disp(Y_CO2(end))

% Plot
subplot(2,1,1);
for i = 1:6
    hold on
    plot(z(:),y(:,i));
end
hold off
legend('$\rm{CH_{4}}$','$\rm{H_{2}O}$','$\rm{H_{2}}$','$\rm{CO}$','$\rm{CO_{2}}$','$\rm{N_{2}}$','location','east','interpreter', 'latex')
ylabel("$\dot n\:/\:\rm{kmol\,h^{-1}}$",'interpreter', 'latex')

subplot(2,1,2);
plot(z(:),y(:,end),'color',"#D95319");
ylabel("$T\:/\:\rm{K}$", 'interpreter', 'latex')
xlabel("$x\:/\:\rm{m}$", 'interpreter', 'latex')
