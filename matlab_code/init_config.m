% Script init_config
%
% Initialize Script for Structure config
% --- Description ---
% Generates Struct config with 3 sub-structs: 
% react : Reactor constants
% const : Physico-Chemical constants
% data : Thermodynamic (Standard Values), Stoich (stoich Matrix)
% --- Output ---
% Struct config


% Substruct react
react.L = 12; %[m] Lengt Reactor
react.d_in = 0.1016; %[m] Diameter inner tube
react.d_out = 0.1322; %[m] Diameter outer tube

react.A = pi * (react.d_in/2)^2; %[m^2/s] Cross-section tube
% Catalyst Dimensions : Pellet : Raschig Ring
react.A = pi * (react.d_in/2)^2; %[m^2] Cross-section tube
react.d_pi = 0.0084; %[m] Inner Catalyst ring Diameter

react.Vr = react.A * react.L; %[m^3] Volume Reactor
% Calculation of bed void fraction epsilon based on Dixon Relationship
react.epsilon = 0.4 + 0.05 * react.d_pi/react.d_in + 0.412 * ...
                react.d_pi^2/react.d_in^2; %[-] Void fraction Kat.-Bed
react.Twall = 1100; %[K] Temperature Wall, here fixed
react.rho_s = 2355; %[kg/m^3] Density of solid material in fixed bed
react.rho_b = react.rho_s * (1 - react.epsilon); %[kg/m^3] Dens. fixed bed
react.lambda_s = 0.3489; %[W/(m K)] radial thermal conductivity fixed bed 
% solid material
react.em = 0.8; % Emmissivity, for heat transfer

% Substruct const
const.R = 8.3145; %[J/(K mol)] Univ id. Gas constant
const.T0 = 298.15; %[K] T-Standard

% Substruct data
% Stoich
data.M = [-1 1 0 0 -1 1; -1 1 -1 1 -2 2; 3 -3 1 -1 4 -4; ...
          1 -1 -1 1 0 0; 0 0 1 -1 1 -1]; %stoich. matrix
data.M_hin = data.M(:,[1,3,5]); % TD (cp = f(T))
data.MW = [16.043, 18.02, 2.016, 28.01, 44.01, 28.01] * 1e-3; % Molecular Weight in [kg/mol]
data.T_bi = [111.7, 373.15, 0, 81.5, 194.6, 77.3]; % Normal boiling 
% Temperatures in [K]
data.cp_coef = zeros(5,4);
data.cp_coef(4,:) = [30.848,-12.84*1e-3,27.87*1e-6,-12.71*1e-9]; % Kohlenmonoxid
data.cp_coef(1,:) = [19.2380000000000,52.0900000000000*1e-3,11.9660000000000*1e-6,-11.3090000000000*1e-9]; % Methan
data.cp_coef(2,:) = [32.2200000000000,1.92250000000000*1e-3,10.5480000000000*1e-6,-3.59400000000000*1e-9]; % Wasser
data.cp_coef(3,:) = [27.1240000000000,9.26700000000000*1e-3,-13.7990000000000*1e-6,7.64000000000000*1e-9]; % Wasserstoff
data.cp_coef(5,:) = [19.7800000000000,73.3900000000000*1e-3,-55.9800000000000*1e-6,17.1400000000000*1e-9]; % Kohlendioxid
data.cp_coef(6,:) = [31.128, -13.556*1e-3, 26.777*1e-6, -11.673*1e-9]; % Stickstoff

% Data from Baerns, Anhang 2 "Tabelle zu Reinstoffdaten"
data.HB0 = [-74850, -241820, 0, -110540, -393500]; % At Standard-Temp. T = 298.15 K
% Methan, Wasser, Wasserstoff, Kohlenmonoxid, Kohlendioxid
data.SB0 = [-80.5467, -44.3736,	0,	89.6529, 2.9515];


config.react = react;
config.data = data;
config.const = const;

clear const data react
