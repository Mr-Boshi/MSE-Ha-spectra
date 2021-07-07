#%% IMPORT
from scipy import constants as const
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from math import sqrt
from math import log
from math import exp
from copy import deepcopy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


#%% FUNCTIONS
def e_to_v(E):
	# E must be in keV

	e_charge    = const.physical_constants['elementary charge'][0]
	proton_mass = const.physical_constants['proton mass'][0]
	e_to_v_coeff = sqrt(2*e_charge/proton_mass)
	V = e_to_v_coeff*(E * 1e3) ** 0.5
	return V # in m/s


def json_to_bsrate(filename, Energy, ne, te):
	# E must be in keV
	# Beamstop rates are in m3/s

	with open(filename, 'r') as json_file:
		beamstop_data = json.load(json_file)
		s_ne_int = interp2d(beamstop_data['e'], beamstop_data['n'], beamstop_data['sen'], kind='cubic',
		                    bounds_error=False, fill_value=0.)
		st = np.array(beamstop_data['st'])/beamstop_data['sref']
		s_t_int = interp1d(beamstop_data['t'], st, kind='cubic',
    	                bounds_error=False, fill_value=0.)

		beamstop_rate = np.zeros((len(Energy), len(ne)))
		for j in range(0, len(Energy)):
			for i in range(0, len(ne)):
				beamstop_rate[j, i] = s_ne_int(Energy[j]*1e3, ne[i])*s_t_int(te[i])
				if beamstop_rate[j,i] <=0.0:
					beamstop_rate[j,i] = 0.0
						
		return beamstop_rate


#%% CLASSES
class plasma_interpolated:
	def __init__(self, r, ne, te, ni, ti):
		self.ne = interp1d(r, ne, kind='cubic', bounds_error=False, fill_value=0.)
		self.ni = interp1d(r, ni, kind='cubic', bounds_error=False, fill_value=0.)
		self.te = interp1d(r, te, kind='cubic', bounds_error=False, fill_value=0.)
		self.ti = interp1d(r, ti, kind='cubic', bounds_error=False, fill_value=0.)

class beam_interpolated:
	def __init__(self, r, density):
		self.e0  = interp1d(r, density[0,:], kind='cubic', bounds_error=False, fill_value=0.)
		self.e02 = interp1d(r, density[1,:], kind='cubic', bounds_error=False, fill_value=0.)
		self.e03 = interp1d(r, density[2,:], kind='cubic', bounds_error=False, fill_value=0.)





#%% SCRIPT BEGINS HERE
os.system('cls' if os.name == 'nt' else 'clear')

# Setting physical constants
h_plank     = const.physical_constants['Planck constant'][0]
e_mass      = const.physical_constants['electron mass'][0]
proton_mass = const.physical_constants['proton mass'][0]
e_charge    = const.physical_constants['elementary charge'][0]
l_speed     = const.physical_constants['speed of light in vacuum'][0]
mu_0        = const.physical_constants['vacuum mag. permeability'][0]
pi          = const.pi

z_p  = 1
z_c6 = 6
Zeff = 3

# Tokamak parameters
I_pl  = 1.e6
R0    = 1.5
a     = 0.67
B_t   = 2
q_a = 2*pi*a**2*B_t/(R0*I_pl*mu_0)

# Ne_av = 4.5e19

#%% Beam parameters
E_0                       = 60 													# Main component energy in keV
Energy                    = np.array([E_0,E_0/2,E_0/3],float)
Composition               = np.array([0.89, 0.045, 0.06],float)
Neutralization_efficiency = np.array([0.43, 0.7, 0.8],float)

Velocity = e_to_v(Energy) 														# Particles velocities in m/s

I_0  = 6.1																		# Ion current in Amp
De   = 0.08																		# Beam diameter at 1/e in meters
D_05 = De/(sqrt(2*sqrt(log(2))))

Beam_cs      = pi*(D_05/2)**2
j_beam_total = I_0/(Composition[0]*Beam_cs)										# Beam total current density in A/m^2

j_beam        = j_beam_total*Composition
j_atoms       = j_beam*Neutralization_efficiency
j_atoms_total = sum(j_atoms)													# Beam total atom density in 1/(s*m^2)

N0_atoms = j_atoms/(e_charge*Velocity)											# Beam density in m-3

#%% Spectral data
lambda_Ha = 6562.79 # A
lambda_C2 = 6582.88 # A

upper_viewport = np.array([795, 1085], float)
lower_viewport = np.array([835, 1067], float)

# %% Reading and interpolating T-15 data
file_name = 't-15_data_full.txt'
T15_data  = pd.read_table(file_name, sep=r"\s+").to_numpy()

r_array  = T15_data[:, 0]														# In m
ne_array = T15_data[:, 1]*1e19													# In m-3
Te_array = T15_data[:, 2]*1e3													# In eV
ni_array = T15_data[:, 3]*1e19													# In m-3
Ti_array = T15_data[:, 4]*1e3													# In eV

rho_array       = r_array/max(r_array)
if not min(rho_array) == 0:
	zer0            = (np.argmin(abs(rho_array)))
	rho_array[zer0] = 0																# Setting the closest to center point to be the center

T15_inter = plasma_interpolated(rho_array, ne_array, ni_array, Te_array, Ti_array)

#%% Beam-stopping calculation
# Use 'ADAS' to get rates from json or or 'SOS' to use rates stored in file
source_key='ADAS'

if source_key=='SOS':
	# Using rates stored in data_file
	Beam_stop_rate = T15_data[:, 5:8]
	BS0_int  = interp1d(rho_array, Beam_stop_rate[:,0], kind='cubic', bounds_error=False, fill_value=0.)
	BS02_int = interp1d(rho_array, Beam_stop_rate[:,1], kind='cubic', bounds_error=False, fill_value=0.)
	BS03_int = interp1d(rho_array, Beam_stop_rate[:,2], kind='cubic', bounds_error=False, fill_value=0.)

	rho            = np.linspace(1, -1, num=100, endpoint=True)
	rho_dif        = a*abs(np.diff(rho))										# in meters
	N0_array       = np.zeros((len(Energy),len(rho)))
	N0_array[:, 0] = N0_atoms

	for i in range(0,len(rho_dif)):
		qrat = np.array([BS0_int(rho[i+1]), BS02_int(rho[i+1]), BS03_int(rho[i+1])], float)
		N0_array[:, i+1] = N0_array[:, i] * (1-T15_inter.ne(rho[i+1])*rho_dif[i]*qrat)

elif source_key=='ADAS':
	# Using beam stopping rates from ADAS (with CHERAB parser)
	beamstop_h_json_file = r'beam\stopping\h\h\1_default.json'
	beamstop_c_json_file = r'beam\stopping\h\c\6_default.json'

	# Fractions of plasma composition species
	f_c = (ne_array/ni_array - z_p)/(z_c6-z_p)
	f_h = 1-f_c

	sum_zf  = z_p * f_h + z_c6 * f_c
	sum_z2f = z_p**2 * f_h + z_c6**2 * f_c

	# Equivalent electron density for each plasma specie
	ne_equ_h = (ne_array / sum_zf) * (sum_z2f / z_p)
	ne_equ_c = (ne_array / sum_zf) * (sum_z2f / z_c6)

	bs_rate_h = json_to_bsrate(beamstop_h_json_file, Energy, ne_equ_h, Ti_array)
	bs_rate_c = json_to_bsrate(beamstop_c_json_file, Energy, ne_equ_c, Ti_array)


	bs_rate_1  = np.zeros(bs_rate_c.shape)
	bs_rate_1 += z_p * bs_rate_h * f_h / sum_zf
	bs_rate_1 += z_c6 * bs_rate_c * f_c / sum_zf

	bs_rate = (bs_rate_h + bs_rate_c) / sum_zf

	BS0_int  = interp1d(rho_array, bs_rate[0,:], kind='cubic', bounds_error=False, fill_value=0.)
	BS02_int = interp1d(rho_array, bs_rate[1,:], kind='cubic', bounds_error=False, fill_value=0.)
	BS03_int = interp1d(rho_array, bs_rate[2,:], kind='cubic', bounds_error=False, fill_value=0.)

	# Beam-stopping calculation

	rho                 = np.linspace(1, -1, num=100, endpoint=True)
	rho_dif             = a*abs(np.diff(rho))										# in meters
	Ne_Sbs              = np.zeros((len(Energy), len(rho)))
	Ne_Sbs_sum          = np.zeros((len(Energy), len(rho)))

	for i in range(1,len(rho_dif)):
		qrat = np.array([BS0_int(rho[i+1]), BS02_int(rho[i+1]), BS03_int(rho[i+1])], float)
		Ne_Sbs[:, i] = T15_inter.ne(rho[i+1])*rho_dif[i]*qrat
		Ne_Sbs_sum[:, i] = (1/Velocity)*np.sum(Ne_Sbs, axis=1)

	Ne_Sbs_sum[:, -1] = Ne_Sbs_sum[:, -2]
	N0_atoms_array = np.ones(Ne_Sbs_sum.shape)
	for i in range(0,3):
		N0_atoms_array[i, :] *= N0_atoms[i]

	N0_array = N0_atoms_array * np.exp(-Ne_Sbs_sum)

else:
	raise ValueError('''Value of source code must be 'SOS' or 'ADAS'.''')

# Creating interpolants with beam density
dnb_inter = beam_interpolated(rho, N0_array)

# %% MSE H_alpha exitation rates




# %% Plasma profiles plotting

# plt.subplot(2 ,2, 1)
# plt.plot(rho_array, ne_array, 'o', rho, T15_inter.ne(rho), '-')
# plt.grid(True)
# plt.ylabel('N_e')
# plt.xlabel('rho')

# plt.subplot(2, 2, 2)
# plt.plot(rho_array, Te_array, 'o', rho, T15_inter.te(rho), '-')
# plt.grid(True)
# plt.ylabel('T_e')

# plt.subplot(2, 2, 3)
# plt.plot(rho_array, ni_array, 'o', rho, T15_inter.ni(rho), '-')
# plt.grid(True)
# plt.ylabel('N_i')


# plt.subplot(2, 2, 4)
# plt.plot(rho_array, Ti_array, 'o', rho, T15_inter.ti(rho), '-')
# plt.grid(True)
# plt.ylabel('T_i')

# plt.subplots_adjust(left=0.08, bottom=0.13, right=0.98,
#                     top=0.96, wspace=0.285, hspace=0.330)
# plt.show()

# %% Beam-stopping rate profiles (From file)

# plt.plot(rho_array, Beam_stop_rate[:, 0], 'o', rho, BS0_int(rho), '-')
# plt.plot(rho_array, Beam_stop_rate[:, 1], 'o', rho, BS02_int(rho), '-')
# plt.plot(rho_array, Beam_stop_rate[:, 2], 'o', rho, BS03_int(rho), '-')
# plt.grid(True)
# plt.ylabel('Beam-stopping rate')
# plt.xlabel('rho')
# plt.show()

# %% Beam-stopping rate profiles (From ADAS)

# plt.subplot(1, 3, 1)
# plt.plot(rho, BS0_int(rho) / bs0_rate_int(rho), '-')
# plt.grid(True)
# plt.ylabel('Beam-stopping rate, E0')
# plt.xlabel('rho')

# plt.subplot(1, 3, 2)
# plt.plot(rho, BS02_int(rho) / bs02_rate_int(rho), '-')
# plt.grid(True)
# plt.ylabel('Beam-stopping rate, E0/2')
# plt.xlabel('rho')

# plt.subplot(1, 3, 3)
# plt.plot(rho, BS03_int(rho) / bs03_rate_int(rho), '-')
# plt.grid(True)
# plt.ylabel('Beam-stopping rate, E0/3')
# plt.xlabel('rho')
# plt.show()

# %% Beam density plots (From file)


plt.plot(rho, dnb_inter.e0(rho), '-', rho, dnb_inter.e02(rho), '-', rho, dnb_inter.e03(rho), '-')
plt.grid(True)
plt.ylabel('Beam density')
plt.xlabel('rho')
plt.show()

