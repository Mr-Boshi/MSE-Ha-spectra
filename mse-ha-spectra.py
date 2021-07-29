#%% IMPORT
from scipy import constants as const
from numpy import pi

import os
import numpy as np
import matplotlib.pyplot as plt

from classes import *

#%% SCRIPT BEGINS HERE
os.system('cls' if os.name == 'nt' else 'clear')

# Setting physical constants
h_plank     = const.physical_constants['Planck constant'][0]
e_mass      = const.physical_constants['electron mass'][0]
proton_mass = const.physical_constants['proton mass'][0]
e_charge    = const.physical_constants['elementary charge'][0]
l_speed     = const.physical_constants['speed of light in vacuum'][0]
mu_0        = const.physical_constants['vacuum mag. permeability'][0]
epsilon_0   = const.physical_constants['vacuum electric permittivity'][0]
a_bohr      = const.physical_constants['Bohr radius'][0]

# Setting charges of plasma instanses
z_p  = 1
z_c6 = 6
# Zeff = 3

# Setting radial grid
rho = np.linspace(1, -1, num=400, endpoint=True)

#%% Creating tokamak object with T-15 parameters
I_pl  = 1.e6																	# In Amp
R0    = 1.5																		# In m
a     = 0.67																	# In m
B_t   = 2																		# In T
# q_a = 2*pi*a**2*B_t/(R0*I_pl*mu_0)

T15_data = tokamak_data(a, R0, I_pl, B_t)

# %% Creating plasma object with parameters from file
file_name = 't-15_data_full.txt'
T15_data_from_file  = np.loadtxt(file_name, skiprows = 1)
r_array  = T15_data_from_file[:, 0]												# In m
ne_array = T15_data_from_file[:, 1]*1e19										# In m-3
Te_array = T15_data_from_file[:, 2]*1e3											# In eV
ni_array = T15_data_from_file[:, 3]*1e19										# In m-3
Ti_array = T15_data_from_file[:, 4]*1e3											# In eV

rho_array       = r_array/max(r_array)
if not min(rho_array) == 0:
	zer0            = (np.argmin(abs(rho_array)))
	rho_array[zer0] = 0															# Setting the closest to center point to be the center

T15_plasma_int = plasma_interpolated(rho_array, ne_array, Te_array, ni_array, Ti_array)


#%% Creating Beam-class object
I_0         = 6.1																# Ion current in Amp
r_e0        = 0.032																# Beam diameter at 1/e in meters
diver_rad   = np.deg2rad(0.6)													# Beam divergence in radians
E_0         = 60 																# Main component energy in keV
Composition = np.array([0.89, 0.045, 0.06],float)

dnb = beam_class(E_0, Composition, r_e0, I_0, diver_rad, T15_data, T15_plasma_int, z_c6)

#%% Creating viewport objects for upper and lower viewports and chords object
lambda_Ha = 6562.79 # A
lambda_C2 = 6582.88 # A

# View port location, (r, z) in meters, r=0 at rho=0
upper_coordinates = np.array([0.795, 1.085], float)
lower_coordinates = np.array([0.835, 1.067], float)

# Lens diameter
f_lens0 = 0.15																	# in meters
d_lens0 = 0.02																	# in meters

f_hes_in  = 0.394																# in meters
d_hes_out = 0.26																# in meters

slit = np.array([0.2, 20], float)												# Slit width and height in mm

upper_viewport = viewport_class(upper_coordinates, f_lens0, d_lens0, f_hes_in, d_hes_out, slit)
lower_viewport = viewport_class(lower_coordinates, f_lens0, d_lens0, f_hes_in, d_hes_out, slit)

chord_rho    = np.linspace(-0.90,0.98, num=15, endpoint=True)
upper_chords = chord_class(chord_rho, upper_viewport, T15_data)
lower_chords = chord_class(chord_rho, lower_viewport, T15_data)

#%% Creating MSE spectra object
h_alpha = mse_spectre(lambda_Ha)

lambda_array = np.linspace(lambda_Ha-20, lambda_Ha+100, num=2000, endpoint=True)
mse_spectrum = h_alpha.stark_spectra([0,1,2], upper_viewport, upper_chords, lambda_array, dnb, T15_data)

full_spectrum  = mse_spectrum.full(energy = 'sum', component = 'sum')
sigma_spectrum = mse_spectrum.sigma(energy = 'sum', component = 'sum')
pi_spectrum    = mse_spectrum.pi(energy = 'sum', component = 'sum')

full_spectrum_energy  = mse_spectrum.full(energy = 'non', component = 'sum')
sigma_spectrum_energy = mse_spectrum.sigma(energy = 'non', component = 'sum')
pi_spectrum_energy    = mse_spectrum.pi(energy = 'non', component = 'sum')

#%% Testing
den2 = dnb.density_2d(1, T15_data, upper_viewport, upper_chords, rho)


for LOS in range(0,15):
	plt.plot(rho, den2[LOS,:])
# plt.legend(['1', '3', '6', '8', '10', '12', '15'])
plt.grid(True)
plt.show()

# %% Plasma profiles plotting

# plt.subplot(2 ,2, 1)
# plt.plot(rho_array, ne_array, 'o', rho, T15_plasma_int.ne(rho), '-')
# plt.grid(True)
# plt.ylabel('N_e')
# plt.xlabel('rho')

# plt.subplot(2, 2, 2)
# plt.plot(rho_array, Te_array, 'o', rho, T15_plasma_int.te(rho), '-')
# plt.grid(True)
# plt.ylabel('T_e')

# plt.subplot(2, 2, 3)
# plt.plot(rho_array, ni_array, 'o', rho, T15_plasma_int.ni(rho), '-')
# plt.grid(True)
# plt.ylabel('N_i')


# plt.subplot(2, 2, 4)
# plt.plot(rho_array, Ti_array, 'o', rho, T15_plasma_int.ti(rho), '-')
# plt.grid(True)
# plt.ylabel('T_i')

# plt.subplots_adjust(left=0.08, bottom=0.13, right=0.98,
#                     top=0.96, wspace=0.285, hspace=0.330)
# plt.show()

# %% Beam data

# plt.subplot(2, 2, 1)
# plt.plot(rho, dnb.bsrate(1,rho), rho, dnb.bsrate(2,rho), rho, dnb.bsrate(3,rho))
# plt.grid(True)
# plt.ylabel('Beam-stopping rate')
# plt.xlabel('rho')

# plt.subplot(2, 2, 2)
# plt.plot(rho, dnb.density(1,rho), rho, dnb.density(2,rho), rho, dnb.density(3,rho))
# plt.grid(True)
# plt.ylabel('Beam density, E0/2')
# plt.xlabel('rho')

# plt.subplot(2, 2, 3)
# plt.plot(rho, dnb.exrate(1,rho), rho, dnb.exrate(2,rho), rho, dnb.exrate(3,rho))
# plt.grid(True)
# plt.ylabel('H_alpha exitation rate')
# plt.xlabel('rho')

# plt.subplot(2, 2, 4)
# plt.plot(rho, dnb.halpha(1,rho), rho, dnb.halpha(2,rho), rho, dnb.halpha(3,rho))
# plt.grid(True)
# plt.ylabel('H_alpha intensity')
# plt.xlabel('rho')
# plt.show()

# %% MSE-spectra for 4 chords
chord_number = [0, 5, 10, -1]
subplot_iterator = 0
for item in chord_number:
	subplot_iterator += 1
	plt.subplot(2 ,2, subplot_iterator)
	plt.plot(lambda_array, full_spectrum[item,:])
	plt.plot(lambda_array, sigma_spectrum[item,:], '--')
	plt.plot(lambda_array, pi_spectrum[item,:], '--')
	plt.legend(['Full', 'Sigma', 'Pi'])
	plt.xlabel('Lambda, A')
	plt.ylabel('Intencity')
	plt.title(f'Chord at rho = {chord_rho[item]:.3f}')
	plt.grid(True)

plt.tight_layout()
plt.show()

# %% MSE-spectra for central chord by components
plt.subplot(2, 2, 1)
plt.plot(lambda_array, full_spectrum_energy[0,8,:])
plt.plot(lambda_array, sigma_spectrum_energy[0,8,:], '--')
plt.plot(lambda_array, pi_spectrum_energy[0,8,:], '--')
plt.legend(['Full', 'Sigma', 'Pi'])
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(lambda_array, full_spectrum_energy[1,8,:])
plt.plot(lambda_array, sigma_spectrum_energy[1,8,:], '--')
plt.plot(lambda_array, pi_spectrum_energy[1,8,:], '--')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(lambda_array, full_spectrum_energy[2,8,:])
plt.plot(lambda_array, sigma_spectrum_energy[2,8,:], '--')
plt.plot(lambda_array, pi_spectrum_energy[2,8,:], '--')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(lambda_array, np.sum(full_spectrum_energy[:,8,:],axis = 0))
plt.plot(lambda_array, np.sum(sigma_spectrum_energy[:,8,:],axis = 0), '--')
plt.plot(lambda_array, np.sum(pi_spectrum_energy[:,8,:],axis = 0), '--')
plt.grid(True)
plt.show()
