#%% IMPORT
from scipy import constants as const
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from math import sqrt
from math import log

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

def json_to_rate(filename, type, Energy, ne, te):
	# E must be in keV
	# Beamstop rates are in m3/s

	with open(filename, 'r') as json_file:
		file_data = json.load(json_file)
	if type == 'emission':
		data = file_data['3 -> 2']
	else:
		data = file_data
	
	s_ne_int = interp2d(data['e'], data['n'], data['sen'], kind='cubic',
	                    bounds_error=False, fill_value=0.)
	st = np.array(data['st'])/data['sref']
	s_t_int = interp1d(data['t'], st, kind='cubic',
                    bounds_error=False, fill_value=0.)

	rate = np.zeros((len(Energy), len(ne)))
	for j in range(0, len(Energy)):
		for i in range(0, len(ne)):
			rate[j, i] = s_ne_int(Energy[j]*1e3, ne[i])*s_t_int(te[i])
			if rate[j,i] <=0.0:
				rate[j,i] = 0.0
					
	return rate

def set_neutralization_efficiensy(filename):
	neutralization_data  = pd.read_table(filename, sep=r"\s+").to_numpy()
	beam_energy = neutralization_data[:,0]
	efficiensy =  neutralization_data[:,1]

	return interp1d(beam_energy, efficiensy, kind='cubic', bounds_error=False, fill_value='extrapolate')

def deg(angle):
	return 180 * angle / pi

def comp_to_list(component):
	if not type(component) == list:
		try:
			cup = list()
			cup.append(component)
			component = cup
		except:
			raise TypeError('The first argument must be a list.')
		else:
			component = sorted(component)

	if not component in [[0], [1], [2], [0, 1], [0,2], [1,2], [0,1,2]]:
		raise ValueError('Wrong argument. Accepted arguments are: 0, 1, 2, 3.')
	else:
		return component
	
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

class beam:
	def __init__(self, E_0, Composition, r_e0, I_0, divergence_rad, T15_inter):
		Energy                    = np.array([E_0,E_0/2,E_0/3],float)
		Velocity = e_to_v(Energy) 												# Particles velocities in m/s
		
		Neutralization_efficiency = Neutralization_efficiency_int(Energy)

		r_05         = r_e0/(sqrt(2*sqrt(log(2))))
		Beam_cs      = pi*(r_05/2)**2
		j_beam_total = I_0/(Composition[0]*Beam_cs)								# Beam total current density in A/m^2
		j_atoms      = j_beam_total*Composition*Neutralization_efficiency
		N0_atoms     = j_atoms/(e_charge*Velocity)								# Beam density in m-3 on entering the plasma

		# Using beam stopping rates from ADAS (with CHERAB parser)
		beamstop_h_json_file = r'beam\stopping\h\h\1_default.json'
		beamstop_c_json_file = r'beam\stopping\h\c\6_default.json'

		#%% Fractions of plasma composition species (used for ADAS rates calculation)
		rho = np.linspace(1, -1, num=200, endpoint=True)
		
		f_c = (T15_inter.ne(rho)/T15_inter.ni(rho) - z_p)/(z_c6-z_p)
		f_h = 1-f_c

		sum_zf  = z_p * f_h + z_c6 * f_c
		sum_z2f = z_p**2 * f_h + z_c6**2 * f_c

		# Equivalent electron density for each plasma specie
		ne_equ_h = (T15_inter.ne(rho) / sum_zf) * (sum_z2f / z_p)
		ne_equ_c = (T15_inter.ne(rho) / sum_zf) * (sum_z2f / z_c6)

		bs_rate_h = json_to_rate(beamstop_h_json_file, 'beam-stop', Energy, ne_equ_h, T15_inter.ti(rho))
		bs_rate_c = json_to_rate(beamstop_c_json_file, 'beam-stop', Energy, ne_equ_c, T15_inter.ti(rho))

		bs_rate = (bs_rate_h + bs_rate_c) / sum_zf
		BS_int  = beam_interpolated(rho, bs_rate)

		# Beam-stopping calculation
		rho_dif             = a*abs(np.diff(rho))								# in meters
		Ne_Sbs              = np.zeros((len(Energy), len(rho)))					# Preallocation
		Ne_Sbs_sum          = np.zeros((len(Energy), len(rho)))					# Preallocation

		for i in range(1,len(rho_dif)):
			qrat             = np.array([BS_int.e0(rho[i+1]), BS_int.e02(rho[i+1]), BS_int.e03(rho[i+1])], float)
			Ne_Sbs    [:, i] = T15_inter.ne(rho[i+1])*rho_dif[i]*qrat
			Ne_Sbs_sum[:, i] = (1/Velocity)*np.sum(Ne_Sbs, axis=1)

		Ne_Sbs_sum[:, -1] = Ne_Sbs_sum[:, -2]
		N0_atoms_array = np.ones(Ne_Sbs_sum.shape)
		for i in range(0,3):
			N0_atoms_array[i, :] *= N0_atoms[i]

		N0_array = N0_atoms_array * np.exp(-Ne_Sbs_sum)
		Density  = beam_interpolated(rho, N0_array)

		# MSE H_alpha exitation rates
		h_alpha_h_json_file = r'beam\emission\h\h\1_default.json'
		h_alpha_c_json_file = r'beam\emission\h\c\6_default.json'

		h_alpha_rate_h = json_to_rate(h_alpha_h_json_file, 'emission', Energy, ne_equ_h, T15_inter.ti(rho))
		h_alpha_rate_c = json_to_rate(h_alpha_c_json_file, 'emission', Energy, ne_equ_c, T15_inter.ti(rho))

		h_alpha_rate = (h_alpha_rate_h + h_alpha_rate_c) / sum_zf
		h_a_rate_inter = beam_interpolated(rho, h_alpha_rate)

		h_alpha_intens      = np.zeros(((len(Energy), len(rho))))
		
		h_alpha_intens[0,:] = h_a_rate_inter.e0(rho)*Density.e0(rho)*T15_inter.ne(rho)
		h_alpha_intens[1,:] = h_a_rate_inter.e02(rho)*Density.e02(rho)*T15_inter.ne(rho)
		h_alpha_intens[2,:] = h_a_rate_inter.e03(rho)*Density.e03(rho)*T15_inter.ne(rho)
		Intensity     = beam_interpolated(rho, h_alpha_intens)

		self.r_e0         = r_e0
		self.divergence   = divergence_rad
		self.i_0          = I_0
		self.energy       = Energy
		self.velocity     = Velocity
		self.composition  = Composition
		self._density     = Density
		self._bsrate      = BS_int
		self._exrate      = h_a_rate_inter
		self._intensity   = Intensity
		self._d_array     = np.linspace(-0.15, 0.15, num=200, endpoint=True)

	def density(self,component=0, rho=0.):
		if not component in [0, 1, 2, 3]:
			raise ValueError('Wrong argument. Accepted arguments are: 0, 1, 2, 3.')
		else:
			if component == 0:
				return self._density.e0(rho)+self._density.e02(rho)+self._density.e03(rho)
			elif component == 1:
				return self._density.e0(rho)
			elif component==2:
				return self._density.e02(rho)
			elif component==3:
				return self._density.e03(rho)

	def bsrate(self,component=0, rho=0.):
		if not component in [0, 1, 2, 3]:
			raise ValueError('Wrong argument. Accepted arguments are: 0, 1, 2, 3.')
		else:
			if component == 0:
				return self._bsrate.e0(rho)+self._bsrate.e02(rho)+self._bsrate.e03(rho)
			elif component == 1:
				return self._bsrate.e0(rho)
			elif component==2:
				return self._bsrate.e02(rho)
			elif component==3:
				return self._bsrate.e03(rho)
	
	def exrate(self,component=0, rho=0.):
		if not component in [0, 1, 2, 3]:
			raise ValueError('Wrong argument. Accepted arguments are: 0, 1, 2, 3.')
		else:
			if component == 0:
				return self._exrate.e0(rho)+self._exrate.e02(rho)+self._exrate.e03(rho)
			elif component == 1:
				return self._exrate.e0(rho)
			elif component==2:
				return self._exrate.e02(rho)
			elif component==3:
				return self._exrate.e03(rho)

	def halpha(self,component=0, rho=0.):
		if not component in [0, 1, 2, 3]:
			raise ValueError('Wrong argument. Accepted arguments are: 0, 1, 2, 3.')
		else:
			if component == 0:
				return self._intensity.e0(rho)+self._intensity.e02(rho)+self._intensity.e03(rho)
			elif component == 1:
				return self._intensity.e0(rho)
			elif component==2:
				return self._intensity.e02(rho)
			elif component==3:
				return self._intensity.e03(rho)
	
	def den_pfofile(self, rho):
		rho_array         = np.array(rho,float, ndmin=1)
		r_e_array         = r_e0+ (np.abs((rho_array-1)*a))*np.tan(diver_rad/2)
		den_profile_array = np.zeros((len(rho_array), len(self._d_array)))
		for i in range(0, len(rho_array)):
			den_profile             = np.exp(-0.5*(self._d_array/(r_e_array[i]/np.sqrt(2)))**2)
			den_profile_sum         = np.sum(den_profile)
			den_profile_array[i, :] = den_profile / den_profile_sum

		if len(rho_array)==1:
			den_profile_array = den_profile_array[0,:]

		return den_profile_array

class chord_class:
	def __init__(self, rho, viewport, d_lens0, f_lens0):
		self.rho          = rho
		self.r            = rho*a												# r = 0 at rho = 0
		self._r_ported     = (-1)*self.r + viewport[0]							# r = 0 at viewport[0], direction of the asis is reverced
		self.angles       = np.arctan(self._r_ported / viewport[1])				# in radians
		self._center_angle = np.arctan(viewport[0] / viewport[1])

		self.lens_d = d_lens0 * np.sqrt((pi/4) * np.cos(self.angles - self._center_angle))								# light diameter of lens

		self.length = viewport[1] / np.cos(self.angles)
		self.scale  = self.length / f_lens0 - 1

		# Solid angle of the lens and on the fiber for given LOS
		self.omega       = (pi/4) * (d_lens0 * self.lens_d) / self.length**2
		self.fiber_omega = self.omega * self.scale**2

		# Averaged aperture angles of the lens and MSE Ha measurements
		self.aperture       = np.sqrt(self.omega)
		self.fiber_aperture = np.sqrt(self.fiber_omega)
		self.fiber_distance = self.length / self.scale							# distances between lens and fibers

class spectrum():
	def __init__(self, sigma, pi, lambda_array):
		self._sigma   = sigma 
		self._pi      = pi
		self._full    = np.concatenate([sigma, pi], axis=2)
		self._lambdas = lambda_array
	
	def summizer(self, pi_or_sigma, energy, component):
		if energy=='sum':
			if component == 'sum':
				spectrum = np.sum(pi_or_sigma, 0)
				return np.sum(spectrum, 1)
			else:
				return np.sum(pi_or_sigma, 0)
		else:
			if component == 'sum':
				return np.sum(pi_or_sigma, 2)
			else:
				return pi_or_sigma

	def sigma(self, energy='sum', component='sum'):
		result = self.summizer(self._sigma, energy, component)
		return result
	
	def pi(self, energy='sum', component='sum'):
		result = self.summizer(self._pi, energy, component)
		return result

	def full(self, energy, component):
		result = self.summizer(self._full, energy, component)
		return result



class mse_spectre:
	def __init__(self, line_lambda):
		self.sigma_intensity = None
		self.pi_intensity    = None
		self.line            = line_lambda
		
		# Stark spliting MSE Ha statistical weights supplied by E. Delabie for JET like plasmas
		                            # [    Sigma group   ][        Pi group            ]
		# STARK_STATISTICAL_WEIGHTS = [0.586167, 0.206917, 0.153771, 0.489716, 0.356513]
		# SIGMA / PI = 0.56
		self.sigma_rel_int   = np.array([0.206917, 0.586167, 0.206917], 'float')										# Sigma_-1, Sigma_0, Sigma_1
		self.sigma_rel_shift = np.array([-1, 0, 1], 'float')

		self.pi_rel_int   = np.array([0.356513, 0.489716, 0.153771, 0.153771, 0.489716, 0.356513], 'float') / 2			# Pi_-4, Pi_-3, Pi_-2, Pi_2, Pi_3, Pi_4
		self.pi_rel_shift = np.array([-4, -3, -2, 2, 3, 4], 'float')

	def stark_width(self, component, dnb, chords):
		component = comp_to_list(component)
		rho       = chords.rho

		# Broading of MSE Ha component of DNB due to beam divergence
		dlambda_beam = self.line * (dnb.velocity/l_speed)*np.sin(dnb.divergence)

		# Broading of MSE Ha component of NBI beam due to lens aperture
		dlambda_lens_formula = lambda velocity, aperture : self.line * (velocity/l_speed)*np.sin(aperture)
		ful_lam_formula      = lambda l1, l2: np.sqrt(l1**2 + l2**2 + lambda_slit**2)

		ful_delta            = np.zeros((len(component), len(rho)))
		for i in range(0,len(component)):
			dlambda_beam   = dlambda_lens_formula(dnb.velocity[component[i]], dnb.divergence)
			dlambda_lens   = dlambda_lens_formula(dnb.velocity[component[i]], chords.aperture)
			ful_delta[i,:] = ful_lam_formula(dlambda_beam, dlambda_lens)
	
		return ful_delta

	def stark_intensity(self, dnb, component, chords):
		rho = chords.rho
		component = comp_to_list(component)
		rel_sigma_pi = 1														# Needs a fact-checking
		ful_rel_sum  = rel_sigma_pi * np.sum(self.pi_rel_int) + np.sum(self.sigma_rel_int)

		stark_intens_formula = lambda rel_int, halpha : rel_int * halpha / ful_rel_sum

		sigma_intensity = np.zeros((len(component),len(self.sigma_rel_int), len(rho)))
		for j in range(0, len(component)):
			for i in range(0,len(self.sigma_rel_int)):
				sigma_intensity[j,i,:] = stark_intens_formula(self.sigma_rel_int[i], dnb.halpha(component[j]+1, rho))


		pi_intensity = np.zeros((len(component),len(self.pi_rel_int), len(rho)))
		for j in range(0, len(component)):
			for i in range(0,len(self.pi_rel_int)):
				pi_intensity[j,i,:] = stark_intens_formula(self.pi_rel_int[i], dnb.halpha(component[j]+1, rho))

		return sigma_intensity, pi_intensity

	def stark_doppler(self, component, chords):
		component = comp_to_list(component)
		rho       = chords.rho

		# Doppler shifts of MSE Ha line from A due to beam observation angles
		doppler_ang_formula = lambda velocity, angles : self.line * (1 + (velocity/l_speed) * np.sin(angles))

		dlambda_doppler = np.zeros((len(component), len(rho)))
		for i in range(0,len(component)):
			dlambda_doppler[i,:] = doppler_ang_formula(dnb.velocity[component[i]], chords.angles)						# in A
		
		return dlambda_doppler

	def stark_shift(self, component, chords):
		component       = comp_to_list(component)
		rho             = chords.rho
		dlambda_doppler = self.stark_doppler(component, chords)

		# Lorentz electric field strength
		E_lor = np.zeros(len(component))

		e_lor_formula = lambda velocity : B_t * velocity * np.sin(pi/2)			# in V / m
		for i in range(0,3):
			E_lor[i] = e_lor_formula(dnb.velocity[component[i]])

		# Regular Stark energy splitting between 2 nearest components of MSE spectrum
		stark_delta_lambda = np.zeros((len(component), len(rho)))

		# lambda_shifted must be in A, e_lor in V/m
		stark_dlambda_formula = lambda lambda_shifted, e_lor : (3 * e_charge * a_bohr * (lambda_shifted/1.e10)**2 * e_lor) / (2 * h_plank * l_speed) 
		for i in range(0,len(component)):
			stark_delta_lambda[i,:] = stark_dlambda_formula(dlambda_doppler[i,:], E_lor[i])
		
		return stark_delta_lambda

	def stark_lambda(self, component, chords):
		component       = comp_to_list(component)
		rho             = chords.rho
		dlambda_doppler = self.stark_doppler(component, chords)
		stark_dlambda   = self.stark_shift(component, chords)

		stark_lambda_formula = lambda doppler_shift, shift_multiplier, delta_lambda : doppler_shift / (1 + shift_multiplier * 1.e10 * delta_lambda / self.line)
		stark_lambda_sigma = np.zeros((len(component), len(rho), len(self.sigma_rel_shift)))
		for j in range (0, len(component)):
			for i in range(0,len(rho)):
				stark_lambda_sigma[j,i,:] = stark_lambda_formula(dlambda_doppler[component[j],i],self.sigma_rel_shift, stark_dlambda[component[j],i])

		stark_lambda_pi = np.zeros((len(component), len(rho), len(self.pi_rel_shift)))
		for j in range (0, len(component)):
			for i in range(0,len(rho)):
				stark_lambda_pi[j,i,:] = stark_lambda_formula(dlambda_doppler[component[j],i],self.pi_rel_shift, stark_dlambda[component[j],i])

		return stark_lambda_sigma, stark_lambda_pi

	def stark_spectra(self, component, chords, lambda_array):
		component                    = comp_to_list(component)
		rho                          = chords.rho
		widths                       = self.stark_width(component, dnb, chords)
		lambda_0_sigma, lambda_0_pi  = self.stark_lambda(component, chords)
		sigma_intensity, pi_intensity= self.stark_intensity(dnb, component, chords)
		
		hauss_contour_formula = lambda lambda_0, intensity, width : intensity * np.exp(-(2 * np.sqrt(np.log(2)) * (lambda_array - lambda_0) / width)**2)

		sigma_spectrum = np.zeros((len(component),len(rho),len(self.sigma_rel_int),len(lambda_array)))

		for k in range(0, len(component)):
			for j in range(0, len(rho)):
				for i in range(0, len(self.sigma_rel_int)):
					sigma_spectrum[k,j,i,:] = hauss_contour_formula(lambda_0_sigma[component[k], j, i], sigma_intensity[k,i,j], widths[component[k],j])

		pi_spectrum = np.zeros((len(component),len(rho),len(self.pi_rel_int),len(lambda_array)))
		for k in range(0, len(component)):
			for j in range(0, len(rho)):
				for i in range(0, len(self.pi_rel_int)):
					pi_spectrum[k,j,i,:] = hauss_contour_formula(lambda_0_pi[component[k], j, i], pi_intensity[k,i,j], widths[component[k],j])
		
		return spectrum(sigma_spectrum, pi_spectrum, lambda_array)


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

# Some radial grid
rho = np.linspace(1, -1, num=200, endpoint=True)

#%% Setting neutralisation efficiensy interpollant
neutralisation_data = r'neutralization.dat'
Neutralization_efficiency_int = set_neutralization_efficiensy(neutralisation_data)

#%% Beam parameters
E_0                       = 60 													# Main component energy in keV
Energy                    = np.array([E_0,E_0/2,E_0/3],float)
Composition               = np.array([0.89, 0.045, 0.06],float)

I_0   = 6.1																		# Ion current in Amp
# De    = 0.08																	# Beam diameter at 1/e in meters
r_e0  = 0.032																	# Beam diameter at 1/e in meters

diver_deg = 0.6																	# Beam divergence in degrees
diver_rad = diver_deg*2*pi/360													# Beam divergence in radians

#%% Spectral data
lambda_Ha = 6562.79 # A
lambda_C2 = 6582.88 # A

# View port location, (r, z) in meters, r=0 at rho=0
upper_viewport = np.array([0.795, 1.085], float)
lower_viewport = np.array([0.835, 1.067], float)

# Lens diameter
f_lens0 = 0.15																	# in meters
d_lens0 = 0.02																	# in meters

f_hes_in       = 0.394															# in meters
d_hes_out      = 0.26															# in meters
scale_hes      = d_hes_out / f_hes_in
dispersion_hes = 5.27/scale_hes													# A/mm

slit_l         = 0.2															# in mm
slit_h         = 20																# in mm
lambda_slit    = slit_l * dispersion_hes										# in A

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
	rho_array[zer0] = 0															# Setting the closest to center point to be the center

T15_inter = plasma_interpolated(rho_array, ne_array, Te_array, ni_array, Ti_array)

#%% Creating Beam-class object
dnb = beam(E_0, Composition, r_e0, I_0, diver_rad, T15_inter)

#%% MSE spectra
chord_rho    = np.linspace(-1,1, num=15, endpoint=True)
upper_chords = chord_class(chord_rho, upper_viewport, d_lens0, f_lens0)

h_alpha = mse_spectre(lambda_Ha)
h_alpha.stark_width([0,1,2], dnb, upper_chords)
h_alpha.stark_intensity(dnb, [0,1,2], upper_chords)

lambda_array = np.linspace(lambda_Ha-20, lambda_Ha+100, num=2000, endpoint=True)
mse_spectrum = h_alpha.stark_spectra([0,1,2], upper_chords, lambda_array)

full_spectrum = mse_spectrum.full(energy = 'sum', component = 'sum')
sigma_spectrum = mse_spectrum.sigma(energy = 'sum', component = 'sum')
pi_spectrum = mse_spectrum.pi(energy = 'sum', component = 'sum')

full_spectrum_energy = mse_spectrum.full(energy = 'non', component = 'sum')
sigma_spectrum_energy = mse_spectrum.sigma(energy = 'non', component = 'sum')
pi_spectrum_energy = mse_spectrum.pi(energy = 'non', component = 'sum')

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
# chord_number = [0, 5, 10, -1]
# subplot_iterator = 0
# for item in chord_number:
# 	subplot_iterator += 1
# 	plt.subplot(2 ,2, subplot_iterator)
# 	plt.plot(lambda_array, full_spectrum[item,:])
# 	plt.plot(lambda_array, sigma_spectrum[item,:], '--')
# 	plt.plot(lambda_array, pi_spectrum[item,:], '--')
# 	plt.legend(['Full', 'Sigma', 'Pi'])
# 	plt.xlabel('Lambda, A')
# 	plt.ylabel('Intencity')
# 	plt.title(f'Chord at rho = {chord_rho[item]:.3f}')
# 	plt.grid(True)

# plt.tight_layout()
# plt.show()

# %% MSE-spectra for central chord by components
# plt.subplot(2, 2, 1)
# plt.plot(lambda_array, full_spectrum_energy[0,8,:])
# plt.plot(lambda_array, sigma_spectrum_energy[0,8,:], '--')
# plt.plot(lambda_array, pi_spectrum_energy[0,8,:], '--')
# plt.legend(['Full', 'Sigma', 'Pi'])
# plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(lambda_array, full_spectrum_energy[1,8,:])
# plt.plot(lambda_array, sigma_spectrum_energy[1,8,:], '--')
# plt.plot(lambda_array, pi_spectrum_energy[1,8,:], '--')
# plt.grid(True)

# plt.subplot(2, 2, 3)
# plt.plot(lambda_array, full_spectrum_energy[2,8,:])
# plt.plot(lambda_array, sigma_spectrum_energy[2,8,:], '--')
# plt.plot(lambda_array, pi_spectrum_energy[2,8,:], '--')
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(lambda_array, np.sum(full_spectrum_energy[:,8,:],axis = 0))
# plt.plot(lambda_array, np.sum(sigma_spectrum_energy[:,8,:],axis = 0), '--')
# plt.plot(lambda_array, np.sum(pi_spectrum_energy[:,8,:],axis = 0), '--')
# plt.grid(True)
# plt.show()
