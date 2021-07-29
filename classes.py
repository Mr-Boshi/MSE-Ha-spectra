#%% IMPORT
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy import constants as const
from numpy import pi

import numpy as np
import json

#%% CLASSES
class tokamak_data:
	def __init__(self, a, R0, I_pl, B_t):
		self.a   = a
		self.R0  = R0
		self.Ipl = I_pl
		self.Bt  = B_t

class plasma_interpolated:
	def __init__(self, r, ne, te, ni, ti):
		self.ne = interp1d(r, ne, kind='cubic', bounds_error=False, fill_value=0.)
		self.ni = interp1d(r, ni, kind='cubic', bounds_error=False, fill_value=0.)
		self.te = interp1d(r, te, kind='cubic', bounds_error=False, fill_value=0.)
		self.ti = interp1d(r, ti, kind='cubic', bounds_error=False, fill_value=0.)

class multirow_interpolant:
	def __init__(self, r, table):
		# number_of_components = len(table)
		self._interpolants = list()
		for i in range(len(table)):
			self._interpolants.append(interp1d(r, table[i,:], kind='cubic', bounds_error=False, fill_value=0.))
	
	def get(self, row, rho):
		if type(row) is int:
			return self._interpolants[row](rho)
		else:
			overrow = np.zeros((len(row),len(rho)))
			for i in row:
				overrow[i,:] = self._interpolants[row[i]](rho)

			return overrow
	
	def summed(self, rho):
		overall = np.zeros(len(rho))
		for interpolant in self._interpolants:
			overall =+ interpolant(rho)
		
		return overall

class viewport_class:
	def __init__(self, coordinates, f_lens, d_lens, f_in, d_out, slit):
		# View port location, (r, z) in meters, r=0 at rho=0
		self.coordinates = coordinates
		self.f_lens      = f_lens
		self.d_lens      = d_lens
		self.f_in        = f_in
		self.d_out       = d_out
		self.slit        = slit
		self.slit_l      = slit[0]
		self.slit_h      = slit[-1]

		self.scale       = d_out / f_in
		self.dispersion  = 5.27/self.scale										# in A/mm
		self.lambda_slit = self.slit_l * self.dispersion						# in A

class chord_class:
	def __init__(self, rho, viewport, tokamakdata):
		self.rho           = rho
		self.r             = rho*tokamakdata.a									# r = 0 at rho = 0
		self._r_ported     = (-1)*self.r + viewport.coordinates[0]				# r = 0 at viewport[0], direction of the asis is reverced
		self.angles        = np.arctan(self._r_ported / viewport.coordinates[1])# in radians
		self._center_angle = np.arctan(viewport.coordinates[0] / viewport.coordinates[1])

		self.lens_d = viewport.d_lens * np.sqrt((pi/4) * np.cos(self.angles - self._center_angle))						# light diameter of lens

		self.length = viewport.coordinates[1] / np.cos(self.angles)
		self.scale  = self.length / viewport.f_lens - 1

		# Solid angle of the lens and on the fiber for given LOS
		self.omega       = (pi/4) * (viewport.d_lens * self.lens_d) / self.length**2
		self.fiber_omega = self.omega * self.scale**2

		# Averaged aperture angles of the lens and MSE Ha measurements
		self.aperture       = np.sqrt(self.omega)
		self.fiber_aperture = np.sqrt(self.fiber_omega)
		self.fiber_distance = self.length / self.scale							# distances between lens and fibers

class spectrum_class:
	def __init__(self, sigma, pic, lambda_array):
		self._sigma   = sigma 
		self._pi      = pic
		self._full    = np.concatenate([sigma, pic], axis=2)
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

class beam_class:
	def __init__(self, E_0, Composition, r_e0, I_0, divergence_rad, tokamakdata, plasma_int, impurity_charge):
		self.r_e0         = r_e0
		self.divergence   = divergence_rad
		self.i_0          = I_0
		self.composition  = Composition
		
		e_charge = const.physical_constants['elementary charge'][0]
		z_p      = 1
		z_imp    = impurity_charge

		Energy        = np.array([E_0,E_0/2,E_0/3],float)
		Velocity      = self.e_2_v(Energy) 											# Particles velocities in m/s
		self.energy   = Energy
		self.velocity = Velocity

		Neutralization_efficiency = self.neutralization_efficiensy(Energy)

		r_05         = r_e0/(np.sqrt(2*np.sqrt(np.log(2))))
		Beam_cs      = pi*(r_05/2)**2
		j_beam_total = I_0/(Composition[0]*Beam_cs)								# Beam total current density in A/m^2
		j_atoms      = j_beam_total*Composition*Neutralization_efficiency
		N0_atoms     = j_atoms/(e_charge*Velocity)								# Beam density in m-3 on entering the plasma

		# Using beam stopping rates from ADAS (with CHERAB parser)
		beamstop_h_json_file = r'beam\stopping\h\h\1_default.json'
		beamstop_c_json_file = r'beam\stopping\h\c\6_default.json'

		#%% Fractions of plasma composition species (used for ADAS rates calculation)
		rho = np.linspace(1, -1, num=200, endpoint=True)
		
		f_imp = (plasma_int.ne(rho)/plasma_int.ni(rho) - z_p)/(z_imp-z_p)
		f_h = 1-f_imp

		sum_zf  = z_p * f_h + z_imp * f_imp
		sum_z2f = z_p**2 * f_h + z_imp**2 * f_imp

		# Equivalent electron density for each plasma specie
		ne_equ_h = (plasma_int.ne(rho) / sum_zf) * (sum_z2f / z_p)
		ne_equ_c = (plasma_int.ne(rho) / sum_zf) * (sum_z2f / z_imp)

		bs_rate_h = self.json_to_rate(beamstop_h_json_file, 'beam-stop', ne_equ_h, plasma_int.ti(rho))
		bs_rate_c = self.json_to_rate(beamstop_c_json_file, 'beam-stop', ne_equ_c, plasma_int.ti(rho))

		bs_rate = (bs_rate_h + bs_rate_c) / sum_zf
		BS_int  = multirow_interpolant(rho, bs_rate)

		# Beam-stopping calculation
		rho_dif             = tokamakdata.a * np.abs(np.diff(rho))				# in meters
		Ne_Sbs              = np.zeros((len(Energy), len(rho)))					# Preallocation
		Ne_Sbs_sum          = np.zeros((len(Energy), len(rho)))					# Preallocation

		for i in range(1,len(rho_dif)):
			qrat             = np.array([BS_int.get(0, rho[i+1]), BS_int.get(1, rho[i+1]), BS_int.get(1, rho[i+1])], float)
			Ne_Sbs    [:, i] = plasma_int.ne(rho[i+1])*rho_dif[i]*qrat
			Ne_Sbs_sum[:, i] = (1/Velocity)*np.sum(Ne_Sbs, axis=1)

		Ne_Sbs_sum[:, -1] = Ne_Sbs_sum[:, -2]
		N0_atoms_array = np.ones(Ne_Sbs_sum.shape)
		for i in range(0,3):
			N0_atoms_array[i, :] *= N0_atoms[i]

		N0_array = N0_atoms_array * np.exp(-Ne_Sbs_sum)
		Density  = multirow_interpolant(rho, N0_array)
		

		# MSE H_alpha exitation rates
		h_alpha_h_json_file = r'beam\emission\h\h\1_default.json'
		h_alpha_c_json_file = r'beam\emission\h\c\6_default.json'

		h_alpha_rate_h = self.json_to_rate(h_alpha_h_json_file, 'emission', ne_equ_h, plasma_int.ti(rho))
		h_alpha_rate_c = self.json_to_rate(h_alpha_c_json_file, 'emission', ne_equ_c, plasma_int.ti(rho))

		h_alpha_rate = (h_alpha_rate_h + h_alpha_rate_c) / sum_zf
		h_a_rate_inter = multirow_interpolant(rho, h_alpha_rate)

		h_alpha_intens      = np.zeros(((len(Energy), len(rho))))
		
		for i in range(3):
			h_alpha_intens[i,:] = h_a_rate_inter.get(i,rho)*Density.get(i,rho)*plasma_int.ne(rho)
		
		Intensity     = multirow_interpolant(rho, h_alpha_intens)

		self._density     = Density
		self._bsrate      = BS_int
		self._exrate      = h_a_rate_inter
		self._intensity   = Intensity
		self._d_array     = np.linspace(-tokamakdata.a, tokamakdata.a, num=500, endpoint=True)

	def neutralization_efficiensy(self, energy):
		filename  = r'neutralization.dat'
		neutralization_data  = np.loadtxt(filename)
		beam_energy = neutralization_data[:,0]
		efficiensy =  neutralization_data[:,1]
		efficiensy_int = interp1d(beam_energy, efficiensy, kind='cubic', bounds_error=False, fill_value='extrapolate')

		return efficiensy_int(energy)

	
	def h_from_rho(self, tokamakdata, rho, viewport, chord_angle):
		a = tokamakdata.a
		L = viewport.coordinates[0]
		H = viewport.coordinates[-1]

		return H - (L - rho * a) / np.tan(chord_angle)

	def real_rho(self,tokamakdata, rho, viewport, chord_angle):
		h = self.h_from_rho(tokamakdata, rho, viewport, chord_angle)
		return np.sqrt(rho**2 + h**2)

	def density_2d(self, component, tokamakdata, viewport, chord, rho):
		if not component in [0, 1, 2, 3]:
			raise ValueError('Wrong argument. Accepted arguments are: 0, 1, 2, 3.')
		else:
			density_1d      = np.array([self.density(component, rho) for _ in range(0, len(chord.angles)) ])
			h               = np.array([self.h_from_rho(tokamakdata, rho, viewport, chord.angles[i]) for i in range(0, len(chord.rho))])
			density_profile = self.den_pfofile(tokamakdata,rho)

			density_fraction = np.zeros(density_1d.shape)
			for j in range(0,len(rho)):
				for i in range(0, len(chord.rho)):
					density_fraction[i,j] = density_profile(h[i,j], rho[j])

			return density_1d * density_fraction

	def density(self,component=0, rho=0.):
			if component == 0:
				return self._density.summed(rho)
			else:
				return self._density.get(component, rho)

	def bsrate(self,component=0, rho=0.):
			if component == 0:
				return self._bsrate.summed(rho)
			else:
				return self._bsrate.get(component, rho)
	
	def exrate(self,component=0, rho=0.):
			if component == 0:
				return self._exrate.summed(rho)
			else:
				return self._exrate.get(component, rho)

	def halpha(self,component=0, rho=0.):
		if component == 0:
			return self._intensity.summed(rho)
		else:
			return self._intensity.get(component, rho)
	
	def den_pfofile(self, tokamakdata, rho):
		rho_array         = np.array(rho, float, ndmin=1)
		r_e_array         = self.r_e0 + ((1 - rho_array) * tokamakdata.a) * np.tan(self.divergence / 2)
		
		den_profile_array = np.zeros((len(rho_array), len(self._d_array)))
		for i in range(0, len(rho_array)):
			den_profile             = np.exp(-0.5*(self._d_array/(r_e_array[i]/np.sqrt(2)))**2)
			den_profile_sum         = np.sum(den_profile)
			den_profile_array[i, :] = den_profile / den_profile_sum

		if len(rho_array)==1:
			den_profile_array = interp1d(self._d_array, den_profile_array[0,:], kind='cubic', bounds_error=False, fill_value=0.)
		else:
			den_profile_array = interp2d(self._d_array, rho, den_profile_array, kind='cubic', bounds_error=False, fill_value=0.)

		return den_profile_array

	def e_2_v(self, energy):
		# E must be in keV

		e_charge    = const.physical_constants['elementary charge'][0]
		proton_mass = const.physical_constants['proton mass'][0]
		e_to_v_coeff = np.sqrt(2*e_charge/proton_mass)
		V = e_to_v_coeff*(energy * 1e3) ** 0.5
		return V # in m/s

	def json_to_rate(self, filename, type, ne_eq, ti):
		# E must be in keV
		# Beamstop rates are in m3/s

		Energy = self.energy

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

		rate = np.zeros((len(Energy), len(ne_eq)))
		for j in range(0, len(Energy)):
			for i in range(0, len(ne_eq)):
				rate[j, i] = s_ne_int(Energy[j]*1e3, ne_eq[i])*s_t_int(ti[i])
				if rate[j,i] <=0.0:
					rate[j,i] = 0.0

		return rate

class chord_class:
	def __init__(self, rho, viewport, tokamakdata):
		self.rho           = rho
		self.r             = rho*tokamakdata.a									# r = 0 at rho = 0
		self._r_ported     = (-1)*self.r + viewport.coordinates[0]				# r = 0 at viewport[0], direction of the asis is reverced
		self.angles        = np.arctan(self._r_ported / viewport.coordinates[1])# in radians
		self._center_angle = np.arctan(viewport.coordinates[0] / viewport.coordinates[1])

		self.lens_d = viewport.d_lens * np.sqrt((pi/4) * np.cos(self.angles - self._center_angle))						# light diameter of lens

		self.length = viewport.coordinates[1] / np.cos(self.angles)
		self.scale  = self.length / viewport.f_lens - 1

		# Solid angle of the lens and on the fiber for given LOS
		self.omega       = (pi/4) * (viewport.d_lens * self.lens_d) / self.length**2
		self.fiber_omega = self.omega * self.scale**2

		# Averaged aperture angles of the lens and MSE Ha measurements
		self.aperture       = np.sqrt(self.omega)
		self.fiber_aperture = np.sqrt(self.fiber_omega)
		self.fiber_distance = self.length / self.scale							# distances between lens and fibers

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

	def stark_width(self, component, viewport, beam, chords):
		l_speed   = const.physical_constants['speed of light in vacuum'][0]
		component = self.comp_to_list(component)
		rho       = chords.rho

		# Broading of MSE Ha component of beam due to beam divergence
		dlambda_beam = self.line * (beam.velocity/l_speed)*np.sin(beam.divergence)

		# Broading of MSE Ha component of NBI beam due to lens aperture
		dlambda_lens_formula = lambda velocity, aperture : self.line * (velocity/l_speed)*np.sin(aperture)
		ful_lam_formula      = lambda l1, l2: np.sqrt(l1**2 + l2**2 + viewport.lambda_slit**2)

		ful_delta            = np.zeros((len(component), len(rho)))
		for i in range(0,len(component)):
			dlambda_beam   = dlambda_lens_formula(beam.velocity[component[i]], beam.divergence)
			dlambda_lens   = dlambda_lens_formula(beam.velocity[component[i]], chords.aperture)
			ful_delta[i,:] = ful_lam_formula(dlambda_beam, dlambda_lens)
	
		return ful_delta

	def stark_intensity(self, beam, component, chords):
		rho = chords.rho
		component = self.comp_to_list(component)
		rel_sigma_pi = 1														# Needs a fact-checking
		ful_rel_sum  = rel_sigma_pi * np.sum(self.pi_rel_int) + np.sum(self.sigma_rel_int)

		stark_intens_formula = lambda rel_int, halpha : rel_int * halpha / ful_rel_sum

		sigma_intensity = np.zeros((len(component),len(self.sigma_rel_int), len(rho)))
		for j in range(0, len(component)):
			for i in range(0,len(self.sigma_rel_int)):
				sigma_intensity[j,i,:] = stark_intens_formula(self.sigma_rel_int[i], beam.halpha(component[j], rho))


		pi_intensity = np.zeros((len(component),len(self.pi_rel_int), len(rho)))
		for j in range(0, len(component)):
			for i in range(0,len(self.pi_rel_int)):
				pi_intensity[j,i,:] = stark_intens_formula(self.pi_rel_int[i], beam.halpha(component[j], rho))

		return sigma_intensity, pi_intensity

	def stark_doppler(self, component, beam, chords):
		l_speed   = const.physical_constants['speed of light in vacuum'][0]
		component = self.comp_to_list(component)
		rho       = chords.rho

		# Doppler shifts of MSE Ha line from A due to beam observation angles
		doppler_ang_formula = lambda velocity, angles : self.line * (1 + (velocity/l_speed) * np.sin(angles))

		dlambda_doppler = np.zeros((len(component), len(rho)))
		for i in range(0,len(component)):
			dlambda_doppler[i,:] = doppler_ang_formula(beam.velocity[component[i]], chords.angles)						# in A
		
		return dlambda_doppler

	def stark_shift(self, component, beam, chords, tokamakdata):
		l_speed         = const.physical_constants['speed of light in vacuum'][0]
		e_charge        = const.physical_constants['elementary charge'][0]
		a_bohr          = const.physical_constants['Bohr radius'][0]
		h_plank         = const.physical_constants['Planck constant'][0]
		component       = self.comp_to_list(component)
		rho             = chords.rho
		dlambda_doppler = self.stark_doppler(component, beam, chords)

		# Lorentz electric field strength
		E_lor = np.zeros(len(component))

		e_lor_formula = lambda velocity : tokamakdata.Bt * velocity * np.sin(pi/2)			# in V / m
		for i in range(3):
			E_lor[i] = e_lor_formula(beam.velocity[component[i]])

		# Regular Stark energy splitting between 2 nearest components of MSE spectrum
		stark_delta_lambda = np.zeros((len(component), len(rho)))

		# lambda_shifted must be in A, e_lor in V/m
		stark_dlambda_formula = lambda lambda_shifted, e_lor : (3 * e_charge * a_bohr * (lambda_shifted/1.e10)**2 * e_lor) / (2 * h_plank * l_speed) 
		for i in range(0,len(component)):
			stark_delta_lambda[i,:] = stark_dlambda_formula(dlambda_doppler[i,:], E_lor[i])
		
		return stark_delta_lambda

	def stark_lambda(self, component, beam, chords, tokamakdata):
		component       = self.comp_to_list(component)
		rho             = chords.rho
		dlambda_doppler = self.stark_doppler(component, beam, chords)
		stark_dlambda   = self.stark_shift(component, beam, chords, tokamakdata)

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

	def stark_spectra(self, component, viewport, chords, lambda_array, beam, tokamakdata):
		component                    = self.comp_to_list(component)
		rho                          = chords.rho
		widths                       = self.stark_width(component, viewport, beam, chords)
		lambda_0_sigma, lambda_0_pi  = self.stark_lambda(component, beam, chords, tokamakdata)
		sigma_intensity, pi_intensity= self.stark_intensity(beam, component, chords)
		
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
		
		return spectrum_class(sigma_spectrum, pi_spectrum, lambda_array)

	def comp_to_list(self, component):
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
