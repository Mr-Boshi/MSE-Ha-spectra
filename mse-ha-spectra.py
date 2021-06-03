#%%
from numpy import r_
from scipy import constants as const
import pandas as pd
import matplotlib.pyplot as plt

class multi_var():
	def __init__(self, *simp_vars, **vars):
		if 'si' in vars.keys():
			self.si  = vars['si']
		else:
			self.si=None

		if 'sgs' in vars.keys():
			self.sgs = vars['sgs']
		else:
			self.sgs = None

	def __add__(self, other):
		if isinstance(other, multi_var):
			sgs_val = self.sgs+ other.sgs
			si_val  = self.si + other.si
			return multi_var(si=si_val, sgs=sgs_val)
		elif isinstance(other, (float, int)):
			sgs_val = self.sgs + other
			si_val  = self.si  + other
		else:
			sgs_val = self.sgs
			si_val  = self.si

		return multi_var(si=si_val, sgs=sgs_val)

	def __radd__(self, other):
		return self.__add__(other)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __rtruediv__(self, other):
		if isinstance(other, multi_var):
			sgs_val = other.sgs / self.sgs 
			si_val  = other.si  / self.si  
			return multi_var(si=si_val, sgs=sgs_val)
		elif isinstance(other, (float, int)):
			sgs_val = other / self.sgs
			si_val  = other / self.si 
		else:
			sgs_val = self.sgs
			si_val  = self.si

	def __mul__(self, other):
		if isinstance(other, multi_var):
			sgs_val = self.sgs * other.sgs
			si_val  = self.si  * other.si
			return multi_var(si=si_val, sgs=sgs_val)
		elif isinstance(other, (float, int)):
			sgs_val = self.sgs * other
			si_val  = self.si  * other
		else:
			sgs_val = self.sgs
			si_val  = self.si

		return multi_var(si=si_val, sgs=sgs_val)

	def __truediv__(self, other):
		if isinstance(other, multi_var):
			sgs_val = self.sgs / other.sgs
			si_val  = self.si  / other.si
			return multi_var(si=si_val, sgs=sgs_val)
		elif isinstance(other, (float, int)):
			sgs_val = self.sgs / other
			si_val  = self.si  / other
		else:
			sgs_val = self.sgs
			si_val  = self.si


	def __sub__(self, other):
		if isinstance(other, multi_var):
			sgs_val = self.sgs - other.sgs
			si_val  = self.si  - other.si
			return multi_var(si=si_val, sgs=sgs_val)
		elif isinstance(other, (float, int)):
			sgs_val = self.sgs - other
			si_val  = self.si  - other
		else:
			sgs_val = self.sgs
			si_val  = self.si
	

	def __repr__(self):
		return [self.si, self.sgs]
	def __str__(self):
		return str([self.si, self.sgs])
	def __pow__(self, other):
		return multi_var(si=self.si**other, sgs=self.sgs**other)


def e_to_v(E):
	e_to_v_coeff = 1.38e6
	V = e_to_v_coeff*(E * 1e3) ** 0.5
	return multi_var(si=V*100, sgs=V)


#%%
h_plank  = const.physical_constants['Planck constant'][0]
e_mass   = const.physical_constants['electron mass'][0]
e_charge = const.physical_constants['elementary charge'][0]
l_speed  = const.physical_constants['speed of light in vacuum'][0]
mu_0     = const.physical_constants['vacuum mag. permeability'][0]
pi       = const.pi

#%%
z_p  = 1
z_c6 = 6
Zeff = 3

I_pl  = 1.e6
R0    = 1.5
a     = 0.67
B_t   = 2
Ne_av = 4.5e19

E_0  = 60
E_02 = E_0/2
E_03 = E_0/3

V_0  = e_to_v(E_0)
V_02 = e_to_v(E_02)
V_03 = e_to_v(E_03)


q_a = 2*pi*a**2*B_t/(R0*I_pl*mu_0)

# %%
file_name = 't-15_data.txt'
T15_data = pd.read_table(file_name, sep=r"\s+").to_numpy()

r_array=T15_data[:,0]
ne_array = T15_data[:, 2]
Te_array = T15_data[:, 3]
Ti_array = T15_data[:, 4]
ni_array = T15_data[:, 5]

plt.plot(r_array, ne_array)
plt.show()


