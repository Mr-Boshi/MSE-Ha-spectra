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
