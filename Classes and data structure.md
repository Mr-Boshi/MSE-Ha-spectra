# Classes and data structures in mse-ha-spectra

There are 8 classes for now:
## tokamak_data
1. Atributes:
A class that stores common tokamak paremeters.
   * a  
   * R0 
   * Ipl
   * Bt 
2. Methods:
   None
   
## plasma_interpolated
A class with interpolated at $\rho$-grid plasma parameters.
1. Atributes:
   * ne
   * ni
   * te
   * ti
2. Methods:
   None

## beam_interpolated
A class that takes array shaped [3, ...] and creates 3 attributes with 1-d interpolants from each row.
1. Atributes:
   * e0
   * e02
   * e03

2. Methods:
   None

## viewport_class
A class that stores information about viewport location and lens parameters.
1. Atributes:
   * coordinates
   * f_lens     
   * d_lens     
   * f_in       
   * d_out      
   * slit       
   * slit_l     
   * slit_h     
   * scale      
   * dispersion 
   * lambda_slit
2. Methods:
   None

## beam_class
A big class that stores all data concearning DNB parametes
1. Atributes:
   * r_e0        
   * divergence  
   * i_0         
   * energy      
   * velocity    
   * composition 
   * _density    
   * _bsrate     
   * _exrate     
   * _intensity  
   * _d_array    
2. Methods:
   * h_from_rho(self, tokamakdata, rho, viewport, chord_angle)
   * real_rho(self,tokamakdata, rho, viewport, chord_angle)
   * density_2d(self, component, tokamakdata, viewport, chord, rho)
   * density(self,component=0, rho=0.)
   * bsrate(self,component=0, rho=0.)
   * exrate(self,component=0, rho=0.)
   * halpha(self,component=0, rho=0.)
   * den_pfofile(self, tokamakdata, rho)

## chord_class
A class that stores information about the chords
1. Atributes:
   * rho           
   * r             
   * _r_ported     
   * angles        
   * _center_angle 
   * lens_d
   * length
   * scale
   * omega
   * fiber_omega
   * aperture      
   * fiber_aperture
   * fiber_distance   
2. Methods:
   None


## spectrum_class
A subclass neded to store MSE-spectrum and return in in a more usable format
1. Atributes:
   * _sigma  
   * _pi     
   * _full   
   * _lambdas

2. Methods:
   * summizer(self, pi_or_sigma, energy, component)
   * sigma(self, energy='sum', component='sum')
   * pi(self, energy='sum', component='sum')
   * full(self, energy, component)

## mse_spectre
A big class that lets calculate MSE-spectra.
1. Atributes:
   * sigma_intensity 
   * pi_intensity    
   * line            
   * sigma_rel_int  
   * sigma_rel_shift
   * pi_rel_int
   * pi_rel_shift

2. Methods:
   * stark_width(self, component, viewport, dnb, chords)
   * stark_intensity(self, dnb, component, chords)
   * stark_doppler(self, component, chords)
   * stark_shift(self, component, chords)
   * stark_lambda(self, component, chords)
   * stark_spectra(self, component, viewport, chords, lambda_array)

