total_mass_of_ss = 2.7 * 1e27
total_mass_of_terrestrial = 1.18 * 1e25
total_mass_of_gas_planets =  2.26 * 1e27
total_mass_of_minor_bodies = 4.282 * 1e26

import scipy as sp

data = sp.genfromtxt("minerals.csv", delimiter=",")

print(data[:10])
