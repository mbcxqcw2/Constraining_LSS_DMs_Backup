import numpy as np

Mpc_si = 3.08567758E22
Mpc_cgs = Mpc_si*100.

kpc_si = Mpc_si/1000.
pc_si = Mpc_si/1E6

au_si = 149597870700

c_si = 299792458

G_si = 6.67384E-11
Myr_si = 3.1556926E13
yr_si = Myr_si/1E6
Msol_si = 1.9891E30
Msol_cgs = Msol_si*1000.

kb_si = 1.3806488E-23
h_si = 6.62606957E-34
m_e_si = 9.10938291E-31
mp_si = 1.67E-27

# Planck Cosmology
omegaMatter = 0.3089
omegaBaryon = 0.0486
omegaDm = omegaMatter - omegaBaryon
omegaLambda = 0.6911
Hubble = 67.74
Hubble_si = Hubble*1000.0/Mpc_si
rho_c_si = 3*Hubble_si**2/(8*np.pi*G_si)
rho_c_cgs = rho_c_si/1000.

