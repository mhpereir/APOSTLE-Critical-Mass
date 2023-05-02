import numpy as np

from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from astropy import constants as const


def T200(v200):
    mu = 0.6
    return (mu * const.m_p / 2 / const.k_B * v200**2).decompose()

def V200(m200,r200):
    return np.sqrt(const.G * m200 / r200).decompose()

def R200(m200, rho200):
    return np.power( m200/(4/3*np.pi*rho200), 1/3 ).decompose()

def RHO200(z):
    rho_crit = cosmo.critical_density(z)
    rho200   = rho_crit * 200
    return rho200.decompose()

def M200(t200, z):
    mu = 0.6
    constant = 2 * const.k_B / mu / const.m_p / const.G 
    
    return ( (t200.to(u.K) * constant * (4/3 * np.pi * RHO200(z))**(-1/3) )**(3/2) ).decompose()

def A(c):
    return np.log(1+c) - c/(1+c)

def NFW_profile(rtil, c, rho):
    return rho / (3 * A(c) * rtil * (1/c + rtil)**2)

def M_enc(r,c, rho, r200):
    return 4 * np.pi * r**2. * NFW_profile(r/r200,c, rho).to(u.kg / u.kpc**3).value
