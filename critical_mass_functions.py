from __future__ import division
from scipy.special import cbrt, gammainc, erf
from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev
import cosmology_functions as cf
import params

from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from scipy.optimize import newton_krylov, brentq, curve_fit

import numpy as np

from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from astropy import constants as const

def find_iso_mcrit(in_):
    z, c, T_b = in_[0], in_[1], in_[2]
    return brentq(iso_func, 7,10, args=(c,z, T_b),  rtol=1e-3)

def find_iso_mcrit_2(in_):
    z, c, rho_c, T_b = in_[0], in_[1], in_[2], in_[3]
    return brentq(iso_func_2, 7,10, args=(c, z, rho_c, T_b),  rtol=1e-3)


def find_iso_mcrit_lud(in_):
    z, T_b = in_[0], in_[1]
    return brentq(iso_func_ludlow, 5,12, args=(z, T_b),  rtol=1e-3)

def find_iso_mcrit_lud_2(in_):
    z, T_b, rho_c = in_[0], in_[1], in_[2]
    return brentq(iso_func_ludlow_2, 5,12, args=(z, T_b, rho_c),  rtol=1e-3)

def find_iso_mcrit_lud_2_mgas_norm(in_):
    z, T_b, rho_c = in_[0], in_[1], in_[2]
    return brentq(iso_func_ludlow_2_mgas_norm, 6.5,8, args=(z, T_b, rho_c),  rtol=1e-3)

def find_iso_mcrit_lud_3(in_): #t_b changes if t200 < 10,000 K
    z, m200_upper, rho_c = in_[0], in_[1], in_[2]
    return brentq(iso_func_ludlow_3, 6, m200_upper, args=(z, rho_c),  rtol=1e-3)

 
def find_Trho_mcrit(in_):
    z, m200_upper, c = in_[0], in_[1], in_[2]
    print("running: ", z, m200_upper)
    return brentq(Trho_func, 7, m200_upper, args=(c,z),  rtol=1e-3)

def find_Trho_mcrit_2(in_):
    z, m200_upper, c, rho_c = in_[0], in_[1], in_[2], in_[3]
    print("running: ", z, m200_upper)
    return brentq(Trho_func_2, 7, m200_upper, args=(c,z, rho_c),  rtol=1e-3)

def find_Trho_mcrit_2_adb(in_):
    z, m200_upper, c, rho_c = in_[0], in_[1], in_[2], in_[3]
    print("running: ", z, m200_upper)
    return brentq(Trho_func_2_adb, 7, m200_upper, args=(c,z, rho_c),  rtol=1e-3)


def find_Trho_conce(in_):
    z, c_upper, logm200 = in_[0], in_[1], in_[2]
    print("running: ", z, c_upper)
    return brentq(Trho_func_concentration, 2.2, c_upper, args=(logm200,z),  rtol=1e-3)


def find_Trho_mcrit_lud(in_):
    z, m200_upper = in_[0], in_[1]
    print("running: ", z, m200_upper)
    return brentq(Trho_func_ludlow, 7, m200_upper, args=(z),  rtol=1e-3)

def find_Trho_mcrit_lud_adb(in_):
    z, m200_upper = in_[0], in_[1]
    print("running: ", z, m200_upper)
    return brentq(Trho_adb_func_ludlow, 7, m200_upper, args=(z),  rtol=1e-3)


def find_Trho_mcrit_lud_2(in_):
    z, m200_upper, rho_c = in_[0], in_[1], in_[2]
    print("running: ", z, m200_upper)
    return brentq(Trho_func_ludlow_2, 7, m200_upper, args=(z, rho_c),  rtol=1e-3)

def find_Trho_mcrit_lud_2_adb(in_):
    z, m200_upper, rho_c = in_[0], in_[1], in_[2]
    print("running: ", z, m200_upper)
    return brentq(Trho_adb_func_ludlow_2, 7.5, m200_upper, args=(z, rho_c),  rtol=1e-3)

def find_Trho_mcrit_lud_2_adb_norm_gas(in_):
    z, m200_upper, rho_c = in_[0], in_[1], in_[2]
    print("running: ", z, m200_upper)
    return brentq(Trho_adb_func_ludlow_2_norm_gas, 7.5, m200_upper, args=(z, rho_c),  rtol=1e-3)


def find_central_density(in_):
    rho_norm, z, logm200 = 1, in_[0], in_[1]
    print("running: ", z, logm200)
    return Trho_func_ludlow_central(rho_norm, logm200, z)

def find_central_density_adb(in_):
    z, logm200 = in_[0], in_[1]
    print("running: ", z, logm200)
    return Trho_func_ludlow_central_adb(logm200, z)

def find_central_density_c(in_):
    rho_norm, c, z, logm200 = 1, in_[0], in_[1], in_[2]
    print("running: ", z, logm200)
    return Trho_func_concentration_central(rho_norm, c,logm200, z)

def find_central_density_c_adb(in_):
    c, z, logm200 = in_[0], in_[1], in_[2]
    print("running: ", z, logm200)
    return Trho_func_concentration_central_adb(c,logm200, z)


def find_mgas_lud(in_):
    logm200,z = in_[0], in_[1]
    print("running: ", z, logm200)
    return mgas_ludlow(logm200, z)

def find_mgas_c(in_):
    logm200, z, c = in_[0], in_[1], in_[2]
    print("running: ", z, c, logm200)
    return mgas_c(logm200, z, c)

# funcitons for finding critical mass

def mgas_integrand(r, rho):
    '''
    linear radius integrand
    '''
    return 4 * np.pi * r**2 * rho(r)

def mgas_integrand_logr(logr, rho):
    '''
    log radius integrand
    '''
    return 4 * np.pi * np.power(10, 3*logr) * rho(logr) * np.log(10)

def iso_func(logm200, c, z, T_b):
    '''
    zero finding function that we solve to obtain critical mass for the isothermal model
    for a given c,z, critical mass (logm200) is when output of this function is zero
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    log_rtil_range = np.arange(-2,0.01,0.01)
    r_til_range    = np.power(10, log_rtil_range)
    rho_iso_range  = np.asarray([rho_rel_iso(rtil, c, T_b, t200) * rho_bar_b  for rtil in r_til_range]) #n_Total
    
    rho_iso_range = rho_iso_range *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    rho_iso_func  = interp1d(x=log_rtil_range, y=rho_iso_range, bounds_error=True)
    
    dx            = 0.01
    x_range       = np.arange(-2,0,dx)
    y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_iso_func(x_range) * np.log(10)
    mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
    
    #mgas = quad(mgas_integrand_logr, -2, 0, args=(rho_iso_func))[0] * r200.to(u.kpc).value**3
    
    #print(z, logm200, omega_b/omega_m)
    #print(mgas/m200.to(u.kg).value)
    
    return (omega_b/omega_m - mgas/m200.to(u.kg).value)


def iso_func_2(logm200, c, z, rho_c, T_b):
    '''
    zero finding function that we solve to obtain critical mass for the isothermal model
    for a given c,z, critical mass (logm200) is when output of this function is zero
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    log_rtil = -2
    r_til    = 10**-2.0
    rho_iso  = rho_rel_iso(r_til, c, T_b, t200) * rho_bar_b  #n_Total
    
    #rho_iso_range = rho_iso_range #*const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    return (rho_iso - rho_c)



def iso_func_ludlow(logm200, z, T_b):
    '''
    zero finding function that we solve to obtain critical mass for the isothermal model
    for a given c,z, critical mass (logm200) is when output of this function is zero
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    #T_b       = T_b * u.K
    
    log_rtil_range = np.arange(-2,0.01,0.01)
    r_til_range    = np.power(10, log_rtil_range)
    rho_iso_range  = np.asarray([rho_rel_iso(rtil, c, T_b, t200) * rho_bar_b  for rtil in r_til_range]) #n_Total
    
    rho_iso_range = rho_iso_range *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    rho_iso_func  = interp1d(x=log_rtil_range, y=rho_iso_range, bounds_error=True)
    
    dx            = 0.01
    x_range       = np.arange(-2,0,dx)
    y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_iso_func(x_range) * np.log(10)
    mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
    
    #mgas = quad(mgas_integrand_logr, -2, 0, args=(rho_iso_func))[0] * r200.to(u.kpc).value**3
    
    #print(z, logm200, omega_b/omega_m)
    #print(mgas/m200.to(u.kg).value)
    
    return (omega_b/(omega_m) - mgas/m200.to(u.kg).value)



def iso_func_ludlow_2(logm200, z, T_b, rho_c):
    '''
    zero finding function that we solve to obtain critical mass for the isothermal model
    for a given c,z, critical mass (logm200) is when output of this function is zero
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    #T_b       = T_b * u.K
    
    log_rtil = -2.
    r_til    = 10**log_rtil
    rho_iso  = rho_rel_iso(r_til, c, T_b, t200) * rho_bar_b  #n_Total
    
    rho_iso = rho_iso # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    return (rho_iso - rho_c)


def iso_func_ludlow_2_mgas_norm(logm200, z, T_b, rho_c):
    '''
    zero finding function that we solve to obtain critical mass for the isothermal model
    for a given c,z, critical mass (logm200) is when output of this function is zero
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    print(logm200)
    
    def finding_gas_mass(rho_norm_til):
        
        rho_norm = rho_norm_til * rho_bar_b * const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
        
        log_rtil_range = np.arange(-2,0.01,0.01)
        r_til_range    = np.power(10, log_rtil_range)
        
        
        rho_iso_range  = np.asarray([rho_rel_iso_norm_r200(rtil, c, T_b, t200, rho_norm)  for rtil in r_til_range]) #n_Total
                
        rho_iso_func  = interp1d(x=log_rtil_range, y=rho_iso_range, bounds_error=True)
        
        dx            = 0.01
        x_range       = np.arange(-2,0,dx)
        y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_iso_func(x_range) * np.log(10)
        mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
        
        print(mgas/m200.to(u.kg).value)
        
        return (mgas/m200.to(u.kg).value - omega_b/omega_m)
    
    rho_norm_til2 =  brentq(finding_gas_mass, 1e-6, 1e3, rtol=1e-3)
    
    log_rtil = -2
    r_til    = 10**-2.0
    rho_iso  = rho_rel_iso_norm_r200(r_til, c, T_b, t200, rho_norm_til2 * rho_bar_b)  #n_Total
    
    #rho_iso_range = rho_iso_range #*const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    return (rho_iso - rho_c)


def iso_func_ludlow_3(in_):
    z,logm200, rho_c = in_[0], in_[1], in_[2]
    '''
    zero finding function that we solve to obtain critical mass for the isothermal model
    for a given c,z, critical mass (logm200) is when output of this function is zero
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    T_b = 10**4 * u.K
    if t200 <= 10**4 * u.K:
        T_b = t200
        
        
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    #T_b       = T_b * u.K
    
    log_rtil = -2.
    r_til    = 10**log_rtil
    rho_iso  = rho_rel_iso(r_til, c, T_b, t200) * rho_bar_b  #n_Total
    
    rho_iso = rho_iso # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    print(z, logm200, rho_iso, np.log10(T_b.value))
    return (rho_iso / rho_c)


def iso_func_find_T200(logm200_mcrit_lud, z):
    
    def find_minimiz(logtb):
        tb = np.power(10, logtb)
        in_ = [z, tb*u.K]
        return logm200_mcrit_lud - find_iso_mcrit_lud(in_)
    
    out = brentq(find_minimiz, 3.5, 4.5)
    
    return out
    
    

def Trho_func(logm200, c, z):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil_range = np.arange(-2,0.01,0.1)
    r_til_range    = np.power(10, log_rtil_range)
    rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                       1e-7, 1e4, rtol=1e-3) for rtil in r_til_range]) #n_Total
    
    rho_rel_range = rho_rel_range *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range, bounds_error=True)
    
    dx            = 0.01
    x_range       = np.arange(-2,0,dx)
    y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func(x_range) * np.log(10)
    mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
    
    #mgas          = quad(mgas_integrand_logr, -2, 0, args=(rho_rel_func))[0] * r200.to(u.kpc).value**3
    #print(mgas/m200.to(u.kg).value)
    
    return (omega_b/omega_m - mgas/m200.to(u.kg).value)



def Trho_func_2(logm200, c, z, rho_c):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil = -2.
    r_til    = 10**log_rtil
    rho_rel  = brentq(F_inv1(r_til, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                       1e-7, 1e4, rtol=1e-3) #n_Total
    
    rho_rel = rho_rel # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    
    return (rho_rel - rho_c)


def Trho_func_2_adb(logm200, c, z, rho_c):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init_adb(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil = -2.
    r_til    = 10**log_rtil
    rho_rel  = brentq(F_inv1(r_til, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                       1e-7, 1e4, rtol=1e-3) #n_Total
    
    rho_rel = rho_rel # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    
    return (rho_rel - rho_c)



def Trho_func_concentration(c, logm200, z):
    '''
    zero finding function that we solve to obtain concentration for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil_range = np.arange(-2,0.11,0.1)
    r_til_range    = np.power(10, log_rtil_range)
    rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                       1e-7, 1e4, rtol=1e-3) for rtil in r_til_range]) #n_Total
    
    rho_rel_range = rho_rel_range *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range, bounds_error=True)
    
    dx            = 0.01
    x_range       = np.arange(-2,0,dx)
    y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func(x_range) * np.log(10)
    mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
    
    #mgas          = quad(mgas_integrand_logr, -2, 0, args=(rho_rel_func))[0] * r200.to(u.kpc).value**3
    #print(mgas/m200.to(u.kg).value)
    
    return (omega_b/omega_m - mgas/m200.to(u.kg).value)



def Trho_func_concentration_central(rho_norm, c, logm200, z):
    '''
    zero finding function that we solve to obtain concentration for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil = -2
    r_til    = 10**-2.0
    try:
        rho_rel  = brentq(F_inv1(r_til, t200.value, rho_norm*rho_bar_b, T_rho, c, g, np.infty), 
                                        1e-7, 1e5, rtol=1e-3)#n_Total
    except Exception as E:
        print(E)
        rho_rel = np.infty
    
    rho_rel = rho_rel # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)    
    
    return rho_rel





def Trho_func_concentration_central_adb(c, logm200, z):
    '''
    zero finding function that we solve to obtain concentration for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init_adb(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil = -2
    r_til    = 10**-2.0
    try:
        rho_rel  = brentq(F_inv1(r_til, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                        1e-7, 1e5, rtol=1e-3)#n_Total
    except Exception as E:
        print(E)
        rho_rel = np.infty
    
    rho_rel = rho_rel # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)    
    
    return rho_rel


def Trho_func_ludlow(logm200, z):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil_range = np.arange(-2,0.11,0.05)
    r_til_range    = np.power(10, log_rtil_range)
    rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                       1e-7, 1e4, rtol=1e-3) for rtil in r_til_range]) #n_Total
    
    rho_rel_range = rho_rel_range *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range, bounds_error=True)
    
    dx            = 0.001
    x_range       = np.arange(-2,0+dx/2,dx)
    y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func(x_range) * np.log(10)
    mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
    
    #mgas          = quad(mgas_integrand_logr, -2, 0, args=(rho_rel_func))[0] * r200.to(u.kpc).value**3
    #print(mgas/m200.to(u.kg).value)
    
    return (omega_b/omega_m - mgas/m200.to(u.kg).value)


def Trho_adb_func_ludlow(logm200, z):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init_adb(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil_range = np.arange(-2,0.11,0.05)
    r_til_range    = np.power(10, log_rtil_range)
    rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                       1e-7, 1e4, rtol=1e-3) for rtil in r_til_range]) #n_Total
    
    rho_rel_range = rho_rel_range *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range, bounds_error=True)
    
    dx            = 0.001
    x_range       = np.arange(-2,0+dx/2,dx)
    y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func(x_range) * np.log(10)
    mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
    
    #mgas          = quad(mgas_integrand_logr, -2, 0, args=(rho_rel_func))[0] * r200.to(u.kpc).value**3
    #print(mgas/m200.to(u.kg).value)
    
    return (omega_b/omega_m - mgas/m200.to(u.kg).value)



def Trho_func_ludlow_2(logm200, z, rho_c):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, c)
    
    log_rtil = -2.
    r_til    = 10**log_rtil
    rho_rel  = brentq(F_inv1(r_til, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                       1e-7, 1e4, rtol=1e-3) #n_Total
    
    #rho_rel = rho_rel # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    
    return (rho_rel - rho_c)


def Trho_adb_func_ludlow_2(logm200, z, rho_c):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value 
    
    T_rho = T_rho_init_adb(z)
    
    #print(z, logm200, c)
    
    fudge_factor = 1
    
    log_rtil = -2.
    r_til    = 10**log_rtil
    rho_rel  = brentq(F_inv1(r_til, t200.value, rho_bar_b * fudge_factor, T_rho, c, g, np.infty), 
                                       1e-7, 1e4, rtol=1e-3) #n_Total
    
    #print( brentq(F_inv1(1, t200.value, rho_bar_b*fudge_factor, T_rho, c, g, np.infty), 
                                       #1e-7, 1e4, rtol=1e-3)/rho_bar_b )
    
    #rho_rel = rho_rel # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    
    return (rho_rel - rho_c)



def Trho_adb_func_ludlow_2_norm_gas(logm200, z, rho_c):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''

    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init_adb(z)  #lambda x: np.zeros(len(x)) + 10**4 #
    
    print(z, logm200)
    
    def finding_gas_mass(rho_norm_til):
        
        rho_norm = rho_norm_til * rho_bar_b 
        
        log_rtil_range = np.arange(-2,0.0,0.01)
        r_til_range    = np.power(10, log_rtil_range)
        
        rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_norm, T_rho, c, g, 1), 
                                       1e-7, 1e4, rtol=1e-3) for rtil in r_til_range]) #n_Total
        
        rho_rel_range = rho_rel_range * const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
               
        rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range , bounds_error=True)
        
        dx            = 0.01
        x_range       = np.arange(-2,0,dx)
        y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func (x_range) * np.log(10)
        mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
        
        #print('mgas/m200:', mgas/m200.to(u.kg).value)
        
        return (mgas/m200.to(u.kg).value - omega_b/omega_m)
    
    
    rho_norm_til2 =  brentq(finding_gas_mass, 1e-3, 1e3, rtol=1e-3)
    
    print('rho_norm_til2', rho_norm_til2)
    
    log_rtil = -2.
    r_til    = 10**log_rtil
    rho_rel  = brentq(F_inv1(r_til, t200.value, rho_norm_til2* rho_bar_b, T_rho, c, g, 1), 
                                       1e-7, 1e4, rtol=1e-3) #n_Total
    
    #rho_rel = rho_rel # *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    print(rho_rel)
    return (rho_rel - rho_c)



def Trho_func_ludlow_central(rho_norm, logm200, z):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    uses ludlow concentration,
    require a normalizing density at infinity as input
    outputs central density at 1% of virial radius
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil = -2
    r_til    = 10**-2.0
    try:
        if z <= 11.5:
        
            rho_rel  = brentq(F_inv1(r_til, t200.value, rho_norm*rho_bar_b, T_rho, c, g, np.infty), 
                                        1e-7, 1e4, rtol=1e-3) #n_Total
            
        else:
            
            def finding_gas_mass(rho_norm_til):
                
                rho_norm = rho_norm_til * rho_bar_b 
                
                log_rtil_range = np.arange(-2,0.0,0.01)
                r_til_range    = np.power(10, log_rtil_range)
                
                rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_norm, T_rho, c, g, 1), 
                                            1e-7, 1e4, rtol=1e-3) for rtil in r_til_range]) #n_Total
                
                rho_rel_range = rho_rel_range * const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
                    
                rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range , bounds_error=True)
                
                dx            = 0.01
                x_range       = np.arange(-2,0,dx)
                y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func (x_range) * np.log(10)
                mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
                
                #print('mgas/m200:', mgas/m200.to(u.kg).value)
                
                return (mgas/m200.to(u.kg).value - omega_b/omega_m)
            
            
            rho_norm_til2 =  brentq(finding_gas_mass, 1e-3, 1e3, rtol=1e-3)
            
            print('rho_norm_til2', rho_norm_til2)
            
            log_rtil = -2.
            r_til    = 10**log_rtil
            rho_rel  = brentq(F_inv1(r_til, t200.value, rho_norm_til2* rho_bar_b, T_rho, c, g, 1), 
                                            1e-7, 1e4, rtol=1e-3) #n_Total
            
    except Exception as E:
        print(E)
        rho_rel =  np.infty
    
    rho_rel = rho_rel# *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    return rho_rel



def Trho_func_ludlow_central_adb(logm200, z):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    uses ludlow concentration,
    assumes gas fraction is universal baryon fraction, so it doesn't require a normalizing density as input
    outputs central density at 1% of virial radius

    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init_adb(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil = -2
    r_til    = 10**-2.0
    try:
        if z <= 11.5:
        
            rho_rel  = brentq(F_inv1(r_til, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                        1e-7, 1e4, rtol=1e-3) #n_Total
            
        else:
            
            
            def finding_gas_mass(rho_norm_til):
                
                rho_norm = rho_norm_til * rho_bar_b 
                
                log_rtil_range = np.arange(-2,0.0,0.01)
                r_til_range    = np.power(10, log_rtil_range)
                
                rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_norm, T_rho, c, g, 1), 
                                            1e-7, 1e4, rtol=1e-3) for rtil in r_til_range]) #n_Total
                
                rho_rel_range = rho_rel_range * const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
                    
                rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range , bounds_error=True)
                
                dx            = 0.01
                x_range       = np.arange(-2,0,dx)
                y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func (x_range) * np.log(10)
                mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
                
                #print('mgas/m200:', mgas/m200.to(u.kg).value)
                
                return (mgas/m200.to(u.kg).value - omega_b/omega_m)
            
            
            rho_norm_til2 =  brentq(finding_gas_mass, 1e-3, 1e3, rtol=1e-3)
            
            print('rho_norm_til2', rho_norm_til2)
            
            log_rtil = -2.
            r_til    = 10**log_rtil
            rho_rel  = brentq(F_inv1(r_til, t200.value, rho_norm_til2* rho_bar_b, T_rho, c, g, 1), 
                                            1e-7, 1e4, rtol=1e-3) #n_Total
            
    except Exception as E:
        print(E)
        rho_rel =  np.infty
    
    rho_rel = rho_rel# *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
    
    return rho_rel


def mgas_ludlow(logm200, z):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
    
    logm200_range, c_nfw_ludlow = ludlow_concentration(z)
    c_nfw_ludlow_interp         = interp1d(x=logm200_range, y=c_nfw_ludlow, bounds_error=True, kind='linear')
    
    c = c_nfw_ludlow_interp(logm200)
    
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil_range = np.arange(-2,0.11,0.05)
    r_til_range    = np.power(10, log_rtil_range)
    
    fudge_factor = 1
    
    try:
        rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_bar_b*fudge_factor, T_rho, c, g, np.infty), 
                                        1e-7, 1e6, rtol=1e-3) for rtil in r_til_range]) #n_Total
    
        rho_rel_range = rho_rel_range *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
        
        rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range, bounds_error=True)
        
        dx            = 0.001
        x_range       = np.arange(-2,0+dx/2,dx)
        y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func(x_range) * np.log(10)
        mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
        
        #mgas          = quad(mgas_integrand_logr, -2, 0, args=(rho_rel_func))[0] * r200.to(u.kpc).value**3
        #print(mgas/m200.to(u.kg).value)
        
        return np.log10(mgas/const.M_sun.to(u.kg).value)
    
    except Exception as E:
        print(E)
        print('failed at: ', logm200, c)
        
        return np.infty
    
def mgas_c(logm200, z, c):
    '''
    zero finding function that we solve to obtain critical mass for the T-rho EOS model
    
    expect very long run time for this....
    '''
    m200   = np.power(10,logm200) * const.M_sun
    rho200 = RHO200(z)
        
    r200   = R200(m200, rho200)
    v200   = V200(m200,r200)
    t200   = T200(v200).to(u.K)
    
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    T_rho = T_rho_init(z)
    
    #print(z, logm200, omega_b/omega_m)
    
    log_rtil_range = np.arange(-2,0.11,0.05)
    r_til_range    = np.power(10, log_rtil_range)
    try:
        rho_rel_range  = np.asarray([brentq(F_inv1(rtil, t200.value, rho_bar_b, T_rho, c, g, np.infty), 
                                        1e-7, 1e6, rtol=1e-3) for rtil in r_til_range]) #n_Total
    
        rho_rel_range = rho_rel_range *const.m_p.to(u.kg).value * (u.cm**-3).to(u.kpc**-3)
        
        rho_rel_func  = interp1d(x=log_rtil_range, y=rho_rel_range, bounds_error=True)
        
        dx            = 0.001
        x_range       = np.arange(-2,0+dx/2,dx)
        y_vals        = 4 * np.pi * np.power(10, 3*x_range) * rho_rel_func(x_range) * np.log(10)
        mgas          = np.trapz(x=x_range, y=y_vals, dx=dx)* r200.to(u.kpc).value**3
        
        #mgas          = quad(mgas_integrand_logr, -2, 0, args=(rho_rel_func))[0] * r200.to(u.kpc).value**3
        #print(mgas/m200.to(u.kg).value)
        
        return np.log10(mgas/const.M_sun.to(u.kg).value)
    
    except Exception as E:
        print(E)
        print('failed at: ', logm200, c)
        
        return np.infty

# generic equations

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


#Analytic solution for isothermal model

def rho_rel_iso(r_til, c, tb, t200):## RELHIC gas density for isothermal gas
    f_c   = np.log(1+c) - c/(1+c)
    return np.exp( 2*(t200/tb) * np.log(1+c*r_til) / (f_c*r_til) )

def rho_rel_iso_norm_r200(r_til, c, tb, t200, rho_norm):## RELHIC gas density for isothermal gas
    f_c   = np.log(1+c) - c/(1+c)
    return rho_norm * np.exp( 2/f_c*(t200/tb) *( np.log(1+c*r_til) / (r_til) - np.log(1+c*1) / (1)) )


# T-rho equilibrium model

def g(r,c):  ## M(r)/r2 (normalized M and r)
    '''
    integrand of G
    '''
    return (1/r/r) * (np.log(1+c*r) - c*r/(1+c*r)) / (np.log(1+c) - c/(1+c))

def G(r_til, t200,c, g_func, r_norm):
    
    #print(r_til, r_norm)
    
    #dx      = abs(r_til - r_norm)/1e6
    #x_range = np.arange(r_til, r_norm, dx) 
    #y_vals  = g_func(x_range, c)
    
    #int_    = np.trapz(x=x_range, y=y_vals, dx=dx)
    
    return 2*t200*(quad(g_func, r_til, r_norm, args=(c))[0])

#def f(rho, T_rho):
    #'''
    #integrand of F
    #'''
    #dx      = 1e-8
    #dT_drho = (T_rho(rho+dx) - T_rho(rho-dx))/(2*dx)
    #print(rho.shape, T_rho(rho).shape)
    #return (T_rho(rho)/rho + dT_drho)

#def F(rho, rho_norm, T_rho):
    
    #print(rho_norm, rho)
    
    #dx      = abs(rho_norm - rho)
    #x_range = np.arange(rho_norm, rho, dx) 
    #y_vals  = f(x_range, T_rho)
    #int_    = np.trapz(x=x_range, y=y_vals, dx=dx)
    
    #return int_ #quad(f, rho_norm, rho, args=(T_rho), limit=300)[0]


#def f_log(logrho, T_rho): ORIGINAL (no change to dT/drho)
    
    #dx      = 1e-9
    #rho     = np.power(10, logrho)
    #dT_drho = (T_rho(rho+dx) - T_rho(rho-dx))/(2*dx)
    
    #return (T_rho(rho).flatten()*np.power(10, -logrho) + dT_drho.flatten())  * np.power(10, logrho) * np.log(10)


def f_log(logrho, T_rho):
    
    dx      = 1e-9
    rho     = np.power(10, logrho)
    dT_drho = (T_rho(rho+dx) - T_rho(rho-dx))/(2*dx)
    
    return (T_rho(rho).flatten()*np.power(10, -logrho) + dT_drho.flatten())  * np.power(10, logrho) * np.log(10)



def F(rho, rho_norm, T_rho):
    
    #print(rho_norm, rho)
    dx      = abs(np.log10(rho_norm) - np.log10(rho))/100
    x_range = np.arange(np.log10(rho_norm), np.log10(rho)+dx/2, dx) 
    y_vals  = f_log(x_range, T_rho)
    int_    = np.trapz(x=x_range, y=y_vals, dx=dx)
    
    return int_ #quad(f, rho_norm, rho, args=(T_rho), limit=300)[0]

def F_inv1(r_til, t200, rho_norm, T_rho, c, g_func, r_norm):
    def F_inv2(rho):
        #print(rho, F(rho, rho_norm, T_rho), G(r_til, t200, c, g_func, r_norm))
        return F(rho, rho_norm, T_rho) - G(r_til, t200, c, g_func, r_norm)
    return F_inv2




# T-rho EOS

T_rho_BL2020  = np.genfromtxt('./A1_table_BL_2020.txt', delimiter=',')

z_BL2020      = T_rho_BL2020[1:,0]
log_nH_BL2020 = T_rho_BL2020[0,1:]
log_T_BL2020  = T_rho_BL2020[1:,1:]

t_from_z       = cosmo.lookback_time(z_BL2020).value
T_rho_interp_2 = interp2d(log_nH_BL2020, t_from_z, log_T_BL2020, kind='cubic', bounds_error=False)



def Trho_temp(nT):
    nH=np.asarray(nT) * 0.75
    out=np.zeros(len(nH))
    
    out[nH<=0.15] = 10**4.0
    out[nH> 0.15] = 10**4.0 * (nH[nH> 0.15]/0.15) ** 0.15
    
    return out

# def T_rho_init(z):  #n_Total
    
#     def T_rho2(rho):
#         rho = np.asarray(rho) * 0.75  ##input is expected to be total density, 
#                                       ##but function takes only fractional hydrogen density

#         if rho.shape:
#             return 10**np.array([T_rho_interp_2(np.log10(rrho), cosmo.lookback_time(z).value) for rrho in rho])
#         else:
#             return 10**T_rho_interp_2(np.log10(rho), cosmo.lookback_time(z).value)
    
#     return T_rho2

def T_rho_init_adb(z):  #n_Total
    
    mu = 0.6
    cs_eos  = 9.4 * u.km/u.s
    rho_eos = 0.1 * const.m_p / u.cm**3
    
    kappa = 2/3 * (mu**(4/3)) *const.m_p / ((0.1*u.cm**(-3))**(1/3) * const.k_B) * (9.4 *u.km/u.s)**2

    T_adiab   = lambda nH: kappa.to(u.cm*u.K).value * (nH/0.75) **(1/3)
    
    if z<= 11.5:
        def T_rho2(nT):
            nH = np.asarray(nT) * 0.75  ### converts here from N_TOTAL to N_HYDROGEN (factor of 3/4)

            if nH.shape: #checks that it's an array (else it's a float)
                arr_1 = 10**np.array([T_rho_interp_2(np.log10(nnH), cosmo.lookback_time(z).value) for nnH in nH]).flatten()
                arr_2 = T_adiab(nH) 
                arr_final = np.zeros(arr_1.shape)

                arr_final[arr_1 >= arr_2] = arr_1[arr_1 >= arr_2]
                arr_final[arr_1 < arr_2]  = arr_2[arr_1 < arr_2]
                return arr_final
            else:
                val_1 = 10**T_rho_interp_2(np.log10(nH), cosmo.lookback_time(z).value)
                val_2 = T_adiab(nH)
                if val_1 >= val_2:
                    return val_1
                else:
                    return val_2
        return T_rho2

    else:
        return Trho_temp


def T_rho_init(z):  #n_Total
    
    if z <= 11.5:
    
        def T_rho2(nT):
            nH = np.asarray(nT) * 0.75  ### converts here from N_TOTAL to N_HYDROGEN (factor of 3/4)
            
            if nH.shape: #checks that it's an array (else it's a float)
                arr_1 = 10**np.array([T_rho_interp_2(np.log10(nnH), cosmo.lookback_time(z).value) for nnH in nH]).flatten()

                return arr_1
            else:
                val_1 = 10**T_rho_interp_2(np.log10(nH), cosmo.lookback_time(z).value)
                
                return val_1
    
    else:
        
        def T_rho2(nT):
            nH = np.asarray(nT) * 0.75
            
            if nH.shape:
                arr_1 = np.zeros(len(nH)) + np.power(10, 4.0)
                return arr_1
            else:
                val_1 = np.power(10, 4.0)
                return val_1
    
    return T_rho2





#### ludlow concentration


def ludlow_concentration(redshift):
    
    Database      = params.Database
    name          = params.name
    #redshift      = params.redshift
    MassMin       = params.MassMin
    MassMax       = params.MassMax
    numGridPoints = params.numGridPoints
    N_int         = params.N_int

    #warnings.filterwarnings("ignore", category = RuntimeWarning, append = 1)

    ###############################
    #   LOAD THE POWER SPECTRA    #
    ###############################

    Database = "./PowerSpectra/"
    if (name == "WMAP-1"):
        PS_name = Database + "WMAP1_camb_matterpower_z0_extrapolated.dat"
    elif (name == "WMAP-3"):
        PS_name = Database + "WMAP3_camb_matterpower_z0_extrapolated.dat"
    elif (name == "WMAP-5"):
        PS_name = Database + "WMAP5_camb_matterpower_z0_extrapolated.dat"
    elif (name == "WMAP-7"):
        PS_name = Database + "WMAP7_camb_matterpower_z0_extrapolated.dat"
    elif (name == "WMAP-9"):
        PS_name = Database + "WMAP9_camb_matterpower_z0_extrapolated.dat"
    elif (name == "Planck"):
        PS_name = Database + "Planck_camb_matterpower_z0_extrapolated.dat"
    elif (name == "COCO"):
        PS_name = Database + "COCO_camb_matterpower_z0_extrapolated.dat"
    elif (name == "Millennium"):
        PS_name = Database + "Millennium_camb_matterpower_z0_extrapolated.dat"

    ###############################
    #   READ THE POWER SPECTRA    #
    ###############################

    # Cosmology info in the first 5 lines

    info_file    = open(PS_name, "r")
    cosmo        = info_file.readlines()

    OmegaBar     = float(cosmo[0])
    OmegaMatter  = float(cosmo[1])
    hubble       = float(cosmo[2])
    n_spec       = float(cosmo[3])
    sigma_8      = float(cosmo[4])

    info_file.close()

    # Calculate the critical density at this redshift
    Rhocrit_z  = cf.Rhocrit_z(OmegaMatter, 1.-OmegaMatter, 0.) # units: 1e10 M_solar/Mpc^3/h^2
    Rhocrit_z *= 1e10 # units: M_solar/Mpc^3/h^2

    # Print cosmology
    #print("You have selected the ", name, " power spectrum")
    #print("OmegaB = %f, OmegaM = %f, OmegaL = %f"%(OmegaBar, OmegaMatter, 1.-OmegaMatter) )
    #print("h [100 km/s/Mpc] = %f"%(hubble) )
    #print("Spectral Index = %f"%(n_spec) )
    #print("Sigma (8 Mpc/h) = %f"%(sigma_8) )
    #print("z = %f"%(redshift) )

    # Read in P(k) and k
    PowerSpectrum    = np.genfromtxt(PS_name, skip_header = 5)
    Pk_file, k_file  = PowerSpectrum[:,0], PowerSpectrum[:,1]


    ###############################
    #     CALCULATE SIGMA (M)     #
    ###############################

    # Interpolate the power spectrum 
    Pk_interp   = interp1d(k_file, Pk_file)

    # Omega_m at this redshift
    #Omz         = cf.Omz(OmegaMatter, 1.-OmegaMatter, redshift)
    Omz          = OmegaMatter

    # Mean matter density at this redshift
    Rhomean_z   = Rhocrit_z * Omz # M_solar/Mpc/h^2

    dlogm       = (np.log10(MassMax) - np.log10(MassMin)) / (numGridPoints-1)
    logM0       = np.log10(MassMin) + np.arange(numGridPoints)*dlogm + 0.5*dlogm

    filter_Mass  = 10**logM0
    R            = cbrt(filter_Mass / (4/3 * np.pi * Rhomean_z)) # Mpc/h

    k_min    = k_file.min() * 1.10
    k_max    = k_file.max() * 0.90

    # Tophat
    def TopHat(k, r):
        return 3.0/(k*r)**2 * (np.sin(k*r)/(k*r) - np.cos(k*r))

    def SigmaIntegrand(k, r):
        return k**2 * Pk_interp(k) * TopHat(k,r)**2

    # Integration function
    def integratePk(kmin, kmax, r):

        log_k_min   = np.log10(kmin) 
        log_k_max   = np.log10(kmax) 

        # Size of intervals    
        dlogk       = (log_k_max - log_k_min)/(N_int - 1)
        tot_sum     = 0. 
        for ii in range(N_int):
            logk     = log_k_min + dlogk*ii
            sum_rect = SigmaIntegrand(10**logk, r) * 10**logk * np.log(10)
            tot_sum  = tot_sum + sum_rect
        # Add contributions from the ends of the integration interval
        log_k_min    = log_k_min - dlogk
        sum_rect_min = SigmaIntegrand(10**log_k_min, r) * 10**log_k_min * np.log(10)
        log_k_max    = log_k_max + dlogk
        sum_rect_max = SigmaIntegrand(10**log_k_max, r) * 10**log_k_max * np.log(10)
        
        sigma_sq     = (tot_sum + 0.5*sum_rect_min + 0.5*sum_rect_max) * dlogk
        sigma_sq    /= (2*np.pi**2)
        return sigma_sq

    Sigma_Sq     = np.zeros(len(filter_Mass))

    # Sigma^2(M)
    Sigma_Sq     = integratePk(k_min, k_max, R)
    # Sigma (M)
    Sigma        = np.sqrt(Sigma_Sq)

    # Normalize rms
    sig_interp   = interp1d(filter_Mass, Sigma)
    MassIn8Mpc   = 4/3 * np.pi * 8**3 * Rhomean_z
    sig_8        = sig_interp(MassIn8Mpc)
    normalise    = sig_8 / sigma_8
    Sigma       /= normalise

    # dlog Sigma^2(M) / dlog M
    Sigma_Sq     = Sigma**2
    logSigma_Sq  = np.log10(Sigma_Sq)
    logMass      = np.log10(filter_Mass)
    derivSigma   = np.diff(logSigma_Sq) / np.diff(logMass)

    ###############################
    #  CALCULATE c(M) RELATION    #
    ###############################
    # free model parameters (Ludlow et al, 2016)
    A           = 650. / 200
    f           = 0.02
    delta_sc    = 1.686
    # shape parameter for Einasto concentrations
    alpha_rho   = 0.18

    #c_array     = 10**(np.arange(0,50-2.7*(redshift+1)) * 4./99.)

    c_array     = 10**(np.arange(0,100) * 4./99.)
    delta_sc_0  = delta_sc / cf.linear_growth_factor(OmegaMatter, 1.-OmegaMatter, redshift)
    c_ein       = np.zeros(len(filter_Mass))
    c_nfw       = np.zeros(len(filter_Mass))

    OmegaL      = 1.-OmegaMatter
    sig2_interp = splrep(logM0-10., Sigma_Sq, k=1)

    for jj in range(numGridPoints):
        # Einasto
        M2          = gammainc(3.0 / alpha_rho, 2.0 / alpha_rho) / gammainc(3.0 / alpha_rho, 2.0 * c_array**alpha_rho / alpha_rho)
        rho_2       = 200. * c_array**3 * M2
        rhoc        = rho_2 / (200. * A)
        z2          = (1. / OmegaMatter *(rhoc* (OmegaMatter*(1+redshift)**3 + OmegaL) - OmegaL))**0.3333 - 1.
        delta_sc_z2 = delta_sc / cf.linear_growth_factor(OmegaMatter, OmegaL, z2)

        sig2fM      = splev(logM0[jj] -10. + np.log10(f), sig2_interp)
        sig2M       = Sigma_Sq[jj]
        sig2Min     = splev(np.log10(M2), sig2_interp)

        arg         = A*rhoc/c_array**3 - (1.-erf( (delta_sc_z2-delta_sc_0) / np.sqrt(2.*(sig2fM-sig2M)) ))
        mask        = np.isinf(arg) | np.isnan(arg)
        arg         = arg[mask == False]
        c_array     = c_array[mask==False]
        conc_interp = interp1d(arg, c_array)
        c_ein[jj]   = conc_interp(0.)
        
        
        # NFW
        M2          = (np.log(2.)-0.5) / (np.log(1.+c_array)-c_array/(1.+c_array))
        rho_2       = 200. * c_array**3 * M2
        rhoc        = rho_2 / (200. * A)
        z2          = (1. / OmegaMatter * (rhoc * (OmegaMatter*(1+redshift)**3 + OmegaL) - OmegaL))**0.33333 - 1.
        delta_sc_z2 = delta_sc / cf.linear_growth_factor(OmegaMatter, OmegaL, z2)
        
        arg         = A*rhoc/(c_array**3) - (1.-erf((delta_sc_z2-delta_sc_0) / (np.sqrt(2.*(sig2fM-sig2M)))))
        
        #print(np.shape(rhoc))
        #print(np.shape(c_array))
        #print(np.shape(delta_sc_z2))
        
        mask        = np.isnan(arg) | np.isinf(arg)
        arg         = arg[mask==False]
        c_array     = c_array[mask==False]
        
        try:
            indx = np.min(np.argwhere(c_array >= 50/(1+redshift/2)))
        except:
            indx = -1
            
        conc_interp = interp1d(arg[0:indx],c_array[0:indx])
        c_nfw[jj]   = conc_interp(0.)
    
    return np.log10(filter_Mass), c_nfw
