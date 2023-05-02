from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from scipy.optimize import brentq

import numpy as np

from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from astropy import constants as const

from ludlow_concentration import *
from cosmology_functions import *


def isothermal_mgas_ratio(logm200, c, z, T_b):
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


def isothermal_central_density(logm200, c, z, rho_c, T_b):
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
    return (rho_iso - rho_c)


# def iso_func_find_T200(logm200_mcrit_lud, z):
    
#     def find_minimiz(logtb):
#         tb = np.power(10, logtb)
#         in_ = [z, tb*u.K]
#         return logm200_mcrit_lud - find_mcrit_isothermal_baryon_frac_ludlow_concentration(in_)
    
#     out = brentq(find_minimiz, 3.5, 4.5)
    
#     return out
    
    

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
    
    T_rho = Trho_init(z)
    
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
    
    T_rho = Trho_init(z)
    
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
    
    T_rho = Trho_EOS(z)
    
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
    
    T_rho = Trho_init(z)
    
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



def Trho_func_concentration_central(c, logm200, z):
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
    
    T_rho = Trho_init(z)
    
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
    
    T_rho = Trho_EOS(z)
    
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
    
    T_rho = Trho_init(z)
    
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
    
    T_rho = Trho_EOS(z)
    
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
    
    T_rho = Trho_init(z)
    
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
    
    T_rho = Trho_EOS(z)
    
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
    
    T_rho = Trho_EOS(z)  #lambda x: np.zeros(len(x)) + 10**4 #
    
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



def Trho_func_ludlow_central(logm200, z):
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
    
    T_rho = Trho_init(z)
    
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
    
    T_rho = Trho_EOS(z)
    
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
    
    T_rho = Trho_init(z)
    
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
    
    T_rho = Trho_init(z)
    
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

T_rho_BL2020  = np.genfromtxt('./data/Trho_BL_2020.txt', delimiter=',')

z_BL2020      = T_rho_BL2020[1:,0]
log_nH_BL2020 = T_rho_BL2020[0,1:]
log_T_BL2020  = T_rho_BL2020[1:,1:]

t_from_z       = cosmo.lookback_time(z_BL2020).value
T_rho_interp_2 = interp2d(log_nH_BL2020, t_from_z, log_T_BL2020, kind='cubic', bounds_error=False)

def Trho_ISO_EOS(nT):
    '''
    intermediate function that defines the Temperature-Density relation as 
    Isothermal at 10^4 K except at high density (nH>0.15), where the forced EOS takes over.
    '''
    nH=np.asarray(nT) * 0.75
    out=np.zeros(len(nH))
    
    out[nH<=0.15] = 10**4.0
    out[nH> 0.15] = 10**4.0 * (nH[nH> 0.15]/0.15) ** 0.15
    
    return out

def Trho_EOS(z):  #n_Total
    '''
    main function that defines the Temperature-density relation (including the EOS effect enforced in APOSTLE).
    it returns either of two functions:
    Trho2 if z<=11.5, which corresponds to an interpolation function of the datatable from Benitez-Llambay 2020 + polytropic EOS effects.
    Trho_ISO_EOS, Isothermal at 10^4 K except at high density (nH>0.15), where the forced EOS takes over.
    '''
    mu = 0.6
    cs_eos  = 9.4 * u.km/u.s
    rho_eos = 0.1 * const.m_p / u.cm**3
    
    kappa = 2/3 * (mu**(4/3)) *const.m_p / ((0.1*u.cm**(-3))**(1/3) * const.k_B) * (9.4 *u.km/u.s)**2

    T_EOS   = lambda nH: kappa.to(u.cm*u.K).value * (nH/0.75) **(1/3)
    
    if z<= 11.5:
        def Trho2(nT):
            nH = np.asarray(nT) * 0.75  ### converts here from N_TOTAL to N_HYDROGEN (factor of 3/4)

            if nH.shape: #checks that it's an array (else it's a float)
                arr_1 = 10**np.array([T_rho_interp_2(np.log10(nnH), cosmo.lookback_time(z).value) for nnH in nH]).flatten()
                arr_2 = T_EOS(nH) 
                arr_final = np.zeros(arr_1.shape)

                arr_final[arr_1 >= arr_2] = arr_1[arr_1 >= arr_2]
                arr_final[arr_1 < arr_2]  = arr_2[arr_1 < arr_2]
                return arr_final
            else:
                val_1 = 10**T_rho_interp_2(np.log10(nH), cosmo.lookback_time(z).value)
                val_2 = T_EOS(nH)
                if val_1 >= val_2:
                    return val_1
                else:
                    return val_2
        return Trho2

    else:
        return Trho_ISO_EOS


def Trho_init(z):  #n_Total
    '''
    Input (redshift) -> Output Function (nT [density])

    main function that defines the Temperature-density relation. Similar to Trho_EOS, except it does not include the 
    effect of the enforced EOS that is used in APOSTLE. Instead, the Temperature-density relation is assumed to remain isothermal
    beyond the data included in the datatable of Benitez-Llambay et al. 2020.

    it returns either of two functions:
    Trho2 if z<=11.5, which corresponds to an interpolation function of the datatable from Benitez-Llambay 2020.
    Trho_ISO_EOS, Isothermal at 10^4 K.
    '''
    if z <= 11.5:
    
        def Trho2(nT):
            nH = np.asarray(nT) * 0.75  ### converts here from N_TOTAL to N_HYDROGEN (factor of 3/4)
            
            if nH.shape: #checks that it's an array (else it's a float)
                arr_1 = 10**np.array([T_rho_interp_2(np.log10(nnH), cosmo.lookback_time(z).value) for nnH in nH]).flatten()

                return arr_1
            else:
                val_1 = 10**T_rho_interp_2(np.log10(nH), cosmo.lookback_time(z).value)
                
                return val_1
    
    else:
        
        def Trho2(nT):
            nH = np.asarray(nT) * 0.75
            
            if nH.shape:
                arr_1 = np.zeros(len(nH)) + np.power(10, 4.0)
                return arr_1
            else:
                val_1 = np.power(10, 4.0)
                return val_1
    
    return Trho2




