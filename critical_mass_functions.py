from scipy.optimize import brentq
from hydrostatic_gas_functions import *

def find_mcrit_isothermal_mgas_ratio_fixed_concentration(in_):
    '''
    INPUT (redshift, concentration, temperature)

    OUTPUT critical mass 

    Critical mass definition:
    - enclosed gas mass within r200 is equal to the universal baryon fraction

    Assumptions:
    - fixed halo concentration (c)
    - temperature-density is constant (isothermal, T_b [K]) 
    - gas density at infinity is equal to mean baryon density
    '''
    z, c, T_b = in_[0], in_[1], in_[2]
    return brentq(isothermal_mgas_ratio, 7,10, args=(c,z, T_b),  rtol=1e-3)

def find_mcrit_isothermal_central_density_fixed_concentration(in_):
    '''
    INPUT (redshift, concentration, central density threshold, temperature)

    OUTPUT critical mass 

    Critical mass definition:
    - when the central density of gas (central means at r=0.01*r200) equals central density threshold

    Assumptions:
    - fixed halo concentration (c)
    - temperature-density is constant (isothermal, T_b [K]) 
    - gas density at infinity is equal to mean baryon density
    '''
    z, c, T_b, rho_c = in_[0], in_[1], in_[2], in_[3]
    return brentq(isothermal_central_density, 7,10, args=(c, z, rho_c, T_b),  rtol=1e-3)

def find_mcrit_isothermal_baryon_frac_ludlow_concentration(in_):
    '''
    INPUT (redshift, temperature)

    OUTPUT critical mass

    Critical mass definition:
    - enclosed gas mass within r200 is equal to the universal baryon fraction

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density is constant (isothermal, T_b [K]) 
    - gas density at infinity is equal to mean baryon density
    '''
    z, T_b = in_[0], in_[1]
    return brentq(iso_func_ludlow, 5,12, args=(z, T_b),  rtol=1e-3)

def find_mcrit_isothermal_central_density_ludlow_concentration(in_):
    '''
    INPUT (redshift, temperature, central density threshold)

    OUTPUT critical mass

    Critical mass definition:
    - when the central density of gas (central means at r=0.01*r200) equals central density threshold

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density is constant (isothermal, T_b [K]) 
    - gas density at infinity is equal to mean baryon density
    '''
    z, T_b, rho_c = in_[0], in_[1], in_[2]
    return brentq(iso_func_ludlow_2, 5,12, args=(z, T_b, rho_c),  rtol=1e-3)

def find_mcrit_isothermal_mgas_norm_ludlow_concentration(in_):
    '''
    INPUT (redshift, temperature, central density threshold)
    
    OUTPUT critical mass

    Critical mass definition:
    - when the central density of gas (central means at r=0.01*r200) equals central density threshold

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density is constant (isothermal, T_b [K]) 
    - gas density at r200 is set such that the enclosed gas mass is equal to the universal baryon fraction
    '''
    z, T_b, rho_c = in_[0], in_[1], in_[2]
    return brentq(iso_func_ludlow_2_mgas_norm, 6.5,8, args=(z, T_b, rho_c),  rtol=1e-3)

def find_mcrit_isothermal_central_density_ludlow_concentration_varying_temperature(in_): #t_b changes if t200 < 10,000 K
    '''
    INPUT (redshift, upper limit of m200 for zero finder, central density threshold)
    
    OUTPUT critical mass

    Critical mass definition:
    - when the central density of gas (central means at r=0.01*r200) equals central density threshold

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density is constant (isothermal, T_b [K] = T_200, when T200 < 10^4 K, else T_b = 10^4 K) 
    - gas density at infinity is equal to mean baryon density   
    '''
    z, m200_upper, rho_c = in_[0], in_[1], in_[2]
    return brentq(iso_func_ludlow_3, 6, m200_upper, args=(z, rho_c),  rtol=1e-3)


def find_Trho_mcrit(in_):
    '''
    INPUT (redshift, upper limit of m200 for zero finder, concentration)

    OUTPUT critical mass 

    Critical mass definition:
    - enclosed gas mass within r200 is equal to the universal baryon fraction

    Assumptions:
    - fixed halo concentration (c)
    - temperature-density relationg given by Trho_init(z) function
    - gas density at infinity is equal to mean baryon density
    '''
    z, m200_upper, c = in_[0], in_[1], in_[2]
    print("running: ", z, m200_upper)
    return brentq(Trho_func, 7, m200_upper, args=(c,z),  rtol=1e-3)

def find_Trho_mcrit_2(in_):
    '''
    INPUT (redshift, upper limit of m200 for zero finder, concentration, central density threshold)

    OUTPUT critical mass 

    Critical mass definition:
    - central gas density is equal to input value for central density threshold

    Assumptions:
    - fixed halo concentration (c)
    - temperature-density relationg given by Trho_init(z) function
    - gas density at infinity is equal to mean baryon density
    '''
    z, m200_upper, c, rho_c = in_[0], in_[1], in_[2], in_[3]
    print("running: ", z, m200_upper)
    return brentq(Trho_func_2, 7, m200_upper, args=(c,z, rho_c),  rtol=1e-3)

def find_Trho_mcrit_2_adb(in_):
    '''
    INPUT (redshift, upper limit of m200 for zero finder, concentration, central density threshold)

    OUTPUT critical mass 

    Critical mass definition:
    - central gas density is equal to input value for central density threshold

    Assumptions:
    - fixed halo concentration (c)
    - temperature-density relationg given by Trho_EOS(z) function
    - gas density at infinity is equal to mean baryon density
    '''
    z, m200_upper, c, rho_c = in_[0], in_[1], in_[2], in_[3]
    print("running: ", z, m200_upper)
    return brentq(Trho_func_2_adb, 7, m200_upper, args=(c,z, rho_c),  rtol=1e-3)



def find_Trho_conce(in_):
    '''
    PURPOSE: Recover the concentration of halo, assuming its m200 is a critical mass

    INPUT (redshift, upper limit of concentration, critical mass)

    OUTPUT concentration

    Critical mass definition:
    - enclosed gas mass within r200 is equal to the universal baryon fraction

    Assumptions:
    - fixed halo concentration (c)
    - temperature-density relationg given by Trho_init(z) function
    - gas density at infinity is equal to mean baryon density
    '''
    z, c_upper, logm200 = in_[0], in_[1], in_[2]
    print("running: ", z, c_upper)
    return brentq(Trho_func_concentration, 2.2, c_upper, args=(logm200,z),  rtol=1e-3)


def find_Trho_mcrit_lud(in_):
    '''
    INPUT (redshift, upper limit of concentration)

    OUTPUT critical mass

    Critical mass definition:
    - enclosed gas mass within r200 is equal to the universal baryon fraction

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density relationg given by Trho_init(z) function
    - gas density at infinity is equal to mean baryon density
    '''
    z, m200_upper = in_[0], in_[1]
    print("running: ", z, m200_upper)
    return brentq(Trho_func_ludlow, 7, m200_upper, args=(z),  rtol=1e-3)

def find_Trho_mcrit_lud_adb(in_):
    '''
    INPUT (redshift, upper limit of concentration)

    OUTPUT critical mass

    Critical mass definition:
    - enclosed gas mass within r200 is equal to the universal baryon fraction

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density relationg given by Trho_EOS(z) function
    - gas density at infinity is equal to mean baryon density
    '''
    z, m200_upper = in_[0], in_[1]
    print("running: ", z, m200_upper)
    return brentq(Trho_adb_func_ludlow, 7, m200_upper, args=(z),  rtol=1e-3)


def find_Trho_mcrit_lud_2(in_):
    '''
    INPUT (redshift, upper limit of concentration, central density threshold)

    OUTPUT critical mass

    Critical mass definition:
    - central gas density is equal to input value for central density threshold

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density relationg given by Trho_init(z) function
    - gas density at infinity is equal to mean baryon density
    '''
    z, m200_upper, rho_c = in_[0], in_[1], in_[2]
    print("running: ", z, m200_upper)
    return brentq(Trho_func_ludlow_2, 7, m200_upper, args=(z, rho_c),  rtol=1e-3)

def find_Trho_mcrit_lud_2_adb(in_):
    '''
    INPUT (redshift, upper limit of concentration, central density threshold)

    OUTPUT critical mass

    Critical mass definition:
    - central gas density is equal to input value for central density threshold

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density relationg given by Trho_EOS(z) function
    - gas density at infinity is equal to mean baryon density
    '''
    z, m200_upper, rho_c = in_[0], in_[1], in_[2]
    print("running: ", z, m200_upper)
    return brentq(Trho_adb_func_ludlow_2, 7.5, m200_upper, args=(z, rho_c),  rtol=1e-3)

def find_Trho_mcrit_lud_2_adb_norm_gas(in_):
    '''
    INPUT (redshift, upper limit of concentration, central density threshold)

    OUTPUT critical mass

    Critical mass definition:
    - central gas density is equal to input value for central density threshold

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density relationg given by Trho_EOS(z) function
    - gas density at r200 is set such that the encloser gas mass is equal to the universal baryon fraction
    '''
    z, m200_upper, rho_c = in_[0], in_[1], in_[2]
    print("running: ", z, m200_upper)
    return brentq(Trho_adb_func_ludlow_2_norm_gas, 7.5, m200_upper, args=(z, rho_c),  rtol=1e-3)


def find_central_density(in_):
    '''
    INPUT (redshift, m200)

    OUTPUT central gas density

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density relationg given by Trho_init(z) function
    - gas density at infinity equals the mean baryon density for z<11.5
    - gas density at r200 is set such that the enclosed gas mass is equal to the universal baryon fraction for z > 11.5
    '''
    z, logm200 = in_[0], in_[1]
    print("running: ", z, logm200)
    return Trho_func_ludlow_central(logm200, z)

def find_central_density_adb(in_):
    '''
    INPUT (redshift, m200)

    OUTPUT central gas density

    Assumptions:
    - halo concentration given by Ludlow+16 relation
    - temperature-density relationg given by Trho_EOS(z) function
    - gas density at infinity equals the mean baryon density for z<11.5
    - gas density at r200 is set such that the enclosed gas mass is equal to the universal baryon fraction for z > 11.5
    '''
    z, logm200 = in_[0], in_[1]
    print("running: ", z, logm200)
    return Trho_func_ludlow_central_adb(logm200, z)

def find_central_density_c(in_):
    '''
    INPUT (redshift, concentration, m200)

    OUTPUT central gas density

    Assumptions:
    - halo concentration is fixed to (c)
    - temperature-density relationg given by Trho_init(z) function
    - gas density at infinity equals the mean baryon density
    '''
    z, c, logm200 = in_[0], in_[1], in_[2]
    print("running: ", z, logm200)
    return Trho_func_concentration_central(c,logm200, z)

def find_central_density_c_adb(in_):
    '''
    INPUT (redshift, concentration, m200)

    OUTPUT central gas density

    Assumptions:
    - halo concentration is fixed to (c)
    - temperature-density relationg given by Trho_EOS(z) function
    - gas density at infinity equals the mean baryon density
    '''
    z, c, logm200 = in_[0], in_[1], in_[2]
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
