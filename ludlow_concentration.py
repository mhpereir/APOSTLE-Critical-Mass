from __future__ import division
from scipy.special import cbrt, gammainc, erf
from scipy.interpolate import interp1d, splrep, splev
import params

import numpy as np


#### ludlow concentration
'''
This script is from Aaron Ludlow. A script to calculate the Concentration-Mass-Redshift relation 
to higher redshifts than was possible using the equations provided in Ludlow+2016 (https://doi.org/10.1093/mnras/stw1046).
'''


def ludlow_concentration(redshift):
    '''
    Modified version (by Matthew Pereira Wilson) of a function provided by Aaron Ludlow to 
    calculate the concentration-mass-redshift relation. 
    '''
    
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
    Rhocrit_z  = Rhocrit_z(OmegaMatter, 1.-OmegaMatter, 0.) # units: 1e10 M_solar/Mpc^3/h^2
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
    #Omz         = Omz(OmegaMatter, 1.-OmegaMatter, redshift)
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
    delta_sc_0  = delta_sc / linear_growth_factor(OmegaMatter, 1.-OmegaMatter, redshift)
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
        delta_sc_z2 = delta_sc / linear_growth_factor(OmegaMatter, OmegaL, z2)

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
        delta_sc_z2 = delta_sc / linear_growth_factor(OmegaMatter, OmegaL, z2)
        
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





# Constants

G = 43.020 # Mpc/(1e10 M_solar) km^2 / s^2

# Evolution of the Hubble constant

def E(Omega_m0, Omega_l0, z):
    return np.sqrt( Omega_l0 + Omega_m0 * (1+z)**3 )

# Evolution of Omega_m

def Omz(Omega_m0, Omega_l0, z):
    return Omega_m0 * (1+z)**3 / (E(Omega_m0, Omega_l0, z)**2)

# Evolution of Omega_l

def Olz(Omega_m0, Omega_l0, z):
    return Omega_l0 / (E(Omega_m0, Omega_l0, z)**2)

# Evolution of Rho_critical

def Rhocrit_z(Omega_m0, Omega_l0, z):
    Rhocrit_0 = 3.0/(8.0 * np.pi * G) * 1e4 
    return Rhocrit_0 * (E(Omega_m0, Omega_l0, z)**2)

# Real-space Tophat in Fourier space 

def TopHat(k, R):

    # k: wavenumber
    # R: filter size in Mpc/h

    return 3.0/(k*R)**2 * (np.sin(k*R)/(k*R) - np.cos(k*R))

# Integrand that goes in calculation of Sigma(M)

def SigmaIntegrand(k, Pk, R):
    return k**2 * Pk * TopHat(k, R)**2

def linear_growth_factor(Omega_m0, Omega_l0, z):
    if len(np.atleast_1d(z)) == 2:
        z1    = z[0]
        z2    = z[1] # z2 > z1                                                                                            

    if (len(np.atleast_1d(z)) == 1) or (len(np.atleast_1d(z)) > 2):
        z1    = 0.
        z2    = z    # z2 > z1                                                                                           
             
    Omega_lz1 = Omega_l0 / (Omega_l0 + Omega_m0 * (1.+z1)**3)
    Omega_mz1 = 1. - Omega_lz1
    gz1       = (5./2.) * Omega_mz1 / (Omega_mz1**(4./7.) - Omega_lz1 + (1. + Omega_mz1/2.) * (1. + Omega_lz1/70.))
    Omega_lz2 = Omega_l0 / (Omega_l0 + Omega_m0 * (1.+z2)**3)
    Omega_mz2 = 1. - Omega_lz2
    gz2       = (5./2.) * Omega_mz2 / (Omega_mz2**(4./7.) - Omega_lz2 + (1. + Omega_mz2/2.) * (1. + Omega_lz2/70.))
    return (gz2 / (1.+z2)) / (gz1 / (1+z1))
