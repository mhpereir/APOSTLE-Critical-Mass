import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from matplotlib import cm

plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 3

from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from astropy import constants as const

from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from scipy.optimize import newton_krylov, brentq, curve_fit

from critical_mass_functions_2 import *

from multiprocessing import Pool 
from time import time



z_BL_2020   = [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0]

Tcr_Trho_BL_2020 = [4.34, 4.41, 4.40, 4.35, 4.31, 4.26, 4.21, 4.19, 4.16, 4.12, 4.12]
Tcr_iso_BL_2020  = [4.46, 4.43, 4.42, 4.41, 4.43, 4.42, 4.41, 4.43, 4.42, 4.42, 4.42]


#fig,ax = plt.subplots(figsize=(10,6))

#ax.plot(z_BL_2020,   Tcr_Trho_BL_2020 , label=r'T$(\rho)$, BL+20', color='C0', marker='o')
#ax.plot(z_BL_2020,   Tcr_iso_BL_2020 , label=r'Isothermal, BL+20', color='C1', marker='o')

#ax.set_xlabel('Redshift')
#ax.set_ylabel(r'log$_{10}$(T$_{critical}$ [K])')

#ax.set_ylim([4,4.6])
#ax.set_xlim([0,10])

#ax.xaxis.set_major_locator(MultipleLocator(2))
#ax.xaxis.set_minor_locator(MultipleLocator(0.4))

#ax.yaxis.set_major_locator(MultipleLocator(0.1))
#ax.yaxis.set_minor_locator(MultipleLocator(0.02))

#ax.tick_params( right='on', top='on', which='both', direction='in' )
#ax.tick_params( which='major', length=7)
#ax.tick_params( which='minor', length=4)

#ax.legend()

m200_Trho_range  = []
m200_iso_range   = []

for i,z in enumerate(z_BL_2020):
    
    t200 = np.power(10, Tcr_Trho_BL_2020[i])*u.K
    v200_temp = lambda t200: np.sqrt(t200/(10**4*u.K)) * (17*u.km/u.s)
    
    m200_temp = v200_temp(t200)**3 / (const.G * 10 * cosmo.H(z))
    m200_temp = np.log10(m200_temp.decompose() / const.M_sun)

    #print('m200', m200_temp)
    
    m200_Trho_range.append(m200_temp)

for i,z in enumerate(z_BL_2020):
    
    t200 = np.power(10, Tcr_iso_BL_2020[i])*u.K
    v200_temp = lambda t200: np.sqrt(t200/(10**4*u.K)) * (17*u.km/u.s)
    
    m200_temp = v200_temp(t200)**3 / (const.G * 10 * cosmo.H(z))
    m200_temp = np.log10(m200_temp.decompose() / const.M_sun)

    #print('m200', m200_temp)
    
    m200_iso_range.append(m200_temp)


#fig,ax = plt.subplots(figsize=(10,6))

#ax.plot(z_BL_2020,   m200_Trho_range , label=r'T$(\rho)$, BL+20', color='C0', marker='o')
#ax.plot(z_BL_2020,   m200_iso_range , label=r'Isothermal, BL+20', color='C1', marker='o')

#ax.set_xlabel('Redshift')
#ax.set_ylabel(r'log$_{10}$(M$_{critical}$ [M$_\odot$])')

#ax.set_ylim([7.4,10])
#ax.set_xlim([0,10])

#ax.xaxis.set_major_locator(MultipleLocator(2))
#ax.xaxis.set_minor_locator(MultipleLocator(0.4))

#ax.yaxis.set_major_locator(MultipleLocator(0.5))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))

#ax.tick_params( right='on', top='on', which='both', direction='in' )
#ax.tick_params( which='major', length=7)
#ax.tick_params( which='minor', length=4)

#ax.legend()


#fig,ax = plt.subplots(figsize=(8,6), tight_layout=True)

#lognh_range = np.arange(-7,3.01,0.2)
#nh_range    = np.power(10, lognh_range )

#for z in [0,1,2,4,10,12]:
    #rho_crit  = cosmo.critical_density(z)
    #omega_b   = cosmo.Ob(z)
    #omega_m   = cosmo.Om(z)
    #rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    #rhob_rhobar = nh_range / rho_bar_b
    
    #T_rho = T_rho_init_adb(z)
    
    #ax.plot(lognh_range, np.log10(T_rho(nh_range)), label='z={}'.format(z))
    
#ax.hlines(xmin=-7,xmax=10,y=np.log10(2e4), linestyle='--', color='magenta')

#ax.legend(loc='lower right')

#ax.set_xlabel(r'log$_{10}(n_H/cm^{-3})$')
#ax.set_ylabel(r'log$_{10}(T/K)$')

#ax.set_xlim([-7.05,1.05])
#ax.set_ylim([3,4.75])

#ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.xaxis.set_minor_locator(MultipleLocator(0.2))

#ax.yaxis.set_major_locator(MultipleLocator(0.25))
#ax.yaxis.set_minor_locator(MultipleLocator(0.05))

#ax.tick_params( right='on', top='on', which='both', direction='in' )
#ax.tick_params( which='major', length=7)
#ax.tick_params( which='minor', length=4)


z_range         = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11.5])
z_range_iso     = np.asarray([11.501, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0])

p = Pool(4)

#calculate critical mass ALEJANDRO
# Mb/M200, Trho1, c=10

m200_c10_upper  = [9.75, 9.6, 9.35, 9.1,  8.9, 8.7, 8.5, 8.35, 8.25, 8.15, 8.1, 8.0]
c10_range       = np.zeros(len(m200_c10_upper)) + 10

#mcrit_mb_trho1_c10 = p.map(find_Trho_mcrit, zip(z_range, m200_c10_upper))
mcrit_mb_trho1_c10 = [9.702796622321443, 9.521194484923953, 9.293882529958184, 9.043414200181324,
                      8.833905997645841, 8.64899079436759,  8.475317478659877, 8.3380229530776,
                      8.222141479899348, 8.11792456514029,  8.045753680304372, 7.94741349550805]

#print('mcrit_mb_trho1_c10', mcrit_mb_trho1_c10)

c10_range_iso          = np.zeros(len(z_range_iso)) + 10
T_b_range_iso          = np.zeros(len(z_range_iso)) + 7000 * u.K#10**4. *u.K
mcrit_mb_trho1_c10_iso = p.map(find_iso_mcrit, zip(z_range_iso, c10_range_iso, T_b_range_iso))

T_b_range_iso           = np.zeros(len(z_range_iso)) + 10**4. *u.K
mcrit_mb_trho1_c10_iso2 = p.map(find_iso_mcrit, zip(z_range_iso, c10_range_iso, T_b_range_iso))



# calculate critical mass MATTHEW 1:
# Central density, Trho1, c=10

m200_upper_limit = [9.75,  9.6, 9.3, 9.1, 8.9, 8.7, 8.5, 8.35, 8.25, 8.15, 8.1,  8.0]
c_range          = np.zeros(len(z_range)) + 10
rhoc_range       = np.zeros(len(z_range)) + 10

#mcrit_rhoc_trho1_c10 = p.map(find_Trho_mcrit_2, zip(z_range, m200_upper_limit, c_range, rhoc_range))
mcrit_rhoc_trho1_c10 = [9.69998756096183, 9.49807505767062, 9.244681262655533, 8.974156217226158, 
                        8.744975416435661, 8.543140909474198, 8.34781194930181, 8.19359552612821, 
                        8.059550515185819, 7.940439184701531, 7.857483972344295, 7.735176544786474]

#print('mcrit_rhoc_trho1_c10', mcrit_rhoc_trho1_c10)

c10_range_iso   = np.zeros(len(z_range_iso)) + 10
T_b_range_iso   = np.zeros(len(z_range_iso)) + 10**4. *u.K
rhoc_range_iso  = np.zeros(len(z_range_iso)) + 10

mcrit_rhoc_trho1_c10_iso = p.map(find_iso_mcrit_2, zip(z_range_iso, c10_range_iso, rhoc_range_iso, T_b_range_iso))



# calculate critical mass MATTHEW 2:
# Central density, Trho1, c=lud

m200_upper_limit = [9.76, 9.62, 9.44, 9.24, 9.05, 8.86, 8.70, 8.57, 8.45, 8.35, 8.30, 8.20]
rhoc_range       = np.zeros(len(z_range)) + 10

#mcrit_rhoc_trho1_clud = p.map(find_Trho_mcrit_lud_2, zip(z_range, m200_upper_limit, rhoc_range))
mcrit_rhoc_trho1_clud = [9.625996334703931, 9.523020890684315, 9.339633175001987, 9.116677391573958,
                         8.906775528612863, 8.724634598445034, 8.537348265325987, 8.391379392493079,
                         8.261028394775606, 8.146863572520518, 8.064465072776754, 7.947260936318887]

#print('mcrit_rhoc_trho1_clud', mcrit_mb_trho1_c10)

rhoc_range_iso       = np.zeros(len(z_range_iso)) + 10
m200_upper_limit_iso = [8.10, 8.10, 8.05, 8.00, 7.95, 7.90, 7.80, 7.70]

mcrit_rhoc_trho1_clud_iso = p.map(find_Trho_mcrit_lud_2, zip(z_range_iso, m200_upper_limit_iso, rhoc_range_iso))


# calculate critical mass MATTHEW 3:
# Central density, Trho2, c=lud

m200_upper_limit = np.asarray([9.76, 9.62, 9.44, 9.24, 9.05, 8.86, 8.70, 8.57, 8.45, 8.35, 8.30, 8.20])+0.2
rhoc_range       = np.zeros(len(z_range)) + 10



mcrit_rhoc_trho2_clud = p.map(find_Trho_mcrit_lud_2_adb, zip(z_range, m200_upper_limit, rhoc_range))
#mcrit_rhoc_trho2_clud = [9.683248781376223, 9.559800018934089, 9.377850491269776, 9.156967263870925,
                         #8.955999924317716, 8.783561603553167, 8.60446749190951, 8.470337706549596, 
                         #8.351650503672467, 8.245764558083058, 8.163865666656054, 8.053880940995196]


#print('mcrit_rhoc_trho2_c10', mcrit_rhoc_trho2_clud)

rhoc_range_iso       = np.zeros(len(z_range_iso)) + 10
m200_upper_limit_iso = np.asarray([8.10, 8.10, 8.05, 8.00, 7.95, 7.90, 7.80, 7.70])
T_b_range_iso        = np.zeros(len(z_range_iso)) + 10**4. *u.K

m200_upper_limit_iso_mgas_norm = np.asarray([8.40, 8.40, 8.35, 8.30, 8.25, 8.20, 8.00, 7.90])

#old method, boundary condition at infinity
#mcrit_rhoc_trho2_clud_iso = p.map(find_Trho_mcrit_lud_2_adb, zip(z_range_iso, m200_upper_limit_iso, rhoc_range_iso))

#isothermal all the way
#mcrit_rhoc_trho2_clud_iso = p.map(find_iso_mcrit_lud_2_mgas_norm, zip(z_range_iso, T_b_range_iso, rhoc_range_iso))

#isothermal + EOS
mcrit_rhoc_trho2_clud_iso = p.map(find_Trho_mcrit_lud_2_adb_norm_gas, zip(z_range_iso, m200_upper_limit_iso_mgas_norm, rhoc_range_iso))

print(mcrit_rhoc_trho2_clud_iso)

z_range_m200     = np.arange(0,20.51,0.1)
m200_t200_10000K = np.log10( (M200(1e4 *u.K, z_range_m200)/const.M_sun).decompose() )
m200_t200_7000K  = np.log10( (M200(7e3 *u.K, z_range_m200)/const.M_sun).decompose() )



tegmark_data = np.genfromtxt('Tegmark_1997.txt', delimiter=', ', skip_header=1)

logz_tegmark = tegmark_data[:,0]
z_tegmark    = np.power(10, logz_tegmark)-1
logT_tegmark = tegmark_data[:,1]
m200_t200_Tegmark = np.log10( (M200( np.power(10, logT_tegmark)  *u.K, z_tegmark)/const.M_sun).decompose() )

fig,ax = plt.subplots(figsize=(7,5), tight_layout=True, ncols=2, sharey=True,
                      gridspec_kw={'width_ratios': [1, (20-11.5)/11.5],'wspace':0, 'hspace':0})


ax[0].plot(z_range, mcrit_mb_trho1_c10,    linewidth=4, color='darkviolet',       linestyle='-', label='BLF20', zorder=5)
#ax[0].plot(z_range, mcrit_rhoc_trho1_c10,  linewidth=2.4, color='skyblue', linestyle='-')#, label='Line2')
#ax[0].plot(z_range, mcrit_rhoc_trho1_clud, linewidth=3, color='gray',    linestyle='-')#, label='Line3')
ax[0].plot(z_range, mcrit_rhoc_trho2_clud, linewidth=4, color='k',       linestyle='--', label='APOSTLE Fiducial', zorder=6)

#ax[0].plot(z_range_m200, m200_t200_10000K, color='gray', linestyle=':', linewidth=3)
ax[0].plot(z_range_m200, m200_t200_7000K,  color='magenta', linestyle=':', linewidth=3, label=r'T$_{200}=7e3K$')
#ax[0].plot(z_tegmark, m200_t200_Tegmark,  color='navy', linestyle='-', linewidth=3)


#ax[0].set_xlabel(r'Redshift', size=18)
ax[0].set_ylabel(r'log$_{10}$(M$_{200}$/M$_\odot$)', size=18)

ax[0].annotate((r'Redshift'),(0.73,-0.12),xycoords='axes fraction',fontsize=18,ha='left',color='k')


ax[0].xaxis.set_major_locator(MultipleLocator(2))
ax[0].xaxis.set_minor_locator(MultipleLocator(0.4))

ax[0].yaxis.set_major_locator(MultipleLocator(0.5))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.1))

ax[0].tick_params( right='on', top='on', which='both', direction='in' )
ax[0].tick_params( which='major', length=7, width=1.5)
ax[0].tick_params( which='minor', length=4, width=1.2)

ax[0].set_xlim([0,11.5])
ax[0].set_ylim([6.5,9.8])

ax[0].legend(loc='lower left', fontsize=16)

ax[0].annotate((r'z<z$_{reion}$'),(.65,.90),xycoords='axes fraction',fontsize=18,ha='left',color='k')
ax[1].annotate((r'z>z$_{reion}$'),(.15,.90),xycoords='axes fraction',fontsize=18,ha='left',color='k')

#ax[1].annotate((r'Before Reionization'),(.07,.90),xycoords='axes fraction',fontsize=18,ha='left',color='k')

print('MW Mcrit Fiducial')
print(mcrit_rhoc_trho2_clud)
print(mcrit_rhoc_trho2_clud_iso)

print('BL Mcrit')
print(mcrit_mb_trho1_c10)
print(mcrit_mb_trho1_c10_iso)

#ax[1].plot(z_range_iso, mcrit_mb_trho1_c10_iso,    linewidth=4, color='darkviolet',       linestyle='-', zorder=5) #old and incorrect..
#ax[1].plot(z_range_iso, mcrit_mb_trho1_c10_iso2,   linewidth=2.4, color='skyblue',  linestyle='--')
#ax[1].plot(z_range_iso, mcrit_rhoc_trho1_c10_iso,  linewidth=2.4, color='skyblue', linestyle='--')
#ax[1].plot(z_range_iso, mcrit_rhoc_trho1_clud_iso, linewidth=3, color='gray',    linestyle='--')
ax[1].plot(z_range_iso, mcrit_rhoc_trho2_clud_iso, linewidth=4, color='k',       linestyle='--', zorder=6)


#ax[1].plot(z_range_m200, m200_t200_10000K, color='gray', linestyle=':', linewidth=3, label=r'T$_{200}=1e4K$')
ax[1].plot(z_range_m200, m200_t200_7000K, color='darkviolet', linestyle='-', linewidth=4)#, label=r'T$_{200}=7e3K$')
ax[1].plot(z_tegmark, m200_t200_Tegmark,  color='navy', linestyle=':', linewidth=4)#, label=r'Tegmark97')

#ax[1].legend(bbox_to_anchor=(0.99,0.85), fontsize=16)


ax[1].xaxis.set_major_locator(MultipleLocator(2))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.4))

ax[1].yaxis.set_major_locator(MultipleLocator(0.5))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))

ax[1].tick_params( right='on', top='on', which='both', direction='in' )
ax[1].tick_params( which='major', length=7, width=1.5)
ax[1].tick_params( which='minor', length=4, width=1.2)

ax[1].set_xlim([11.5,20])

fig.savefig('./images/critical_mass.png', dpi=300, bbox_inches='tight')

plt.show()


