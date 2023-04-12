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

from critical_mass_functions import *

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





fig,ax = plt.subplots(figsize=(7,6), tight_layout=True)

lognh_range = np.arange(-7,3.01,0.2)
nh_range    = np.power(10, lognh_range )

for z in [0,1,2,4,10,12]:
    rho_crit  = cosmo.critical_density(z)
    omega_b   = cosmo.Ob(z)
    omega_m   = cosmo.Om(z)
    rho_bar_b = (rho_crit * omega_b / const.m_p ).to(u.cm**-3).value
    
    rhob_rhobar = nh_range / rho_bar_b
    
    T_rho = T_rho_init_adb(z)
    
    ax.plot(lognh_range, np.log10(T_rho(nh_range)), label='z={}'.format(z))
    
ax.hlines(xmin=-7,xmax=10,y=np.log10(2e4), linestyle='--', color='magenta')

ax.legend(loc='lower right')

ax.set_xlabel(r'log$_{10}(n_H/cm^{-3})$')
ax.set_ylabel(r'log$_{10}(T/K)$')

ax.set_xlim([-7.05,1.05])
ax.set_ylim([3,4.75])

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))

ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

ax.tick_params( right='on', top='on', which='both', direction='in' )
ax.tick_params( which='major', length=7)
ax.tick_params( which='minor', length=4)

p = Pool(10)

# mgas & concentration vs m200

z = 0

m200_range = np.arange(8, 10.01,0.1)
c05_range   = np.zeros(len(m200_range)) + 5
c10_range   = np.zeros(len(m200_range)) + 10
c15_range   = np.zeros(len(m200_range)) + 15
c20_range   = np.zeros(len(m200_range)) + 20
z_range     = np.zeros(len(m200_range)) + z

omega_b   = cosmo.Ob(z)
omega_m   = cosmo.Om(z)

mgas_c05_range = p.map(find_mgas_c, zip(m200_range, z_range, c05_range))
mgas_c10_range = p.map(find_mgas_c, zip(m200_range, z_range, c10_range))
mgas_c15_range = p.map(find_mgas_c, zip(m200_range, z_range, c15_range))
mgas_c20_range = p.map(find_mgas_c, zip(m200_range, z_range, c20_range))

print(m200_range)

print(mgas_c05_range)

mgas_lud_range = p.map(find_mgas_lud, zip(m200_range, z_range))

print(mgas_lud_range)

c05_range   = np.zeros(len(m200_range)) + 5
c10_range   = np.zeros(len(m200_range)) + 10
c15_range   = np.zeros(len(m200_range)) + 15
c20_range   = np.zeros(len(m200_range)) + 20

#rho_norm, c, z, logm200

c05_central_rho  = p.map(find_central_density_c, zip(c05_range, z_range, m200_range))
c10_central_rho  = p.map(find_central_density_c, zip(c10_range, z_range, m200_range))
c15_central_rho  = p.map(find_central_density_c, zip(c15_range, z_range, m200_range))
c20_central_rho  = p.map(find_central_density_c, zip(c20_range, z_range, m200_range))
clud_central_rho = p.map(find_central_density,  zip(z_range, m200_range))

print(m200_range)

print(np.log10(clud_central_rho))

fig,ax = plt.subplots(figsize=(7,9), nrows=2, sharex=True, gridspec_kw={'wspace':0., 'hspace':0.0} )

cm = plt.cm.get_cmap('RdYlBu')

#ax.plot(m200_range, mgas_c05_range-m200_range, color=cm( (5 -1)/14 ),  linewidth=3, zorder=6, label='c=5')
#ax.plot(m200_range, mgas_c10_range-m200_range, color=cm( (10 -1)/14 ), linewidth=3, zorder=5, label='c=10')
#ax.plot(m200_range, mgas_c15_range-m200_range, color=cm( (15 -1)/14 ), linewidth=3, zorder=4, label='c=15')



#ax.annotate((r'$\Omega_b/\Delta$'),(.08,.05),xycoords='axes fraction',fontsize=18,ha='left',color='k')
#ax.hlines(xmin=-10, xmax=10, y=np.log10(omega_b/200), linestyle='--', color='k')

#ax[0].annotate((r'Star-forming'),(.90,.20),xycoords='axes fraction',fontsize=18,ha='left',color='k', rotation=90)


#z_ = np.empty((1, 100, 4), dtype=float)
#rgb = colors.colorConverter.to_rgb('C0')
#z_[:,:,:3] = rgb
#z_[:,:,-1] = np.linspace(0, 0.7, 100)[None,:] #alpha chanel 

#xmin, xmax, ymin, ymax = 9.6, 10, -10, 10
#im = ax[0].imshow(z_, aspect='auto', extent=[xmin, xmax, ymin, ymax], origin='lower')

ax[0].plot(m200_range, np.log10(c05_central_rho), color='k', linewidth=5)
ax[0].plot(m200_range, np.log10(c10_central_rho), color='k', linewidth=5)
ax[0].plot(m200_range, np.log10(c15_central_rho), color='k', linewidth=5)
ax[0].plot(m200_range, np.log10(c20_central_rho), color='k', linewidth=5)

ax[0].plot(m200_range, np.log10(c05_central_rho), color='darkorange', linewidth=4, label='c=5')
ax[0].plot(m200_range, np.log10(c10_central_rho), color='darkturquoise', linewidth=4, label='c=10')
ax[0].plot(m200_range, np.log10(c15_central_rho), color='C0', linewidth=4, label='c=15')
ax[0].plot(m200_range, np.log10(c20_central_rho), color='navy', linewidth=4, label='c=20')

ax[0].plot(m200_range, np.log10(clud_central_rho), color='k', linewidth=4, label=r'c(M$_{200}$,z=0)')

print(m200_range)

print(np.log10(clud_central_rho))

ax[0].hlines(xmin=-10, xmax=100, y=1, linestyle='-.', color='k', linewidth=2, zorder=100)

rect1 = patches.Rectangle((0, 1), 15, 15, color ='C0', alpha=0.2)
ax[0].add_patch(rect1)

ax[1].plot(m200_range, mgas_c20_range-m200_range, color='navy',  linewidth=4, zorder=8, label='c=20')
ax[1].plot(m200_range, mgas_c15_range-m200_range, color='C0',  linewidth=4, zorder=7, label='c=15')
ax[1].plot(m200_range, mgas_c10_range-m200_range, color='darkturquoise', linewidth=4, zorder=6, label='c=10')
ax[1].plot(m200_range, mgas_c05_range-m200_range, color='darkorange', linewidth=4, zorder=5, label='c=5')
ax[1].plot(m200_range, mgas_lud_range-m200_range, color='k', linewidth=4, zorder=8, label=r'c($M_{200}$,z=0)')


ax[1].plot(m200_range, mgas_c05_range-m200_range, color='k', linewidth=5, zorder=1)
ax[1].plot(m200_range, mgas_c10_range-m200_range, color='k', linewidth=5, zorder=2)
ax[1].plot(m200_range, mgas_c15_range-m200_range, color='k', linewidth=5, zorder=3)
ax[1].plot(m200_range, mgas_c20_range-m200_range, color='k', linewidth=5, zorder=4)

ax[1].annotate((r'$\Omega_{bar}/\Omega_M$'),(.08,.79),xycoords='axes fraction',fontsize=18,ha='left',color='k')
ax[1].hlines(xmin=-10, xmax=10, y=np.log10(omega_b/omega_m), linestyle='--', color='k', zorder=100, linewidth=2.5)

ax[1].set_xlabel(r'log$_{10}$(M$_{200}$/M$_\odot$)')

ax[1].set_ylabel(r'log$_{10}$(M$_{gas}$/M$_{200}$)')
ax[0].set_ylabel(r'log$_{10}$($n_{c}$/cm$^{-3}$)')

ax[0].set_xlim([8.90,10])
ax[0].set_ylim([-2.99,2])
ax[1].set_ylim([-2.65,-0.2])

ax[0].yaxis.set_major_locator(MultipleLocator(1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.2))

ax[1].yaxis.set_major_locator(MultipleLocator(0.5))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))



ax[1].xaxis.set_major_locator(MultipleLocator(0.2))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.04))

ax[1].tick_params( right='on', top='on', which='both', direction='in' )
ax[1].tick_params( which='major', length=7, width=1.5)
ax[1].tick_params( which='minor', length=4, width=1.2)

ax[0].tick_params( right='on', top='on', which='both', direction='in' )
ax[0].tick_params( which='major', length=7, width=1.5)
ax[0].tick_params( which='minor', length=4, width=1.2)

#ax[0].legend(bbox_to_anchor=(0.55,0.47), fontsize=16)
ax[0].legend(loc='center left', fontsize=16)

fig.savefig('./images/mgas_concentration_2.png', dpi=300, bbox_inches='tight')

plt.show()
    
