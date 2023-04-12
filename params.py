# Power spectra

# Path to power spectra

Database = "./PowerSpectra/"

# Name of power spectrum

#name     = "WMAP-1"
#name     = "WMAP-3"
#name     = "WMAP-5"
#name     = "WMAP-7"
#name     = "WMAP-9"
name     = "Planck"
#name     = "COCO"
#name     = "Millennium"

# Choose redshift at which to calculate c(M) relation

redshift  = 10.00

# Range of masses to calculate c(M) relation over

MassMin   = 1e-2
MassMax   = 1e17

# Number of mass grid points to generate for c(M) relation

numGridPoints = 200

# Number of intervals to integrate P(k) (recommended >= 500)

N_int        = 1000
