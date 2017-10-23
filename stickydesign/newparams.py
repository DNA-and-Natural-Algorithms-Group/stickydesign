# n.b.: All parameters arrays here are arranged to fit the binary
# representation used in stickydesign.  A, C, G, and T are given values of
# 0,1,2,3, thus taking up 2 bits, and these bits are then put in 5' to 3'
# order.  Thus, for example, 5 would correspond to 5'-CC-3', and 2 would
# correspond to 5'-AG-3'.

import numpy as np

# yapf: disable

# dG and dS parameters from SantaLucia, 2004.
nndG37 = np.array([-1.00, -1.44, -1.28, -0.88,
                   -1.45, -1.84, -2.17, -1.28,
                   -1.30, -2.24, -1.84, -1.44,
                   -0.58, -1.30, -1.45, -1.00])
nndS = np.array([-0.0213, -0.0224, -0.0210, -0.0204,
                 -0.0227, -0.0199, -0.0272, -0.0210,
                 -0.0222, -0.0244, -0.0199, -0.0224,
                 -0.0213, -0.0222, -0.0227, -0.0213])

# Coaxial stacking parameter dGs at 37 C are from Protozanova 2004
# (doi:10.1016/j.jmb.2004.07.075), table 2.
coax_prot_tab2_dG37 = np.array([-1.04, -2.04, -1.29, -1.27,
                                -0.78, -1.97, -1.44, -1.29,
                                -1.66, -2.70, -1.97, -2.04,
                                -0.12, -1.66, -0.78, -1.04])

coax_prot_tab1_dG37 = np.array([-1.11, -1.81, -1.06, -1.34,
                                -0.55, -1.44, -0.91, -1.06,
                                -1.43, -2.17, -1.44, -1.81,
                                -0.19, -1.43, -0.55, -1.11])

# Coaxial stacking parameter dSs for Protozanova parameters are taken
# from Zhang, 2009 supplementary information, which discusses this
# formulaic approach.
coaxdS = 0.0027 / (1-310.15*0.0027) * coax_prot_tab2_dG37
coaxddS = coaxdS-nndS
coaxddG37 = coax_prot_tab2_dG37-nndG37
# correction term rather than absolute dG

# Coaxial stacking parameters from Pyshni, Ivanova 2002
coax_pyshni_dG37 = np.array([-1.22, -2.10, -1.19, -1.89,
                             -1.20, -1.96, -1.17, -1.35,
                             -1.46, -2.76, -1.25, -2.29,
                             -0.85, -2.21, -1.06, -1.93])

coax_pyshni_dS = 0.001 * np.array([-26.0, -30.1, -27.9, -29.0,
                                   -33.8, -21.4, -14.9, -15.3,
                                   -25.4, -35.6, -20.6, -35.6,
                                   -28.2, -45.6, -31.7, -48.3])
coax_pyshni_ddS = coax_pyshni_dS - nndS
coax_pyshni_ddG37 = coax_pyshni_dG37 - nndG37

# Coaxial stacking parameters from Peyret 2000. The nick here is on the
# reverse complement strand.  The TT and AG values are averages.
# These are currently unused.
coax_peyret_dG37 = np.array([-1.5, -1.9, -1.15, -2.2,
                             -1.7, -2.6, -0.8, -2.4,
                             -2.6, -3.3, -1.6, -3.0,
                             -1.6, -2.9, -1.7, -2.55])

coax_peyret_dS = 0.001 * np.array([-44.7, -27.8, -(25.4+56.9)/2.0, -59.2,
                                   -46.8, -52.0, -26.4, -33.5,
                                   -48.5, -31.4, -20.6, -55.5,
                                   -26.2, -34.1, -33.1, -(23.1+77.6)/2])

coax_peyret_ddS = coax_peyret_dS - nndS
coax_peyret_ddG37 = coax_peyret_dG37 - nndG37

dangle5dG37 = np.array([-0.51, -0.96, -0.58, -0.50,
                        -0.42, -0.52, -0.34, -0.02,
                        -0.62, -0.72, -0.56,  0.48,
                        -0.71, -0.58, -0.61, -0.10])
dangle5dH = np.array([0.2, -6.3, -3.7, -2.9,
                      0.6, -4.4, -4.0, -4.1,
                     -1.1, -5.1, -3.9, -4.2,
                     -6.9, -4.0, -4.9, -0.2])
dangle3dG37 = np.array([-0.12,  0.28, -0.01,  0.13,
                        -0.82, -0.31, -0.01, -0.52,
                        -0.92, -0.23, -0.44, -0.35,
                        -0.48, -0.19, -0.50, -0.29])
dangle3dH = np.array([-0.5,  4.7, -4.1, -3.8,
                      -5.9, -2.6, -3.2, -5.2,
                      -2.1, -0.2, -3.9, -4.4,
                      -0.7,  4.4, -1.6,  2.9])
intmmdG37 = np.array([0.61,  0.88,  0.14,  0.00,  # ij
                      0.77,  1.33,  0.00,  0.64,  # *k
                      0.02,  0.00, -0.13,  0.71,
                      0.00,  0.73,  0.07,  0.69,
                      0.43,  0.75,  0.03,  0.00,
                      0.79,  0.70,  0.00,  0.62,
                      0.11,  0.00, -0.11, -0.47,
                      0.00,  0.40, -0.32, -0.12,
                      0.17,  0.81, -0.25,  0.00,
                      0.47,  0.79,  0.00,  0.62,
                     -0.52,  0.00, -1.11,  0.08,
                      0.00,  0.98, -0.59,  0.45,
                      0.69,  0.92,  0.42,  0.00,
                      1.33,  1.05,  0.00,  0.97,
                      0.74,  0.00,  0.44,  0.43,
                      0.00,  0.75,  0.34,  0.68])
intmmdS = np.array([1.7,   4.6,  -2.3,   0.0,
                    14.6,  -4.4,   0.0,   0.2,
                    -2.3,   0.0,  -9.5,   0.9,
                    0.0,  -6.2,  -8.3, -10.8,
                    -4.2,   3.7,  -2.3,   0.0,
                    -0.6,  -7.2,   0.0,  -4.5,
                    -13.2,   0.0, -15.3, -11.7,
                    0.0,  -6.1,  -8.0, -15.8,
                    -9.8,  14.2,  -1.0,   0.0,
                    -3.8,   8.9,   0.0,   5.4,
                    3.2,   0.0, -15.8,  10.4,
                    0.0,  13.5, -12.3,  -8.4,
                    12.9,   8.0,   0.7,   0.0,
                    20.2,  16.4,   0.0,   0.7,
                    7.4,   0.0,   3.6,  -1.7,
                    0.0,   0.7,  -5.3,  -1.5])/1000.0
# go from cal/(molK) to kcal/(molK)

dangle5dS = (dangle5dH - dangle5dG37) / 310.15
dangle3dS = (dangle3dH - dangle3dG37) / 310.15
initdG37 = 1.96
initdS = 0.0057
tailcordG37 = 0.8
looppenalty = 3.6

# yapf: enable
