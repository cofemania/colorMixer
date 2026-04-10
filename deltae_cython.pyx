# deltae_cython.pyx
import math

def delta_e_cie2000(double L1, double a1, double b1,
                    double L2, double a2, double b2):
    cdef double kL = 1.0
    cdef double kC = 1.0
    cdef double kH = 1.0
    cdef double delta_Lp = L2 - L1
    cdef double L_bar = (L1 + L2) / 2.0
    cdef double C1 = math.hypot(a1, b1)
    cdef double C2 = math.hypot(a2, b2)
    cdef double C_bar = (C1 + C2) / 2.0
    cdef double C_bar7 = C_bar**7
    cdef double root_term = math.sqrt(C_bar7 / (C_bar7 + 25**7))
    cdef double a1p = a1 + (a1 / 2.0) * (1.0 - root_term)
    cdef double a2p = a2 + (a2 / 2.0) * (1.0 - root_term)
    cdef double C1p = math.hypot(a1p, b1)
    cdef double C2p = math.hypot(a2p, b2)
    cdef double delta_Cp = C2p - C1p
    cdef double C_barp = (C1p + C2p) / 2.0
    cdef double h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    cdef double h2p = math.degrees(math.atan2(b2, a2p)) % 360.0
    cdef double delta_hp = h2p - h1p
    if C1p * C2p != 0:
        if delta_hp > 180.0:
            delta_hp -= 360.0
        elif delta_hp < -180.0:
            delta_hp += 360.0
    cdef double H_barp = (h1p + h2p) / 2.0
    if abs(delta_hp) > 180.0:
        H_barp += 180.0
    cdef double T = 1.0 - 0.17 * math.cos(math.radians(H_barp - 30.0)) \
                    + 0.24 * math.cos(math.radians(2.0 * H_barp)) \
                    + 0.32 * math.cos(math.radians(3.0 * H_barp + 6.0)) \
                    - 0.20 * math.cos(math.radians(4.0 * H_barp - 63.0))
    cdef double delta_Hp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(delta_hp) / 2.0)
    cdef double S_L = 1.0 + (0.015 * (L_bar - 50.0)**2) / math.sqrt(20.0 + (L_bar - 50.0)**2)
    cdef double S_C = 1.0 + 0.045 * C_barp
    cdef double S_H = 1.0 + 0.015 * C_barp * T
    cdef double delta_theta = 30.0 * math.exp(-((H_barp - 275.0) / 25.0)**2)
    cdef double R_C = 2.0 * math.sqrt(C_barp**7 / (C_barp**7 + 25**7))
    cdef double R_T = -R_C * math.sin(2.0 * math.radians(delta_theta))
    cdef double term1 = (delta_Lp / (kL * S_L))**2
    cdef double term2 = (delta_Cp / (kC * S_C))**2
    cdef double term3 = (delta_Hp / (kH * S_H))**2
    cdef double term4 = R_T * (delta_Cp / (kC * S_C)) * (delta_Hp / (kH * S_H))
    return math.sqrt(term1 + term2 + term3 + term4)