import numpy as np
import pandas as pd
from numba import njit

GM = 1.32712440018e20

@njit(fastmath=True)
def transfer_time_numba(r1, r2, ve, m_dry, T, mp0):

    dt = 3600.0
    max_t = 3000*86400.0
    v1 = np.sqrt(GM/r1)
    v2 = np.sqrt(GM/r2)

    dv_tot = ve * np.log((m_dry+mp0)/m_dry)

    dv_acc = 0.5*(dv_tot + (v1 - v2))
    dv_brk = 0.5*(dv_tot - (v1 - v2))
    if dv_acc < 0.0 or dv_brk < 0.0:
        return -1.0

    mdot = T/ve

    r_f0 = r1
    r_f1 = 0.0
    v_f0 = 0.0
    v_f1 = v1
    m_f = m_dry + mp0
    dv_f = 0.0

    r_b0 = 0.0
    r_b1 = r2
    v_b0 = -v2
    v_b1 = 0.0
    m_b = m_dry
    dv_b = dv_brk

    t = 0.0

    while t < max_t:
        rf = np.sqrt(r_f0*r_f0 + r_f1*r_f1)
        rb = np.sqrt(r_b0*r_b0 + r_b1*r_b1)

        if rf > rb:
            return 2.0*t/86400.0/7.0

        if dv_f < dv_acc and m_f > m_b:
            a_th_f0 =  (T/m_f)*(r_f0/rf)
            a_th_f1 =  (T/m_f)*(r_f1/rf)
            m_f -= mdot*dt
            dv_f += T/m_f*dt
        else:
            a_th_f0 = 0.0
            a_th_f1 = 0.0

        inv_rf3 = 1.0/(rf*rf*rf)
        a_gr_f0 = -GM*r_f0*inv_rf3
        a_gr_f1 = -GM*r_f1*inv_rf3

        v_f0 += (a_th_f0 + a_gr_f0)*dt
        v_f1 += (a_th_f1 + a_gr_f1)*dt

        r_f0 += v_f0*dt
        r_f1 += v_f1*dt

        if dv_b > 0.0 and m_b < m_f:
            a_th_b0 = -(T/m_b)*(r_b0/rb)
            a_th_b1 = -(T/m_b)*(r_b1/rb)
            m_b += mdot*dt
            dv_b -= T/m_b*dt
        else:
            a_th_b0 = 0.0
            a_th_b1 = 0.0

        inv_rb3 = 1.0/(rb*rb*rb)
        a_gr_b0 = -GM*r_b0*inv_rb3
        a_gr_b1 = -GM*r_b1*inv_rb3

        v_b0 += (a_th_b0 + a_gr_b0)*dt
        v_b1 += (a_th_b1 + a_gr_b1)*dt

        r_b0 += v_b0*dt
        r_b1 += v_b1*dt

        t += dt

    return -1.0


au = 1.496e8 * 1e3

drives = pd.read_csv('drives.csv', index_col=0)
ks = drives.index

R = 10**np.linspace(0, np.log10(30), num=51)[1:]
t_R = np.zeros((len(ks), len(R)))
p_R = np.zeros((len(ks), len(R)))

for k in ks:
    ve = drives.loc[k, 'EV'] * 1e3
    m_dry = drives.loc[k, 'dry_mass'] * 1e3
    T = drives.loc[k, 'thrust']
    print(drives.loc[k,'name'])

    for j in range(len(R)):
        r1 = 1.0 * au
        r2 = R[j] * au

        prop_cap = drives.loc[k,'tank_cap']
        rb = drives.loc[k,'RP_bins'] - 1
        pms = np.unique((10**np.linspace(0,np.log10(200),num=100)).astype(int))
        pms = pms[pms<prop_cap]
        tts = np.zeros(len(pms))

        for i in range(len(tts)):
            mp0 = pms[i] * 1e5
            tts[i] = transfer_time_numba(r1, r2, ve, m_dry, T, mp0)

        mask = tts > 0
        p_good = pms[mask]
        t_good = tts[mask]

        if sum(mask)>0:
            expect_t = np.trapz(t_good, p_good) / (p_good.max() - p_good.min())
            expect_p = p_good[np.argmin(np.abs(t_good - expect_t))]
        else:
            expect_t = -1
            expect_p = -1

        t_R[k, j] = expect_t
        p_R[k, j] = expect_p

np.save('t_R.npy', t_R)
np.save('p_R.npy', p_R)
