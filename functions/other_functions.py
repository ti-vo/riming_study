import numpy as np

def rimed_mass_fraction_dmitri(vd_mean_pcor_sealevel):
    """
    Function to compute rime mass fraction from mean Doppler velocity

    :param vd_mean_pcor_sealevel: Mean Doppler velocity, corrected for air density (sea level)
    :return: Rimed mass fraction derived from Dmitri Moisseev's fit (Kneifel & Moisseev 2020)
    """
    p1 = 0.07018
    p2 = -0.5427
    p3 = 1.265
    p4 = -0.5018
    p5 = -0.0584

    vd_mean_pcor_pos = -1 * vd_mean_pcor_sealevel

    # Stefan: Basically I would anyways only use MDV>1.5 m/s for the analysis,
    # because a rimed dendrite cannot be distinguished from a non-rimed snowflake
    # when both fall at a speed of 1 m/s

    np.putmask(vd_mean_pcor_pos, vd_mean_pcor_pos < 1.5, np.nan)

    rmf_dmitri = p1 * vd_mean_pcor_pos**4 + p2 * vd_mean_pcor_pos**3 + p3 * vd_mean_pcor_pos**2 \
                 + p4 * vd_mean_pcor_pos + p5

    # filter out negative rimed mass fractions (vd > ca. - 0.65 m / s)
    neg_rf, neg_rf_dmitri_ind = np.where(rmf_dmitri < 0)
    rmf_dmitri[neg_rf_dmitri_ind] = np.nan

    # some very few rain events are misclassified by CLOUDNET as ice and also Dmitri's
    # polynomial fit actually bends up after 3 m/s, so we could just set everything falling faster
    # to a rimed mass fraction of 1.
    rf = np.where(vd_mean_pcor_pos > 2.5, 0.85, rmf_dmitri)

    return rf

