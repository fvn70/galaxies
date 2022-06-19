import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u

def stage1():
    df = pd.read_csv('groups.tsv', sep='\t').dropna()
    d = df.groupby('features').mean_mu.mean()
    print(d[1], d[0])


def stage2():
    df = pd.read_csv('groups.tsv', sep='\t').dropna()
    s0 = df.mean_mu[df.features == 0]
    s1 = df.mean_mu[df.features == 1]

    # Shapiro-Wilk Normality test
    sh0 = stats.shapiro(s0)[1]
    sh1 = stats.shapiro(s1)[1]

    # Fligner-Killeen Homogeneity test
    fk = stats.fligner(s0, s1)[1]

    # one-way ANOVA test
    ow = stats.f_oneway(s0, s1)[1]
    print(sh1, sh0, fk, ow)


def stage3():
    df1 = pd.read_csv('isolated_galaxies.tsv', sep='\t')
    df2 = pd.read_csv('galaxies_morphology.tsv', sep='\t')
    s1 = df1.n
    s2 = df2.n

    bins = np.linspace(0, 12, 20)
    plt.hist(s1, bins, alpha=0.5, edgecolor='k', label='isolated galaxies')
    plt.hist(s2, bins, alpha=0.5, edgecolor='k', label='group galaxies')
    plt.title(" Sérsic index")
    plt.ylabel("Count")
    plt.xlabel("n")
    plt.legend()
    plt.show()

    k1 = df1.n[df1.n > 2].count() / df1.n.count()
    k2 = df2.n[df2.n > 2].count() / df2.n.count()
    p = stats.ks_2samp(s1, s2)[1]

    print(k2, k1, p)


def stage4():
    df_gal = pd.read_csv('galaxies_morphology.tsv', sep='\t')
    df_gr = pd.read_csv('groups.tsv', sep='\t').dropna()

    df = df_gal.groupby('Group').mean()
    df.columns = ['mean_n', 'mean_T']
    df.reset_index(inplace=True)
    df = df.merge(df_gr)

    # Plot scatterplots
    plt.scatter(df.mean_n, df.mean_mu)
    plt.xlabel('<n>', size=16)
    plt.ylabel(r'$\mu_{IGL, r}$ (mag~arcsec$^{-2}$)', size=16)
    plt.show()
    plt.scatter(df.mean_T, df.mean_mu)
    plt.xlabel('<T>', size=16)
    plt.ylabel(r'$\mu_{IGL, r}$ (mag~arcsec$^{-2}$)', size=16)
    plt.show()

    # Shapiro-Wilk Normality test
    sh_mu = stats.shapiro(df.mean_mu)[1]
    sh_n = stats.shapiro(df.mean_n)[1]
    sh_T = stats.shapiro(df.mean_T)[1]

    # Pearson correlation coefficients
    r_n = stats.pearsonr(df.mean_mu, df.mean_n)[1]
    r_T = stats.pearsonr(df.mean_mu, df.mean_T)[1]

    print(sh_mu, sh_n, sh_T, r_n, r_T)


def median_separ(df, r):
      df = df[df.Group == r].reset_index()
      n = df.shape[0]
      s = []
      for i in range(n):
          p1 = SkyCoord(ra=df.loc[i].RA * u.degree, dec=df.loc[i].DEC * u.degree, frame="fk5")
          for j in range(i+1, n):
              p2 = SkyCoord(ra=df.loc[j].RA * u.degree, dec=df.loc[j].DEC * u.degree, frame="fk5")
              r = p1.separation(p2).radian
              s.append(r)
      ad = pd.Series(s)
      return ad.median()


def cadd(cosmo, row):
    # my_cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    d = cosmo.angular_diameter_distance(row['z']).to(u.kpc).value
    r = d * row['Rm']
    return r


def stage5():
    df = pd.read_csv('groups.tsv', sep='\t').dropna()
    df_coo = pd.read_csv('galaxies_coordinates.tsv', sep='\t')

    df['Rm'] = df.apply(lambda row: median_separ(df_coo, row['Group']), axis=1)

    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    df['R'] = df.apply(lambda row: cadd(cosmo, row), axis=1)

    # Plot a scatterplot for the projected median separation and
    # the IGL mean surface brightness
    plt.scatter(df.R, df.mean_mu)
    plt.gca().invert_yaxis()
    plt.xlabel('<R (kpc)>', size=16)
    plt.ylabel(r'$\mu_{IGL, r}$ (mag~arcsec$^{-2}$)', size=16)
    plt.show()

    # median separation for the HCG 2 group
    r = df[df.Group == 'HCG 2'].R[1]

    # Shapiro-Wilk Normality test
    sh_R = stats.shapiro(df.R)[1]
    sh_mu = stats.shapiro(df.mean_mu)[1]

    # Pearson correlation coefficients
    r_n = stats.pearsonr(df.mean_mu, df.R)[1]

    print(r, sh_R, sh_mu, r_n)


stage5()
