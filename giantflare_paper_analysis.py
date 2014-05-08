

### SCRIPT THAT MAKES PLOTS FOR GIANT FLARE PAPER
#
# Needs the following data files
#
# - 1806.dat --> ASCII file with RXTE giant flare data
# - 1806_rxte_tseg=3s_dt=0.5s_df=2.66hz_30000sims_savgall.dat: powers at 625Hz for 30000 light curves simulated from
#   the RXTE data with the light curve smoothed to 0.01s resolution, such that the 625 Hz signal is smoothed out.
#       --> remake with function make_rxte_sims() if necessary
#
#
#
# Dependencies:
#   numpy
#   cPickle
#
#   generaltools.py in UTools repository
#   lightcurve.py in UTools repository
#   giantflare.py in SAScripts repository
#
#   giantflares.py also depends on
#       powerspectrum.py in UTools repository
#       envelopes.py in UTools repository


import generaltools as gt
import numpy as np
import cPickle as pickle

import lightcurve
import giantflare

from pylab import *
rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)


def compute_pval(allstack, allstack_sims):

    """
    Given the output of giantflare.make_stacks() for both a data set (in allstacks) and
    n simulations (in allstack_sims), compute the p-values in dependence of the number
    of averaged cycles.

    """

    ### compute p-values for data using allstack (data) and allstack_sims (simulations):
    pvals = []
    for i in range(len(allstack)):
        savgtemp = allstack[i]
        allstack_temp = np.array([a[i] for a in allstack_sims])
        allstack_max = np.array([np.max(a) for a in allstack_temp])
        allstack_sort = np.sort(allstack_max)
        len_allstack = np.float(len(allstack_sort))
        ind_allstack = allstack_sort.searchsorted(max(savgtemp))
        pvals.append((len_allstack - ind_allstack)/len_allstack)

    return np.array(pvals)


def pvalues_error(pval, n):
    """
    Compute theoretical error on p-values in the limit of small p-values.
    pval can be a single value or a numpy array, but it'd better not be a list!

    """
    assert isinstance(pval, list) is False, "p-values must not be stored in list, but either be a " \
                                            "single float or a numpy array"

    perr_val = np.sqrt(np.array(pval)*(1.0-np.array(pval))/np.float(n))
    return perr_val



##### FIRST BIT: RXTE ANALYSIS AND PLOTS! ###############

def load_rxte_data(datadir="./", tstart=196.1, climits=[10,200]):

    """
     Load Giant Flare RXTE data from file.
     Default value for tstart is the one from Strohmayer+Watts 2006 (determined by squinting at Figure 1).
     Default channels to be included are 10-200, also from Strohmayer+Watts 2006
    """

    data = gt.conversion('%s1806.dat'%datadir)
    time = np.array([float(x) for x in data[0]])
    channels = np.array([float(x) for x in data[1]])
    time = np.array([t for t,c in zip(time, channels) if climits[0] <= c <= climits[1]])
    time = time - time[0]

    ### start time used by Anna's analysis, used for consistency with Strohmayer+Watts 2006
    tmin = time.searchsorted(tstart)
    tnew = time[tmin:]

    return tnew

def rxte_pvalues():

    ### load RXTE data
    tnew = load_rxte_data()

    ### compute powers at 625 Hz for time segments of 3s duration, binned frequency resolution of 2.66 Hz,
    ### starting every 0.5(ish) seconds apart
    ### lcall: light curves of all segments
    ### psall: periodograms of all segments
    ### mid: mid-point time stamps of all segments
    ### savg: power at 625 Hz for all segments
    ### xerr: error in x-direction (= 1/2 of tseg)
    ### ntrials: number of frequencies in each bin(?)
    ### sfreqs: five frequency bins around 625 Hz (so that I can check whether I have the right bin), for all segments
    ### spowers: powers to the corresponding frequency bins in sfreqs, for all segments

    lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(tnew, nsteps=15, tseg=3.0, df=2.66, fnyquist=2000.0, stack=None,
                                      setlc=True, freq=625.0)

    ### stack up periodograms at the same phase at consecutive cycles, up to averaging 9 cycles:
    allstack = giantflare.make_stacks(savg, 10, 15)


    ### load powers at 625 Hz from 30000 simulations with the QPO smoothed out:
    ### if file doesn't exist, load with function make_rxte_sims() below, but be warned that it takes a while
    ### (like a day or so) to run!
    savgall_sims = gt.getpickle("1806_rxte_tseg=3s_dt=0.5s_df=2.66hz_30000sims_savgall.dat")

    ### savgall_sims should be the direct output of giantflare.simulations, which means the first dimension
    ### of the array are the individual segments, the second dimension the simulations.
    ### Thus, for use with make_stacks, we need to transpose it.
    assert np.shape(savgall_sims)[0] < np.shape(savgall_sims)[1], "savgall_sims should be 315 by 30000, but isn't!"

    savgall_sims = np.transpose(savgall_sims)

    ### make stacks of all simulations in the same way as for the real data
    ### note that this could take a while and use a lot of memory!
    allstack_sims = []
    for s in savgall_sims:
        allstack_sims.append(giantflare.make_stacks(s, 10, 15))


    ### Compute p-values from data (allstack) and simulations (allstack_sims)
    pvals = compute_pval(allstack, allstack_sims)

    ### Compute theoretical errors on p-values; sort of a fudge, but I think okay as a general idea:
    pvals_err = pvalues_error(pvals, len(savgall_sims))

    ### list of cycles averaged for plot
    cycles = np.arange(len(pvals))+1

    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)

    errorbar(cycles, pvals, yerr=pvals_err, color='black', ls="solid", marker="o")

    return



def make_rxte_sims(tnew, nsims=30000,save=True, fout="1806_rxte_tseg=3.0_df=2.66_dt=0.5_f=625Hz_savgall.dat"):

    """
    Make 30000 simulated light curves, with the original RXTE giant flare light curve smoothed out to a 0.01s
    resolution, such that the 625Hz QPO is definitely no longer in the smoothed light curve.
    Then add instrumental noise using a Poisson distribution, and run the same analysis as for the original
    RXTE light curve, with 3s long segments, a frequency resolution of 625 Hz and 0.5s between segment start times.

    tnew: array of input photon arrival times, can be e.g. output of load_rxte_data()

    Returns an array of 315 by 30000, i.e. 30000 simulated powers at 625 Hz for each segment. Note that in order
    to stick this into make_stacks(), one needs to take the transpose!

    """

    savgall = giantflare.simulations(tnew, nsims=nsims, tcoarse = 0.01, tfine =0.5/1000.0, freq=625.0, nsteps=10,
                                     tseg=3.0, df = 2.66, fnyquist=1000.0, stack=None, setlc=False, set_analysis=True,
                                     maxstack=9, qpo=False)

    if save:
        f = open(fout, "w")
        pickle.dump(savgall, f)
        f.close()

    return savgall



######## SECOND BIT: RHESSI ANALYSIS

def load_rhessi_data(datadir="./", tstart=80.0, tend=236.0, climits=[100.0, 200.0], seglimits=[0.0,7.0]):
    """
     Load Giant Flare RHESSI data from file.
     Default values for tstart/tend are from Watts+Strohmayer 2006 (determined by squinting at figure in paper).
     Default channels (climits) to be included are 100-200, also in Watts+Strohmayer 2006,
     Default segments (seglimits) to be included are 0-7, as in Watts+Strohmayer 2006
    """

    data = gt.conversion('%s1806_rhessi.dat'%datadir)
    time = np.array([float(x) for x in data[0]])
    channels = np.array([float(x) for x in data[1]])
    segments = np.array([float(s) for s in data[2]])
    time = np.array([t for t,c,s in zip(time, channels, segments) if climits[0] <= c <= climits[1] and
                                                                     seglimits[0] <= s <= seglimits[1]])

    tmin_ind = time.searchsorted(tstart)
    tmax_ind = time.searchsorted(tend)

    tnew = time[tmin_ind:tmax_ind]

    return tnew






######################################################################################################################
####### ALL PLOTS ####################################################################################################
######################################################################################################################

def plot_lightcurves(datadir="./"):

    """
     Figure 1 from Huppenkothen et al (in prep)
    """

    ### load RXTE data, entire data set
    times_rxte = load_rxte_data(datadir="./", tstart=0.0, climits=[10,200])

    ### make RXTE light curve
    lc_rxte = lightcurve.Lightcurve(times_rxte, timestep=0.1)

    ### load RHESSI data, entire data set
    times_rhessi = load_rhessi_data(datadir="./", tstart=0.0, tend=400.0, climits=[100,200], seglimits=[0,7])

    ### make RHESSI light curve
    lc_rhessi = lightcurve.Lightcurve(times_rhessi, timestep=0.1)

    ###  now make plot
    fig = figure(figsize=(24,9))
    subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.1, hspace=0.3)
    ax = fig.add_subplot(121)

    plot(lc_rxte.time, np.log10(lc_rxte.countrate), lw=2, color="black", linestyle="steps-mid")
    axis([35, 350, 1.0, 4.5])
    xlabel("Time since trigger [s]", fontsize=24, labelpad=15)
    ylabel(r"$\log_{10}{(\mathrm{Count rate}}$ [counts/s]", fontsize=24)
    title("RXTE Light Curve, 4-90 keV")

    arrow(188, 4.1, 9*7.54-2.0, 0, fc="k", ec="k", head_width=0.07, head_length=6, lw=3)
    arrow(188+9*7.54, 4.1, -(9*7.54-2.0), 0, fc="k", ec="k", head_width=0.07, head_length=6, lw=3)

    ax.text(223,4.2, "625.6 Hz QPO", verticalalignment='center', horizontalalignment='center', color='black',
            fontsize=26)

    ax2 = fig.add_subplot(122)
    plot(lc_rhessi.time, np.log10(lc_rhessi.countrate), lw=2, color="black", linestyle="steps-mid")

    axis([35, 350, 1.0, 4.5])
    xlabel("Time since trigger [s]", fontsize=24, labelpad=15)
    ylabel(r"$\log_{10}{(\mathrm{Count rate}}$ [counts/s]", fontsize=24)
    title("RHESSI Light Curve, 100-200 keV")

    arrow(52, 3.1, 200-52, 0, fc="k", ec="k", head_width=0.07, head_length=6, lw=3)
    arrow(200, 3.1, -(200-52), 0, fc="k", ec="k", head_width=0.07, head_length=6, lw=3)

    ax2.text(127,3.2, "626.5 Hz QPO", verticalalignment='center', horizontalalignment='center', color='black',
            fontsize=26)

    savefig("f1.eps", format='eps')
    close()

    return
