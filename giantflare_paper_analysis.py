

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

import lightcurve
import giantflare

def load_rxte_data(datadir="./", climits=[10,200]):

    """
     Load Giant Flare RXTE data from file.
    """

    data = gt.conversion('%s1806.dat'%datadir)
    time = np.array([float(x) for x in data[0]])
    channels = np.array([float(x) for x in data[1]])
    time = np.array([t for t,c in zip(time, channels) if climits[0] <= c <= climits[1]])
    time = time - time[0]

    ### start time used by Anna's analysis, used for consistency with Strohmayer+Watts 2006
    tmin = time.searchsorted(196.1)
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
    allstack = giantflare.make_stacks(savg, 9, 15)


    ### load powers at 625 Hz from 30000 simulations with the QPO smoothed out:
    ### if file doesn't exist, load with function make_rxte_sims() below, but be warned that it takes a while
    ### (like a day or so) to run!
    savgall_sims = gt.getpickle("1806_rxte_tseg=3s_dt=0.5s_df=2.66hz_30000sims_savgall.dat")




    return


def make_rxte_sims(tnew):

    savgall = giantflare.simulations(tnew, nsims=30000, tcoarse = 0.01, tfine =0.5/2000.0, freq=625.0, nsteps=15,
                                     tseg=3.0, df = 2.66, fnyquist=2000.0, stack='all', setlc=False, set_analysis=True,
                                     maxstack=9, qpo=False)

    return savgall
