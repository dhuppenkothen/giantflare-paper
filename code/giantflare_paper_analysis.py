

### SCRIPT THAT MAKES PLOTS FOR GIANT FLARE PAPER
#
# First part contains RXTE analysis, second part RHESSI
# Last part makes all plots, provided all data files are there.
#
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


import numpy as np
import glob
import scipy.interpolate
import scipy.ndimage

from utils import *

import lightcurve
import giantflare

from pylab import *
import matplotlib.cm as cm
from matplotlib.patches import FancyArrow

rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)


#########################################################
##### FIRST BIT: RXTE ANALYSIS AND PLOTS! ###############
#########################################################

def load_rxte_data(datadir="../data/", tstart=196.1, climits=[10,200]):

    """
     Load Giant Flare RXTE data from file.
     Default value for tstart is the one from Strohmayer+Watts 2006 (determined by squinting at Figure 1).
     Default channels to be included are 10-200, which is what Strohmayer+Watts 2006 write, but didn't do.
    """

    print("Loading data file %s1806.dat."%datadir)
    data = conversion('%s1806.dat'%datadir)
    time = np.array([float(x) for x in data[0]])
    channels = np.array([float(x) for x in data[1]])
    time = np.array([t for t,c in zip(time, channels) if climits[0] <= c <= climits[1]])
    time = time - time[0]

    ### start time used by Anna's analysis, used for consistency with Strohmayer+Watts 2006
    tmin = time.searchsorted(tstart)
    tnew = time[tmin:]

    return tnew

def rxte_pvalues(datadir="../data/", maxpowfile="../data/sgr1806_rxte_simulated_maxpowers.txt", savgallroot = None):

    """
    Makes Figure 2 of the paper.
    Uses a file with simulations from make_rxte_sims(). If you don't want to use the provided file,
    make it yourself,
    but be warned that it takes a while!

    """


    ### load RXTE data
    print("Loading data file %s1806.dat."%datadir)

    tnew = load_rxte_data(datadir)

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
        giantflare.search_singlepulse(tnew, nsteps=10, tseg=3.0, df=2.66, fnyquist=1000.0, stack=None,
                                      setlc=True, freq=624.0)

    ### stack up periodograms at the same phase at consecutive cycles, up to averaging 9 cycles:
    allstack = giantflare.make_stacks(savg, 10, 10)



    if not maxpowfile is None:
        print("Loading simulation data files in %s"%maxpowfile)
        maxp_all = np.loadtxt(maxpowfile)


    else:
        ### load powers at 625 Hz from 40000 simulations with the QPO smoothed out:
        ### if file doesn't exist, load with function make_rxte_sims() below, but be warned that it takes a while
        ### (like a day or so) to run!

        savgallfiles = glob.glob("%s*savgall.dat"%savgallroot)
        print("Loading all simulation data files: " + str(savgallfiles))
        maxp_all = []
        for s in savgallfiles:
            savgall_sims = np.loadtxt(s)


            ### savgall_sims should be the direct output of giantflare.simulations, which means the first dimension
            ### of the array are the individual segments, the second dimension the simulations.
            ### Thus, for use with make_stacks, we need to transpose it.
            if np.shape(savgall_sims)[0] < np.shape(savgall_sims)[1]:
                savgall_sims = np.transpose(savgall_sims)

            ### make stacks of all simulations in the same way as for the real data
            ### note that this could take a while and use a lot of memory!
            allstack_sims = []
            for s in savgall_sims:
                allstack_sims.append(giantflare.make_stacks(s, 10, 10))

            maxp_temp = []
            for i in xrange(len(allstack)):
                amax = np.array([np.max(a[i]) for a in allstack_sims])
                maxp_temp.append(amax)

            maxp_temp = np.transpose(np.array(maxp_temp))
            maxp_all.extend(maxp_temp)

        maxp_all = np.array(maxp_all)

        print("Saving maximum powers across segments in file %s_maxpowers.txt"%savgallroot)
        np.savetxt("%s_maxpowers.txt"%savgallroot)


    pvals = []
    for i,a in enumerate(allstack):
        sims = maxp_all[:,i]
        #print("sims " + str(sims))

        sims_sort = np.sort(sims)
        len_sims = np.float(len(sims_sort))
        ind_sims = sims_sort.searchsorted(max(a))

        pvals.append((len_sims-ind_sims)/len(sims))

    nsims = np.shape(maxp_all)[0]
    #print("nsims: " + str(nsims))
    pvals = np.array(pvals)
    ### Compute theoretical error on p-values
    pvals_err = pvalues_error(pvals, len(sims))



    ### list of cycles averaged for plot
    cycles = np.arange(len(pvals))+1


    #print("pvals_all: " + str(pvals))
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    errorbar(np.arange(len(pvals))+1, pvals, yerr=pvals_err, fmt="-o", lw=3, color="black", markersize=12)
    xlabel("Number of averaged cycles", fontsize=20)
    ylabel("P-value of maximum power", fontsize=20)
    title("SGR 1806-20, RXTE data, p-value from %i simulations"%nsims)
    print("Saving figure with RXTE p-values in f5.pdf")
    savefig("f5.pdf", format="pdf")
    close()

    return allstack, pvals, pvals_err



def make_rxte_sims(tnew=None, nsims=30000,save=True, froot="../data/sgr1806_rxte_tseg=3.0_df=2.66_dt=0.5_f=625Hz"):

    """
    Make nsims simulated light curves, with the original RXTE giant flare light curve smoothed out to a 0.01s
    resolution, such that the 625Hz QPO is definitely no longer in the smoothed light curve.
    Then add instrumental noise using a Poisson distribution, and run the same analysis as for the original
    RXTE light curve, with 3s long segments, a frequency resolution of 625 Hz and 0.5s between segment start times.

    tnew: array of input photon arrival times, can be e.g. output of load_rxte_data()

    Returns an array of n by nsims, i.e. nsims simulated powers at 625 Hz for each segment; n depends on segment size.

    For large nsims, this can be very long and tedious to run.
    In this case, the easiest solution is to spawn many smaller runs (~5000 simulations can easily run overnight)
    onto a multi-core system and let them run in parallel.

    """
    if tnew is None:
        tnew = load_rxte_data()

    savgall = giantflare.rxte_simulations(tnew, nsims=nsims, tcoarse=0.01, tfine=0.5/1000.0, freq=624.0, nsteps=10, 
                                          tseg=3.0, df=2.66, set_analysis=True, set_lc = False)
 

    if save:
        print("Saving RXTE simulation results in %s_savgall.dat"%froot)
        np.savetxt("%s_savgall.dat"%froot, savgall)

    return savgall


def stitch_savgall_together(froot="test"):
    """
    Take multiple identical runs of make_rxte_sims() and stitch them together.
    """
    savgfiles = glob.glob("../data/%s*"%froot)

    savgall = []
    for f in savgfiles:
        savgtemp = np.loadtxt(f)

        savgall.extend(savgtemp)

    return savgall



def periodogram_nosignal(froot_in="../data/sgr1806_rxte_tseg=3.0_df=2.66_nsteps=10_f=625Hz_final"):
    """
    Make a periodogram of the seven cycles in the nine-cycle average that don't have the
    strongest signal to check whether they are significant on their own.

    """

    ### load RXTE data
    tnew = load_rxte_data()

    nsteps=10

    ### compute powers at 625 Hz for time segments of 3s duration, binned frequency resolution of 2.66 Hz,
    ### starting every 0.5(ish) seconds apart
    ### details in comments for rxte_pvalues()
    lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(tnew, nsteps=nsteps, tseg=3.0, df=2.66, fnyquist=2000.0, stack=None,
                                      setlc=True, freq=624.0)

    ### make an empty array of zeros
    savg_nosig = np.zeros(10)

    ### these are the cycles without the strongest signal in them
    cycles = [0,1,2,3,4,5,7,8]

    ### loop over cycles and powers at the same phase for these cycles
    for c in cycles:
        savg_nosig += np.array(savg[(c*nsteps):((c+1)*nsteps)])

    ### divide by number of cycles
    savg_nosig = savg_nosig/np.float(len(cycles))
    savg_nosig = savg_nosig[0]
    print("power in the seven cycles w/out signal: " + str(savg_nosig))

    ### find all datafiles with string froot_in in their filename
    savgfiles = glob.glob("%s*"%froot_in)

    savgall_sims = []
    for f in savgfiles:
        savgtemp = np.loadtxt(f)
        print("np.shape(savgtemp): " + str(np.shape(savgtemp)))
        savgall_sims.extend(savgtemp)

    print("np.shape(savg_sims): " + str(np.shape(savgtemp)))
    savg_nosig_sims = []

    for s in savgall_sims:
        stemp = 0
        for c in cycles:
            #print("len(stemp) " + str(len(stemp)))
            stemp += np.array(s[c])
            #print("stemp: " + str(stemp))

        stemp = stemp/np.float(len(cycles))

        savg_nosig_sims.append(stemp)

    savg_nosig_sims = np.array(savg_nosig_sims)

    savg_nosig_sims_sorted = np.sort(savg_nosig_sims)
    print(np.shape(savg_nosig_sims_sorted))

    n_sig = savg_nosig_sims_sorted.searchsorted(np.max(savg_nosig))

    pval = np.float(len(savg_nosig_sims)-n_sig)/np.float(np.max(np.shape(savg_nosig_sims)))

    return pval, savg_nosig_sims_sorted



def rxte_qpo_sims_singlecycle(nsims=1000, froot = "../data/sgr1806_rxte"):


    """
    Make nsims simulations with a periodic signal added exactly at 624 Hz, then compute powers at that frequency
    for each simulation in segments in the rotational cycle with the strongest signal in the real data, and compare
    the simulated powers with those observed. Makes Figure 6 of the paper.
    """

    coarsesteps = 10
    finesteps = 750 ## starting approx every 0.01s apart

    tnew = load_rxte_data()

    tstart_all = [211.4, 241.5]
    amp_all = [0.15, 0.22]

    freq_all = [625.0 for t in tstart_all]
    randomphase_all = [False for t in tstart_all]

    length_all = [0.5, 0.5]

    nqpo = len(tstart_all)
    qpoparams = [{"freq":f, "amp":a, "tstart":t, "randomphase":r, "length":l} for f,a,t,r,l in \
        zip(freq_all, amp_all, tstart_all, randomphase_all, length_all)]

    lcsimall, savgall, pssimall = giantflare.make_qpo_simulations(tnew, nqpo, qpoparams, nsims=nsims, tcoarse=0.01,
                                              tfine=0.5/1000.0, freq=624.0, set_analysis=False, set_lc=True)

    savgall_coarsesteps, savgall_finesteps = [], []

    pvals_all, mid_coarse_all, mid_fine_all, savg_coarse_all, savg_fine_all = [], [], [], [], []
    ### load simulated maxpowers without QPO signal
    print("Loading simulated maximum powers across all segments, %s_simulated_maxpowers.txt."%froot)

    maxp_all = np.loadtxt("%s_simulated_maxpowers.txt"%froot)
    #print("shape(maxp_all): " + str(np.shape(maxp_all)))


    for j,lc in enumerate(lcsimall):

        print("I am on simulation %i"%j)
        ### first step: compute p-values for simulations
        lcall, psall, mid_coarse, savg_coarse, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(lc, nsteps=coarsesteps, tseg=3.0, df=2.66, fnyquist=1000.0, stack=None,
                                          setlc=True, freq=624.0)
        mid_coarse_all.append(mid_coarse)
        savg_coarse_all.append(savg_coarse)

        allstack = giantflare.make_stacks(savg_coarse, 10, 10)

        pvals = []
        for i,a in enumerate(allstack):

            maxp = np.max(a)

            sims = np.sort(maxp_all[:,i])
            #print("len(sims): " + str(len(sims)))


            max_ind = sims.searchsorted(maxp)

            pvals.append(float(len(sims)-max_ind)/float(len(sims)))

        pvals_all.append(pvals)


        minind = np.array(lc.time).searchsorted(236.0)
        maxind = np.array(lc.time).searchsorted(236.0+(1.0/0.132)*2)

        lcnew = lightcurve.Lightcurve(lc.time[minind:maxind], counts=lc.counts[minind:maxind])

        lcall, psall, mid_fine, savg_fine, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(lcnew, nsteps=finesteps, tseg=0.5, df=2.00, fnyquist=1000.0, stack=None,
                                          setlc=True, freq=624.0)
        mid_fine_all.append(mid_fine)
        savg_fine_all.append(savg_fine)

    pvals_all = np.array(pvals_all)

    ### real data, plot of strongest cycle:

    minind = tnew.searchsorted(236.0)
    maxind = tnew.searchsorted(236.0+(1.0/0.132)*2)

    tnew_small = tnew[minind:maxind]

    lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(tnew_small, nsteps=finesteps, tseg=0.5, df=2.00, fnyquist=1000.0, stack=None,
                                          setlc=True, freq=624.0)


    ### compute quantiles:
    sq_all = compute_quantiles(savg_fine_all, quantiles=[0.05,0.5,0.95])

    print("Saving data output for cycle with the strongest QPO signal in a bunch of files"
          "of type %s_strongestcycle*"%froot)

    np.savetxt("%s_strongestcycle_quantiles.txt"%froot, sq_all)
    np.savetxt("%s_strongestcycle_midbins.txt"%froot, mid_fine_all[0])
    np.savetxt("%s_strongestcycle_savgfine.txt"%froot, savg_fine_all)
    np.savetxt("%s_strongestcycle_pvals_all.txt"%froot, pvals_all)

    fig = figure(figsize=(12,9))
    #subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.1, hspace=0.2)

    ax = fig.add_subplot(111)

    ### plot data
    plot(mid, savg, lw=2, color="black", label="observed data")

    ### plot mean from simulations:
    plot(mid_fine_all[0], sq_all[:,1], lw=2, color="red", label="mean of simulations")

    ### plot quantiles as shaded area:
    fill_between(mid_fine_all[0], sq_all[:,0], sq_all[:,2], color="red", alpha=0.6, label="95\% quantiles from simulations")

    maxy = np.max([np.max(savg), np.max(sq_all[:,2])])

    legend(loc="upper right", prop={"size":16})
    axis([238.0, 246.0, 0, maxy+2])
    xlabel("Time since trigger [s]", fontsize=20)
    ylabel("Leahy Power", fontsize=20)

    #ax2 = fig.add_subplot(122)
    #pvals_hist = []
    #for i in range(10):
    #    p = pvals_all[:,i]
    #    h,bins = np.histogram(np.log10(p), bins=15, range=[-5.1,0.0])
    #    pvals_hist.append(h[::-1])

#    ax2.imshow(np.transpose(pvals_hist), cmap=cm.hot, extent=[0.5,10.5,-5.1,0.0])
#    ax2.set_aspect(2)

#    pvals_data = loadtxt("%s_pvals_all.txt"%froot)

#    plot(np.arange(10)+1, np.log10(pvals_data), "-o", lw=3, color="cyan", ms=10)

#    axis([0.5,10.5,-5.1,0.0])
#    xlabel("Number of averaged cycles", fontsize=20)
#    ylabel("log(p-value)", fontsize=20)

    print("Saving figure in f6.pdf")
    savefig("f6.pdf", format="pdf")
    close()


    return mid_coarse_all, savg_coarse_all, mid_fine_all, savg_fine_all, pvals_all



########################################################################################################################
######## SECOND BIT: RHESSI ANALYSIS ###################################################################################
########################################################################################################################

def load_rhessi_data(datadir="../data/", tstart=80.0, tend=236.0, climits=[100.0, 200.0], seglimits=[0.0,7.0]):
    """
     Load Giant Flare RHESSI data from file.
     Default values for tstart/tend are from Watts+Strohmayer 2006 (determined by squinting at figure in paper).
     Default channels (climits) to be included are 100-200, also in Watts+Strohmayer 2006,
     Default segments (seglimits) to be included are 0-7, as in Watts+Strohmayer 2006
    """
    print('Loading RHESSI data from %s1806_rhessi.dat'%datadir)
    data = conversion('%s1806_rhessi.dat'%datadir)
    time = np.array([float(x) for x in data[0]])
    channels = np.array([float(x) for x in data[1]])
    segments = np.array([float(s) for s in data[2]])
    time = np.array([t for t,c,s in zip(time, channels, segments) if climits[0] <= c <= climits[1] and
                                                                     seglimits[0] <= s <= seglimits[1]])

    tmin_ind = time.searchsorted(tstart)
    tmax_ind = time.searchsorted(tend)

    tnew = time[tmin_ind:tmax_ind]

    return tnew



def make_rhessi_sims(tnew=None, tseg_all=None, nsims=30000,save=True, froot="../data/sgr1806_rhessi_test"):

    """
    Make nsims simulated light curves, with the original RHESSI giant flare light curve smoothed out to a 0.01s
    resolution, such that the 625Hz QPO is definitely no longer in the smoothed light curve.
    Then add instrumental noise using a Poisson distribution, and run the same analysis as for the original
    RXTE light curve, with 3s long segments, a frequency resolution of 626 Hz and 0.5s between segment start times.

    tnew: array of input photon arrival times, can be e.g. output of load_rhessi_data()

    Returns an array of n by nsims, i.e. nsims simulated powers at 625 Hz for each segment; n depends on segment size.

    For large nsims, this can be very long and tedious to run.
    In this case, the easiest solution is to spawn many smaller runs (~5000 simulations can easily run overnight)
    onto a multi-core system and let them run in parallel.

    NOTE: THIS IS ALMOST THE SAME AS make_rxte_sims, but it loops, because the RHESSI data is tested for various
    segment lengths.

    It is possible to run this function, but in practice, it's probably both quicker and more efficient to run
    several instances of giantflare.rxte_simulations() with various values of tseg and df.

    """
    if tnew is None:
        tnew = load_rhessi_data()

    if tseg_all is None:
        ### all tsegs to test
        tseg_all = [0.5, 1.0, 1.5, 2.0, 2.5]

    ### use frequency resolution
    df_all = 1.0/np.array(tseg_all)


    for tseg,df in zip(tseg_all, df_all):
        savgall = giantflare.rxte_simulations(tnew, nsims=nsims, tcoarse=0.01, tfine=0.5/1000.0, freq=626.0, nsteps=30,
                                                tseg=tseg, df=df, set_analysis=True, set_lc = False)

#    savgall = giantflare.simulations(tnew, nsims=nsims, tcoarse = 0.01, tfine =0.5/1000.0, freq=624.0, nsteps=10,
#                                     tseg=3.0, df = 2.66, fnyquist=1000.0, stack=None, setlc=False, set_analysis=True,
#                                     maxstack=9, qpo=False)

        if save:
            print("Saving output of simulations in %s_tseg=%.2f_df=%.2f_savgall.txt"%(froot, tseg, df))
            np.savetxt("%s_tseg=%.2f_df=%.2f_savgall.txt"%(froot, tseg, df), savgall)

    return savgall

def rhessi_simulations_results(tnew=None, tseg_all=[0.5, 1.0, 2.0, 2.5], df_all=[2.0, 1.0, 1.0, 1.0],
                               freq = [627.0, 627.0, 626.5,626.0], classical=False,
                               froot_in="../data/sgr1806_rhessi", froot_out="../data/sgr1806_rhessi", plotdist=False, maxpowers=True):
    """
    Take several simulation runs made with make_rhessi_sims() and read them out one after the other,
    to avoid memory problems when running make_stacks for very large runs.

    NOTE: This requires simulation files run with the EXACT SAME parameters as the ones used for running
    giantflare.search_singlepulse() on the data below. You'd better make sure this is true. If not,
    the comparison will not be correct.

    For RHESSI, I use several segment lengths and correspondingly, also different frequency resolutions.

    """

    ### if tnew isn't given, read in:
    tnew = load_rhessi_data()

    savg_data, allstack_data = [], []

    pvals_all, perr_all = [], []

    colours= ["navy", "magenta", "cyan", "orange", "mediumseagreen", "black", "blue", "red"]

    ### plot p-values
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)

    #print("tseg_all: " + str(tseg_all))
    #print("df_all: " + str(df_all))
    #print("freq: " + str(freq))
    ### loop over all values of the segment lengths and frequency resolutions
    for tseg, df,f,c in zip(tseg_all, df_all,freq, colours):
    ### extract maximum powers from data.
    ### Note: The RHESSI QPO is at slightly higher frequency, thus using 626.0 Hz
        lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(tnew, nsteps=30, tseg=tseg, df=df, fnyquist=1000.0, stack=None,
                                      setlc=True, freq=f)


        for k,s in enumerate(spowers):
            print("index: %i, f: %.3f, P: %.3f"%(k, sfreqs[k,0], np.max(spowers[k,:])))


        savg_data.append(savg)
        ### make averaged powers for consecutive cycles, up to 19, for each of the nsteps segments per cycle:
        allstack = giantflare.make_stacks(savg, 19, 30)

        allstack_data.append(allstack)

        if classical is False:
            if maxpowers is True:
                print("%s*_tseg=%.1f*maxpowers.txt"%(froot_in,tseg))
                maxp_all_files = glob.glob("%s*_tseg=%.1f*maxpowers.txt"%(froot_in,tseg))
                print(maxp_all_files)
                maxp_all = np.loadtxt(maxp_all_files[0])
            else:
                ### find all datafiles with string froot_in in their filename
                print("%s*_tseg=%.1f*"%(froot_in,tseg))
                savgfiles = glob.glob("%s*_tseg=%.1f*savgall*"%(froot_in,tseg))

                print("Simulation files: " + str(savgfiles))

                maxp_all = []
                for f in savgfiles:

                    ### load simulation output
                    savgtemp = np.loadtxt(f)
                    #print("shape(savgtemp): " + str(np.shape(savgtemp)))

                    ### make averaged powers for each of the 10 cycles
                    allstack_temp = []
                    for s in savgtemp:
                        allstack_temp.append(giantflare.make_stacks(s, 19, 30))

                    maxp_temp = []
                    for i in xrange(len(allstack)):
                        amax = np.array([np.max(a[i]) for a in allstack_temp])
                        maxp_temp.append(amax)

                    maxp_temp = np.transpose(np.array(maxp_temp))
                    maxp_all.extend(maxp_temp)

                maxp_all = np.array(maxp_all)
                #print("shape(maxp_all) " + str(np.shape(maxp_all)))

                np.savetxt("%s_tseg=%.1f_simulated_maxpowers.txt"%(froot_out, tseg), maxp_all)

        pvals = []
        for i,a in enumerate(allstack):

            if classical is False:
                sims = maxp_all[:,i]
                #print("sims: " + str(len(sims)))

                sims_sort = np.sort(sims)
                len_sims = np.float(len(sims_sort))
                ind_sims = sims_sort.searchsorted(np.max(a))
                #print("ind_sims: " + str(ind_sims))
                pvals.append((len_sims-ind_sims)/len(sims))

            else:

                pval_single = pavnosig(np.max(a), float(i+1))
                pvals.append(pval_single)

            ### plot distributions of maximum powers against theoretical expectations?
            if plotdist:

                ### simulated chi-square powers:
                ### degree of freedom is 2*nbins*ncycles, i.e. 2*no of avg frequency bins*no of avg cycles
                ### for df = 2.66, nbins=8
                ### ncycles = i+1 (i.e. index of cycle +1, because index arrays start with zero instead of 1)
                ### size of output simulations is (n_simulations, n_averagedpowers)
                if len_sims > 100000:
                    print("Too many simulations! This will likely break your memory! Not making distribution plots!")
                else:
                    chisquare = np.random.chisquare(2*(i+1), size=(len_sims,len(a)))/(i+1)
                    maxc = np.array([np.max(chi) for chi in chisquare])

                    ### set plotting boundaries
                    minx = 0
                    maxx = np.max([np.max(maxc), np.max(sims)])+1

                    fig = figure(figsize=(12,9))
                    ax = fig.add_subplot(111)
                    ns, bins, patches = hist(sims, bins=100, color="cyan", alpha=0.7,
                                            label=r"maximum powers out of %i segments, %.2e simulations"%(len(a),len_sims),
                                            histtype="stepfilled", range=[minx,maxx], normed=True)

                    nc, bins, patches = hist(maxc, bins=100, color="magenta", alpha=0.7,
                                            label=r"maximum powers, $\chi^2$ expected powers",
                                            histtype="stepfilled", range=[minx,maxx], normed=True)

                    maxy = np.max([np.max(ns), np.max(nc)])
                    axis([minx, maxx, 0, maxy+0.1*maxy])
                    legend(prop={'size':16}, loc='upper right')

                    xlabel("Maximum Leahy powers", fontsize=18)
                    ylabel(r"$p(\mathrm{Maximum Leahy powers})$", fontsize=18)
                    title("Maximum Leahy power distributions for %i averaged cycles"%(i+1), fontsize=18)
                    savefig("%s_tseg=%.1f_maxdist_ncycle%i.png"%(froot_out,tseg, (i+1)), format="png")
                    close()

        pvals = np.array(pvals)
        pvals_all.append(pvals)

        pvals_noul, pvals_ul = [], []

        cycles = list(np.arange(len(pvals)))
        cycles_ul, cycles_noul = [], []
        for i,p in enumerate(pvals):
            if p < 1.0/len_sims:
                pvals_ul.append(1.0/len_sims)
                pvals[i] = 1.0/len_sims
                cycles_ul.append(i)
            else:
                pvals_noul.append(p)
                cycles_noul.append(i)

        pvals = np.log10(np.array(pvals))
        pvals_noul = np.log10(np.array(pvals_noul))
        pvals_ul = np.log10(np.array(pvals_ul))

        cycles = np.array(cycles)
        cycles_noul = np.array(cycles_noul)
        cycles_ul = np.array(cycles_ul)


        ### Compute theoretical error on p-values
        if classical is False:
            pvals_error = pvalues_error(pvals_noul, len(sims))
        else:
            pvals_error = pvalues_error(pvals_noul, 10000000.0)

        perr_all.append(pvals_error)

        log_errors = 0.434*np.array(pvals_error)/np.array(pvals_noul)

        ax.scatter(cycles+1, pvals, marker="o", s=32, color=c)
        ax.plot(cycles+1, pvals, lw=3, color=c)
        errorbar(cycles_noul+1, pvals_noul, yerr=log_errors, fmt="o", lw=0, color=c, markersize=12,
                 label=r"$t_{\mathrm{seg}} = %.1f \, \mathrm{s}$"%tseg)

        for cyc,pv in zip(cycles_ul, pvals_ul):
            ar1 = FancyArrow(cyc+1, pv, 0.0, -1.0, length_includes_head=True,
                color=c, head_width=0.2, head_length=0.3, width=0.001, linewidth=2)
            ax.add_artist(ar1)


    xlabel("Number of averaged cycles", fontsize=20)
    ylabel(r"$\log_{10}{(\mathrm{p-value\; of\; maximum\; power})}$", fontsize=20)
    legend(loc="upper right", prop={"size":16})
    axis([0,20,-7.1, 0.5])

    if classical is False:
        title("SGR 1806-20, RHESSI data, p-values")
        draw()
        tight_layout()
        print("Saving p-values in Figure f7.pdf")
        savefig("f7.pdf", format="pdf")
    else:
        title("SGR 1806-20, RHESSI data, classical p-values")
        draw()
        tight_layout()
        savefig("f7_classical.pdf", format="pdf")
    close()

    if classical is False:
        np.savetxt("%s_pvals_all.txt"%froot_out, pvals_all)
    else:
        np.savetxt("%s_pvals_all_classical.txt"%froot_out, pvals_all)


    return pvals_all



def rhessi_qpo_sims_allcycles(nsims=1000, froot="../data/sgr1806_rhessi"):
    ### first batch of simulations: constant signal in all cycles
    tstart_all = [  84.80569267,   92.38199711,   99.95283222,  107.52754498,
                    115.13365078,  122.68631363,  130.26195049,  137.8398037 ,
                    145.41058636,  152.98174858,  160.5617857 ,  168.1408844 ,
                    175.70906353,  183.28972721,  190.86196136,  198.43483257,
                    206.01998615,  213.58804893,  221.17355442]

    amp_all = [0.1 for t in tstart_all]
    freq_all = [626.5 for t in tstart_all]
    randomphase_all = [False for t in tstart_all]
    length_all = [2.0 for t in tstart_all]

    nqpo = len(tstart_all)
    qpoparams_all = [{"freq":f, "amp":a, "tstart":t, "randomphase":r, "length":l} for f,a,t,r,l in \
        zip(freq_all, amp_all, tstart_all, randomphase_all, length_all)]

    make_rhessi_qpo_sims(nqpo, qpoparams_all, nsims=nsims, froot="%s_allcycles"%froot)

    return

def rhessi_qpo_sims_allcycles_randomised(nsims=1000, froot="../data/sgr1806_rhessi"):

    """
     Same as rhessi_qpo_sims_allcycles above, but randomphase = True for all, and amplitude slightly different
    """

    ### first batch of simulations: constant signal in all cycles
    tstart_all = [  84.80569267,   92.38199711,   99.95283222,  107.52754498,
                    115.13365078,  122.68631363,  130.26195049,  137.8398037 ,
                    145.41058636,  152.98174858,  160.5617857 ,  168.1408844 ,
                    175.70906353,  183.28972721,  190.86196136,  198.43483257,
                    206.01998615,  213.58804893,  221.17355442]

    amp_all = [0.3 for t in tstart_all]
    freq_all = [626.5 for t in tstart_all]
    randomphase_all = [True for t in tstart_all]
    length_all = [2.0 for t in tstart_all]

    nqpo = len(tstart_all)
    qpoparams_all = [{"freq":f, "amp":a, "tstart":t, "randomphase":r, "length":l} for f,a,t,r,l in \
        zip(freq_all, amp_all, tstart_all, randomphase_all, length_all)]

    make_rhessi_qpo_sims(nqpo, qpoparams_all, nsims=nsims, froot="%s_allcyclesrandomised"%froot)

    return

def rhessi_qpo_sims_singlecycle(nsims=1000, froot="../data/sgr1806_rhessi"):

    tstart_all = [99.95283222, 107.52754498, 190.86196136,  198.43483257, 206.01998615,  213.58804893,  221.17355442]
    amp_all = [0.3, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2]

    freq_all = [626.5 for t in tstart_all]
    randomphase_all = [True for t in tstart_all]

    length_all = [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    nqpo = len(tstart_all)
    qpoparams_all = [{"freq":f, "amp":a, "tstart":t, "randomphase":r, "length":l} for f,a,t,r,l in \
        zip(freq_all, amp_all, tstart_all, randomphase_all, length_all)]

    make_rhessi_qpo_sims(nqpo, qpoparams_all, nsims=nsims, froot="%s_singlecycle"%froot)

    return


def make_rhessi_qpo_sims(nqpo, qpoparams, nsims=1000, froot="../data/sgr1806_rhessi_test",
                         freq = [627.0, 627.0, 626.0, 626.5,626.0]):
    """
    This function makes simulations with QPOs for the RHESSI data, to compare the real results to
    what would come out of the analysis given a particular signal.

    For a run of search_singlepulse, with nsteps=30, tseg=2.0, df=1.0, fnyquist=1000, freq=626.0, the locations of
    - the strongest signal is at tstart = 108.027545-0.5
    - second strongest signal, one cycle before: tstart = 100.452832-0.5
    - the last five cycles are at [191.361961365, 198.934832573, 206.519986153, 214.088048935, 221.67355442] - 0.5

    """

    tnew = load_rhessi_data()


    lcsimall, savgall, pssimall = giantflare.make_qpo_simulations(tnew, nqpo, qpoparams, nsims=nsims, tcoarse=0.01,
                                              tfine=0.5/1000.0, freq=626.0, set_analysis=False, set_lc=True)


    savgall_05, savgall_1, savgall_15, savgall_2, savgall_25 = [], [], [], [], []

    for i, lc in enumerate(lcsimall):
        print("I am on simulation %i" %i)
        lcall, psall, mid, savg_05, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(lc, nsteps=30, tseg=0.5, df=2.00, fnyquist=nsims, stack=None,
                                      setlc=True, freq=freq[0])

        savgall_05.append(savg_05)

        lcall, psall, mid, savg_1, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(lc, nsteps=30, tseg=1.0, df=1.00, fnyquist=nsims, stack=None,
                                      setlc=True, freq=freq[1])

        savgall_1.append(savg_1)

        #lcall, psall, mid, savg_15, xerr, ntrials, sfreqs, spowers = \
        #giantflare.search_singlepulse(lc, nsteps=30, tseg=1.5, df=1.00, fnyquist=nsims, stack=None,
        #                              setlc=True, freq=freq[2])

        #savgall_15.append(savg_15)

        lcall, psall, mid, savg_2, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(lc, nsteps=30, tseg=2.0, df=1.00, fnyquist=nsims, stack=None,
                                      setlc=True, freq=freq[3])

        savgall_2.append(savg_2)

        lcall, psall, mid, savg_25, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(lc, nsteps=30, tseg=2.5, df=1.00, fnyquist=nsims, stack=None,
                                      setlc=True, freq=freq[4])
        savgall_25.append(savg_25)


    ### save results to file
    print("Saving data in %s_tseg=0.5_savgall.txt"%froot)
    np.savetxt("%s_tseg=0.5_savgall.txt"%froot, savgall_05)
    print("Saving data in %s_tseg=1.0_savgall.txt"%froot)
    np.savetxt("%s_tseg=1.0_savgall.txt"%froot, savgall_1)
    #print("Saving data in %s_tseg=1.5_savgall.txt"%froot)
    #np.savetxt("%s_tseg=1.5_savgall.txt"%froot, savgall_15)
    print("Saving data in %s_tseg=2.0_savgall.txt"%froot)
    np.savetxt("%s_tseg=2.0_savgall.txt"%froot, savgall_2)
    print("Saving data in %s_tseg=2.5_savgall.txt"%froot)
    np.savetxt("%s_tseg=2.5_savgall.txt"%froot, savgall_25)

    return


def rhessi_qpo_sims_images(tseg_all=[0.5,1.0,2.0,2.5], df_all=[2.0, 1.0, 1.0, 1.0], nbins=30, froot_in="../data/sgr1806_rhessi",
                           froot_sims="allcycles", classical=False, freq=[627.0, 627.0, 626.5,626.0], plotn=8):

    ### if tnew isn't given, read in:
    tnew = load_rhessi_data()

    savg_data, allstack_data = [], []

    pvals_all, perr_all = [], []

    print("froot_in: %s" %froot_in)

    fig = figure(figsize=(24,18))
    subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.1, hspace=0.2)


    ### loop over all values of the segment lengths and frequency resolutions
    for k,(f, tseg, df) in enumerate(zip(freq, tseg_all, df_all)):
        ### extract maximum powers from data.
        ### Note: The RHESSI QPO is at slightly higher frequency, thus using 626.0 Hz
        lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(tnew, nsteps=30, tseg=tseg, df=df, fnyquist=1000.0, stack=None,
                                      setlc=True, freq=f)


        savg_data.append(savg)
        ### make averaged powers for consecutive cycles, up to 19, for each of the nsteps segments per cycle:
        allstack = giantflare.make_stacks(savg, 19, 30)

        allstack_data.append(allstack)

        if classical is False:
            maxp_all = np.loadtxt("%s_tseg=%.1f_simulated_maxpowers.txt"%(froot_in, tseg))

            ### sort powers for each ncycle from smallest to highest, such that I can simply use searchsorted to find
            ### the index of the power that corresponds to the observed one to compute p-value
            #maxp_sorted = np.array([np.sort(maxp_all[:,i]) for i in xrange(len(allstack))])
            ### transpose maxp, such that it's of the shape [nsims, ncycles]
            #maxp_sorted = np.transpose(maxp_all)

            #print("shape(maxp_sorted) " + str(np.shape(maxp_all)))

        ### load fake data, i.e. simulations *WITH* qpo
        qpofiles = glob.glob("%s*_%s_tseg=%.1f*savgall.txt"%(froot_in,froot_sims, tseg))
        print("qpofiles: " + str(qpofiles))
        print("%s*_%s_tseg=%.1f*savgall.txt"%(froot_in,froot_sims, tseg))

        pvals_all, pvals_data, pvals_hist_all = [], [], []
        ### allow for qpo simulations to be broken up into several parts
        for j,q in enumerate(qpofiles):
            savg_qpo = np.loadtxt(q)
            ### make averaged powers for each of the 10 cycles
            allstack_qpo = []
            for s in savg_qpo:
                allstack_qpo.append(giantflare.make_stacks(s, 19, 30))

            allstack_qpo = np.array(allstack_qpo)

        pvals, pvals_hist = [], []
        for i in xrange(len(allstack)):

            if classical is False:
                ### these are the simulations WITHOUT QPO
                sims = maxp_all[:,i]
                sims = np.sort(sims)
                len_sims = np.float(len(sims))
                #print("len_sims %i" %len_sims)
                #print("sims[0:10] " + str(sims[0:10]))
                #print("sims[-10:] " + str(sims[-10:]))


                ind_data = np.float(sims.searchsorted(np.max(allstack[i])))
                pvals_data.append((len_sims-ind_data)/len_sims)

                ### find index in sorted simulations without QPO that correspond to the observed maximum power
                ### for the simulated light curve with QPO
                ind_temp = np.array([np.float(sims.searchsorted(np.max(a))) for a in allstack_qpo[:,i]])
                pvals_temp = (len_sims-ind_temp)/len_sims

                pvals.append(pvals_temp)

            else:
                pvals_data_temp = pavnosig(np.max(allstack[i]), float(i+1))
                print(pvals_data_temp)
                pvals_data.append(pvals_data_temp)
                pvals_temp = [pavnosig(np.max(a), (i+1)) for a in allstack_qpo[:,i]]
                pvals.append(pvals_temp)

            h, bins = np.histogram(np.log10(pvals_temp), bins=nbins, range=[-6.0, 0.0])
            pvals_hist.append(h[::-1])

        pvals_all.append(pvals)
        pvals_hist_all.append(pvals_hist)

        pv_data_corr = [1.0/len_sims if pd == 0 else pd for pd in pvals_data]

        if classical is False:
            np.savetxt("%s_%s_tseg=%.1f_pvals.txt"%(froot_in, froot_sims, tseg), pvals)
        else:
            np.savetxt("%s_%s_tseg=%.1f_pvals_classical.txt"%(froot_in, froot_sims, tseg), pvals)

        ax = fig.add_subplot(2,2,k+1)
        znew = scipy.ndimage.gaussian_filter(np.transpose(pvals_hist), sigma=0.5, order=0)
        ax.imshow(znew, cmap=cm.hot, extent=[0.0,len(allstack), -7.1,0.0], interpolation="bicubic")
        ax.set_aspect(3)

        scatter(np.arange(19)+0.5, np.log10(pv_data_corr), lw=2, facecolor="LightGoldenRodYellow",
                edgecolor="cyan", marker="v", s=50)
        for i,pd in enumerate(pvals_data):
            print("pd: " + str(pd))
            if pd == 0.0:
                ar1 = FancyArrow(i+0.5, np.log10(1.0/len_sims), 0.0, -1.0, length_includes_head=True,
                                 color="cyan", head_width=0.2,
                                 head_length=0.3, width=0.001, linewidth=2)
                ax.add_artist(ar1)


        axis([0,len(allstack), -7.1, 0.0])
        xlabel("Number of averaged cycles", fontsize=24)
        ylabel(r"$\log_{10}{(\mathrm{p-value})}$", fontsize=24)
        title(r"simulated p-values, $t_{\mathrm{seg}} = %.1f$"%tseg)

    draw()
    tight_layout()

    if classical is False:
        print("Saving plot for p-values of simulations with periodic signal in %s in file f%i.pdf "%(froot_sims, plotn))
        savefig("f%i.pdf"%plotn)
    else:
        print("Saving plot for classical p-values of simulations with periodic signal in %s in file f%i.pdf "%(froot_sims, plotn))
        savefig("f%i_classical.pdf"%plotn)

    close()

    return pvals_all, pvals_hist_all




######################################################################################################################
####### ALL PLOTS ####################################################################################################
######################################################################################################################

def plot_averaging_example():

    tnew = load_rxte_data(tstart=198.0)
    tend = tnew[0]+3.0*7.5478

    tend_ind = tnew.searchsorted(tend)

    tnew = tnew[:tend_ind]

    lc = lightcurve.Lightcurve(tnew, timestep=0.01)

    fig = figure(figsize=(18,9))
    subplots_adjust(top=0.9, bottom=0.12, left=0.08, right=0.95, wspace=0.1, hspace=0.2)
    ax = fig.add_subplot(111)


    xt = ((lc.time - lc.time[0])/7.54577)*2.0*np.pi

    plot(xt, lc.countrate/1000.0, lw=2, color="black", linestyle="steps-mid")
    axis([0.0, 6.0*np.pi, 0.0, 9.0])

    ticks = np.arange(np.pi, 6.0*np.pi, np.pi)
    labels = [r"$%i\pi$"%(i+1) for i in range(len(ticks))]
    xticks(ticks, labels)

    ### NEED TO SET XTICKSLABELS PROPERLY
    xlabel("Phase ", fontsize=26)
    ylabel(r"Count rate [$10^{3} \, \mathrm{counts}\, \mathrm{s}^{-1}$]", fontsize=26)

    tstart = [2.5, 3.2]
    color=["navy", "darkred"]

    ylen = 0
    y_offset = 0.1

    for t,c in zip(tstart, color):
        ### first set of shaded areas
        x1 = np.arange(30)/10.0+t
        x2 = np.arange(30)/10.0+t+2.0*np.pi
        x3 = np.arange(30)/10.0+t+4.0*np.pi

        ybottom = np.zeros(len(x1))
        ytop = 6.0*np.ones(len(x1))

        fill_between(x1, ybottom, ytop, color=c, alpha=0.6)
        fill_between(x2, ybottom, ytop, color=c, alpha=0.6)
        fill_between(x3, ybottom, ytop, color=c, alpha=0.6)

        vlines(t+1.5, 6.0, 6.7+ylen, lw=4, color=c)
        vlines(t+1.0+2.0*np.pi, 6.0, 6.7+ylen, lw=4, color=c)
        hlines(6.7+ylen, t+1.5-0.015, t+1.0+2.0*np.pi+0.025, lw=4, color=c)
        middlevx = ((t+1.0+2.0*np.pi) - (t+1.5))/2.0 + t+1.5
        vlines(middlevx, 6.7+ylen, 7.4+ylen, lw=4, color=c)
        ax.text(middlevx-0.1,7.6+(ylen*3.0), r"$P_{\mathrm{avg,1}}$", verticalalignment='center',
                horizontalalignment='center', color=c, fontsize=26)

        vlines(t+2.0+2.0*np.pi, 6.0, 6.7+ylen, lw=4, color=c)
        vlines(t+1.5+4.0*np.pi, 6.0, 6.7+ylen, lw=4, color=c)
        hlines(6.7+ylen, t+2.0*np.pi+2.0-0.015, t+1.5+4.0*np.pi+0.025, lw=4, color=c)
        middlevx = ((t+1.5+4.0*np.pi) - (t+2.0+2.0*np.pi))/2.0 + t+2.0+2.0*np.pi
        vlines(middlevx, 6.7+ylen, 7.4+ylen, lw=4, color=c)
        ax.text(middlevx-0.5, 7.35+(ylen*3.0), r"$P_{\mathrm{avg,2}}$", verticalalignment='bottom',
                horizontalalignment='left', color=c, fontsize=26)


        ylen += y_offset

    draw()
    tight_layout()
    savefig("f1.pdf",format="pdf")
    close()

    return

def maxpower_plot():


    tnew = load_rxte_data(tstart=196.0)
    te_ind = tnew.searchsorted(tnew[0]+10*7.54577)
    tnew  = tnew[:te_ind]


    #fig = figure(figsize=(18,9))
    #subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.0, hspace=0.2)

    ### first plot: light curve, left y-axis
    #ax = fig.add_subplot(111)
    lc = lightcurve.Lightcurve(tnew, timestep=0.05)
    #plot(lc.time, lc.countrate/1000.0, lw=2, color="black", linestyle='steps-mid')
    #axis([tnew[0], tnew[-1], 0, 7])
    #xlabel("Time since trigger [s]", fontsize=26)
    #ylabel(r"Count rate [$\mathrm{counts} \, \mathrm{s}^{-1}$]",fontsize=26)

    #ax2 = subplot(1,1,1, sharex=ax, frameon=False)
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position("right")

    lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(tnew, nsteps=10, tseg=3.0, df=2.66, fnyquist=1000.0, stack=None,
                                      setlc=True, freq=624.0)

    allstack = giantflare.make_stacks(savg, 10, 10)

    print("maximum single power: %f"%np.max(savg))

    #errorbar(mid, savg, xerr=xerr, color='red', marker="o")
    #plot(yRight, '.-g')
    #ylabel("Leahy Power", fontsize=26)


    #savefig("maxpowers_test.png", format="png")
    #close()

    fig, ax1 = plt.subplots(figsize=(12,9))
    subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.0, hspace=0.2)

    ax1.plot(lc.time, lc.countrate/1000.0, lw=2, color="black", linestyle='steps-mid', label="RXTE light curve")

    ax1.set_xlim([tnew[0], tnew[-1]])
    ax1.set_ylim([0,7])
    #ax1.axis([tnew[0], tnew[-1], 0, 7])
    ax1.set_xlabel("Time since trigger [s]", fontsize=26)
    ax1.set_ylabel(r"Count rate [$10^3 \mathrm{counts} \; \mathrm{s}^{-1}$]",fontsize=26)

    ax2 = ax1.twinx()

    ax2.errorbar(mid, savg, xerr=xerr, color='red', fmt="o", ecolor="red", markersize=10, markeredgecolor="red",
                 label="single-cycle powers at 625 Hz")
    ax2.errorbar(mid[:len(allstack[1])], allstack[1], xerr=xerr[:len(allstack[1])], color="cyan", fmt="v",
                 markeredgecolor="cyan",
                 markersize=10, ecolor="cyan", label="two-cycle averaged powers at 625 Hz")

    #plot(yRight, '.-g')
    ax2.set_ylabel("Leahy Power", fontsize=26, rotation=270, labelpad=20)
    ax2.set_xlim([tnew[0], tnew[-1]])
    ax2.set_ylim([0,12])

    legend()

    ### add arrows where the highest powers are
    maxp1 = np.max(savg)
    maxind1 = np.where(savg == maxp1)[0]
    maxtime1 = mid[maxind1]

    maxp2 = np.max(allstack[1])
    maxind2 = np.where(allstack[1] == maxp2)[0]
    maxtime2 = mid[maxind2]

    ax2.arrow(maxtime1, maxp1+0.5+1.3, 0, -1.3, fc="red", ec="red", head_width=0.5, head_length=0.15, lw=3)
    ax2.arrow(maxtime2, maxp1+0.5+1.3, 0, -1.3, fc="cyan", ec="cyan", head_width=0.5, head_length=0.15, lw=3)

    draw()
    tight_layout()
    savefig("f2.pdf", format="pdf")
    close()


    return


def make_analysis_schematic():

    tnew = load_rxte_data(tstart=198.0)
    tend = tnew[0]+3.0*7.5478

    tend_ind = tnew.searchsorted(tend)

    tnew = tnew[:tend_ind]

    lc = lightcurve.Lightcurve(tnew, timestep=0.01)

    ### coarse light curve
    lccoarse = lightcurve.Lightcurve(tnew, timestep=0.1)
    ### use interpolation routine in scipy to interpolate between data points, use linear interpolation
    ### should be non-linear interpolation?
    interpolated_countrate = scipy.interpolate.interp1d(lccoarse.time, lccoarse.countrate, "linear")

    ### make light curve with right time resolution for analysis
    lcnew = lightcurve.Lightcurve(tnew, timestep=0.01)
    minind = lcnew.time.searchsorted(lccoarse.time[0])
    maxind = lcnew.time.searchsorted(lccoarse.time[-1])

    lcnew.time = lcnew.time[minind:maxind]
    lcnew.counts = lcnew.counts[minind:maxind]
    lcnew.countrate = lcnew.countrate[minind:maxind]

    ### interpolate light curve to right resolution
    countrate_new = interpolated_countrate(lcnew.time)
    counts_new = countrate_new*0.01


    fig = figure(figsize=(18,7))
    subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.0, hspace=0.2)


    ### Plot3: simulated light curves
    #ax3 = fig.add_subplot(133)
    ax3 = plt.subplot2grid((1,3),(0,2),rowspan=1)

    cpoisson = np.array([np.random.poisson(cts) for cts in counts_new])
    plot(((lcnew.time - lcnew.time[0])/7.5457)*2.0*np.pi, (cpoisson*100.0)/1000.0,
         lw=1, color="black", linestyle="steps-mid")
    axis([0.01, 6.0*np.pi-0.01, 0.03, 8.97])

    ticks = np.arange(np.pi, 6.0*np.pi, np.pi)
    labels = [r"$%i\pi$"%(i+1) for i in range(len(ticks))]
    xticks(ticks, labels)
    ### NEED TO SET XTICKSLABELS PROPERLY
    #ylabel(r"Count rate [$10^{3} \, \mathrm{counts}\, \mathrm{s}^{-1}]", fontsize=24)

    title("Simulated light curves")

    tstart = [2.5, 3.2]
    color=["navy", "darkred"]

    ylen = 0
    y_offset = 0.3

    for t,c in zip(tstart, color):
        ### first set of shaded areas
        x1 = np.arange(30)/10.0+t
        x2 = np.arange(30)/10.0+t+2.0*np.pi
        x3 = np.arange(30)/10.0+t+4.0*np.pi

        ybottom = np.zeros(len(x1))
        ytop = 6.0*np.ones(len(x1))

        fill_between(x1, ybottom, ytop, color=c, alpha=0.6)
        fill_between(x2, ybottom, ytop, color=c, alpha=0.6)
        fill_between(x3, ybottom, ytop, color=c, alpha=0.6)

        vlines(t+1.5, 6.0, 6.5+ylen, lw=4, color=c)
        vlines(t+1.0+2.0*np.pi, 6.0, 6.5+ylen, lw=4, color=c)
        hlines(6.5+ylen, t+1.5-0.09, t+1.0+2.0*np.pi+0.1, lw=4, color=c)
        middlevx = ((t+1.0+2.0*np.pi) - (t+1.5))/2.0 + t+1.5
        vlines(middlevx, 6.5+ylen, 7.4+ylen, lw=4, color=c)
        ax3.text(middlevx-0.8+ylen*2.8,7.6+(ylen*1.2), r"$P_{\mathrm{sim,1}}$", verticalalignment='center',
                horizontalalignment='center', color=c, fontsize=22)

        vlines(t+2.0+2.0*np.pi, 6.0, 6.5+ylen, lw=4, color=c)
        vlines(t+1.5+4.0*np.pi, 6.0, 6.5+ylen, lw=4, color=c)
        hlines(6.5+ylen, t+2.0*np.pi+2.0-0.09, t+1.5+4.0*np.pi+0.1, lw=4, color=c)
        middlevx = ((t+1.5+4.0*np.pi) - (t+2.0+2.0*np.pi))/2.0 + t+2.0+2.0*np.pi
        vlines(middlevx, 6.5+ylen, 7.4+ylen, lw=4, color=c)
        ax3.text(middlevx-2.1+ylen*2.8, 7.35+(ylen*1.2), r"$P_{\mathrm{sim,2}}$", verticalalignment='bottom',
                horizontalalignment='left', color=c, fontsize=22)


        ylen += y_offset

    ### second panel: smoothed-out lightcurve
    #ax2 = fig.add_subplot(132)
    ax2 = plt.subplot2grid((1,3),(0,1),rowspan=1)


    plot(((lcnew.time - lcnew.time[0])/7.5457)*2.0*np.pi, countrate_new/1000.0, lw=2, color="black")
    axis([0.01, 6.0*np.pi-0.01, 0.03, 8.97])

    ticks = np.arange(np.pi, 6.0*np.pi, np.pi)
    labels = [r"$%i\pi$"%(i+1) for i in range(len(ticks))]
    xticks(ticks, labels)

    xlabel("Phase ", fontsize=24)
    title("Smoothed light curve")
    ### left panel: same as Figure above
    #ax = fig.add_subplot(131)
    ax = plt.subplot2grid((1,3),(0,0),rowspan=1)

    xt = ((lc.time - lc.time[0])/7.54577)*2.0*np.pi

    plot(xt, lc.countrate/1000.0, lw=1, color="black", linestyle="steps-mid")
    axis([0.01, 6.0*np.pi-0.01, 0.03, 8.97])

    ticks = np.arange(np.pi, 6.0*np.pi, np.pi)
    labels = [r"$%i\pi$"%(i+1) for i in range(len(ticks))]
    xticks(ticks, labels)

    title("Original RXTE data")
    ### NEED TO SET XTICKSLABELS PROPERLY
    #xlabel("Phase ", fontsize=24)
    ylabel(r"Count rate [$10^{3} \, \mathrm{counts}\, \mathrm{s}^{-1}$]", fontsize=24)

    tstart = [2.5, 3.2]
    color=["navy", "darkred"]

    ylen = 0
    y_offset = 0.3

    for t,c in zip(tstart, color):
        ### first set of shaded areas
        x1 = np.arange(30)/10.0+t
        x2 = np.arange(30)/10.0+t+2.0*np.pi
        x3 = np.arange(30)/10.0+t+4.0*np.pi

        ybottom = np.zeros(len(x1))
        ytop = 6.0*np.ones(len(x1))

        fill_between(x1, ybottom, ytop, color=c, alpha=0.6)
        fill_between(x2, ybottom, ytop, color=c, alpha=0.6)
        fill_between(x3, ybottom, ytop, color=c, alpha=0.6)

        #vlines(t+1.5, 6.0, 6.7+ylen, lw=4, color=c)
        #vlines(t+1.0+2.0*np.pi, 6.0, 6.7+ylen, lw=4, color=c)
        #hlines(6.7+ylen, t+1.5-0.015, t+1.0+2.0*np.pi+0.025, lw=4, color=c)
        #middlevx = ((t+1.0+2.0*np.pi) - (t+1.5))/2.0 + t+1.5
        #vlines(middlevx, 6.7+ylen, 7.4+ylen, lw=4, color=c)
        #ax.text(middlevx-0.1,7.6+(ylen*3.0), r"$P_{\mathrm{avg,1}}$", verticalalignment='center',
        #        horizontalalignment='center', color=c, fontsize=22)

        #vlines(t+2.0+2.0*np.pi, 6.0, 6.7+ylen, lw=4, color=c)
        #vlines(t+1.5+4.0*np.pi, 6.0, 6.7+ylen, lw=4, color=c)
        #hlines(6.7+ylen, t+2.0*np.pi+2.0-0.015, t+1.5+4.0*np.pi+0.025, lw=4, color=c)
        #middlevx = ((t+1.5+4.0*np.pi) - (t+2.0+2.0*np.pi))/2.0 + t+2.0+2.0*np.pi
        #vlines(middlevx, 6.7+ylen, 7.4+ylen, lw=4, color=c)
        #ax.text(middlevx-0.5, 7.35+(ylen*3.0), r"$P_{\mathrm{avg,2}}$", verticalalignment='bottom',
        #        horizontalalignment='left', color=c, fontsize=22)

        vlines(t+1.5, 6.0, 6.5+ylen, lw=4, color=c)
        vlines(t+1.0+2.0*np.pi, 6.0, 6.5+ylen, lw=4, color=c)
        hlines(6.5+ylen, t+1.5-0.09, t+1.0+2.0*np.pi+0.1, lw=4, color=c)
        middlevx = ((t+1.0+2.0*np.pi) - (t+1.5))/2.0 + t+1.5
        vlines(middlevx, 6.5+ylen, 7.4+ylen, lw=4, color=c)
        ax.text(middlevx-0.8+ylen*2.8,7.6+(ylen*1.2), r"$P_{\mathrm{avg,1}}$", verticalalignment='center',
                horizontalalignment='center', color=c, fontsize=22)

        vlines(t+2.0+2.0*np.pi, 6.0, 6.5+ylen, lw=4, color=c)
        vlines(t+1.5+4.0*np.pi, 6.0, 6.5+ylen, lw=4, color=c)
        hlines(6.5+ylen, t+2.0*np.pi+2.0-0.09, t+1.5+4.0*np.pi+0.1, lw=4, color=c)
        middlevx = ((t+1.5+4.0*np.pi) - (t+2.0+2.0*np.pi))/2.0 + t+2.0+2.0*np.pi
        vlines(middlevx, 6.5+ylen, 7.4+ylen, lw=4, color=c)
        ax.text(middlevx-2.1+ylen*2.8, 7.35+(ylen*1.2), r"$P_{\mathrm{avg,2}}$", verticalalignment='bottom',
                horizontalalignment='left', color=c, fontsize=22)


        ylen += y_offset

    draw()
    tight_layout()
    savefig("f3.pdf", format="pdf")
    close()

    return




def plot_lightcurves(datadir="./"):

    """
     Figure 1 from Huppenkothen et al (in prep)
    """

    ### load RXTE data, entire data set
    times_rxte = load_rxte_data(datadir=datadir, tstart=0.0, climits=[10,200])

    ### make RXTE light curve
    lc_rxte = lightcurve.Lightcurve(times_rxte, timestep=0.1)

    ### load RHESSI data, entire data set
    times_rhessi = load_rhessi_data(datadir=datadir, tstart=0.0, tend=400.0, climits=[100,200], seglimits=[0,7])

    ### make RHESSI light curve
    lc_rhessi = lightcurve.Lightcurve(times_rhessi, timestep=0.1)

    ###  now make plot
    fig = figure(figsize=(24,8))
    subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.98, wspace=0.1, hspace=0.3)
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
    #ylabel(r"$\log_{10}{(\mathrm{Count rate}}$ [counts/s]", fontsize=24)
    title("RHESSI Light Curve, 100-200 keV")

    arrow(52, 3.1, 200-52, 0, fc="k", ec="k", head_width=0.07, head_length=6, lw=3)
    arrow(200, 3.1, -(200-52), 0, fc="k", ec="k", head_width=0.07, head_length=6, lw=3)

    ax2.text(127,3.2, "626.5 Hz QPO", verticalalignment='center', horizontalalignment='center', color='black',
            fontsize=26)

    draw()
    tight_layout()
    savefig("f4.pdf", format='pdf')
    close()

    return

def plot_rxte_pvalues(nsims=100000, froot = "sgr1806_rxte"):

    pvals = np.loadtxt("%s_pvals_all.txt"%froot)

    ### Compute theoretical error on p-values
    pvals_error = pvalues_error(pvals, nsims)

    ### plot p-values
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    #plot(np.arange(len(pvals))+1, pvals,"-o", lw=3, color="black", markersize=12)
    errorbar(np.arange(len(pvals))+1, pvals, yerr=pvals_error, fmt="-o", lw=3, color="black", markersize=12)
    xlabel("Number of averaged cycles", fontsize=20)
    ylabel("P-value of maximum power", fontsize=20)
    title("SGR 1806-20, RXTE data, p-value from %i simulations"%nsims)
    savefig("f5.pdf", format="pdf")
    close()



def plot_rxte_sims_singlecycle(froot="1806_rxte_strongestcycle"):

    """
    Use output of rxte_qpo_sims_singlecycle() to make a plot that shows the cycle with the
    strongest QPO in the RXTE data, and overplots the mean and 0.05/0.95 quantiles as red line and
    shaded area, respectively.

    Plotting in here is slightly different than in rxte_qpo_sims_singlecycle(), owing to the fact
    that eps doesn't allow for transparency natively.


    """


    tnew = load_rxte_data()

    mid_fine = np.loadtxt("%s_midbins.txt"%froot)
    sq_all = np.loadtxt("%s_quantiles.txt"%froot)
    pvals_all = np.loadtxt("%s_pvals_all.txt"%froot)

    minind = tnew.searchsorted(236.0)
    maxind = tnew.searchsorted(236.0+(1.0/0.132)*2)

    tnew_small = tnew[minind:maxind]

    finesteps = int((1.0/0.132)/0.01)
    print("finesteps: %.4f" %finesteps)

    lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(tnew_small, nsteps=finesteps, tseg=0.5, df=2.00, fnyquist=1000.0, stack=None,
                                          setlc=True, freq=624.0)

    fig = figure(figsize=(12,9))
    subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, wspace=0.1, hspace=0.2)

    ax = fig.add_subplot(111)

    ### plot quantiles as shaded area:
    fill_between(mid_fine, sq_all[:,0], sq_all[:,2], color="lightcoral", label="95\% quantiles from simulations",
                 zorder=1)

    plot(mid_fine, sq_all[:,0], lw=2, color="red")
    plot(mid_fine, sq_all[:,2], lw=2, color="red")

    ### plot data
    plot(mid, savg, lw=3, color="black", linestyle="steps-mid", label="observed data")


    ### plot mean from simulations:
    plot(mid_fine, sq_all[:,1], lw=2, color="red", label="mean of simulations")



    maxy = np.max([np.max(savg), np.max(sq_all[:,2])])

    legend(loc="upper right", prop={"size":24})
    axis([238.0, 246.0, 0, maxy+2])
    xlabel("Time since trigger [s]", fontsize=24)
    ylabel("Leahy Power", fontsize=24)



#    ax2 = fig.add_subplot(122)
#    pvals_hist = []
#    for i in range(10):
#        p = pvals_all[:,i]
#        h,bins = np.histogram(np.log10(p), bins=15, range=[-5.1,0.0])
#        pvals_hist.append(h[::-1])

#    ax2.imshow(np.transpose(pvals_hist), cmap=cm.hot, extent=[0.5,10.5,-5.1,0.0])
#    ax2.set_aspect(2)

#    pvals_data = loadtxt("sgr1806_rxte_pvals_all.txt")

#    plot(np.arange(10)+1, np.log10(pvals_data), "-o", lw=3, color="cyan", ms=10)

#    axis([0.5,10.5,-5.1,0.0])
#    xlabel("Number of averaged cycles", fontsize=20)
#    ylabel("log(p-value)", fontsize=20)

    savefig("f6.pdf", format="pdf")
    close()


    return


def plot_rhessi_pvalues(filename="sgr1806_rhessi_pvals_all.txt", tseg=[0.5,1.0,2.0,2.5],
                       nsims=30000):

    """
    Re-makes the p-value plot for the RHESSI data, without having to re-do the entire analysis.
    Figure 4 of Huppenkothen et al, 2014

    filename: a string that has the file with the p-values
    tseg: a list of the segment lengths used for the various p-values
    nsims: the number of simulations from which the p-value was derived, to compute the error
    froot: output file name root
    """

    pvals_all = np.loadtxt(filename)
    print("shape pvals_all: "+ str(np.shape(pvals_all)))

    #pvals_error = [pvalues_error(pvals, nsims) for pvals in pvals_all]
    #print("shape(pvals_error): " + str(np.shape(pvals_error)))

    #log_errors = [0.434*dp/p for dp,p in zip(pvals_error, pvals_all)]
    #print("shape(pvals_error): " + str(np.shape(pvals_error)))

    colours= ["navy", "magenta", "cyan", "orange", "mediumseagreen", "black", "blue", "red"]


    ### plot p-values
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)

    nsims = [10000.0, 10000.0, 1075000.0 , 10000.0]

    for ts, n, pv, c, in zip(tseg, nsims, pvals_all,colours[:len(pvals_all)]):
        pvals = np.array(pv)

        pvals_noul, pvals_ul = [], []

        cycles = list(np.arange(len(pvals)))
        cycles_ul, cycles_noul = [], []
        for i,p in enumerate(pvals):
            if p < 1.0/n:
                pvals_ul.append(1.0/n)
                pvals[i] = 1.0/n
                cycles_ul.append(i)
            else:
                pvals_noul.append(p)
                cycles_noul.append(i)

        pvals = np.log10(np.array(pvals))
        pvals_noul = np.log10(np.array(pvals_noul))
        pvals_ul = np.log10(np.array(pvals_ul))

        cycles = np.array(cycles)
        cycles_noul = np.array(cycles_noul)
        cycles_ul = np.array(cycles_ul)


        pvals_error = pvalues_error(pvals_noul, n)
        log_errors = 0.434*np.array(pvals_error)/np.array(pvals_noul)


        ax.scatter(cycles+1, pvals, marker="o", s=42, color=c, edgecolor="black")
        ax.plot(cycles+1, pvals, lw=3, color=c)
        errorbar(cycles_noul+1, pvals_noul, yerr=log_errors, fmt="o", lw=0, color=c, markersize=10,
                 label=r"$t_{\mathrm{seg}} = %.1f \, \mathrm{s}$"%ts)

        for cyc,p in zip(cycles_ul, pvals_ul):
            ar1 = FancyArrow(cyc+1, p, 0.0, -1.0, length_includes_head=True,
                color=c, head_width=0.2, head_length=0.3, width=0.001, linewidth=2)
            ax.add_artist(ar1)

    xlabel("Number of averaged cycles", fontsize=20)
    ylabel(r"$\log_{10}{(\mathrm{p-value\; of\; maximum\; power})}$", fontsize=20)
    legend(loc="upper right", prop={"size":16})
    axis([0,20,-7.1, 0.5])
    title("SGR 1806-20, RHESSI data, p-values ")
    savefig("f7.pdf", format="pdf")
    close()

    return



def plot_rhessi_qpo_simulations(tnew=None, tseg_all=[0.5,1.0,2.0,2.5], df_all = [2.0, 1.0, 1.0, 1.0, 1.0],
                                freq=[627.0, 627.0, 626.5,626.0]):

    if tnew is None:
        tnew = load_rhessi_data()



    for f,namestr in enumerate(["allcycles", "singlecycle"]):

        fig = figure(figsize=(24,18))
        subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.1, hspace=0.2)



        ### loop over all values of the segment lengths and frequency resolutions
        for k,(tseg, df, fr) in enumerate(zip(tseg_all, df_all, freq)):
            ### extract maximum powers from data.
            ### Note: The RHESSI QPO is at slightly higher frequency, thus using 626.0 Hz
            lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
                giantflare.search_singlepulse(tnew, nsteps=30, tseg=tseg, df=df, fnyquist=1000.0, stack=None,
                                              setlc=True, freq=fr)


            ### make averaged powers for consecutive cycles, up to 19, for each of the nsteps segments per cycle:
            allstack = giantflare.make_stacks(savg, 19, 30)


            maxp_all = np.loadtxt("sgr1806_rhessi_tseg=%.1f_simulated_maxpowers.txt"%(tseg))

            pvals = np.loadtxt("sgr1806_rhessi_%s_tseg=%.1f_pvals.txt"%(namestr, tseg))
            print("pvals file: sgr1806_rhessi_%s_tseg=%.1f_pvals.txt"%(namestr, tseg))

            pvals_data, pv_data_corr, pvals_hist = [], [], []
            len_sims_all = []
            for i in xrange(len(allstack)):

                    ### these are the simulations WITHOUT QPO
                sims = maxp_all[:,i]
                sims = np.sort(sims)
                len_sims = np.float(len(sims))
                print("Using %i simulations."%len_sims)
                #print("len_sims %i" %len_sims)
                #print("sims[0:10] " + str(sims[0:10]))
                #print("sims[-10:] " + str(sims[-10:]))


                ind_data = np.float(sims.searchsorted(np.max(allstack[i])))
                pd = (len_sims-ind_data)/len_sims
                len_sims_all.append(len_sims)
                #if pd == 0.0:
                #    pd = 1.0/len_sims

                pvals_data.append(pd)

                print("pd: " + str(pd))

                if pd == 0:
                    pv_data_corr.append(1.0/len_sims)
                else:
                    pv_data_corr.append(pd)

                h, bins = np.histogram(np.log10(pvals[i]), bins=10, range=[-4.1, 0.0])
                pvals_hist.append(h[::-1])

            #print("shape(pvals): " + str(np.shape(pvals)))
            #print("pvals_data for tseg = %.1f: "%(tseg) + str(pvals_data))

            ax = fig.add_subplot(2,2,k+1)
            znew = scipy.ndimage.gaussian_filter(np.transpose(pvals_hist), sigma=0.5, order=0)
            ax.imshow(znew, cmap=cm.hot, extent=[0.0,len(allstack), -7.1,0.0], interpolation="bicubic")
            ax.set_aspect(3)

            print('len(pvals_data): ' + str(len(pvals_data)))
            print("log10(pvals_data): " + str(np.log10(pvals_data)))
            scatter(np.arange(19)+0.5, np.log10(pv_data_corr), lw=2, facecolor="LightGoldenRodYellow",
                    edgecolor="cyan", marker="v", s=50)
            for i,pd in enumerate(pvals_data):
                print("pd: " + str(pd))
                if pd == 0.0:
                    ar1 = FancyArrow(i+0.5, np.log10(1.0/len_sims), 0.0, -1.0, length_includes_head=True,
                                     color="cyan", head_width=0.2,
                                     head_length=0.3, width=0.001, linewidth=2)
                    ax.add_artist(ar1)


            axis([0,len(allstack), -7.1, 0.0])
            xlabel("Number of averaged cycles", fontsize=24)
            ylabel(r"$\log_{10}{(\mathrm{p-value})}$", fontsize=24)
            title(r"simulated p-values, $t_{\mathrm{seg}} = %.1f$"%tseg)

        draw()
        tight_layout()
        savefig("f%i.pdf"%(f+8), format="pdf")
        close()

    return





def main():



    return


if __name__ == "__main__":





    main()
