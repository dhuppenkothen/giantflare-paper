

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
import glob

import lightcurve
import giantflare

from pylab import *
import matplotlib.cm as cm
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

    """
    Makes Figure 2 of the paper.
    Uses a file with simulations from make_rxte_sims(). If you don't want to use the provided file, make it yourself,
    but be warned that it takes a while!

    """


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
        giantflare.search_singlepulse(tnew, nsteps=10, tseg=3.0, df=2.66, fnyquist=2000.0, stack=None,
                                      setlc=True, freq=625.0)

    ### stack up periodograms at the same phase at consecutive cycles, up to averaging 9 cycles:
    allstack = giantflare.make_stacks(savg, 10, 15)


    ### load powers at 625 Hz from 40000 simulations with the QPO smoothed out:
    ### if file doesn't exist, load with function make_rxte_sims() below, but be warned that it takes a while
    ### (like a day or so) to run!
    savgall_sims = np.loadtxt()

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



def make_rxte_sims(tnew=None, nsims=30000,save=True, fout="1806_rxte_tseg=3.0_df=2.66_dt=0.5_f=625Hz_savgall.dat"):

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
 
#    savgall = giantflare.simulations(tnew, nsims=nsims, tcoarse = 0.01, tfine =0.5/1000.0, freq=624.0, nsteps=10,
#                                     tseg=3.0, df = 2.66, fnyquist=1000.0, stack=None, setlc=False, set_analysis=True,
#                                     maxstack=9, qpo=False)

    if save:
        #f = open(fout, "w")
        #pickle.dump(savgall, f)
        #f.close()
	np.savetxt(fout, savgall)

    return savgall


def stitch_savgall_together(froot="test"):
    """
    Take multiple identical runs of make_rxte_sims() and stitch them together.
    """
    savgfiles = glob.glob("%s*"%froot)

    savgall = []
    for f in savgfiles:
        savgtemp = np.loadtxt(f)

        savgall.extend(savgtemp)

    return savgall


def rxte_simulations_results(tnew=None, froot_in="1806_rxte_tseg=3.0_df=2.66_nsteps=10_f=625Hz_final",
                             froot_out="sgr1806_rxte", plotdist=True):
    """
    Take several simulation runs made with make_rxte_sims() and read them out one after the other,
    to avoid memory problems when running make_stacks for very large runs.

    NOTE: This requires simulation files run with the EXACT SAME parameters as the ones used for running
    giantflare.search_singlepulse() on the data below. You'd better make sure this is true. If not,
    the comparison will not be correct.

    """

    ### if tnew isn't given, read in:
    tnew = load_rxte_data()


    ### extract maximum powers from data.
    ### NOTE: I use 624.0 Hz here as the search frequency, because np.searchsorted finds the first element
    ### in a list *after* the given one. Because of the way I've done the binning, the highest power is actually
    ### at 624.5 Hz rather than 625 Hz. So there.
    lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(tnew, nsteps=10, tseg=3.0, df=2.66, fnyquist=1000.0, stack=None,
                                      setlc=True, freq=624.0)


    ### make averaged powers for consecutive cycles, up to 10, for each of the nsteps segments per cycle:
    allstack = giantflare.make_stacks(savg, 10, 10)


    ### find all datafiles with string froot_in in their filename
    savgfiles = glob.glob("%s*"%froot_in)

    maxp_all = []
    for f in savgfiles:

        ### load simulation output
        savgtemp = np.loadtxt(f)
        print("shape(savgtemp): " + str(np.shape(savgtemp)))

        ### make averaged powers for each of the 10 cycles
        allstack_temp = []
        for s in savgtemp:
            allstack_temp.append(giantflare.make_stacks(s, 10, 10))

        maxp_temp = []
        for i in xrange(len(allstack)):
            amax = np.array([np.max(a[i]) for a in allstack_temp])
            maxp_temp.append(amax)

        maxp_temp = np.transpose(np.array(maxp_temp))
        maxp_all.extend(maxp_temp)

    maxp_all = np.array(maxp_all)
    print("shape(maxp_all) " + str(np.shape(maxp_all)))

    np.savetxt("%s_simulated_maxpowers.txt"%froot_out, maxp_all)

    pvals = []
    for i,a in enumerate(allstack):
        sims = maxp_all[:,i]
        print("sims " + str(sims))

        sims_sort = np.sort(sims)
        len_sims = np.float(len(sims_sort))
        ind_sims = sims_sort.searchsorted(max(a))

        pvals.append((len_sims-ind_sims)/len(sims))


        ### plot distributions of maximum powers against theoretical expectations?
        if plotdist:

            ### simulated chi-square powers:
            ### degree of freedom is 2*nbins*ncycles, i.e. 2*no of avg frequency bins*no of avg cycles
            ### for df = 2.66, nbins=8
            ### ncycles = i+1 (i.e. index of cycle +1, because index arrays start with zero instead of 1)
            ### size of output simulations is (n_simulations, n_averagedpowers)
            chisquare = np.random.chisquare(2*8*(i+1), size=(len_sims,len(a)))/(8.0*(i+1))
            maxc = np.array([np.max(c) for c in chisquare])

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
            savefig("%s_maxdist_ncycle%i.png"%(froot_out, (i+1)))
            close()

    pvals = np.array(pvals)

    np.savetxt("%s_pvals_all.txt"%froot_out, pvals)

    ### Compute theoretical error on p-values
    pvals_error = pvalues_error(pvals, len(sims))

    ### plot p-values
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    #plot(np.arange(len(pvals))+1, pvals,"-o", lw=3, color="black", markersize=12)
    errorbar(np.arange(len(pvals))+1, pvals, yerr=pvals_error, fmt="-o", lw=3, color="black", markersize=12)
    xlabel("Number of averaged cycles", fontsize=20)
    ylabel("P-value of maximum power", fontsize=20)
    title("SGR 1806-20, RXTE data, p-value from %i simulations"%len_sims)
    savefig("%s_pvals.png"%froot_out, format="png")
    close()

    return pvals

def periodogram_nosignal(froot_in="1806_rxte_tseg=3.0_df=2.66_nsteps=10_f=625Hz_final", froot_out="sgr1806_rxte"):
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


def rxte_qpo_sims_singlecycle(nsims=100, froot="sgr1806_rxte"):

    finesteps = int((1.0/0.132)/0.1)
    coarsesteps = 10

    tnew = load_rxte_data()

    tstart_all = []
    amp_all = [3]

    freq_all = [626.5 for t in tstart_all]
    randomphase_all = [True for t in tstart_all]

    length_all = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]

    nqpo = len(tstart_all)
    qpoparams = [{"freq":f, "amp":a, "tstart":t, "randomphase":r, "length":l} for f,a,t,r,l in \
        zip(freq_all, amp_all, tstart_all, randomphase_all, length_all)]

    lcsimall, savgall, pssimall = giantflare.make_qpo_simulations(tnew, nqpo, qpoparams, nsims=nsims, tcoarse=0.01,
                                              tfine=0.5/1000.0, freq=624.0, set_analysis=False, set_lc=True)

    savgall_coarsesteps, savgall_finesteps = [], []




    for lc in lcsimall:
        lcall, psall, mid_coarse, savg_coarse, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(tnew, nsteps=coarsesteps, tseg=3.0, df=2.66, fnyquist=1000.0, stack=None,
                                          setlc=True, freq=624.0)

        lcall, psall, mid_fine, savg_file, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(tnew, nsteps=finesteps, tseg=3.0, df=2.66, fnyquist=1000.0, stack=None,
                                          setlc=True, freq=624.0)



    return


def rxte_highres():

    ### load RXTE data
    tnew = load_rxte_data()

    dt = 0.1
    nsteps = int((1.0/0.132)/dt)

    lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = giantflare.search_singlepulse(tnew, nsteps=nsteps,tseg=3.0,
                                                                                            df=2.66, fnyquist=1000.0,
                                                                                            stack=None, setlc=True,
                                                                                            freq=624.0)

    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)

    colours= ["black", "red", "cyan", "orange", "mediumseagreen", "magenta", "blue", "grey", "yellow"]
    for i in range(9):
        plot(mid[i*nsteps+int(nsteps/2):(i+1)*nsteps+int(nsteps/2)]-mid[i*nsteps+int(nsteps/2)],
             savg[i*nsteps+int(nsteps/2):(i+1)*nsteps+int(nsteps/2)], lw=3, linestyle="steps-mid",
             color=colours[i],label="cycle %i"%i)

        plt.hlines(2.0, 0.0,7.575757, lw=7, color="black", linestyle="dashed", label="average noise level")
        axis([0,1.0/0.132, 0, 12])
        legend(prop={"size":16})
        xlabel("Phase (in periods)", fontsize=18)
        ylabel("Averaged Leahy Power", fontsize=18)

        title("Leahy Power versus rotational phase for all nine cycles")
        savefig("sgr1806_rxte_phaseplot.png", format="png")
        close()


    minind = tnew.searchsorted(235.4792253970644)
    maxind = tnew.searchsorted(235.4792253970644+2*1.0/0.132)

    tnew_small = tnew[minind:maxind]

    dt = 1.0/624.5
    nsteps = (1.0/0.132)/dt

    lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(tnew_small, nsteps=nsteps,tseg=0.5, df=2.00, fnyquist=1000.0, stack=None,
                                      setlc=True, freq=624.0)

    return mid, savg

########################################################################################################################
######## SECOND BIT: RHESSI ANALYSIS ###################################################################################
########################################################################################################################

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



def make_rhessi_sims(tnew=None, tseg_all=None, df_all=None, nsims=30000,save=True, froot="1806_rhessi_test"):

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
        tseg_all = [0.5, 1.0, 1.5, 2.0, 3.0]

    if df_all is None:
        ## corresponding dfs: because the QPO is narrow, I use a frequency resolution of 1 Hz for all but the shortest
        ## segments, where I can't, because the native frequency resolution is 2 Hz already.
        df_all = [2.0, 1.0, 1.0, 1.0, 1.0]


    for tseg,df in zip(tseg_all, df_all):
        savgall = giantflare.rxte_simulations(tnew, nsims=nsims, tcoarse=0.01, tfine=0.5/1000.0, freq=626.0, nsteps=30,
					  tseg=tseg, df=df, set_analysis=True, set_lc = False)

#    savgall = giantflare.simulations(tnew, nsims=nsims, tcoarse = 0.01, tfine =0.5/1000.0, freq=624.0, nsteps=10,
#                                     tseg=3.0, df = 2.66, fnyquist=1000.0, stack=None, setlc=False, set_analysis=True,
#                                     maxstack=9, qpo=False)

        if save:
            np.savetxt("%s_tseg=%.2f_df=%.2f_savgall.txt"%(froot, tseg, df), savgall)

    return savgall

def rhessi_simulations_results(tnew=None, tseg_all=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], df_all=[2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               froot_in="1806_rhessi", froot_out="test", plotdist=True):
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

    ### loop over all values of the segment lengths and frequency resolutions
    for tseg, df in zip(tseg_all, df_all):
    ### extract maximum powers from data.
    ### Note: The RHESSI QPO is at slightly higher frequency, thus using 626.0 Hz
        lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(tnew, nsteps=30, tseg=tseg, df=df, fnyquist=1000.0, stack=None,
                                      setlc=True, freq=626.0)


        savg_data.append(savg)
        ### make averaged powers for consecutive cycles, up to 19, for each of the nsteps segments per cycle:
        allstack = giantflare.make_stacks(savg, 19, 30)

        allstack_data.append(allstack)

        ### find all datafiles with string froot_in in their filename
        print("%s*_tseg=%.1f*"%(froot_in,tseg))
        savgfiles = glob.glob("%s*_tseg=%.1f*"%(froot_in,tseg))

        print("Simulation files: " + str(savgfiles))

        maxp_all = []
        for f in savgfiles:

            ### load simulation output
            savgtemp = np.loadtxt(f)
            print("shape(savgtemp): " + str(np.shape(savgtemp)))

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
        print("shape(maxp_all) " + str(np.shape(maxp_all)))

        np.savetxt("%s_tseg=%.1f_simulated_maxpowers.txt"%(froot_out, tseg), maxp_all)

        pvals = []
        for i,a in enumerate(allstack):
            sims = maxp_all[:,i]
            #print("sims " + str(sims))

            sims_sort = np.sort(sims)
            len_sims = np.float(len(sims_sort))
            ind_sims = sims_sort.searchsorted(max(a))

            pvals.append((len_sims-ind_sims)/len(sims))


            ### plot distributions of maximum powers against theoretical expectations?
            if plotdist:

                ### simulated chi-square powers:
                ### degree of freedom is 2*nbins*ncycles, i.e. 2*no of avg frequency bins*no of avg cycles
                ### for df = 2.66, nbins=8
                ### ncycles = i+1 (i.e. index of cycle +1, because index arrays start with zero instead of 1)
                ### size of output simulations is (n_simulations, n_averagedpowers)
                chisquare = np.random.chisquare(2*8*(i+1), size=(len_sims,len(a)))/(8.0*(i+1))
                maxc = np.array([np.max(c) for c in chisquare])

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

        ### Compute theoretical error on p-values
        pvals_error = pvalues_error(pvals, len(sims))
        perr_all.append(pvals_error)

    np.savetxt("%s_pvals_all.txt"%froot_out, pvals_all)


    colours= ["navy", "magenta", "cyan", "orange", "mediumseagreen", "black", "blue", "red"]

    ### plot p-values
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)

    for pv, pe, c, in zip(pvals_all, perr_all, colours[:len(pvals_all)]):
    #plot(np.arange(len(pvals))+1, pvals,"-o", lw=3, color="black", markersize=12)
        errorbar(np.arange(len(pv))+1, pv, yerr=pe, fmt="-o", lw=3, color=c, markersize=12,
                 label=r"$t_{\mathrm{seg}} = %.1f \, \mathrm{s}$"%tseg)
    xlabel("Number of averaged cycles", fontsize=20)
    ylabel("P-value of maximum power", fontsize=20)
    title("SGR 1806-20, RHESSI data, p-values from %i simulations"%len_sims)
    savefig("%s_pvals.png"%froot_out, format="png")
    close()

    return pvals_all



def rhessi_qpo_sims_allcycles(nsims=1000, froot="1806_rhessi"):
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

def rhessi_qpo_sims_allcycles_randomised(nsims=1000, froot="1806_rhessi"):

    """
     Same as rhessi_qpo_sims_allcycles above, but randomphase = True for all, and amplitude slightly different
    """

    ### first batch of simulations: constant signal in all cycles
    tstart_all = [  84.80569267,   92.38199711,   99.95283222,  107.52754498,
                    115.13365078,  122.68631363,  130.26195049,  137.8398037 ,
                    145.41058636,  152.98174858,  160.5617857 ,  168.1408844 ,
                    175.70906353,  183.28972721,  190.86196136,  198.43483257,
                    206.01998615,  213.58804893,  221.17355442]

    amp_all = [0.1 for t in tstart_all]
    freq_all = [626.5 for t in tstart_all]
    randomphase_all = [True for t in tstart_all]
    length_all = [2.0 for t in tstart_all]

    nqpo = len(tstart_all)
    qpoparams_all = [{"freq":f, "amp":a, "tstart":t, "randomphase":r, "length":l} for f,a,t,r,l in \
        zip(freq_all, amp_all, tstart_all, randomphase_all, length_all)]

    make_rhessi_qpo_sims(nqpo, qpoparams_all, nsims=nsims, froot="%s_allcycles_randomised"%froot)

    return

def rhessi_qpo_sims_singlecycle(nsims=1000, froot="1806_rhessi"):

    tstart_all = [99.95283222, 107.52754498, 190.86196136,  198.43483257, 206.01998615,  213.58804893,  221.17355442]
    amp_all = [0.2, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]

    freq_all = [626.5 for t in tstart_all]
    randomphase_all = [True for t in tstart_all]

    length_all = [0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0]

    nqpo = len(tstart_all)
    qpoparams_all = [{"freq":f, "amp":a, "tstart":t, "randomphase":r, "length":l} for f,a,t,r,l in \
        zip(freq_all, amp_all, tstart_all, randomphase_all, length_all)]

    make_rhessi_qpo_sims(nqpo, qpoparams_all, nsims=nsims, froot="%s_singlecycle"%froot)

    return


def make_rhessi_qpo_sims(nqpo, qpoparams, nsims=1000, froot="1806_rhessi_test"):
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
                                      setlc=True, freq=626.0)

        savgall_05.append(savg_05)

        lcall, psall, mid, savg_1, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(lc, nsteps=30, tseg=1.0, df=1.00, fnyquist=nsims, stack=None,
                                      setlc=True, freq=626.0)

        savgall_1.append(savg_1)

        lcall, psall, mid, savg_15, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(lc, nsteps=30, tseg=1.5, df=1.00, fnyquist=nsims, stack=None,
                                      setlc=True, freq=626.0)

        savgall_15.append(savg_15)

        lcall, psall, mid, savg_2, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(lc, nsteps=30, tseg=2.0, df=1.00, fnyquist=nsims, stack=None,
                                      setlc=True, freq=626.0)

        savgall_2.append(savg_2)

        lcall, psall, mid, savg_25, xerr, ntrials, sfreqs, spowers = \
        giantflare.search_singlepulse(lc, nsteps=30, tseg=3.0, df=1.00, fnyquist=nsims, stack=None,
                                      setlc=True, freq=626.0)
        savgall_25.append(savg_25)


    ### save results to file
    np.savetxt("%s_tseg=0.5_savgall.txt"%froot, savgall_05)
    np.savetxt("%s_tseg=1.0_savgall.txt"%froot, savgall_1)
    np.savetxt("%s_tseg=1.5_savgall.txt"%froot, savgall_15)
    np.savetxt("%s_tseg=2.0_savgall.txt"%froot, savgall_2)
    np.savetxt("%s_tseg=2.5_savgall.txt"%froot, savgall_25)


    return


def rhessi_qpo_sims_images(tseg_all=[0.5,1.0,2.0,2.5], df_all=[2.0, 1.0, 1.0, 1.0], nbins=30, froot_in="sgr1806_rhessi",
                           froot_sims="allcycles"):

    ### if tnew isn't given, read in:
    tnew = load_rhessi_data()

    savg_data, allstack_data = [], []

    pvals_all, perr_all = [], []

    print("froot_in: %s" %froot_in)

    fig = figure(figsize=(24,18))
    subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.1, hspace=0.2)


    ### loop over all values of the segment lengths and frequency resolutions
    for k,(tseg, df) in enumerate(zip(tseg_all, df_all)):
        ### extract maximum powers from data.
        ### Note: The RHESSI QPO is at slightly higher frequency, thus using 626.0 Hz
        lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
            giantflare.search_singlepulse(tnew, nsteps=30, tseg=tseg, df=df, fnyquist=1000.0, stack=None,
                                      setlc=True, freq=626.0)


        savg_data.append(savg)
        ### make averaged powers for consecutive cycles, up to 19, for each of the nsteps segments per cycle:
        allstack = giantflare.make_stacks(savg, 19, 30)

        allstack_data.append(allstack)

        maxp_all = np.loadtxt("%s_tseg=%.1f_simulated_maxpowers.txt"%(froot_in, tseg))

        ### sort powers for each ncycle from smallest to highest, such that I can simply use searchsorted to find
        ### the index of the power that corresponds to the observed one to compute p-value
        #maxp_sorted = np.array([np.sort(maxp_all[:,i]) for i in xrange(len(allstack))])
        ### transpose maxp, such that it's of the shape [nsims, ncycles]
        #maxp_sorted = np.transpose(maxp_all)

        print("shape(maxp_sorted) " + str(np.shape(maxp_all)))

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

                ### these are the simulations WITHOUT QPO
            sims = maxp_all[:,i]
            sims = np.sort(sims)
            len_sims = np.float(len(sims))
            print("len_sims %i" %len_sims)
            print("sims[0:10] " + str(sims[0:10]))
            print("sims[-10:] " + str(sims[-10:]))


            ind_data = np.float(sims.searchsorted(np.max(allstack[i])))
            pvals_data.append((len_sims-ind_data)/len_sims)

            ### find index in sorted simulations without QPO that correspond to the observed maximum power
            ### for the simulated light curve with QPO
            ind_temp = np.array([np.float(sims.searchsorted(np.max(a))) for a in allstack_qpo[:,i]])
            pvals_temp = (len_sims-ind_temp)/len_sims

            pvals.append(pvals_temp)


            h, bins = np.histogram(np.log10(pvals_temp), bins=nbins, range=[-5.0, 0.0])
            pvals_hist.append(h[::-1])

        print("shape(pvals): " + str(np.shape(pvals)))
        print("pvals_data for tseg = %.1f: "%(tseg) + str(pvals_data))

        pvals_all.append(pvals)
        pvals_hist_all.append(pvals_hist)

        #pvals_all = np.array(pvals_all)
        #print("shape pvals_all: " + str(np.shape(pvals)))
        #print("froot_in %s" %froot_in)
        #print("froot_sims: %s" %froot_sims)
        #print("tseg: %f" %tseg)

        np.savetxt("%s_%s_tseg=%.1f"%(froot_in, froot_sims, tseg), pvals)

        ax = fig.add_subplot(2,2,k+1)
        ax.imshow(np.transpose(pvals_hist), cmap=cm.hot, extent=[0,len(allstack), -4.1,0.0])
        ax.set_aspect(3)
        print('len(pvals_data): ' + str(len(pvals_data)))
        scatter(np.arange(19)+0.5, np.log10(pvals_data), lw=1, facecolor="LightGoldenRodYellow",
                edgecolor="cyan", marker="v")
        axis([0,len(allstack), -4.1, 0.0])
        xlabel("Number of averaged cycles", fontsize=20)
        ylabel(r"$\log_{10}{(\mathrm{p-value})}$", fontsize=20)
        title(r"simulated p-values, $t_{\mathrm{seg}} = %.1f$"%tseg)

    savefig("%s_%s_pvals_sims.png"%(froot_in, froot_sims), format="png")
    close()

    return pvals_all, pvals_hist_all

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


def plot_rhessi_pvalues(filename="sgr1806_rhessi_pvals_all.txt", tseg=[0.5,1.0,1.5,2.0,2.5],
                       nsims=10000):

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

    pvals_error = [pvalues_error(pvals, nsims) for pvals in pvals_all]
    print("shape(pvals_error): " + str(np.shape(pvals_error)))

    log_errors = [0.434*dp/p for dp,p in zip(pvals_error, pvals_all)]
    print("shape(pvals_error): " + str(np.shape(pvals_error)))

    colours= ["navy", "magenta", "cyan", "orange", "mediumseagreen", "black", "blue", "red"]


    ### plot p-values
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)

    for ts, pv, pe, c, in zip(tseg, pvals_all, log_errors, colours[:len(pvals_all)]):
    #plot(np.arange(len(pvals))+1, pvals,"-o", lw=3, color="black", markersize=12)
        errorbar(np.arange(len(pv))+1, np.log10(pv), yerr=pe, fmt="-o", lw=3, color=c, markersize=12,
                 label=r"$t_{\mathrm{seg}} = %.1f \, \mathrm{s}$"%ts)
    xlabel("Number of averaged cycles", fontsize=20)
    ylabel(r"$\log_{10}{(\mathrm{p-value\; of\; maximum\; power})}$", fontsize=20)
    legend(loc="upper right", prop={"size":16})
    axis([0,20,-4.2, 0.5])
    title("SGR 1806-20, RHESSI data, p-values from %i simulations"%nsims)
    savefig("f4.eps", format="eps")
    close()

    return



def main():



    return


if __name__ == "__main__":





    main()