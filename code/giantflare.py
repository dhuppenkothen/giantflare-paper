
import numpy as np
import matplotlib.pyplot as plt

import utils
import lightcurve
import powerspectrum
import scipy.interpolate






def search_singlepulse(time, nsteps=7, tseg=0.2, df = 2.66, freq = 627.0, fnyquist=4096.0, stack='all', setlc=False, norm='leahy',
                       period = 1.0/0.132):

    """
    This function takes a light curve (time) or list of photon arrival times (time), chops it up in nsteps segments
    per rotational cycle of the neutron star, each of length tseg, and computes periodograms of each segments
    with frequency resolution of df up to a Nyquist frequency of fnyquist.
    If setlc == True, return light curves and power spectra of the segments as well as the powers.

    It's a bit of a crappy piece of a hack, but it works.

    Some notes:
    - the power extracted is that of the first frequency bin *after* the number defined in freq (look up
    numpy.searchsorted for details of the exact procedure)
    - at the moment, only integer multiples of the frequency resolution are possible (because non-integer binning
    screws with the distributions of powers), such that the actual bin frequency won't always be that defined in df.
    Sorry about that!



    """

    ### check whether input is an object of type Lightcurve, or a bunch of photon arrival times
    if isinstance(time, lightcurve.Lightcurve):
        duration = time.time[-1] - time.time[0]
        tmax = time.time[0] + tseg
        tmin = time.time[0]

        tend = time.time[-1]


    else:
        duration = time[-1] - time[0]
        tmax = time[0]+ tseg
        tmin = time[0]

        tend = time[-1]

    lcall = []
    psall = []
    maxpows = []
    nfreq, bnfreq = [], []

    ## 1/period of the neutron star
    f_neutron = 1.0/period
    p = 1.0/f_neutron

    ### distance between start times of individual segments
    dt = p/float(nsteps)

    ## frequency to be extracted
    f = freq
    ## corresponding period
    period = 1.0/f
    dtsmall = 5.0*period
    sfreqs, spowers = [], []
    maxpows_mean, maxpows_err = [], []


    i = 1

    ### make a while loop, run through light curve until the end time of each segment tmax is larger than the
    ### end of the light curve tend
    while tmax < tend:

        if isinstance(time, lightcurve.Lightcurve):
            minind = np.array(time.time).searchsorted(tmin)
            maxind = np.array(time.time).searchsorted(tmax)

            lc = lightcurve.Lightcurve(time.time[minind:maxind], counts=time.counts[minind:maxind], tseg = tseg)

        else:

            minind = time.searchsorted(tmin)
            maxind = time.searchsorted(tmax)

            tnew = time[minind:maxind]
            lc = lightcurve.Lightcurve(tnew, timestep=0.5/fnyquist, tseg=tseg)

        if setlc: 
            lcall.append(lc)
        nps = int(tseg/dtsmall)



        ps = powerspectrum.PowerSpectrum(lc, norm=norm)

        ### There seems to be a bug in rebinps, which makes the binned powers every so slightly non-Chisquare
        ### distributed! I'm only binning multiple integers of the original frequency resolution
        if df > 1.1*ps.df:
            binps = powerspectrum.PowerSpectrum()
            n = int(df/ps.df)
            binfreq, binpowers = utils.rebin_lightcurve(ps.freq, ps.ps, n=n, type="average")
            binps.freq = binfreq
            binps.ps = binpowers
            binps.df = binfreq[1] - binfreq[0]
            binps.nphots = binpowers[0]
            binps.n = 2*len(binfreq)
            #binps = ps.rebinps(df)
        else:
            binps = ps

        if setlc:
            psall.append(binps)

        mpind = np.searchsorted(np.array(binps.freq), f)

        maxpow = binps.ps[mpind]
        sfreqs.append(binps.freq[mpind-2:mpind+3])
        spowers.append(binps.ps[mpind-2:mpind+3])
        maxpows.append(maxpow)
        nfreq.append(len(ps.freq))
        bnfreq.append(len(binps.freq))
        ntrials = [n/b for n,b in zip(nfreq, bnfreq)]
        ntrials = np.mean(ntrials)

        tmin = tmin+dt
        tmax = tmax+dt
        i = i + 1

    sfreqs = np.array(sfreqs)
    sfreqs = sfreqs.transpose()

    spowers = np.array(spowers)
    spowers = spowers.transpose()

    savg = maxpows
    mid = np.array([l.time[0] + l.tseg/2.0 for l in lcall])
    xerr = np.array([l.tseg/2.0 for l in lcall])


    if stack == None:
            print("I am here in search_singlepulse")
            if setlc:
                return lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers #maxpows_mean, maxpows_err, xerr, 

            else:
                return mid, savg, xerr, ntrials

    if stack == "all":
            stack = np.zeros((nsteps, len(psall[0].freq)))
            i = 0
            cycles = len(psall)/nsteps
            for n in range(nsteps):
                stack[n,:] = np.sum([psall[i].ps for i in np.arange(cycles)*nsteps+n], axis=0)/float(cycles)

            return lcall, psall, mid, savg, xerr, ntrials, stack

    if type(stack) == int or type(stack) == float:
        allstack = []
        allstackmaxpows = []
        cycles = int(np.ceil((len(psall))/float(nsteps)))-1
        for n in range(nsteps):
             stacktemp = []
             stacktempmaxpows = []
             for c in range(cycles - (stack-1)):
                 stacktemp.append(np.sum([psall[n+(c+i)*nsteps].ps for i in range(stack)], axis=0))
                 stacktempmaxpows.append(np.sum([savg[n+(c+i)*nsteps] for i in range(stack)])/float(stack))
             allstackmaxpows.append(stacktempmaxpows)
             allstack.append(stacktemp)
        allstackmaxpows = np.array(allstackmaxpows)
        nasm = allstackmaxpows.transpose()
        nasm = nasm.flatten()


        return lcall, psall, mid, savg, xerr, ntrials, allstack, nasm



def rxte_simulations(time, nsims=1000, tcoarse = 0.1, tfine=0.5/1000.0, freq=625.0, nsteps=10, tseg=3.0,
                     df=2.66, set_analysis=True, set_lc =False):

    """
     Much simpler version of function simulations() below, for RXTE data, without the fancy inclusion
     of QPOs and all that jazz. Straight-up simulations of giant flare light curves from the smoothed out
     version of the real data.
    """

    ### coarse light curve
    lccoarse = lightcurve.Lightcurve(time, timestep=tcoarse)
    ### use interpolation routine in scipy to interpolate between data points, use linear interpolation
    ### should be non-linear interpolation?
    interpolated_countrate = scipy.interpolate.interp1d(lccoarse.time, lccoarse.countrate, "linear")

    ### make light curve with right time resolution for analysis
    lcnew = lightcurve.Lightcurve(time, timestep=tfine)

    ### scipy.interpolate cannot interpolate outside the data points, so we loose about 0.5*tcoarse at start
    ### and end of light curve, adjust fine light curve such I can use the output of interp1d to make the
    ### smoothed-out light curve
    minind = lcnew.time.searchsorted(lccoarse.time[0])
    maxind = lcnew.time.searchsorted(lccoarse.time[-1])

    lcnew.time = lcnew.time[minind:maxind]
    lcnew.counts = lcnew.counts[minind:maxind]
    lcnew.countrate = lcnew.countrate[minind:maxind]

    ### interpolate light curve to right resolution
    countrate_new = interpolated_countrate(lcnew.time)

    ### go back to counts by multiplying by time resolution
    counts_new = countrate_new*lcnew.res

    lcsimall, savgall, pssimall = [], [], []

    for n in xrange(nsims):
        print("I am on simulation %i" %n)
        cpoisson = np.array([np.random.poisson(cts) for cts in counts_new])
        lcsim = lightcurve.Lightcurve(lcnew.time, counts=cpoisson)

        lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
            search_singlepulse(lcsim, nsteps=nsteps, tseg=tseg, df=df, fnyquist=0.5/tfine, stack=None,
                               setlc=True, freq=freq)

        if set_lc:
            lcsimall.append(lcsim)
            pssimall.append(psall)

        savgall.append(savg)

    if set_lc:
        return lcsimall, np.array(savgall), pssimall
    else:
        return np.array(savgall)



def make_qpo_simulations(time, nqpo, qpoparams, nsims=1000, tcoarse = 0.1, tfine=0.5/1000.0, freq=625.0, nsteps=10,
                         tseg=3.0, df=2.66, set_analysis=True, set_lc =False):

    """
    Do essentially the same as the function above, except with a periodic signal added.
    The number in nqpo needs to match the length of the list qpoparams.
    qpoparams should be a dictionary with the following keywords:
        - freq: frequency parameter, should be a positive float
        - amp: fractional rms amplitude of the signal
        - tstart: start time in seconds, in the same time framework as the data
        - length: duration in seconds of the QPO signal
        - randomphase: bool: if True, randomise the start phase

    """

    ### coarse light curve
    lccoarse = lightcurve.Lightcurve(time, timestep=tcoarse)
    ### use interpolation routine in scipy to interpolate between data points, use linear interpolation
    ### should be non-linear interpolation?
    interpolated_countrate = scipy.interpolate.interp1d(lccoarse.time, lccoarse.countrate, "linear")

    ### make light curve with right time resolution for analysis
    lcnew = lightcurve.Lightcurve(time, timestep=tfine)

    ### scipy.interpolate cannot interpolate outside the data points, so we loose about 0.5*tcoarse at start
    ### and end of light curve, adjust fine light curve such I can use the output of interp1d to make the
    ### smoothed-out light curve
    minind = lcnew.time.searchsorted(lccoarse.time[0])
    maxind = lcnew.time.searchsorted(lccoarse.time[-1])

    lcnew.time = lcnew.time[minind:maxind]
    lcnew.counts = lcnew.counts[minind:maxind]
    lcnew.countrate = lcnew.countrate[minind:maxind]

    ### interpolate light curve to right resolution
    countrate_new = interpolated_countrate(lcnew.time)

    ### go back to counts by multiplying by time resolution
    counts_new = countrate_new*lcnew.res


    if set_analysis:
        savgall, pssimall = [], []

    if set_lc:
        lcsimall = []

    for n in xrange(nsims):

        qpo_counts = np.zeros(len(counts_new))
        for qpopars in qpoparams:

            ### phase shift in terms of shifting the *start time* of the signal in
            ### *rotational phase* of the neutron star!
            if qpopars["randomphase"] is True:
                phaseshift = np.random.random() - 0.5
            else:
                phaseshift = 0.0

            ### adjust start time for phase shift
            tstart = qpopars["tstart"] + phaseshift

            ### compute end time of the signal
            tend = tstart + qpopars["length"]

            ### find indices in time array that correspond to the start and end of the signal
            sig_minind = lcnew.time.searchsorted(tstart)
            sig_maxind = lcnew.time.searchsorted(tend)

            ### make a dummy array of zeros for the signal
            sig = np.zeros(len(lcnew.time))

            ### replace the zeros corresponding to time stamps with signal with ones.
            sig[sig_minind:sig_maxind] = np.ones(len(lcnew.time[sig_minind:sig_maxind]))

            ### phase shift of the periodic signal *itself*
            phase = np.random.rand()*2.0*np.pi

            ### convert fractional rms amplitude
            amp = qpopars["amp"]*np.sqrt(2.0)

            ### now make a periodic signal
            per = np.sin(2.0*np.pi*qpopars["freq"]*lcnew.time + phase)
            per = np.array(per)

            ### multiply by amplitude and cast into zeros/ones array that restricts the signal in time
            qpo_counts += amp*sig*per

        counts_wqpo = counts_new*(1.0+qpo_counts)


        print("min(counts_qpo): %f" %np.min(counts_wqpo))

        ### check for counts < 0 that screw up Poisson distribution, set to zero:
        zero_ind = np.where(counts_wqpo < 0)[0]
        if len(zero_ind) != 0:
            for z in zero_ind:
                counts_wqpo[z] = 0.0

        cp = np.array([np.random.poisson(x) for x in counts_wqpo])

        lcsim = lightcurve.Lightcurve(lcnew.time, counts=cp)

        if set_lc:
            lcsimall.append(lcsim)

        else:
            lcsimall = None

        if set_analysis:
            lcall, psall, mid, savg, xerr, ntrials, sfreqs, spowers = \
                search_singlepulse(lcsim, nsteps=nsteps, tseg=tseg, df=df, fnyquist=0.5/tfine, stack=None,
                                setlc=True, freq=freq)

            savgall.append(savg)

            if set_lc:
                pssimall.append(psall)

        else:
            pssimall = None
            savgall = None

    return lcsimall, savgall, pssimall


### make simulated giant flare light curves such that the 625 Hz QPO is smoothed out
### for adding a signal of specified length back, set qpo=True, and following kwargs:
### - mu: time of maximum signal (also ~start time if skewness is large and positive
### - sigma: width of the skew-normal signal
### - alpha: skewness parameter, large and positive for sharp rise times, large and negative for sharp fall times, 0 for symmetry
### - f0: frequency of the QPO
### - amp: total rms amplitude
### - nqpo: number of signals, if >1, then mu, sigma, alpha, and amp need to be lists
def simulations(time, nsims=1000, tcoarse = 0.1, tfine =0.5/4096.0, freq=627.0, nsteps=7, tseg=0.2, df = 2.66,
                fnyquist=4096.0, stack='all', setlc=True, set_analysis=True, maxstack=19, qpo=False, **kwargs):
 
    lccoarse = lightcurve.Lightcurve(time, timestep = tcoarse)
    f = scipy.interpolate.interp1d(lccoarse.time, lccoarse.countrate, "linear")
    
    lcnew = lightcurve.Lightcurve(time, timestep = tfine)

    minind = lcnew.time.searchsorted(lccoarse.time[0])
    maxind = lcnew.time.searchsorted(lccoarse.time[-1])

    lcnew.time = lcnew.time[minind:maxind]
    lcnew.counts = lcnew.counts[minind:maxind]
    lcnew.countrate = lcnew.countrate[minind:maxind]
    cnew = f(lcnew.time)

#    cnew = np.ones(len(lcnew.time))*1500.0
#    for n in range(nsims):

    #cpoisson = np.array([np.random.poisson(x, size=nsims) for x in cnew])

    #cpoisson = cpoisson.transpose()

    lcsimall, savgall, nasmall, psall =[], [], [], []

    #print("kwargs: " + str(kwargs))
    #print("kwargs keys: " + str(kwargs.keys()))
    if not "nqpo" in kwargs.keys():
        nqpo = 1
    else:
        nqpo = kwargs["nqpo"]

    if qpo:
        if nqpo == 1:
            if 'mu' in kwargs.keys() and 'sigma' in kwargs.keys() and 'alpha' in kwargs.keys():
                if random_phase:
                    phaseshift = np.random.random() - 0.5
                else:
                    phaseshift = 0.0
                tstart = kwargs["mu"][0] - kwargs["sigma"][0]/2.0 + phaseshift
                tend = tstart+ kwargs["sigma"][0]
                env_minind = lcnew.time.searchsorted(tstart)
                env_maxind = lcnew.time.searchsorted(tend)
                env = np.zeros(len(lcnew.time))
                env[env_minind:env_maxind] = np.ones(len(lcnew.time[env_minind:env_maxind]))

                #env = envelopes.skewed_normal(lcnew.time, kwargs['mu'], kwargs['sigma'], kwargs['alpha'], 1.0)
                env = env/np.max(env)
        else:
            env = []
            
            for m,s,a in zip(kwargs["mu"], kwargs["sigma"], kwargs["alpha"]):
                if random_phase:
                    phaseshift = np.random.random() - 0.5
                else:
                    phaseshift = 0.0
                tstart = m- s/2.0 + phaseshift
                tend = tstart + s
                env_minind = lcnew.time.searchsorted(tstart)
                env_maxind = lcnew.time.searchsorted(tend)
                env_temp = np.zeros(len(lcnew.time))
                env_temp[env_minind:env_maxind] = np.ones(len(lcnew.time[env_minind:env_maxind]))


    #env_temp = envelopes.skewed_nortal(lcnew.time, m,s,a, 1.0)
                env_temp = env_temp/np.max(env_temp)
                env.append(env_temp)
    else:
        env = np.zeros(len(lcnew.time))



    for c in range(nsims):
        print("I am on simulation " + str(c))
#        if qpo:
#            if 'mu' in kwargs.keys() and 'sigma' in kwargs.keys() and 'alpha' in kwargs.keys():
#                env = envelopes.skewed_normal(lcnew.time, kwargs['mu'], kwargs['sigma'], kwargs['alpha'], 1.0)
#                env = env/np.max(env)
#            else:
#                env = np.zeros(len(lcnew.time))
        if qpo:
            if nqpo == 1:
                phase = np.random.rand()*2.0*np.pi
                #tmin = lcnew.time[0]
                #tmax = lcnew.time[-1] - 2.0*tau
                #tstart = np.random.rand()*(tmax-tmin) + tmin
                #print('start time: ' + str(tstart))
                #tsind = lcnew.time.searchsorted(tstart)
                amp = kwargs['amp']*np.sqrt(2.0)
                #decay = np.exp(-lcnew.time[tsind:]/tau)
                per = np.sin(2.0*np.pi*kwargs['f0']*lcnew.time + phase)
                signal = amp*env*per
                crate_wqpo = cnew*(1.0+signal)
                counts_qpo = crate_wqpo*tfine
#            print(len(cnew)) 
            else:
                signal = np.zeros(len(lcnew.time))
                for a,e in zip(kwargs["amp"], env):
                    phase = np.random.rand()*2.0*np.pi
                    amp = a*np.sqrt(2.0)
                    #decay = np.exp(-lcnew.time[tsind:]/tau)
                    per = np.sin(2.0*np.pi*kwargs['f0']*lcnew.time + phase)
                    signal = signal + a*e*per

                crate_wqpo = cnew*(1.0+signal)
                counts_qpo = crate_wqpo*tfine


        else:
            counts_qpo = cnew
        cp = np.array([np.random.poisson(x) for x in counts_qpo])

        #cp = cnew
        lcsim = lightcurve.Lightcurve(lcnew.time, counts=cp)
        
        if setlc:
            lcsimall.append(lcsim)


        if set_analysis:

            if stack == None:
                print("I am here")
                lcall, psallsim, mid, savgsim, xerr, ntrials, sfreqs, spowers = search_singlepulse(lcsim, nsteps=nsteps,
                                                                                             tseg=tseg, df=df,
                                                                                             stack=stack, freq=freq,
                                                                                             setlc=True)

                #midsim, savgsim, xerrsim, ntrialssim = search_singlepulse(lcsim, nsteps=nsteps, tseg=tseg, df=df,
                #                                                          stack=stack, freq=freq)
                #print("len savgsim: " + str(len(savgsim)))
#               lcall, psallsim, midsim, savgsim, xerrsim, ntrialssim = search_singlepulse(lcsim, nsteps=nsteps, tseg=tseg, df=df, stack=stack, freq=freq)
            elif stack == "all":
                lcall, psallsim, midsim, savgsim, xerrsim, ntrialssim, allstacksim = search_singlepulse(lcsim,
                                                                                                        nsteps=nsteps,
                                                                                                        tseg=tseg,
                                                                                                        df=df,
                                                                                                        stack=stack,
                                                                                                        freq=freq)
            else:
                lcall, psallsim, midsim, savgsim, xerrsim, ntrialssim, allstacksim, nasmsim = search_singlepulse(lcsim,
                                                                                              nsteps=nsteps, tseg=tseg,
                                                                                              df=df, stack=stack,
                                                                                              freq=freq)

 
            savgall.append(savgsim)
            psall.append(psallsim)

            #nasmall.append(nasmsim)

    if set_analysis:
        savgall = np.array(savgall)
        #print(savgall)
        savgall = savgall.transpose()


            #nasmall = np.array(nasmall)
            #nasmall = np.transpose(nasmall)

            #savgmax = [max(s) for s in savgall]
            #nasmmax = [max(s) for s in nasmall]

    if setlc:
                return lcsimall, savgall, psall#, nasmsim#, savgmax, nasmmax
    else:
                return savgall

        
    #return lcsimall

def make_ps_stacks(psall, ncycle, nsteps):

    psstack = []
    for stack in np.arange(ncycle)+1:

        allstackmaxpows = []
        cycles = int(np.ceil((len(psall))/float(nsteps)))-1
        for n in range(nsteps):
             stacktempmaxpows = []
             for c in range(cycles - (stack-1)):
                 stacktempmaxpows.append(np.sum([psall[n+(c+i)*nsteps].ps for i in range(stack)], axis=0)/float(stack))
             allstackmaxpows.extend(stacktempmaxpows)
        allstackmaxpows = np.array(allstackmaxpows)
        #nasm = allstackmaxpows.transpose()
        #nasm = nasm.flatten()
        psstack.append(allstackmaxpows)
    return psstack



def make_stacks(savg, ncycle, nsteps):

    allstack = []
    for stack in np.arange(ncycle)+1:

        allstackmaxpows = []
        cycles = int(np.floor((len(savg))/float(nsteps)))
        for n in range(nsteps):
             stacktemp = []
             stacktempmaxpows = []
             for c in range(cycles - (stack-1)):
                 stacktempmaxpows.append(np.sum([savg[n+(c+i)*nsteps] for i in range(stack)])/float(stack))
             allstackmaxpows.append(stacktempmaxpows)
        allstackmaxpows = np.array(allstackmaxpows)
        nasm = allstackmaxpows.transpose()
        nasm = nasm.flatten()
        allstack.append(nasm)
    return allstack
 
def compute_maxpowers(savgall, ncycle, nsteps, froot="text"):


    allstack_sims = []
    for s in savgall:
        allstack = make_stacks(s, ncycle, nsteps)
        allstack_sims.append(allstack)

    maxp_all = []
    for i in xrange(len(allstack)):
        amax = np.array([np.max(a[i]) for a in allstack_sims])
        maxp_all.append(amax)

    maxp_all = np.array(maxp_all)
    maxp_all = np.transpose(maxp_all)
    print("Saving maximum powers across all segments in %s_simulated_maxpowers.txt"%froot)
    np.savetxt("%s_simulated_maxpowers.txt"%froot, maxp_all)
    return maxp_all

