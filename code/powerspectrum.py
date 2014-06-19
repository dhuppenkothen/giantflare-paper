##### POWER SPECTRUM CLASS ####################################
#
# Stripped-down version of my power spectrum class. For full experience,
# clone #https://dhuppenkothen@bitbucket.org/dhuppenkothen/utools.git  ,
# but be aware that there's a lot of junk code in that repository.
#
# BEWARE OF THE REBINNING METHOD! It works, but because it is capable of
# dealing with bin frequencies that aren't integer multiples of the
# original frequency resolution, chopping up bins and re-distributing powers
# will almost certainly have an effect on the distributions!
# It's a small effect (I checked), but it's there.
#
#
#
#




import numpy as np
import scipy
import scipy.optimize
import lightcurve



class PowerSpectrum(lightcurve.Lightcurve):
    def __init__(self, lc = None, counts = None, nphot=None, norm='leahy', verbose=False):
        """
        Make an object of class PowerSpectrum.
        Inputs are either an object of type lightcurve.Lightcurve, or
        times (keyword lc) and associated counts per bin (counts).
        The number of photons in the light curve can be independently specified (nphot),
        otherwise it will be inferred from the data.

        norm is a string setting the normalisation of the power spectrum.
        Default is 'leahy' (normalised by 2/nphot, but currently, 'rms' and 'var' (or 'variance') are also supported.



        """


        ### normalisation of the power spectrum. One of either 'leahy', 'rms', 'var', or, equivalently 'variance'
        self.norm = norm

        ### check whether whatever is given in lc is an object of class Lightcurve
        if isinstance(lc, lightcurve.Lightcurve) and counts == None:             
            pass

        elif not lc == None and not counts == None:
            if verbose == True:
                print "You put in a standard light curve (I hope). Converting to object of type Lightcurve"
            lc = lightcurve.Lightcurve(lc, counts, verbose=verbose)
        else:
            self.freq = None
            self.ps = None
            self.df = None
            return

        ### is nphot given? If not, infer from counts
        if nphot == None:
            nphots = np.sum(lc.counts)
        else:
            nphots = nphot

        ### The meat of the code: compute power spectrum
        ### 1) compute the number of bins in the light curve
        nel = np.round(lc.tseg/lc.res)
        ### 2) compute frequency resolution, defined by the length of the light curve
        df = 1.0/lc.tseg
        ### 3) use scipy.fft to make Fast Fourier transform of data
        fourier= scipy.fft(lc.counts)
        ### compute conjuage of FFT
        f2 = fourier.conjugate()
        ### multiply FFT with conjugate, gives us amplitudes squared
        ff = f2*fourier
        ### we're interested in real part only
        fr = np.array([x.real for x in ff])
        ### apply standard (Leahy) normalisation
        ps = 2.0*fr[0: (nel/2 )]/nphots


        ### various useful normalisations (Default is Leahy)
        if norm.lower() in ['leahy']:
            self.ps = ps
            
        elif norm.lower() in ['rms']:
            self.ps = ps/(df*nphots)
        ### put frequency to mid-frequencies
        elif norm.lower() in ['variance', 'var']:
            self.ps = ps*nphots/len(lc.counts)**2.0

        ### compute list of frequencies
        freq = np.arange(len(ps))*df
        self.freq = np.array([f+(freq[1]-freq[0])/2.0 for f in freq])
        ### frequency resolution
        self.df = self.freq[1] - self.freq[0]
        ### number of photons in light curve
        self.nphots = nphots
        ### number of bins in light curve
        self.n = len(lc.counts)

        return

    def rebinps(self, res, verbose=False):
        ### frequency range of power spectrum
        flen = (self.freq[-1] - self.freq[0])
        ### calculate number of new bins in rebinned spectrum
        bins = np.floor(flen/res) 
        ### calculate *actual* new resolution
        self.bindf = flen/bins
        ### rebin power spectrum to new resolution
        binfreq, binps, dt = self._rebin_new(self.freq, self.ps, res, method='mean')
        newps = PowerSpectrum()
        newps.freq = binfreq
        newps.ps = binps
        newps.df = dt
        newps.nphots = binps[0]
        newps.n = 2*len(binps)
        return newps


##########################################################################################
##########################################################################################



### add up a number of periodograms or average them
## psall: list of periograms (note: *must* have same resolution!)
## method:
##     'average', 'avg', 'mean': average periodograms
##     'add', 'sum': add them up
def add_ps(psall, method='avg'):

    pssum = np.zeros(len(psall[0].ps))
    for x in psall:
        pssum = pssum + x.ps

    if method.lower() in ['average', 'avg', 'mean']:
        pssum = pssum/len(psall)

    psnew = PowerSpectrum()
    psnew.freq = psall[0].freq
    psnew.ps = pssum
    psnew.n = psall[0].n
    psnew.df = psall[0].df
    psnew.norm = psall[0].norm
    return psnew
