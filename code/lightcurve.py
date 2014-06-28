#!/usr/bin/env python
#####################
#
# Class definition for the light curve class. 
# Stripped-down version. For the full experience,
# clone
# https://dhuppenkothen@bitbucket.org/dhuppenkothen/utools.git
#
# BE CAUTIOUS WITH THE REBIN ROUTINE. ONE DAY, I WILL TEST
# THIS PROPERLY, BUT AS IT IS, I WOULDN'T TRUST IT!
#
#

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('PS')

import numpy
import math
import numpy as np
import fractions




#### BRAND-NEW CLASS IMPLEMENTATION!!!!

class Lightcurve(object):
    def __init__(self, time, counts = None, timestep=1.0, tseg=None, verbose = False, tstart = None):

        if counts == None:
            if verbose == True:
                print "You put in time of arrivals."
                print "Time resolution of light curve: " + str(timestep)
            ### TOA has a list of photon times of arrival
            self.toa = time
            self.ncounts = len(self.toa)
            self.tstart = tstart
            self.makeLightcurve(timestep, tseg = tseg,verbose=verbose)
            
        else:
            self.time = np.array(time)
            self.counts = np.array(counts)
            self.res = time[1] - time[0]
            self.countrate = [t/self.res for t in self.counts]
            self.tseg = self.time[-1] - self.time[0] + self.res

    def makeLightcurve(self, timestep, tseg=None, verbose=False):

        ### if self.counts exists, this is already a light curve, so abort
        try:
            self.counts
            raise Exception("You can't make a light curve out of a light curve! Use rebinLightcurve for rebinning.")
        except AttributeError:

            ## tstart is an optional parameter to set a starting time for the light curve
            ## in case this does not coincide with the first photon
            if self.tstart == None:
                ## if tstart is not set, assume light curve starts with first photon
                tstart = self.toa[0]
            else:
                tstart = self.tstart
            ### number of bins in light curve

            ## compute the number of bins in the light curve
            ## for cases where tseg/timestep are not integer, computer one
            ## last time bin more that we have to subtract in the end
            if tseg:
                timebin = np.ceil(tseg/timestep)
                frac = (tseg/timestep) - int(timebin - 1)
            else:
                timebin = np.ceil((self.toa[-1] - self.toa[0])/timestep)
                frac = (self.toa[-1] - self.toa[0])/timestep - int(timebin - 1)
            #print('tstart: ' + str(tstart))

            tend = tstart + timebin*timestep

            ### make histogram
            ## if there are no counts in the light curve, make empty bins
            if self.ncounts == 0:
                print("No counts in light curve!")
                timebins = np.arange(timebin+1)*timestep + tstart
                counts = np.zeros(len(timebins)-1)
                histbins = timebins
                self.res = timebins[1] - timebins[0]
            else:
                timebins = np.arange(timebin+1)*timestep + tstart
                counts, histbins = np.histogram(self.toa, bins=timebin, range = [tstart, tend])
                self.res = histbins[1] - histbins[0]

            #print("len timebins: " + str(len(timebins)))
            if frac > 0.0:
                self.counts = np.array(counts[:-1])
            else:
                self.counts = np.array(counts) 
            ### time resolution of light curve
            if verbose == True:
                print "Please note: "
                print "You specified the time resolution as: " + str(timestep)+ "."
                print "The actual time resolution of the light curve is: " + str(self.res) +"."

            self.countrate = self.counts/self.res
            self.time = np.array([histbins[0] + 0.5*self.res + n*self.res for n in range(int(timebin))])
            if frac > 0.0:
                self.time = np.array(self.time[:-1])
            else:
                self.time = self.time
            self.tseg = self.time[-1] - self.time[0] + self.res

    def saveLightcurve(self, filename):
        """ This method saves a light curve to file. """
        lfile = open(filename, 'w')
        lfile.write("# time \t counts \t countrate \n")
        for t,c,cr in zip(self.time, self.counts, self.countrate):
            lfile.write(str(t) + "\t" + str(c) + "\t" + str(cr) + "\n")
        lfile.close()

    def plot(self, filename, plottype='counts'):
        if plottype in ['counts']:
            plt.plot(self.time, self.counts, lw=3, color='navy', linestyle='steps-mid')
            plt.ylabel('counts', fontsize=18)
        elif plottype in ['countrate']:
            plt.plot(self.time, self.countrate)
            plt.ylabel('countrate', fontsize=18)
        plt.xlabel('time [s]', fontsize=18)
        plt.title('Light curve for observation ' + filename)
        plt.savefig(str(filename) + '.ps')
        plt.close()

    def rebinLightcurve(self, newres, method='sum', verbose = False, implementation="new"):
        ### calculate number of bins in new light curve
        nbins = math.floor(self.tseg/newres)+1
        self.binres = self.tseg/nbins
        print "New time resolution is: " + str(self.binres)

        if implementation in ["o", "old"]:
            self.bintime, self.bincounts, self.binres = self._rebin(self.time, self.counts, nbins, method, verbose=verbose)
        else:
            print("I am here")
            self.bintime, self.bincounts, self.binres = self._rebin_new(self.time, self.counts, newres, method)





    def _rebin_new(self, time, counts, dtnew, method='sum'):


        try:
            step_size = float(dtnew)/float(self.res)
        except AttributeError:
            step_size = float(dtnew)/float(self.df)

        output = []
        for i in numpy.arange(0, len(counts), step_size):
            total = 0
            #print "Bin is " + str(i)

            prev_frac = int(i+1) - i
            prev_bin = int(i)
            #print "Fractional part of bin %d is %f"  %(prev_bin, prev_frac)
            total += prev_frac * counts[prev_bin]

            if i + step_size < len(time):
                # Fractional part of next bin:
                next_frac = i+step_size - int(i+step_size)
                next_bin = int(i+step_size)
                #print "Fractional part of bin %d is %f"  %(next_bin, next_frac)
                total += next_frac * counts[next_bin]

            #print "Fully included bins: %d to %d" % (int(i+1), int(i+step_size)-1)
            total += sum(counts[int(i+1):int(i+step_size)])
            output.append(total)

        tnew = np.arange(len(output))*dtnew + time[0]
        if method in ['mean', 'avg', 'average', 'arithmetic mean']:
            cbinnew = output
            cbin = np.array(cbinnew)/float(step_size)
        elif method not in ['sum']:
            raise Exception("Method for summing or averaging not recognized. Please enter either 'sum' or 'mean'.")
        else:
            cbin = output

        return tnew, cbin, dtnew


    ### this method rebins a light curve to a new number of bins 'newbins'
    def _rebin(self, time, counts, newbins, method = 'sum', verbose = False):

        ### nbins is the number of bins in the new light curve
        nbins = int(newbins)
        ### told is the _old_ time resolution
        told = time[1] - time[0]

        ### tseg: length of the entire segment
        tseg = time[-1] - time[0] #+ told
        #print "tseg: " + str(tseg)

        if verbose == True:
            print "nbins: " + str(nbins)
            print "told: " + str(told)
            print "tseg: " + str(tseg)

        ### move times to _beginning_ of each bin
        btime = np.array(time) - told/2.0

        ### dt: new time resolution
        dt = float(tseg)/float(nbins)

        ### check whether old time resolution is larger than new time resolution
        if dt <= told:
            if verbose == True:
                print "Old time resolution bigger than new time resolution."
                print "That's not implemented yet. Returning power spectrum with original resolution."
            return time, counts, told


        ### tnew is the ratio of new to old bins
        tnew = dt/told

        #print "dt: " + str(dt)
        #print "told: " + str(told)
        #print "tnew: " + str(tnew)

        ### new array with bin midtimes
        bintime = [time[0] + 0.5*dt + t*dt for t in range(nbins)]

        ### this fraction is useful, because I can define a number of old bins until the 
        ### boundaries of the old and new bins match up again
        ### this makes it easier in some cases
        tnewfrac = fractions.Fraction(tnew)

        top = tnewfrac.numerator
        bottom = tnewfrac.denominator
        
        #print "top: " + str(top)
        #print "bottom: " + str(bottom)


        ### if the fraction turns out insanely big (happens due to rounding errors), then I do
        ### only one iteration (since tseg/tseg = 1)
        if top > tseg:
            top = tseg
            bottom = nbins
#            print "top: " + str(top)
#            print "bottom: " + str(bottom)
        cbin = []

        ### now iterate over all cycles
#        print "int(tseg/top): " + str(int(nbins/bottom))
#        print("nbins: " + str(nbins)) 

        for i in range(int(nbins/bottom)):

        ### I need this index to remember where I left off during the iteration
            before_ind = 0
#            print "i: " + str(i)
            ### for each cycle iterate through the number of new bins in that cycle
            for j in range(bottom):
                # print "j: " + str(j)
                ### in the first round, start at the lower edge of the bin:
                if before_ind == 0:
                    #print "tnew: " + str(tnew)
                    ## this is the first index to use
                    i0 = int(i*top)
                    #print "i0: " + str(i0)
                    ### first I sum up all complete old bins in that new bin
                    aint = sum(counts[i0:int(i0+math.floor(tnew))])
                    #print "lower index: " + str(i0)
                    #print "upper index: " + str(int(i0+math.floor(tnew)))
                    #print "values to sum: " + str(counts[i0:int(i0+math.floor(tnew))])

                    ### then I set the index of the old bin that is _not_ completely in the new bin
                    fracind = int(i0 + math.floor(tnew) )
                    #print "fracind 1 : " + str(fracind)


                    ### frac is the fraction of the old bin that's in the new bin
                    frac = tnew - math.floor(tnew)
                    #print "tnew fractional part: "  + str(tnew- math.floor(tnew))

                    ### if frac is not zero, then compute fraction of counts that goes into my new bin
                    if frac < 1.0e-10:
                        frac =0
                    if not frac == 0:
                        afrac = frac*counts[fracind]
                        #print "afrac: " + str(afrac)
                        cbin.append(aint + afrac) ### append to array with new counts
                    else:
                        cbin.append(aint)
                        #print "cbin: " + str(cbin[-1])

                    ### reset before_ind for next iteration in j
                    before_ind = fracind
                    #print "before_ind 1 : " + str(before_ind)
                else:

                    ### This new bin doesn't start at the same position as the old bin, hence I start with the fraction
                    ### afrac1 is the rest of the preceding old bin that was split up
                    afrac1 = (1.0 - frac)*counts[before_ind]
                    # print "afrac1: " + str(afrac1)
                    ### 1.0-frac of the bin already done, so define new length for the rest: ttemp 
                    ttemp = tnew - (1.0 - frac)
                    ### take integer part of ttemp and sum up
                    aint = sum(counts[before_ind+1:before_ind+1+int(math.floor(ttemp))])
                    ### fracind is the index of the last old bin that is split up between the current new bin and the next
                    fracind = np.int(before_ind + 1 + math.floor(ttemp))
                    #print "fracind 2 : " + str(fracind)
                    ### redefine frac
                    frac = ttemp - math.floor(ttemp)
                    #print "frac: " + str(frac)
                    if frac < 1.0e-10:
                        frac = 0
                    ### if frac is not zero, calculate the part of the old bin that will be in the current new bin
                    if not frac == 0:
                        #print("fracind2: " + str(fracind))
                        afrac2 = frac*counts[int(fracind)]
                        #print "afrac2: " + str(afrac2)
                        cbin.append(afrac1 + aint + afrac2)
                    else:
                        cbin.append(afrac1+aint)
                    #print "cbin: " + str(cbin[-1])
                    before_ind = fracind

        if method in ['mean', 'avg', 'average', 'arithmetic mean']:
            cbinnew = cbin
            cbin = [c/tnew for c in cbinnew]
        elif method not in ['sum']:
            raise Exception("Method for summing or averaging not recognized. Please enter either 'sum' or 'mean'.")
        return bintime, cbin, dt

###############################################################
