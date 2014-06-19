##### USEFUL FUNCTIONS FOR GIANT FLARE ANALYSIS
#
# Not the actual analysis tools, just a bunch of
# useful functions.
#
#
#
#



__author__ = 'daniela'


from __future__ import with_statement
from collections import defaultdict

import numpy as np


def conversion(filename):
    """
    This is a useful little function that reads
    data from file and stores it in a dictionary
    with as many lists as the file had columns.
    The dictionary has the following architecture
    (e.g. for a file with three columns):

    {'0':[1st column data], '1':[2nd column data], '2':[3rd column data]}

    NOTE: Each element of the lists is still a *STRING*, because
    the function doesn't make an assumption about what type of data
    you're trying to read! Numbers need to be converted before using them!
    """

    f=open(filename, 'r')
    output_lists=defaultdict(list)
    for line in f:
        if not line.startswith('#'):
             line=[value for value in line.split()]
             for col, data in enumerate(line):
                 output_lists[col].append(data)
    return output_lists



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



def pavnosig(power, nspec, nsim=1.0e9):

    """
    Compute the probability of observing a power in a periodogram,
    given a chi-squared distribution with 2*nspec degrees of freedom,
    where nspec is the number of frequency bins or periodograms averaged.
    THIS ASSUMES A FLAT POWER SPECTRUM WITH WHITE NOISE ONLY!

    The number of simulations sets whether the code will use simulated draws
    from a chi-squared distribution to compute the result, or an
    analytical expression from Groth 1975
    """


    if power*nspec > 30000:
        print("Probability of no signal too miniscule to calculate.")
        return 0.0

    else:
        fn = pavnosigfun(power, nspec, nsim)

        print("Pr(Averaged power lies above P if no signal present) = %.4e" %fn)
        return fn



def pavnosigfun(power, nspec, nsim = 1.0e6):

    """
    Basic calculation. If number of sims is small, use
    simulated powers to derive p-value.
    If nsim is too big, use Anna's implementation of
    the analytical expression from Groth 1975.

    """

    if nsim < 1.0e7:

        chisquare = np.random.chisquare(2*nspec, size=nsim)/nspec

        print("The mean of the distribution is %f" %np.mean(chisquare))
        print("The variance of the distribution is %f" %np.var(chisquare))

        pval_ind = np.where(power < chisquare)[0]
        pval = len(pval_ind)/nsim

    else:

        pval = pavnosigfun_idl(power, nspec)

    return pval




def pavnosigfun_idl(power, nspec):

    """
    Python version of Anna Watts' function that computes
    the probability of observing a given power under the assumption
    of a chi-squared distribution with 2 degrees of freedom, modified
    by nspec binned frequencies or periodograms.

    """

    sum = 0.0
    m = nspec-1

    pn = power*nspec

    while m >= 0:

        #print("m %i" %m)
        s = 0.0
        for i in xrange(int(m)-1):
            #print("i %i" %i)
            #print("m-i: %f" %(m-i))
            s += np.log(float(m-i))
            #print("s: %f" %s)


        #print("s: %f" %s)
        logterm = m*np.log(pn/2.0) - pn/2.0 - s
        #print("logterm: %f" %logterm)
        term = np.exp(logterm)
        #print("term: %f" %term)
        ratio = sum/term
        #print("ratio: %f" %ratio)

        if ratio > 1.0e15:
            return sum

        sum += term
        m -= 1

    #print("The probability of observing P = %f under the assumption of noise is %.7f" %(power, sum))

    return sum
