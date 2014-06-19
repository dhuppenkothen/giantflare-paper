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

