
=================================================================================
Re-Analysis of the RXTE and RHESSI data of the 2004 Giant Flare from SGR 1906-20
=================================================================================

This repository contains all code and basic data to reproduce the results and
figures from Huppenkothen, Watts + Levin (accepted for publication in ApJ). 

You can find the repository at its `github address <https://github.com/dhuppenkothen/giantflare-paper/>`_.
Use ::

    git clone git@github.com:dhuppenkothen/giantflare-paper.git

to clone the repository to your local disc, or set up your ssh client.

Only the basic (processed) data files (1806.dat and 1806_rhessi.dat) are located in folder Data.
Because the intermediate data products do not readily fit into a repository, you can find the
entire set of intermediate and final data products on `figshare <http://figshare.com/articles/SGR_1806_20_Giant_Flare_Data_and_Simulations/1126082>`_.
The folder Documents contains both the paper as accepted in ApJ as well as an ipython notebook
that will walk you through the analysis step by step. You can also access the notebook 
via `nbviewer <http://nbviewer.ipython.org/github/dhuppenkothen/giantflare-paper/blob/master/documents/giantflare-analysis.ipynb>`_.

Requirements 
============

The code requires 

* python 2.7 or later (should be python 3 compliant, but I haven't tested it
* `numpy <http://www.numpy.org>`_
* `scipy <http://www.scipy.org>`_
* `matplotlib <http://www.matplotlib.org>`_

The code is just a bunch of scripts, so no install instructions are provided.






