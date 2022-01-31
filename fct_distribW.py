# Functions for estimating the asymptotic null distributions of the rank statistics, depending on the score-generating
# functions

# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

import numpy as np
import random
from scipy.stats import distributions
rds = 40
random.seed(rds)



def sftplus(x,beta = 1e2):
    return (1. /beta) * np.log(1. + np.exp(beta*x))

def sig(x):
    return 1. / (1. + np.exp(-x))

def scoring_RTB(rank_x, N, u0):
    return  (sftplus(rank_x / (N + 1) - u0) + u0 * sig(1e2 * (rank_x / (N + 1)-u0)))



def scoring_RTB_(rank_x, N, u0):
    if rank_x / (N + 1.) >= u0: return rank_x / (N + 1.)
    else: return 0.0



""""  Functions estimating the null threshold for the hypothesis testing """


def thresh_rangeRTB_nulld(n,m,range_RTB, alpha):
    T = []
    N = n+m
    for u in range_RTB:
        phi = [scoring_RTB_(i, N, u) for i in np.arange(1, N + 1, 1)]
        phi2 = [scoring_RTB_(i, N, u)**2 for i in np.arange(1, N + 1, 1)]
        sphi = np.sum(phi)
        sphibar = (1. / N) * sphi
        var =  (N/((N-1)*n))*((1./N)*np.sum(phi2) - sphibar**2)
        exp = sphibar
        thres = distributions.norm.isf(alpha, exp, np.sqrt(var))
        T.append(thres)
    return T


def thresh_rangeRTB_null_CB(n,m,range_RTB, alpha):
    T = []
    N = n+m
    p = n/N
    for u in range_RTB:
        phi = [scoring_RTB_(i, N, u)/N for i in np.arange(1, N + 1, 1)]
        intphi = (1-u**2)/2
        sphi = np.sum(phi)
        delta = np.abs(sphi - intphi)
        b = np.sqrt(np.log(1./alpha)/(2*n))
        T.append((sphi + b)*n)
    return T

