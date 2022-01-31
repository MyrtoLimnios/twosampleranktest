# Functions for the estimation of the power of all the tests implemented in the main.py

# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

import numpy as np
import random
import torch

from scipy.stats import rankdata, mannwhitneyu
#rds = 4
#random.seed(rds)


""" Auxiliary functions"""

def sftplus(x,beta = 1e2):
    return (1. /beta) * np.log(1. + np.exp(beta*x))

def sig(x):
    return 1. / (1. + np.exp(-x))

def scoring_RTB(rank_x, N, u0):
    return - (sftplus(rank_x / (N + 1) - u0) + u0 * sig(1e2 * (rank_x / (N + 1)-u0)))


def scoring_RTB_(rank_x, N, u0):
    if rank_x / (N + 1.) >= u0: return rank_x / (N + 1.)
    else: return 0.0


def stat_emp(scoring, sX, sY):
    N = len(sX) + len(sY)

    alldata = np.concatenate((sX, sY))
    ranked = rankdata(alldata)
    rankx = ranked[:len(sX)]

    loss = np.sum([scoring(rx, N) for rx in rankx]) / len(sX)
    return loss


def stat_emp_RTB(sX, sY, u0):
    N = len(sX) + len(sY)
    alldata = np.concatenate((sX, sY))
    ranked = rankdata(alldata)
    rankx = ranked[:len(sX)]
    loss = np.sum([scoring_RTB_(rx, N, u0) for rx in rankx]) / len(sX)

    return loss

""" Main functions """

""" Function estimating the power for rank statistics """

def emp_power_allrank(x_test, scor_test, s_predrk_list,thresh_rtb, alpha, range_RTB, B_pow, subspl_len):

    pwr_RTB = np.zeros((len(range_RTB), len(s_predrk_list)))

    """ For MWW """
    pwr_other_rk = np.zeros((1, len(s_predrk_list)))

    """ Test sample """
    Xtest = x_test[np.where(scor_test == 1)]
    Ytest = x_test[np.where(scor_test == 0)]

    n, m = len(Xtest), len(Ytest)
    ns, ms = subspl_len[0], subspl_len[1]

    sX_, sY_ = [], []
    k = 0

    for ypred in s_predrk_list:
        sx = ypred[np.where(scor_test == 1)].tolist()
        sy = ypred[np.where(scor_test == 0)].tolist()
        sX_.append(sx)
        sY_.append(sy)
        k += 1

    for b in range(B_pow):

        """ subsampling multivariate two samples for the Rank tests """
        idx_x = random.sample(range(n), ns)
        idx_y = random.sample(range(m), ms)

        for j in range(len(s_predrk_list)):

            """ Predicted scores"""
            x, y = np.array(sX_[j])[idx_x], np.array(sY_[j])[idx_y]
            l = min(len(x), len(y))

            if list(x[0:l]) == list(y[0:l]):
                continue
                #print('equal pred')
            else:
                W_emp_rtb = [ max(np.abs(stat_emp_RTB(x, y, u)),np.abs(stat_emp_RTB(y,x, u))) for u in range_RTB]
                for u in range_RTB: print(stat_emp_RTB(x, y, u),stat_emp_RTB(y,x, u))
                print(thresh_rtb)

                for k in range(len(range_RTB)):
                    if W_emp_rtb[k] >= thresh_rtb[k]:
                        pwr_RTB[k, j] += 1. / B_pow

                mww, mww_p = mannwhitneyu(x, y)
                if mww_p <= alpha:
                    pwr_other_rk[0,j] += 1. /B_pow


    pwr_RTB_dict = dict(zip(range_RTB, np.around(pwr_RTB, decimals=4)))
    pwr_RTB_dict['MWW'] = pwr_other_rk[0]


    return pwr_RTB, pwr_RTB_dict, pwr_other_rk[0]


""" Function estimating the power for SoA statistics """

def emp_power_other_onetest(XY, scor, num_other_tests, list_test, alphas_, alpha):
    """ For the other methods  """
    pwr_ = np.zeros(num_other_tests)
    list_names = [str(meth).rsplit('>', 1)[0].rsplit('.', 1)[1] for meth in list_test]

    Xtest = XY[np.where(scor == 1)]
    Ytest = XY[np.where(scor == 0)]
    X_test, Y_test = torch.from_numpy(Xtest.astype(np.float32)), torch.from_numpy(Ytest.astype(np.float32))

    nall, mall = len(X_test), len(Y_test)
    res = dict()

    for i in range(num_other_tests):
        meth = list_test[i]
        test = meth(nall, mall)
        stat_dist, stat = test(X_test, Y_test, alphas=alphas_, ret_matrix=True)
        str_meth = str(meth).rsplit('S', 1)[0].rsplit('.', 1)[1]
        # print(str_meth, stat_dist.numpy(), test.pval(stat))
        res[str_meth] = [stat_dist, test.pval(stat)]
        # print(meth,res[str_meth][1] )

        if res[str_meth][1] <= alpha:
            pwr_[i] += 1.

    list_names = [str(meth).rsplit('S', 1)[0] for meth in list_names]
    pwr_dict = dict(zip(list_names, np.around(pwr_, decimals=4)))

    return pwr_, pwr_dict



