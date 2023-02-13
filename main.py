# Main script to run for the Two-sample problem
# Use of the functions coded in stattest_fct
# Code the probabilistic models in datagenerator for generating the two samples
# The variables are denoted as in the main paper Section Numerical Experiments:
#    "A Bipartite Ranking Approach to the Two-sample Problem"

# author: Myrto Limnios // mail: myli@math.ku.dk

# What it does:
# 1. Samples two data samples from different distribution functions using datagenerator
# 2. Performs a series of bipartite ranking algorithms in the first halves to learning the optimal model:
#               RankNN, RankSVM L2 penalty, rForest
#               All are coded in this projects in their respective .py

# 3. Uses the outputs of 2. to score the second halves to the real line
# 4. Performs the hypothesis test on the obtained univariate two samples
# 5. Compares the results to SoA algorithms: Maximum Mean Discrepancy [Gretton et al. 2012],
#               Energy statistic [Szekely et al. 2004], and Wald-Wolfowitz [Friedman et al. 1979] coded at
#               https://github.com/josipd/torch-two-sample that needed to be updated
#               also compared to Tukey depth adapted from https://github.com/GuillaumeStaermanML/DRPM
# 6. Outputs the numerical pvalue for each sampling loop



import stattest_fct
import datagenerator as data

import pandas as pd
import numpy as np

from numpy.linalg import inv, norm
import torch
import torch_two_sample as twospl

from sklearn.model_selection import train_test_split
from RankNN import RankNetNN
from RankSVM import RankSVM
from RankBoost import RankRB
import treerank as TR

from scipy.stats import rankdata, ranksums, mannwhitneyu
from scipy.stats import norm as statnorm

import datetime
import random

seed =
rng = np.random.default_rng(seed)



'''  generate two-sample data '''
test_size = #Step1/Step2 size split (iee train/test split of the original dataset)
eps =  #discrepancy parameter
theta =   #angle of rotation between samples
distrib = '' #distribution type, related to the probabilistic model if one (used with the models from datagenerator)
sample_type = ''  #location, scale, blobs_loc, blobs_rot, related to the probabilistic model if one
# (used with the models from datagenerator)

epoch =  # epochs in the LTR algo
Ksub = # number of MC sampling loops for the learning algorithms

'''  Test parameters  '''
alpha =  #threshold of the tests
B_pow = #number of MC sampling loops for the power estimation
K_rank =
n_dir =  #number directions for Tukey depth

'''  Benchmark tests parameters '''
alphas_ = [1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 1e2, 1e3]  # hyperparameters for the optimization of the SoA
list_test = [f for _, f in twospl.__dict__.items() if callable(f)]
print(list_test)
list_test.pop(1)
list_test.pop(4)
list_test.pop(0)
num_other_tests = len(list_test)
print(list_test)
n_perm =  #number of permutations for estimation of the null distribution

list_names = ['MMD', 'Energy', 'FR']
eps_range = []

param_range = eps_range

'''   Complete with the path to your directory '''
path_MWW = r'/Users/' + sample_type + '/'


""" Initialization of the algorithmic parameters"""
subspl_len = []
ntst, mtst = 0, 0
n, m = 0, 0
sim, str_param = 0, 0

B_power_rk = 10
name_BR = ['rSVM2', 'rForest', 'rBoost', 'Tukey']

subspl_len = []
ntst, mtst = 0, 0
n, m = 0, 0
sim, str_param = 0, 0


for N in []:
    ntst, mtst = int(N/10), int(N/10)
    n, m = int(4*N/10), int(4*N/10)
    subspl_len = [ntst, mtst]

    print('sizes', n, ntst)

    for d in []:

        print('loop', 'sample size', 'd', d)

        sim = sample_type + str(n) + str(m) + str(d) + str(epoch) + str(Ksub) + str(B_pow) + str(int(ntst)) + str(
            int(mtst))
        str_param = sim


        for eps in np.around(eps_range, decimals=2):
            print('sizes', n, ntst, d, eps)

            '''  Generate the data: XY matrix of the twosamples, with scor=1 for X and scor=0 for Y, q=unit vector '''
            print('Generate datasets')


            pwr_rk = np.zeros((1, len(name_BR)))
            dict_pvalue_MWW = dict(zip(name_BR, [[] for i in range(len(name_BR))]))
            dict_pvalue_RTB7 = dict(zip(name_BR, [[] for i in range(len(name_BR))]))
            dict_pvalue_RTB8 = dict(zip(name_BR, [[] for i in range(len(name_BR))]))
            dict_pvalue_RTB9 = dict(zip(name_BR, [[] for i in range(len(name_BR))]))

            pwr_othr_ = np.zeros((1, len(list_names) - 1))
            dict_pval_othr = dict(zip(list_names, [[] for i in range(len(list_names))]))

            XY_train_, s_train_, mu_X, mu_Y, S_X, S_Y = data.XY_generator(n*B_pow, m*B_pow, d, eps, theta, sample_type, distrib, rng)
            x_test_, scor_test_, _, _, _, _, = data.XY_generator(ntst*B_pow, mtst*B_pow, d, eps, theta, sample_type, distrib, rng)
            s_train, scor_test = np.concatenate((np.ones(n), np.zeros(m))).astype(np.float32), np.concatenate((np.ones(ntst), np.zeros(mtst))).astype(np.float32)

            for b in range(B_pow):
                """ subsampling multivariate two samples for the Rank tests """
                XY_train = np.concatenate((XY_train_[n*b:n*(b+1)], XY_train_[n*B_pow + m*b:n*B_pow + m*(b+1)])).astype(np.float32)
                x_test = np.concatenate((x_test_[ntst*b:ntst*(b+1)], x_test_[ntst*B_pow + mtst*b:ntst*B_pow + mtst*(b+1)])).astype(np.float32)

                print('indices train from', n*b, 'to',  n*(b+1), 'from',  n*B_pow + m*b, 'to', n*B_pow + m*(b+1))
                print('indices test', ntst*b, ntst*B_pow + mtst*b, ntst*B_pow + mtst*(b+1))
                print('len', len(XY_train), len(x_test))
                print('n, m', n, m, ntst, mtst, len(XY_train), len(x_test))

                x_train_ = XY_train
                scor_train_ = s_train

                y_pred_rsvm = np.zeros(ntst + mtst)
                y_pred_tr = np.zeros(ntst + mtst)
                y_pred_rboost = np.zeros(ntst + mtst)
                y_pred_rnn = np.zeros(ntst + mtst)
                proj_XYtest = np.zeros(ntst + mtst)

                s_predrk_list = []

                print("#" * 80, "#{:^78}#".format("TREE"), "#" * 80, sep='\n')
                for kt in range(K_tree):

                    XY_index = np.arange(0, len(s_train))
                    XY_index_train, _, scor_train_t, _ = train_test_split(XY_index, s_train, test_size=0.05,
                                                                          stratify=s_train)
                    xtrain_temp = x_train_[XY_index_train]
                    scor_train_t = 2 * scor_train_t - 1

                    tree = TR.TreeRANK(max_depth=d, verbose=0, C=100.0, penalty='l2', fit_intercept=True)
                    tree.fit(xtrain_temp, scor_train_t)

                    """" Predict """
                    y_pred_tr += tree.predict_scor(x_test) / K_tree

                    if kt < K_rank:
                        rsvm = RankSVM(loss='squared_hinge', penalty='l2', fit_intercept=True, dual=False, tol=1e-6, C=100.,
                                       max_iter=10000)
                        rsvm.fit(xtrain_temp, scor_train_t)

                        rboost = RankBR()
                        rboost.fit(xtrain_temp, scor_train_t)

                        y_pred_rsvm += rsvm.predict(x_test) / K_rank
                        y_pred_rboost += rboost.predict(x_test) / K_rank

                ranker = RankNetNN(input_size=XY_train.shape[1], hidden_layer_sizes=lay, activation=('relu', 'relu',), solver='adam')
                ranker.fit(XY_train, scor_train_transf, epochs=epoch)
                y_pred_rnn = (ranker.predict(x_test) + 1) / 2

                """ TUKEY DEPTH """
                proj_XYtest = depth_test.tukey_depth_proj(XY_train[np.where(scor_test == 1)], x_test, n_dir)

                s_predrk_list = []

                ind = 0
                for ypred in s_predrk_list:
                    sx = ypred[np.where(scor_test == 1)].tolist()
                    sy = ypred[np.where(scor_test == 0)].tolist()

                    if (sx == sy) == True:
                        dict_pvalue_MWW[name_BR[ind]].append(1.0)
                        dict_pvalue_RTB9[name_BR[ind]].append(1.0)
                        dict_pvalue_RTB8[name_BR[ind]].append(1.0)
                        dict_pvalue_RTB7[name_BR[ind]].append(1.0)
                    else:
                        mww, mww_pR = mannwhitneyu(sx, sy, use_continuity=True, alternative='greater')
                        W9, pval9, _, _, _ = stattest_fct.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.9, alpha, asymptotic=True)
                        W8, pval8, _, _, _ = stattest_fct.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.8, alpha, asymptotic=True)
                        W7, pval7, _, _, _ = stattest_fct.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.7, alpha, asymptotic=True)
                        dict_pvalue_MWW[name_BR[ind]].append(mww_pR)
                        dict_pvalue_RTB9[name_BR[ind]].append(pval9)
                        dict_pvalue_RTB8[name_BR[ind]].append(pval8)
                        dict_pvalue_RTB7[name_BR[ind]].append(pval7)

                    ind +=1

                XY_oth = np.concatenate((XY_train, x_test))
                scor_oth = np.concatenate((s_train, scor_test))
                pwr_, pwr_dict, pval_othr = stattest_fct.emp_power_other_onetest_pval(XY_oth, scor_oth, list_test, alphas_, alpha,
                                                                                      npermutations=n_perm)
                pwr_othr_ += pwr_ / B_pow
                for i in range(len(list_names)-1):
                    dict_pval_othr[list_names[i]].append(pval_othr[i])

                dict_pval_othr['Tukey'] = dict_pvalue_MWW['Tukey']

                print('sampling loop', b, d, eps)
                print('MWW', dict_pvalue_MWW)
                print('RTB9', dict_pvalue_RTB9)
                print('RTB8', dict_pvalue_RTB8)
                print('RTB7', dict_pvalue_RTB7)
                print('SOA', dict_pval_othr)

            ''' Power estimation for all methods '''

            df_pvalue_MWW = pd.DataFrame.from_dict(dict_pvalue_MWW)
            df_pvalue_MWW.to_csv(path_MWW + 'pval_MWW_all_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + 'eps' + str(eps) + '.csv')
            df_pvalue_MWW = {}

            df_pvalue_RTB9 = pd.DataFrame.from_dict(dict_pvalue_RTB9)
            df_pvalue_RTB9.to_csv(path_MWW + 'pval_RTB9_all_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + 'eps' + str(eps) + '.csv')
            df_pvalue_RTB9 = {}

            df_pvalue_RTB8 = pd.DataFrame.from_dict(dict_pvalue_RTB8)
            df_pvalue_RTB8.to_csv(path_MWW + 'pval_RTB8_all_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + 'eps' + str(eps) + '.csv')
            df_pvalue_RTB8 = {}

            df_pvalue_RTB7 = pd.DataFrame.from_dict(dict_pvalue_RTB7)
            df_pvalue_RTB7.to_csv(path_MWW + 'pval_RTB7_all_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + 'eps' + str(eps) + '.csv')
            df_pvalue_RTB7 = {}

            df_pvalue_SOA = pd.DataFrame.from_dict(dict_pval_othr)
            df_pvalue_SOA.to_csv(path_MWW + 'pval_SoA_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + 'eps' + str(eps) + '.csv')
            df_pvalue_SOA = {}

            print('loop', 'sample size', 'd', d)


