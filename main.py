# Main script to run for the Two-sample problem
# Use of the functions coded in stattest_fct and fct_distribW
# Code the probabilistic models in datagenerator for generating the two samples
# The variables are denoted as in the main paper Section Numerical Experiments:
#    "A Bipartite Ranking Approach to the Two-sample Problem"

# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

# What it does:
# 1. Samples two data samples from different distribution functions
# 2. Performs a series of bipartite ranking algorithms in the first halves to learning the optimal model:
#               LambdaRankNN, RankNN, RankSVM with L1 and L2 penalties, LinearSVR, Logistic Regression,
#               and possibility to also use RankBoost, AdaBoost, RankSVML with L1 and L2 penalties
#               All are coded in this projects in their respective .py

# 3. Uses the outputs of 2. to score the second halves to the real line
# 4. Performs the hypothesis test on the obtained univariate two samples
# 5. Compares the results to SoA algorithms: Maximum Mean Discrepancy [Gretton et al. 2012],
#               Energy statistic [Szekely et al. 2004], and Wald-Wolfowitz [Friedman et al. 1979] coded at
#               https://github.com/josipd/torch-two-sample that needed to be updated

# NB: Initial implementation for RankNN and LambdaRankNN from https://github.com/liyinxiao/LambdaRankNN. Modified for
#       the two-sample procedure as detailed in the companion paper



import stattest_fct
import datagenerator as data
import fct_distribW

import pandas as pd
import numpy as np

from numpy.linalg import inv, norm
import torch
import torch_two_sample as twospl

from sklearn.model_selection import train_test_split
from RankNN import LambdaRankNN, RankNetNN
from RankSVM import RankSVM, RankSVMR, RankSVML
from RankBoost import RankB

from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression


import datetime
import random
rds = 4
random.seed(rds)
torch.manual_seed(rds)


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

'''  Power estimation parameters for rank statistics  '''
range_RTB = []  # values of u_0 of the RTB score-generating function
range_rk = range_RTB.append('MWW')  #list of the score-generating functions
B_pow = #number of MC sampling loops for the power estimation

'''  Benchmark tests parameters '''
alphas_ = [1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 1e2, 1e3]  # hyperparameters for the optimization of the SoA
list_test = [f for _, f in twospl.__dict__.items() if callable(f)]
print(list_test)
list_test.pop(1)
list_test.pop(4)
list_test.pop(0)
num_other_tests = len(list_test)
print(list_test)

list_names = ['MMD', 'Energy', 'FR']
eps_range = []

param_range = eps_range

'''   Complete with the path to your directory '''
path_RTB = r'/Users/' + sample_type + '/'
path_others = r'/Users/'+ sample_type + '/'
path = r'/Users/' + sample_type + '/'


k = 0

""" Choice of Bipartite Ranking algorithms """
rank_rSVM = True
rank_RNN = True
other_test = True


""" Initialization of the algorithmic parameters"""
subspl_len = []
ntst, mtst = 0, 0
n, m = 0, 0
sim, str_param = 0, 0

B_power_rk = 10

df_pred_lrnn = {}
df_pred_rnn = {}
df_pred_rsvm2 = {}
df_pred_rsvm = {}
df_pred_rsvr = {}
df_pred_rsvl = {}
df_pred_star = {}

df_pred_true = {}


for nsub in [500]:  ###  sample sizes

    ntst, mtst = int(nsub/5), int(nsub/5)
    n, m = int(4*nsub/5), int(4*nsub/5)
    subspl_len =[ntst, mtst]


    print('sizes', n, ntst, nsub)

    for d in [6]:

        print('loop', 'sample size', nsub, 'd', d)

        sim = sample_type + str(n) + str(m) + str(d) + str(epoch) + str(Ksub) + str(B_power_rk) + str(int(ntst)) + str(
            int(mtst))
        str_param = sim

        lay = (int(2*d), int(d),)

        '''  Loop on eps and save the results in a csv file '''
        res_rk_rsvm = []
        res_rk_rsvm2 = []
        res_rk_rsvr = []
        res_rk_rsvl = []
        res_rk_lrnn = []
        res_rk_rnn = []
        res_rk_star = []
        res_all_others = []

        df_rk_lrnn = {}
        df_rk_rnn = {}
        df_rk_rsvm2 = {}
        df_rk_rsvm = {}
        df_rk_rsvr = {}
        df_rk_rsvl = {}
        df_rk_star = {}
        df_others = {}

        for eps in np.around(eps_range, decimals=2):
            print('sizes', n, ntst, nsub, d, eps)

            '''  Generate the data: XY matrix of the twosamples, with scor=1 for X and scor=0 for Y, q=unit vector '''
            print('Generate datasets')

            XY_train, s_train, mu_X, mu_Y, S_X, S_Y = data.XY_generator(n, m, d, eps, theta, sample_type, distrib,  plot_data = False)
            x_test, scor_test, mu_X, mu_Y, S_X, S_Y = data.XY_generator(ntst, mtst, d, eps, theta, sample_type, distrib, plot_data=False)

            subspl_len = [ntst, mtst]

            if eps > 0.0:
                """ Compute the best parameter"""
                orig_star = - mu_X.T @ inv(S_X) @ mu_X + mu_Y.T @ inv(S_X) @ mu_Y
                theta_star = np.dot(inv(S_X), (mu_X - mu_Y)) / np.sqrt(norm(np.dot(inv(S_X), (mu_X - mu_Y))))
                print('theta star', theta_star)
                y_predstar = np.dot(x_test, theta_star)
            else:
                y_predstar = np.zeros(len(scor_test))

            """ Thresholds of the rank tests"""
            thresh_rtb = fct_distribW.thresh_rangeRTB_null_CB(ntst, mtst, range_RTB, alpha)
            print('thresh table W', thresh_rtb)

            thresh_rtb = fct_distribW.thresh_rangeRTB_nulld(ntst, mtst, range_RTB, alpha)
            print('thresh table W', thresh_rtb)

            pwr_cv = np.zeros((len(range_rk), 5))
            pwr_cv_othr = np.zeros(num_other_tests)

            for b in range(B_power_rk):
                """ subsampling multivariate two samples for the Rank tests """
                idx_x, _, scor_train, _ = train_test_split(range(int(n+m)), s_train, test_size=1/10, stratify=s_train,
                                                                                                 random_state=rds)
                x_train = XY_train[idx_x]
                ns = int(np.sum(scor_train))
                ms = int(len(scor_train) - ns)

                print('subsampling n, m' , ns, ms)

                y_pred_rsvm, y_pred_rsvm2 = np.zeros(len(scor_test)), np.zeros(len(scor_test))
                y_pred_rsvr, y_pred_rsvl = np.zeros(len(scor_test)), np.zeros(len(scor_test))
                y_pred_lrnn, y_pred_rnn = np.zeros(len(scor_test)), np.zeros(len(scor_test))

                y_rnn = []
                y_rsvm = []
                y_rsvr = []
                y_rsvl = []
                s_predrk_list = []

                if rank_rSVM == True:
                    for ksub in range(Ksub):

                        idx = np.random.permutation(len(scor_train))
                        x_train_, scor_train_ = x_train[idx], scor_train[idx]
                        scor_train_transf = 2 * scor_train_ - 1

                        rSVM = RankSVM(loss='squared_hinge', penalty='l1', #random_state=rds,
                                        fit_intercept=True, dual = False, tol=1e-6, C=100.)
                        rSVM.fit(x_train_, scor_train_transf)

                        pred = (rSVM.predict(x_test) + 1) / 2
                        y_pred_rsvm += pred / Ksub
                        print('pred', np.sum(pred - scor_test))


                        rSVM2 = RankSVM(loss='squared_hinge', penalty='l2',   #random_state=rds,
                            fit_intercept=True, dual = False, tol=1e-6, C=100.)
                        rSVM2.fit(x_train_, scor_train_transf)
                        pred = (rSVM2.predict(x_test) + 1)/2
                        y_pred_rsvm2 += pred / Ksub
                        print('pred2', np.sum(pred - scor_test))

                        rSVMR = LinearSVR(loss='epsilon_insensitive', tol=1e-6, C=100.)
                        rSVMR.fit(x_train_, scor_train_transf)
                        pred = (rSVMR.predict(x_test) + 1)/2
                        y_pred_rsvr += pred / Ksub
                        print('pred_rsvr', np.sum(pred - scor_test))

                        rSVML = LogisticRegression(penalty='l2',  tol=1e-6, C=100.)
                        rSVML.fit(x_train_, scor_train_transf)
                        pred = (rSVML.predict(x_test) + 1) / 2
                        y_pred_rsvl += pred / Ksub
                        print('pred_rsvr', np.sum(pred - scor_test))


                    y_rsvm = [y_pred_rsvm, y_pred_rsvm2, y_pred_rsvr, y_pred_rsvl]
                    df_pred_rsvm2[str(b)] = y_pred_rsvm2
                    df_pred_rsvm[str(b)] = y_pred_rsvm
                    df_pred_rsvr[str(b)] = y_pred_rsvr
                    df_pred_rsvl[str(b)] = y_pred_rsvl
                    df_pred_star[str(b)] = y_predstar
                    df_pred_true[str(b)] = scor_test

                if rank_RNN == True :
                    scor_train_transf = 2*scor_train -1
                    lambdaranker = LambdaRankNN(input_size=x_train.shape[1], hidden_layer_sizes=lay,
                                                activation=('relu', 'relu',), solver='adam')
                    lambdaranker.fit(x_train, scor_train_transf, epochs=epoch)
                    y_pred_lrnn = (lambdaranker.predict(x_test) + 1) / 2


                    '''  train model LambdaRank '''
                    ranker = RankNetNN(input_size=x_train.shape[1], hidden_layer_sizes=lay,
                                       activation=('relu', 'relu',),
                                      solver='adam')
                    ranker.fit(x_train, scor_train_transf, epochs=epoch)
                    y_pred_rnn = (ranker.predict(x_test) + 1) / 2
                    print(np.sum(y_pred_rnn), np.sum(y_pred_lrnn))

                    y_rnn = [y_pred_lrnn, y_pred_rnn]

                    df_pred_lrnn[str(b)] = y_pred_lrnn
                    df_pred_rnn[str(b)] = y_pred_rnn

                if rank_rSVM == True:
                    if rank_RNN == True :
                        s_predrk_list = [ y_pred_rsvm , y_pred_rsvm2, y_pred_rsvr, y_pred_rsvl , y_pred_lrnn, y_pred_rnn , y_predstar ]
                    else:
                        s_predrk_list = [y_pred_rsvm, y_pred_rsvm2, y_pred_rsvr, y_pred_rsvl, y_predstar ]
                else:
                    s_predrk_list = [ y_pred_lrnn, y_pred_rnn, y_predstar]


                ''' Power estimation for all methods '''

                pwr_RTB, pwr_RTB_dict, pwr_other_rk = stattest_fct.emp_power_allrank(x_test, scor_test, s_predrk_list,
                                                                                     thresh_rtb, alpha, range_RTB, 1, subspl_len)


                pwr_rk = np.concatenate((pwr_RTB, np.array([pwr_other_rk])), axis=0)
                pwr_cv += pwr_rk

                XY_oth = np.concatenate((x_train,x_test))
                scor_oth = np.concatenate((scor_train,scor_test))
                pwr_, _ = stattest_fct.emp_power_other_onetest(XY_oth, scor_oth, num_other_tests, list_test, alphas_, alpha)
                pwr_cv_othr += pwr_

                print('pwr_rank', pwr_rk)
                print('pwr_', pwr_)
                print('sampling loop', b, ns, ms, d, eps)

            pwr_rk = pwr_cv / B_power_rk
            pwr_ = pwr_cv_othr / B_power_rk
            pwr_dict = dict(zip(list_names, np.around(pwr_, decimals=4)))

            print('pwr_cv_rk', pwr_rk)
            print('pwr_cv_othr', pwr_dict)


            if rank_rSVM == True :
                pwr_rsvm = [pwr_rk[k, 0] for k in range(len(pwr_rk))]
                pwr_rk_dict_rsvm = dict(zip(range_rk, pwr_rsvm))
                res_rk_rsvm.append(pwr_rk_dict_rsvm)

                pwr_rsvm2= [pwr_rk[k, 1] for k in range(len(pwr_rk))]
                pwr_rk_dict_rsvm2 = dict(zip(range_rk, pwr_rsvm2))
                res_rk_rsvm2.append(pwr_rk_dict_rsvm2)

                pwr_rsvr= [pwr_rk[k, 2] for k in range(len(pwr_rk))]
                pwr_rk_dict_rsvr = dict(zip(range_rk, pwr_rsvr))
                res_rk_rsvr.append(pwr_rk_dict_rsvr)

                pwr_rsvl = [pwr_rk[k, 3] for k in range(len(pwr_rk))]
                pwr_rk_dict_rsvl = dict(zip(range_rk, pwr_rsvl))
                res_rk_rsvl.append(pwr_rk_dict_rsvl)

            if rank_RNN == True :
                if rank_rSVM == True : j = 4
                else : j = 0
                pwr_lrnn = [pwr_rk[k, j] for k in range(len(pwr_rk))]
                pwr_rk_dict_lrnn = dict(zip(range_rk, pwr_lrnn))
                res_rk_lrnn.append(pwr_rk_dict_lrnn)

                pwr_rnn = [pwr_rk[k, j+1] for k in range(len(pwr_rk))]
                pwr_rk_dict_rnn = dict(zip(range_rk, pwr_rnn))
                res_rk_rnn.append(pwr_rk_dict_rnn)

                pwr_star = [pwr_rk[k, j+2] for k in range(len(pwr_rk))]
                pwr_rk_dict_star = dict(zip(range_rk, pwr_star))
                res_rk_star.append(pwr_rk_dict_star)

            if other_test == True:
                res_all_others.append(pwr_dict)
                print('loop', pwr_rk, pwr_dict)

        df_all = [
            df_pred_lrnn,
            df_pred_rnn,
            df_pred_rsvm2,
            df_pred_rsvm,
            df_pred_rsvr,
            df_pred_rsvl,
            df_pred_star,
            df_pred_true]
        name = ['lRNN', 'RNN', 'rSVM2', 'rSVM', 'rSVR', 'rSVL', 'star', 'true']

        k = 0
        for df in df_all:
            df_ = pd.DataFrame(df)
            df_.to_csv(
                path + 'pred_' + name[k] + '_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            k += 1

        if rank_rSVM == True :
            res_rk_rsvm_dict = dict(zip(param_range, res_rk_rsvm))
            res_rk_rsvm_dict2 = dict(zip(param_range, res_rk_rsvm2))
            res_rk_rsvr_dict = dict(zip(param_range, res_rk_rsvr))
            res_rk_rsvl_dict = dict(zip(param_range, res_rk_rsvl))

            print(res_rk_rsvm_dict)

            df_rk_rsvm = pd.DataFrame.from_dict(res_rk_rsvm_dict)
            df_rk_rsvm.to_csv(path_RTB + 'power_Rk_rSVM_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_rk_rsvm = {}

            df_rk_rsvm2 = pd.DataFrame.from_dict(res_rk_rsvm_dict2)
            df_rk_rsvm2.to_csv(path_RTB + 'power_Rk_rSVM2_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_rk_rsvm2 = {}

            df_rk_rsvr = pd.DataFrame.from_dict(res_rk_rsvr_dict)
            df_rk_rsvr.to_csv(path_RTB + 'power_Rk_rSVR_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_rk_rsvr = {}

            df_rk_rsvl = pd.DataFrame.from_dict(res_rk_rsvl_dict)
            df_rk_rsvl.to_csv(path_RTB + 'power_Rk_rSVL_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_rk_rsvl = {}


            print('rSVM l1', res_rk_rsvm_dict)
            print('rSVM l2', res_rk_rsvm_dict2)

            print('rSVR', res_rk_rsvr_dict)
            print('rSVL', res_rk_rsvl_dict)


        if rank_RNN == True :
            res_rk_lrnn_dict = dict(zip(param_range, res_rk_lrnn))
            res_rk_rnn_dict = dict(zip(param_range, res_rk_rnn))
            res_rk_star_dict = dict(zip(param_range, res_rk_star))

            df_rk_lrnn = pd.DataFrame.from_dict(res_rk_lrnn_dict)
            df_rk_lrnn.to_csv(path_RTB + 'power_Rk_lRNN_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_rk_lrnn = {}

            df_rk_rnn = pd.DataFrame.from_dict(res_rk_rnn_dict)
            df_rk_rnn.to_csv(path_RTB + 'power_Rk_RNN_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_rk_rnn = {}

            df_rk_star = pd.DataFrame.from_dict(res_rk_star_dict)
            df_rk_star.to_csv(path_RTB + 'power_Rk_star_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_rk_star = {}


            print('lRNN', res_rk_lrnn_dict)
            print('RNN', res_rk_rnn_dict)
            print('star', res_rk_star_dict)



        if other_test == True:
            res_all_others_dict = dict(zip(param_range, res_all_others))

            df_others = pd.DataFrame.from_dict(res_all_others_dict)
            df_others.to_csv(
                path_others + 'power_others_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_others = {}

            print('Others results', res_all_others_dict)


        print('loop', 'sample size', nsub, 'd', d)


