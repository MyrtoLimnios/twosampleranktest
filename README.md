

Code used for the numerical experiments of the companion working paper, Section Numerical Experiments:
   "A Bipartite Ranking Approach to the Two-sample Problem" S. Clémençon, M. Limnios*, N. Vayatis.
   
# Information

Use the main.py script to run for the Two-sample problem.

The functions coded in stattest_fct and fct_distribW are needed for the execution.

Code the probabilistic models in datagenerator for generating the two samples.

The variables are denoted as in the main paper Section Numerical Experiments.

author: Myrto Limnios // mail: myli@math.ku.dk


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


# Requirements

Python librairies: numpy, pandas, scipy, sklearn, random, tensorflow and torch for the SoA comparison algorithms
