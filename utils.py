import os
import numpy as np
from numpy.linalg import pinv
from copy import copy
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_files_superlist(dir_list, suffix_list):

    assert len(dir_list) == 1 or len(dir_list) == len(suffix_list), 'len(dir_list) must be equal to 1 or len(suffix_list)'

    files_list = os.listdir(dir_list[0])
    names_list = [f.split(suffix_list[0])[0] for f in files_list if f.endswith(suffix_list[0])]

    files_superlist = [[f + suffix_list[i] for f in names_list] for i in range(len(suffix_list))]
    files_superlist_t = [[f + suffix_list[i] for i in range(len(suffix_list))] for f in names_list]

    for i in range(len(suffix_list)):

        cur_dir = dir_list[0]
        if len(dir_list) == len(suffix_list):
            cur_dir = dir_list[i]

        assert False not in [os.path.exists(os.path.join(cur_dir, f)) for f in files_superlist[i]], "Some file no found from suffix %s" % suffix_list[i]

    return (names_list, files_superlist, files_superlist_t)


# reads scores by compute similarities

def read_sim_scores(scores_file):

    from pickle import loads

    with open(scores_file, 'r') as f:
        s = f.read()

    dict = {}
    dict.update(zip(['in1_dir', 'in1_files_list', 'in2_dir', 'in2_files_list', 'scores'], loads(s)))

    return dict


# intra-class correlation

def ICC_rep_anova(Y):
    '''
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y)**2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(nb_conditions), np.ones((nb_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, pinv(np.dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals**2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((np.mean(Y, 0) - mean_Y)**2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) /
    #            (mean square subjeT + (k-1)*-mean square error)
    ICC = (MSR - MSE) / (MSR + dfc * MSE)

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    return ICC, r_var, e_var, session_effect_F, dfc, dfe


def correct_for_covariates(covariates, data):
    """Correct data for covariates with linear regression

    Parameters
    ----------
    covariates : numpy.array (n_samples, n_covariates)
    data: numpy.array (n_samples, n_features)

    Returns
    -------
    corrected: numpy.array (n_samples, features)
    """

    ols = LinearRegression(fit_intercept=True, normalize=False)
    ols_list = [copy(ols.fit(covariates, data[:, i])) for i in range(data.shape[1])]
    data_corr = np.zeros(data.shape)
    for i, ols in enumerate(ols_list):
        data_corr[:, i] = data[:, i] - ols.predict(covariates)

    return data_corr


def re_index(data_df, imageid_df):
    """Re-indexes a dataset with the image-id

    Parameters
    ----------
    data_df: pandas.Dataframe
        The dataset containing imagetransid

    imageid_df: pandas.Dataframe
        The dataset containing imageid and imagetransid

    Returns
    -------
    The initial dataframe data_df re-indexed with the image-id
    """

    data_list = []
    iid_list = []

    for index, row in data_df.iterrows():

        iidtrans = row['IMAGEID_TRANS_R1']
        row2 = imageid_df.loc[imageid_df['imageidtrans'] == iidtrans]
        if row2.values.size > 0:
            data_list.append(row)
            iid_list.append(row2['imageid'].values[0])

    return pd.DataFrame(data=data_list, index=iid_list)  # index the list with image-id


def histedges_equalN(x, nbin=10):
    """Computes histogram with bin-edges containing equal samples each

    Parameters
    ----------
    x : iterable
        Data
    nbin : integer
        Number of bins

    Returns
    -------
    List of bin-edges
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def update_params(params_base, params_update):
    """Update the parameters of a dictionary and return the updated version
    Parameters:
    ----------
    params_base : dict
        base parameters
    params_update : dict
        parameters to be added / modified to the base ones
    Return
    ------
    d : dict
        updated parameters
    """
    d = params_base
    d.update(params_update)
    return copy(d)


def compute_stratified_class(rs_df, num_classes=30, sex_var='BASE_SEX_R1', age_var='BASE_AGE_R1'):
    """Returns a series with class info stratified by sex and age
    Parameters
    ----------
        rs_df : pandas dataframe
            contains all data
        num_classes: int
            number of classes to create. Maximum should be the number of expected samples in the smallest fold.
            For N=1000 and nested CV of 1/4, 1/3, the max number is approx 80. Choose 20 or 30 to be safe.
        sex_var : string
            id of sex variable
        age_var : string
            id of age variable
    Returns
    -------
        pandas series with stratified class variable
    """
    y = np.zeros(rs_df.shape[0], dtype=np.int)

    sex = rs_df[sex_var].astype('category').cat.codes # get numeric values
    sex_codes = sex.unique()
    nbins0 = np.int((num_classes * (sex == sex_codes[0]).sum()) / rs_df.shape[0]) # proportional bins for men and women accoding to prevalence
    nbins1 = np.int((num_classes * (sex == sex_codes[1]).sum()) / rs_df.shape[0])
    bins0 = histedges_equalN(rs_df.loc[sex == sex_codes[0], age_var].values, nbins0)
    bins1 = histedges_equalN(rs_df.loc[sex == sex_codes[1], age_var].values, nbins1)

    bins0[-1] += 1.e-3  # slightly extend last bin to avoid highest sample falling in new bin by digitize function below
    bins1[-1] += 1.e-3

    y[sex == sex_codes[0]] = np.digitize(rs_df.loc[sex == sex_codes[0], age_var], bins0)
    y[sex == sex_codes[1]] = np.digitize(rs_df.loc[sex == sex_codes[1], age_var], bins1) + nbins0  # to not overlap with men's class-ids

    return pd.Series(y, index=rs_df.index)


