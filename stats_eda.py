import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# creating mean difference test function


def mean_diff_test(data, group_col, value_col, alpha=0.05):
    """
    Perform a mean difference test between two groups.

    Parameters:
    data (pd.DataFrame): The input dataframe containing the data.
    group_col (str): The column name representing the groups.
    value_col (str): The column name representing the values to compare.
    alpha (float): Significance level for the test.

    Returns:
    dict: A dictionary containing the test statistic, p-value, and conclusion.
    """
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError("The group column must contain exactly two unique values.")

    group1 = data[data[group_col] == groups[0]][value_col]
    group2 = data[data[group_col] == groups[1]][value_col]

    # Check for normality
    _, p1 = stats.shapiro(group1)
    _, p2 = stats.shapiro(group2)

    normal = p1 > alpha and p2 > alpha

    # Check for equal variances
    _, p_var = stats.levene(group1, group2)
    equal_var = p_var > alpha

    if normal:
        if equal_var:
            stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
        else:
            stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    else:
        stat, p_value = stats.mannwhitneyu(group1, group2)

    conclusion = (
        "Reject null hypothesis"
        if p_value < alpha
        else "Fail to reject null hypothesis"
    )

    return {"test_statistic": stat, "p_value": p_value, "conclusion": conclusion}
