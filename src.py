#########################################
# AB Testing
#########################################

# -------------------------------------------------------------------------------------------------------------------

# Business Problem

""""
Company X has recently introduced a new type of bidding, average bidding,
as an alternative to the current type of bidding called maximum bidding.
One of our clients, bombabomba.com, decided to test this new feature and
wants to do an A/B test to see if average bidding converts more than maximum bidding.
"""

# Dataset Story:
""""
In this dataset, which includes the website information of bombabomba.com, 
there is information such as the number of advertisements that users see and click, 
as well as earnings information from here.

There are two separate data sets, the control and test groups.

The max binding strategy was presented to the control group, and the average binding strategy was presented to the test group.
"""

# Features:

# Total Features : 4
# Total Row : 40
# CSV File Size : 26 KB

""""
- Impression : Ad views count
- Click : The number of clicks on the displayed ad
- Purchase : The number of products purchased after the ads clicked
- Earning : Earnings after purchased products
"""

# -------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


A_ = pd.read_excel("ab_testing.xlsx", sheet_name="Control Group")
B_ = pd.read_excel("ab_testing.xlsx", sheet_name="Test Group")


# Missing Value Analysis - Function
def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

# Let's define the auxiliary functions that we will use in the study
def check_df(df, head=5, box=False, column="Purchase"):
    print("--------------------- Shape ---------------------")
    print(df.shape)

    print("---------------------- Types --------------------")
    print(df.dtypes)

    print("--------------------- Head ---------------------")
    print(df.head(head))

    print("--------------------- Missing Value Analysis ---------------------")
    print(missing_values_analysis(df))

    print("--------------------- Quantiles ---------------------")
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("--------------------- BOX PLOT ---------------------")
    if box == True:
        sns.boxplot(x=df[column])
        print(plt.show())

check_df(A_,box=True)

check_df(B_,box=True)

""""
The success criterion for Bombomba.com is the Number of Purchases. 
For this reason, we will examine the Procurement variable. 
There are no missing values in either dataset. 
Also, there are no outliers in the Number of Purchases.

-----> Everything is going well...

"""

#--------------------------------------------------------------------


# Let's start A/B TEST
""""
It is used when it is desired to make a comparison between the mean of two groups.

1. Assumption Check¶
    1- Normality Assumption
    2- Variance Homogeneity

2. Implementation of the Hypothesis
    1- Independent two-sample t-test (parametric test) if assumptions are met.
    2- Mannwhitneyu test if assumptions are not met (non-parametric test).

If the normality is not provided, the mannwhitneyu test should be applied directly.
If the normality assumption is provided but the variance homogeneity is not provided, 
the equal_var parameter can be set to False for the two-sample t-test.
"""

# Let's name them clearly, as we're going to merge and organize the datasets.

A_.columns = [i+"_A" for i in A_.columns]
A_.head()

B_.columns = [i+"_B" for i in B_.columns]
B_.head()

# Merge the dataset
df = pd.concat([A_, B_], axis=1)
df.head()

""""
   Impression_A   Click_A  Purchase_A  Earning_A  Impression_B   Click_B     Purchase_B  Earning_B  
0    82529.4593 6090.0773    665.2113  2311.2771   120103.5038 3216.5480     702.1603    1939.6112  
1    98050.4519 3382.8618    315.0849  1742.8069   134775.9434 3635.0824     834.0543    2929.4058  
2    82696.0235 4167.9657    458.0837  1797.8274   107806.6208 3057.1436     422.9343    2526.2449  
3   109914.4004 4910.8822    487.0908  1696.2292   116445.2755 4650.4739     429.0335    2281.4286  
4   108457.7626 5987.6558    441.0340  1543.7202   145082.5168 5201.3877     749.8604    2781.6975 
"""


# Hypotheses

""""
---- OUR HYPOTHESIS H0 AND H1 ----

H0: M1 = M2 
--> There is no statistical difference between the average purchase earned,
    by the maximum binding strategy and the average purchase achieved by the average binding strategy.

H1: M1 != M2 
--> There is a statistical difference between the average purchase earned, 
    by the maximum binding strategy and the average purchases earned by the average binding strategy.
"""


print(" Mean of purchase of control group: %.4f" %A_['Purchase'].mean(), "\n",
      "Mean of purchase of test group: %.4f"  %B_['Purchase'].mean())

# OUT:  Mean of purchase of control group: 550.8941
# OUT:   Mean of purchase of test group: 582.1061

""""
There is a mathematical difference when looking at the purchasing rates for the two groups. 
Group B, the average binding strategy, seems to be more successful. 
But we do not know whether this difference is statistically significant. 
To understand this, we must apply hypothesis testing.
"""

# Assumption Check

""""
1. Normality Assumption
    -H0: There is no statistically significant difference between sample distribution and theoretical normal distribution

    -H1: There is statistically significant difference between sample distribution and theoretical normal distribution 

The test rejects the hypothesis of normality when the p-value is less than or equal to 0.05.
We do not want to reject the null hypothesis in the tests that might be considered for assumptions.

p-value < 0.05 (H0 rejected)
p-value > 0.05 (H0 not rejected)
"""

# Shapiro-Wilks Test for Control Group
test_st , p_value = shapiro(A_["Purchase_A"])
print('Test statistic = %.4f, P-Value = %.4f' % (test_st, p_value))

# OUT : Test statistic = 0.9773, P-Value = 0.5891

""""
*** P-VALUE = 0.5891 ***
--> Since the p value is not less than 0.05, 
the h0 hypothesis cannot be rejected. That is, the assumption of normality is provided for the control group.
"""

# Shapiro-Wilks Test for Control Group
test_st , p_value = shapiro(B_["Purchase_B"])
print('Test statistic = %.4f, P-Value = %.4f' % (test_st, p_value))

# OUT: Test statistic = 0.9589, p-Value = 0.1541

""""
*** P-VALUE = 0.1541 ***
--> Since the p value is not less than 0.05, 
the h0 hypothesis cannot be rejected. That is, the assumption of normality is provided for the control group.
"""

"""
1. Variance Assumption

    -H0: the compared groups have equal variance.
    -H1: the compared groups do not have equal variance. 
    
We do not want to reject the null hypothesis in the tests that might be considered for assumptions.

p-value < 0.05 (H0 rejected)
p-value > 0.05 (H0 not rejected)
"""

# Levene Test
test_st, p_value = levene(df["Purchase_A"],df["Purchase_B"])
print('Test statistic = %.4f, p-Value = %.4f' %(test_st, p_value))

# OUT : Test statistic = 2.6393, p-Value = 0.1083

""""
*** P-VALUE = 0.1083 *** 
--> Since the p value is not less than 0.05, 
the h0 hypothesis cannot be rejected. That is, the assumption of variance is provided.
"""

# Assumptions provided. Thus, we can apply the independent two-sample t-test (parametric test).

test_st, p_value = ttest_ind(df["Purchase_A"], df["Purchase_B"], equal_var=True)
print('tvalue = %.4f, pvalue = %.4f' %(test_st, p_value))

# OUT : tvalue = -0.9416, pvalue = 0.3493

# Result :
"""
**** P-VALUE = 0.3493 Since the p value is not less than 0.05, the h0 hypothesis cannot be rejected.
---> So, There is no statistically significant difference between, the Control( “maximum bidding”) 
     campaign and Test group(average bidding) campaign.
"""