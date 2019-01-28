import sys
import math

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

tqdm.pandas()

df = pd.read_excel('../data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
df_distance = pd.read_csv('geocode_distance.csv', index_col=0)


def get_distance(a, b):
    return 1 / (math.log(df_distance[str(a)].loc[int(b)] + 1) + 1)


df_VA_Heroin = df[(df['State'] == 'VA') & (df['SubstanceName'] == 'Oxycodone')]
df_VA_Heroin = df_VA_Heroin.reset_index()

df_result = df_VA_Heroin.groupby('FIPS_Combined').first()

df_temp = df_VA_Heroin.copy()

for index, row in df_result.iterrows():
    center = index
    # df_temp = df_VA_Heroin.copy()
    df_temp['distance_%d' % center] = df_temp['FIPS_Combined'].apply(lambda fips: get_distance(center, fips))
    # X = df_temp[['YYYY', 'distance']]
    # y = df_temp['DrugReports'] / df_temp['TotalDrugReportsCounty']
    #
    # reg = LinearRegression().fit(X, y)
    # print(reg.score(X, y))

    # kernel = DotProduct() + WhiteKernel()
    # gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
    # print(gpr.score(X, y))
    # break
    # df_result.loc[index]['score'] = gpr.score(X, y)

    # print(gpr.predict(X[:2,:], return_std=True))
    # print(gpr.intercept_)
    # print(gpr.predict(np.array([[3, 5]])))

# df_temp = df_temp[df_temp['TotalDrugReportsCounty'] > 10].reset_index()

X = df_temp.filter(regex='^(YYYY|distance)', axis=1)
y = df_temp['DrugReports'] / df_temp['TotalDrugReportsCounty']

reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
print(reg.coef_)
print(sorted(abs(reg.coef_), reverse=True))

y2 = reg.predict(X).reshape(-1, 1)
y = y.values.reshape(-1, 1)
arr = np.concatenate((y, y2), axis=1)
