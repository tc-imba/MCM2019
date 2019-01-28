import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing

import numpy as np
from sklearn.linear_model import LinearRegression

df_data = pd.DataFrame()
for i in range(7):
    year = '1%d' % i
    filename = 'data/ACS_1%d_5YR_DP02/ACS_1%d_5YR_DP02_with_ann.csv' % (i, i)
    df_temp = pd.read_csv(filename, header=[0, 1])
    df_temp.columns = df_temp.columns.get_level_values(0)
    df_temp = df_temp.filter(regex='^(HC01|GEO.id2)', axis=1)
    columns = df_temp.columns.tolist()
    columns = ['YYYY', 'State'] + columns
    df_temp = df_temp.reindex(columns=columns)
    df_temp['YYYY'] = '201%d' % i
    df_temp['State'] = df_temp['GEO.id2'].apply(lambda fips: str(fips)[0:2])
    # df_temp['GEO.id2'] = df_temp['GEO.id2'].apply(lambda fips: '%d,201%d' % (fips, i))
    df_data = df_data.append(df_temp, sort=False)


def preprocess_data(x):
    if isinstance(x, str) and x == '(X)':
        return 0
    if not isinstance(x, str) and np.isnan(x):
        return 0
    return x


df_data = df_data.applymap(preprocess_data)
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, None]
states = [21, 39, 42, 51, 54, None]
states_dict = {
    21: 'KY',
    39: 'OH',
    42: 'PA',
    51: 'VA',
    54: 'WV'
}


def apply_pca(state=None, year=None, n_components=10):
    df = df_data.copy()
    if state:
        df = df[df['State'] == str(state)]
    if year:
        df = df[df['YYYY'] == str(year)]
    df = df.reset_index(drop=True)

    X = df[df.columns[3:]].astype(np.float64)
    X = preprocessing.scale(X)

    pca = PCA(n_components=n_components)
    pca.fit(X)
    # print(pca.explained_variance_ratio_)

    A = pca.transform(X)

    df_result = pd.concat([df[df.columns[0:3]], pd.DataFrame(A)], axis=1)
    df_result['key'] = df_result['YYYY'].astype(str) + df_result['GEO.id2'].astype(str)
    df_result.set_index('key', inplace=True)
    df_result.drop(columns=['YYYY', 'State', 'GEO.id2'], inplace=True)

    df_test = pd.read_excel('data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
    df_test = df_test.groupby(['YYYY', 'FIPS_Combined']).first().reset_index()
    # df_test = df_test[df_test['YYYY'] == 2010].reset_index()
    df_test['key'] = df_test['YYYY'].astype(str) + df_test['FIPS_Combined'].astype(str)
    df_test.set_index('key', inplace=True)
    df_test = df_test.filter(['TotalDrugReportsCounty'], axis=1)

    df_result = df_result.join(df_test)
    df_result.dropna(inplace=True)

    result_arr = []
    year_name = year and str(year) or 'All'
    state_name = state and states_dict[state] or 'All'

    for i in range(1, n_components + 1):
        X = df_result[df_result.columns[:i]].values
        y = df_result[df_result.columns[-1]].values

        data_length = X.shape[0]
        train_data_length = data_length * 2 // 3

        reg_score = []
        for j in range(10):
            arr = np.arange(data_length)
            np.random.shuffle(arr)

            train_X = X[arr[:train_data_length]]
            train_y = y[arr[:train_data_length]]
            test_X = X[arr[train_data_length:]]
            test_y = y[arr[train_data_length:]]

            reg = LinearRegression().fit(train_X, train_y)
            reg_score.append(reg.score(test_X, test_y))
            # print(reg.score(train_X, train_y))
            # print(reg.score(test_X, test_y))
            # print(reg.coef_)
            # print(reg.intercept_)

        explained_variance_ratio_sum = np.sum(pca.explained_variance_ratio_[:i])
        score = np.average(reg_score)

        result = [year_name, state_name, i, explained_variance_ratio_sum, score]
        result_arr.append(result)
        print(result)

    df = pd.DataFrame(result_arr, columns=['YYYY', 'State', 'components', 'ratio', 'score'])
    return df


df_plot = pd.DataFrame(columns=['YYYY', 'State', 'components', 'ratio', 'score'])
for i, state in enumerate(states):
    df_plot = df_plot.append(apply_pca(state=state, n_components=10), ignore_index=True)

df_plot.to_csv('result/pca_state.csv', index=False)

df_plot = pd.DataFrame(columns=['YYYY', 'State', 'components', 'ratio', 'score'])
for i, year in enumerate(years):
    df_plot = df_plot.append(apply_pca(year=year, n_components=10), ignore_index=True)

df_plot.to_csv('result/pca_year.csv', index=False)
