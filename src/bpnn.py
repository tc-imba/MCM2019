import pandas as pd
import numpy as np
from tqdm import tqdm

total_years = 8
start_year = 2010

training_rate = 1e-6
weight_default = 0
weight_dim = 6
weight_dim_used = 6


class Path:
    def __init__(self, src, dest, dist):
        self.src = src
        self.dest = dest
        self.dist = dist
        self.weight = [weight_default for i in range(weight_dim)]


class County:
    def __init__(self, fips, name):
        self.fips = fips
        self.name = name
        self.paths = []
        self.train_data = [[0 for j in range(weight_dim)] for i in range(total_years)]
        self.predict_data = []
        self.aggregate_data = 0
        self.aggregate_data_size = [0 for i in range(weight_dim)]
        self.transfer_rate = 0

    def add_path(self, dest_county, dist):
        self.paths.append(Path(self, dest_county, dist))

    def set_train_data_drug(self, year, data):
        year = int(year)
        self.train_data[year][0] += data
        if year < total_years - 1:
            self.train_data[year + 1][1] += data

    def set_train_data_pca(self, year, data):
        year = int(year)
        for i, j in enumerate(data):
            self.train_data[year][i + 2] += j

    def prepare_train(self, year):
        self.aggregate_data = self.train_data[year + 1][0]
        self.aggregate_data_size = [0 for i in range(weight_dim)]

    def train(self, year):
        for path in self.paths:
            for i in range(2):
                if self.train_data[year][i] >= 10:
                    path.dest.aggregate_data -= (path.weight[i] * self.train_data[year][i])
                    path.dest.aggregate_data_size[i] += 1
            for i in range(2, weight_dim_used):
                path.dest.aggregate_data -= (path.weight[i] * self.train_data[year][i])
                path.dest.aggregate_data_size[i] += 1

    def back_propagation(self, year):
        for path in self.paths:
            error = path.dest.aggregate_data / weight_dim_used
            for i in range(2):
                if self.train_data[year][i] > 10:
                    e = error / path.dest.aggregate_data_size[i]
                    path.weight[i] = max(-1, path.weight[i] + training_rate * e * self.train_data[year][i])

            for i in range(2, weight_dim_used):
                e = error / path.dest.aggregate_data_size[i]
                path.weight[i] = max(-1, path.weight[i] + training_rate * e * self.train_data[year][i])

    def prepare_predict(self, years=10):
        self.predict_data = [[0 for j in range(weight_dim)] for i in range(years + 2)]
        self.predict_data[0] = self.train_data[-2]
        self.predict_data[1] = self.train_data[-1]

    def predict(self, year):
        for path in self.paths:
            for i in range(weight_dim_used):
                path.dest.predict_data[year + 2][0] += path.weight[i] * self.predict_data[year][i]
        self.predict_data[year + 2][1] = self.predict_data[year + 1][0]
        for i in range(2, weight_dim):
            self.predict_data[year + 2][i] = self.predict_data[year + 1][i]


# df = pd.read_excel('../data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
df_distance = pd.read_csv('geocode_distance.csv', index_col=0)
df_pca = pd.read_csv('../result/pca.csv')


def get_distance(a, b):
    return df_distance[str(a)].loc[int(b)]


# state = 'OH'
# substance_name = 'Buprenorphine'


# state = 'KY'
# substance_name = 'Hydrocodone'


def bpnn(state, substance_name, n_components=0, reverse=False):
    global weight_dim_used
    weight_dim_used = 2 + n_components

    df = pd.read_excel('../data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
    df_county = df[df['State'] == state].groupby('FIPS_Combined').first()
    df = df[(df['State'] == state) & (df['SubstanceName'] == substance_name)].reset_index()

    county_list = dict()

    for fips, row in df_county.iterrows():
        county_list[fips] = County(fips, row['COUNTY'])

    max_min_distance = 0

    for _i, county_i in county_list.items():
        min_distance = float('inf')
        for _j, county_j in county_list.items():
            distance = get_distance(county_i.fips, county_j.fips)
            if distance and distance < min_distance:
                min_distance = distance
            if distance < 40:
                county_i.add_path(county_j, distance)
        # if max_min_distance < min_distance:
        #     print(_i, min_distance)
        max_min_distance = max(max_min_distance, min_distance)

    # print(max_min_distance)

    for index, row in df.iterrows():
        fips = row['FIPS_Combined']
        if reverse:
            year = start_year - 1 + total_years - row['YYYY']
        else:
            year = row['YYYY'] - start_year
        data = row['DrugReports']
        county_list[fips].set_train_data_drug(year, data)

    for index, row in df_pca.iterrows():
        fips = row['FIPS']
        if fips not in county_list:
            continue
        if reverse:
            year = start_year - 1 + total_years - row['YYYY']
        else:
            year = row['YYYY'] - start_year
        data = [row['D%d' % i] for i in range(1, 5)]
        county_list[fips].set_train_data_pca(year, data)

    mean = 0
    for year in range(total_years - 1):
        for _, county_i in county_list.items():
            mean += county_i.train_data[year + 1][0]
    mean = mean / (total_years - 1) / len(county_list)

    print(mean)

    u = 0
    error = 0
    for i in tqdm(range(1000)):
        # u = 0
        # v = 0
        # for year in range(total_years - 1):
        #     for _, county_i in county_list.items():
        #         county_i.prepare_train(year)
        #
        #     for _, county_i in county_list.items():
        #         county_i.train(year)
        #
        #     for _, county_i in county_list.items():
        #         u += county_i.aggregate_data ** 2
        #         v += (county_i.train_data[year + 1][0] - mean) ** 2
        # error = 1 - u / v
        # print(error, u)

        for year in range(total_years - 1):
            for _, county_i in county_list.items():
                county_i.prepare_train(year)

            for _, county_i in county_list.items():
                county_i.train(year)

            for _, county_i in county_list.items():
                county_i.back_propagation(year)

    u = 0
    v = 0
    error = 0
    for year in range(total_years - 1):
        for _, county_i in county_list.items():
            county_i.prepare_train(year)

        for _, county_i in county_list.items():
            county_i.train(year)

        for _, county_i in county_list.items():
            u += county_i.aggregate_data ** 2
            v += (county_i.train_data[year + 1][0] - mean) ** 2
    error = 1 - u / v

    predict_years = 2

    for _, county_i in county_list.items():
        county_i.prepare_predict(predict_years)

    for i in range(predict_years):
        for _, county_i in county_list.items():
            county_i.predict(i)

    result_arr = []
    for _, county_i in county_list.items():
        # if county_i.predict_data[-1] >= 10:
        # print(_, county_i.train_data, county_i.predict_data)
        arr = [_]
        for i in range(total_years):
            arr.append(county_i.train_data[i][0])
        for i in range(2):
            arr.append(county_i.predict_data[2 + i][0])

        result_arr.append(arr)

    df_result = pd.DataFrame(result_arr)
    # df_result.set_index(0, inplace=True)

    df_result.to_csv('../result/bpnn_source_%s_%s.csv' % (state, substance_name), index=False)
    return [state, substance_name, n_components, error]


#
# arr = list()
#
# arr.append(bpnn('KY', 'Hydrocodone', 0, True))
# arr.append(bpnn('KY', 'Hydrocodone', 1, True))
# arr.append(bpnn('OH', 'Hydrocodone', 0, True))
# arr.append(bpnn('OH', 'Hydrocodone', 1, True))
# arr.append(bpnn('PA', 'Hydrocodone', 0, True))
# arr.append(bpnn('PA', 'Hydrocodone', 1, True))
# arr.append(bpnn('VA', 'Hydrocodone', 0, True))
# arr.append(bpnn('VA', 'Hydrocodone', 1, True))
# arr.append(bpnn('WV', 'Hydrocodone', 0, True))
# arr.append(bpnn('WV', 'Hydrocodone', 1, True))
#
# arr.append(bpnn('KY', 'Buprenorphine', 0, False))
# arr.append(bpnn('KY', 'Buprenorphine', 1, False))
# arr.append(bpnn('OH', 'Buprenorphine', 0, False))
# arr.append(bpnn('OH', 'Buprenorphine', 1, False))
# arr.append(bpnn('PA', 'Buprenorphine', 0, False))
# arr.append(bpnn('PA', 'Buprenorphine', 1, False))
# arr.append(bpnn('VA', 'Buprenorphine', 0, False))
# arr.append(bpnn('VA', 'Buprenorphine', 1, False))
# arr.append(bpnn('WV', 'Buprenorphine', 0, False))
# arr.append(bpnn('WV', 'Buprenorphine', 1, False))
#
# df_total = pd.DataFrame(arr, columns=['State', 'Substance', 'N', 'R2'])
# df_total.to_csv('../result/bpnn.csv')


def bpnn_state(substance_name, reverse=False):
    global weight_dim_used
    weight_dim_used = 2
    df = pd.read_excel('../data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
    df = df[df['SubstanceName'] == substance_name].reset_index()
    df_state = df.groupby(['State', 'YYYY'], as_index=False).sum().reset_index()

    state_list = dict()

    for index, row in df_state.iterrows():
        state_name = row['State']
        state_list[state_name] = County(state_name, state_name)

    for _i, state_i in state_list.items():
        for _j, state_j in state_list.items():
            state_i.add_path(state_j, 0)

    for index, row in df_state.iterrows():
        state_name = row['State']
        if reverse:
            year = start_year - 1 + total_years - row['YYYY']
        else:
            year = row['YYYY'] - start_year
        data = row['DrugReports']
        state_list[state_name].set_train_data_drug(year, data)

    mean = 0
    for year in range(total_years - 1):
        for _, state_i in state_list.items():
            mean += state_i.train_data[year + 1][0]
    mean = mean / (total_years - 1) / len(state_list)

    print(mean)

    u = 0
    error = 0
    for i in tqdm(range(1000)):
        # u = 0
        # v = 0
        # for year in range(total_years - 1):
        #     for _, state_i in state_list.items():
        #         state_i.prepare_train(year)
        #
        #     for _, state_i in state_list.items():
        #         state_i.train(year)
        #
        #     for _, state_i in state_list.items():
        #         u += state_i.aggregate_data ** 2
        #         v += (state_i.train_data[year + 1][0] - mean) ** 2
        # error = 1 - u / v
        # print(error, u)

        for year in range(total_years - 1):
            for _, state_i in state_list.items():
                state_i.prepare_train(year)

            for _, state_i in state_list.items():
                state_i.train(year)

            for _, state_i in state_list.items():
                state_i.back_propagation(year)

    u = 0
    v = 0
    error = 0
    for year in range(total_years - 1):
        for _, state_i in state_list.items():
            state_i.prepare_train(year)

        for _, state_i in state_list.items():
            state_i.train(year)

        for _, state_i in state_list.items():
            u += state_i.aggregate_data ** 2
            v += (state_i.train_data[year + 1][0] - mean) ** 2
    error = 1 - u / v

    print(error)

    predict_years = 2

    for _, state_i in state_list.items():
        state_i.prepare_predict(predict_years)

    for i in range(predict_years):
        for _, state_i in state_list.items():
            state_i.predict(i)

    result_arr = []
    for _, state_i in state_list.items():
        # if county_i.predict_data[-1] >= 10:
        # print(_, county_i.train_data, county_i.predict_data)
        arr = [_]
        for i in range(total_years):
            arr.append(state_i.train_data[i][0])
        for i in range(2):
            arr.append(state_i.predict_data[2 + i][0])

        result_arr.append(arr)

    df_result = pd.DataFrame(result_arr)
    df_result.set_index(0, inplace=True)

    return df_result

    # df_result.to_csv('../result/bpnn_source_state_%s.csv' % substance_name, index=False)
    # return [substance_name, error]


def bpnn_state_bidirection(substance_name):
    df_result1 = bpnn_state('Buprenorphine', False)
    df_result2 = bpnn_state('Buprenorphine', True).drop([1, 2, 3, 4, 5, 6, 7, 8], axis=1)
    df_result2.columns = ['a', 'b']
    df_result = pd.concat([df_result2[['b', 'a']], df_result1], axis=1)
    df_result.to_csv('../result/bpnn_source_state_%s.csv' % substance_name, index=False)
    return df_result


df_temp = bpnn_state_bidirection('Buprenorphine')
