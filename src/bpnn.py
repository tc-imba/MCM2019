import pandas as pd
import numpy as np

total_years = 8
start_year = 2010

training_rate = 0.1
weight_default = 0.2


class Path:
    def __init__(self, src, dest, dist):
        self.src = src
        self.dest = dest
        self.dist = dist
        self.weight = [weight_default, weight_default]


class County:
    def __init__(self, fips, name):
        self.fips = fips
        self.name = name
        self.paths = []
        self.train_data = [1 for i in range(total_years)]
        self.predict_data = []
        self.aggregate_data = 0
        self.transfer_rate = 0

    def add_path(self, dest_county, dist):
        self.paths.append(Path(self, dest_county, dist))

    def set_train_data(self, year, data):
        self.train_data[year] += data

    def prepare_train(self, year):
        self.aggregate_data = self.train_data[year + 1]

    def train(self, year):
        for path in self.paths:
            if year > 0:
                path.dest.aggregate_data -= (path.weight[0] * self.train_data[year - 1])
            path.dest.aggregate_data -= (path.weight[1] * self.train_data[year])

    def back_propagation(self, year):
        for path in self.paths:
            error = path.dest.aggregate_data / len(path.dest.paths) / len(path.weight)
            # error_rate = error / path.dest.train_data[year + 1]
            path.weight[1] = max(0, path.weight[1] + training_rate * error / path.dest.train_data[year])
            if year > 0:
                path.weight[0] = max(0, path.weight[0] + training_rate * error / path.dest.train_data[year - 1])

    def calculate_transfer_rate(self):
        self.transfer_rate = 0
        for path in self.paths:
            if path.src != path.dest:
                self.transfer_rate += path.weight[1]
        return self.transfer_rate

    def prepare_predict(self, years=10):
        self.predict_data = [1 for i in range(years + 2)]
        self.predict_data[0] = self.train_data[-2]
        self.predict_data[1] = self.train_data[-1]

    def predict(self, year):
        for path in self.paths:
            path.dest.predict_data[year + 2] += path.weight[0] * self.predict_data[year] + path.weight[1] * \
                                                self.predict_data[year + 1]


df = pd.read_excel('../data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
df_distance = pd.read_csv('geocode_distance.csv', index_col=0)


def get_distance(a, b):
    return df_distance[str(a)].loc[int(b)]


state = 'WV'
# substance_name = 'Oxycodone'
substance_name = 'Heroin'
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
    # year = row['YYYY'] - start_year
    year = start_year - 1 + total_years - row['YYYY']
    data = row['DrugReports']
    county_list[fips].set_train_data(year, data)

mean = 0
for year in range(total_years - 1):
    for _, county_i in county_list.items():
        mean += county_i.train_data[year + 1]
mean = mean / (total_years - 1) / len(county_list)

for i in range(10):
    u = 0
    v = 0
    for year in range(total_years - 1):
        for _, county_i in county_list.items():
            county_i.prepare_train(year)

        for _, county_i in county_list.items():
            county_i.train(year)

        for _, county_i in county_list.items():
            u += county_i.aggregate_data ** 2
            v += (county_i.train_data[year + 1] - mean) ** 2
    error = 1 - u / v
    print(error, u)

    for year in range(total_years - 1):
        for _, county_i in county_list.items():
            county_i.prepare_train(year)

        for _, county_i in county_list.items():
            county_i.train(year)

        for _, county_i in county_list.items():
            county_i.back_propagation(year)

predict_years = 3

for _, county_i in county_list.items():
    county_i.prepare_predict(predict_years)

for i in range(predict_years):
    for _, county_i in county_list.items():
        county_i.predict(i)

for _, county_i in county_list.items():
    # if county_i.predict_data[-1] >= 10:
    print(_, county_i.train_data, county_i.predict_data)
