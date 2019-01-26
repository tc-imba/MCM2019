import pandas as pd
import math

total_years = 8
start_year = 2010
training_rate = 0.1


class Path:
    def __init__(self, src, dest, dist):
        self.src = src
        self.dest = dest
        self.dist = dist
        self.ratio = 0


class County:
    def __init__(self, fips, name):
        self.fips = fips
        self.name = name
        self.paths = []
        self.train_data = [1 for i in range(total_years)]
        self.aggregate_data = 0

    def add_path(self, dest_county, dist):
        self.paths.append(Path(self, dest_county, dist))

    def set_train_data(self, year, data):
        self.train_data[year] += data

    def prepare(self, year):
        self.aggregate_data = self.train_data[year + 1]

    def train(self, year):
        for path in self.paths:
            path.dest.aggregate_data -= path.ratio * self.train_data[year]

    def back_propagation(self, year):
        for path in self.paths:
            error = path.dest.aggregate_data / len(path.dest.paths)
            error_rate = error / path.dest.train_data[year + 1]
            path.ratio = max(path.ratio + training_rate * error_rate, 0)
            # if path.ratio == 0:
            #     print(path.ratio)


df = pd.read_excel('data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
df_distance = pd.read_csv('geocode_distance.csv', index_col=0)


def get_distance(a, b):
    return df_distance[str(a)].loc[int(b)]


df_county = df[df['State'] == 'PA'].groupby('FIPS_Combined').first()
df = df[(df['State'] == 'PA') & (df['SubstanceName'] == 'Heroin')].reset_index()

county_list = dict()

for fips, row in df_county.iterrows():
    county_list[fips] = County(fips, row['COUNTY'])

max_min_distance = 0

for _, county_i in county_list.items():
    min_distance = float('inf')
    for _, county_j in county_list.items():
        distance = get_distance(county_i.fips, county_j.fips)
        if distance and distance < min_distance:
            min_distance = distance
        if distance < 40:
            county_i.add_path(county_j, distance)
    max_min_distance = max(max_min_distance, min_distance)

for index, row in df.iterrows():
    fips = row['FIPS_Combined']
    year = row['YYYY'] - start_year
    data = row['DrugReports']
    county_list[fips].set_train_data(year, data)

mean = 0
for year in range(total_years - 1):
    for _, county_i in county_list.items():
        mean += county_i.train_data[year + 1]
mean = mean / (total_years - 1) / len(county_list)

for i in range(100):
    u = 0
    v = 0
    for year in range(total_years - 1):
        for _, county_i in county_list.items():
            county_i.prepare(year)

        for _, county_i in county_list.items():
            county_i.train(year)

        for _, county_i in county_list.items():
            u += county_i.aggregate_data ** 2
            v += (county_i.train_data[year + 1] - mean) ** 2
    error = 1 - u / v
    print(error)

    for year in range(total_years - 1):
        for _, county_i in county_list.items():
            county_i.prepare(year)

        for _, county_i in county_list.items():
            county_i.train(year)

        for _, county_i in county_list.items():
            county_i.back_propagation(year)

# print(max_min_distance)
