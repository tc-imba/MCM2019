import pandas as pd
import numpy as np
import geopy.distance
from tqdm import tqdm
import sklearn.metrics

tqdm.pandas()

df_geocode = pd.read_csv('geocode_final.csv')
df_geocode.set_index('FIPS_Combined', inplace=True, drop=False)

df_temp = df_geocode[['latitude', 'longitude']]

distance_matrix = sklearn.metrics.pairwise_distances(
    df_temp, metric=lambda a, b: geopy.distance.distance(a, b).miles, n_jobs=8)

df_distance = pd.DataFrame(distance_matrix,
                           columns=df_geocode['FIPS_Combined'],
                           index=df_geocode['FIPS_Combined'])

df_distance.to_csv('geocode_distance.csv')
