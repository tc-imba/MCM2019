from geopy.geocoders import Bing
from geopy.extra.rate_limiter import RateLimiter

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

geolocator = Bing(api_key='AtlqFZkg8aISlN-CNOpURU3oLthMK6g166C9gDCO1sc9Cl5njVi1NcGaqwIW21vd')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.1)

df = pd.read_excel('../data/MCM_NFLIS_Data.xlsx', sheet_name='Data')

df = df.groupby('FIPS_Combined').first().reset_index()

df['name'] = df['COUNTY'] + ', ' + df['State']
df['location'] = df['name'].progress_apply(geocode)
df['latitude'] = df['location'].apply(lambda loc: loc and loc.latitude or 0)
df['longitude'] = df['location'].apply(lambda loc: loc and loc.longitude or 0)

df = df.reindex(columns=['FIPS_Combined', 'State', 'COUNTY', 'latitude', 'longitude'])

df.to_csv('geocode.csv', index=False)
print(df)
