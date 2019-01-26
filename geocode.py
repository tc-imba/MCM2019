from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

geolocator = Nominatim(user_agent='mcm2019-1920446')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

df = pd.read_excel('data/MCM_NFLIS_Data.xlsx', sheet_name='Data')

df = df.groupby('FIPS_Combined').first().reset_index()

df['name'] = df['COUNTY'] + ', ' + df['State']
df['location'] = df['name'].progress_apply(geocode)
df['latitude'] = df['location'].apply(lambda loc: loc and loc.latitude or 0)
df['longitude'] = df['location'].apply(lambda loc: loc and loc.longitude or 0)

df = df.reindex(columns=['FIPS_Combined', 'State', 'COUNTY', 'latitude', 'longitude'])

df.to_csv('geocode.csv', index=False)
print(df)
