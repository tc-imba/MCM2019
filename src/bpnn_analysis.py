import pandas as pd

# state = 'OH'
# substance_name = 'Buprenorphine'
state = 'KY'
substance_name = 'Hydrocodone'
df = pd.read_csv('../result/bpnn_source_%s_%s.csv' % (state, substance_name))

Q1 = df.quantile(0.25)[10]
Q3 = df.quantile(0.75)[10]
IQR = Q3 - Q1
Low = Q1 - 1.5 * IQR
High = Q3 + 1.5 * IQR

df_high = df[df['10'] > High].copy()
df_high.sort_values('10', ascending=False, inplace=True)

df_county = pd.read_excel('../data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
df_county = df_county.groupby('FIPS_Combined').first().reset_index()
df_county.set_index('FIPS_Combined', inplace=True)
df_county = df_county[['COUNTY', 'State']]

df_high.set_index('0', inplace=True)
df_high = df_high[['10']]

df_join = df_high.join(df_county)

for index, row in df_join.iterrows():
    print(row['COUNTY'], row['State'])

df_join.to_csv('../result/bpnn_source_%s_%s_county.csv' % (state, substance_name))
