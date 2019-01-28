import pandas as pd

df = pd.read_excel('../data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
df = df[['State', 'SubstanceName', 'DrugReports']]

df_grouped = df.groupby(['State', 'SubstanceName'],as_index=False).count()

df_grouped.sort_values(by='DrugReports', inplace=True, ascending=False)
df_grouped.to_csv('drug_count.csv', index=False)
