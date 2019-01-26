import pandas as pd
import numpy as np

df = pd.read_excel('data/MCM_NFLIS_Data.xlsx', sheet_name='Data')

df_VA_Heroin = df[(df['State'] == 'VA') & (df['SubstanceName'] == 'Heroin')]



print(len(df_VA_Heroin))

