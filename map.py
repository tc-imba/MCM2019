import plotly.figure_factory
import plotly

import pandas as pd
import numpy as np

df = pd.read_excel('data/MCM_NFLIS_Data.xlsx', sheet_name='Data')
fips_dict = dict()
for index, row in df.iterrows():
    fips_dict[row['FIPS_Combined']] = 0

df_2010_Heroin = df[(df['SubstanceName'] == 'Heroin')]

for index, row in df_2010_Heroin.iterrows():
    fips_dict[row['FIPS_Combined']] += row['DrugReports']

fips = list(fips_dict.keys())
values = list(fips_dict.values())

plotly.tools.set_credentials_file(username='tc-imba', api_key='xNu6lsfY6Twz6LfUmjHa')
endpts = list(np.geomspace(10, 1000, 5))

fig = plotly.figure_factory.create_choropleth(
    fips=fips, values=values,
    scope=['VA', 'OH', 'PA', 'KY', 'WV'],
    binning_endpoints=endpts,
    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
    legend_title='Drug Reports'
)

fig['layout']['legend'].update({'x': 0.25, 'y': 0.75})
fig['layout']['annotations'][0].update({'x': 0.12, 'y': 0.8, 'xanchor': 'left'})

# plotly.offline.plot(fig, filename='map.html')
plotly.plotly.plot(fig, filename='test', fileopt='overwrite', auto_open=False)
# # plotly.plotly.image.save_as(fig, 'test.png', width=1920, height=1080)
