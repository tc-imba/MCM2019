import plotly.figure_factory
import plotly

import pandas as pd
import numpy as np

plotly.tools.set_credentials_file(username='tc-imba', api_key='xNu6lsfY6Twz6LfUmjHa')

fips_dict = dict()

# state = 'OH'
# substance_name = 'Buprenorphine'
state = 'KY'
substance_name = 'Hydrocodone'
df = pd.read_csv('../result/bpnn_source_%s_%s.csv' % (state, substance_name))

colorscale = ["#deebf7", "#b3d2e9", "#6baed6", "#4292c6",
              "#2171b5", "#0b4083", "#08306b"]

for i in range(5):

    for index, row in df.iterrows():
        fips_dict[row[0]] = row[i + 1 + 4]
        # fips_dict[row[0]] = row[10 - i]

    fips = list(fips_dict.keys())
    values = list(fips_dict.values())

    endpts = list(np.geomspace(10, 200, 6))

    fig = plotly.figure_factory.create_choropleth(
        fips=fips, values=values,
        scope=[state],
        binning_endpoints=endpts,
        county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
        legend_title='Drug Reports',
        colorscale=colorscale
    )

    fig['layout']['legend'].update({'x': 0.25, 'y': 0.75})
    fig['layout']['annotations'][0].update({'x': 0.12, 'y': 0.8, 'xanchor': 'left'})

    # plotly.offline.plot(fig, filename='map.html')
    plotly.plotly.plot(fig, filename='%s_%s_%d' % (state, substance_name, i + 1), fileopt='overwrite', auto_open=True)
    # # plotly.plotly.image.save_as(fig, 'test.png', width=1920, height=1080)
