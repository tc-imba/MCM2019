import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey']

substance_name = 'Buprenorphine'

df = pd.read_csv('../result/bpnn_source_state_%s.csv' % substance_name)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
x = np.arange(2008, 2020)

for i, (name, row) in enumerate(df.iterrows()):
    color = colors[i]
    y = row.values
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.grid(True, 'both')
plt.xticks(x)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Drug Report', fontsize=15)
plt.ylim([0, 2500])
plt.legend(loc='upper left', prop={'size': 15})
plt.title('Year vs Drug Report (%s)' % substance_name)
plt.savefig('../result/state_%s.eps' % substance_name)
plt.show()
