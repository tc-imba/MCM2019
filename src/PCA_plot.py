import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey']


def formatter_func(x, pos):
    return '%.1f' % x


df = pd.read_csv('../result/pca_state.csv')
n_components = df['components'].max()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i, (name, group) in enumerate(df.groupby('State')):
    color = colors[i]
    x = group['components']
    y = group['ratio']
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.yscale('log')
plt.grid(True, 'both')
plt.xticks(np.arange(1, n_components + 1))
plt.xlabel('components', fontsize=15)
plt.ylabel('explained variance ratio', fontsize=15)
plt.ylim([0.4, 1])
plt.legend(loc='lower right', prop={'size': 15})
plt.title('PCA components vs explained variance ratio')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(formatter_func))
plt.savefig('../result/pca_state_ratio.eps')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i, (name, group) in enumerate(df.groupby('State')):
    color = colors[i]
    x = group['components']
    y = group['score']
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.grid(True)
plt.xticks(np.arange(1, n_components + 1))
plt.xlabel('components', fontsize=15)
plt.ylabel('coefficient of determination', fontsize=15)
plt.ylim([0, 1])
plt.legend(loc='lower right', prop={'size': 15})
plt.title('PCA components vs coefficient of determination')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(formatter_func))
plt.savefig('../result/pca_state_score.eps')
plt.show()

df = pd.read_csv('../result/pca_year.csv')
n_components = df['components'].max()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i, (name, group) in enumerate(df.groupby('YYYY')):
    color = colors[i]
    x = group['components']
    y = group['ratio']
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.yscale('log')
plt.grid(True, 'both')
plt.xticks(np.arange(1, n_components + 1))
plt.xlabel('components', fontsize=15)
plt.ylabel('explained variance ratio', fontsize=15)
plt.ylim([0.75, 1])
plt.legend(loc='lower right', prop={'size': 15})
plt.title('PCA components vs explained variance ratio')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(formatter_func))
plt.savefig('../result/pca_year_ratio.eps')
plt.show()


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i, (name, group) in enumerate(df.groupby('YYYY')):
    color = colors[i]
    x = group['components']
    y = group['score']
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.grid(True)
plt.xticks(np.arange(1, n_components + 1))
plt.xlabel('components', fontsize=15)
plt.ylabel('coefficient of determination', fontsize=15)
plt.ylim([-1, 1])
plt.legend(loc='lower right', prop={'size': 15})
plt.title('PCA components vs coefficient of determination')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(formatter_func))
plt.savefig('../result/pca_year_score.eps')
plt.show()


df = pd.read_csv('../result/pca_state_2010.csv')
n_components = df['components'].max()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i, (name, group) in enumerate(df.groupby('State')):
    color = colors[i]
    x = group['components']
    y = group['ratio']
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.yscale('log')
plt.grid(True, 'both')
plt.xticks(np.arange(1, n_components + 1))
plt.xlabel('components', fontsize=15)
plt.ylabel('explained variance ratio', fontsize=15)
plt.ylim([0.7, 1])
plt.legend(loc='lower right', prop={'size': 15})
plt.title('PCA components vs explained variance ratio (2010)')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(formatter_func))
plt.savefig('../result/pca_state_2010_ratio.eps')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i, (name, group) in enumerate(df.groupby('State')):
    color = colors[i]
    x = group['components']
    y = group['score']
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.grid(True)
plt.xticks(np.arange(1, n_components + 1))
plt.xlabel('components', fontsize=15)
plt.ylabel('coefficient of determination', fontsize=15)
plt.ylim([-1, 1])
plt.legend(loc='lower right', prop={'size': 15})
plt.title('PCA components vs coefficient of determination (2010)')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(formatter_func))
plt.savefig('../result/pca_state_2010_score.eps')
plt.show()

df = pd.read_csv('../result/pca_year_PA.csv')
n_components = df['components'].max()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i, (name, group) in enumerate(df.groupby('YYYY')):
    color = colors[i]
    x = group['components']
    y = group['ratio']
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.yscale('log')
plt.grid(True, 'both')
plt.xticks(np.arange(1, n_components + 1))
plt.xlabel('components', fontsize=15)
plt.ylabel('explained variance ratio', fontsize=15)
plt.ylim([0.5, 1])
plt.legend(loc='lower right', prop={'size': 15})
plt.title('PCA components vs explained variance ratio (PA)')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(formatter_func))
plt.savefig('../result/pca_year_PA_ratio.eps')
plt.show()


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i, (name, group) in enumerate(df.groupby('YYYY')):
    color = colors[i]
    x = group['components']
    y = group['score']
    h = plt.plot(x, y, label=name,
                 linestyle='-', linewidth=1, color=colors[i],
                 marker='o', markersize=5, markerfacecolor=colors[i])

plt.grid(True)
plt.xticks(np.arange(1, n_components + 1))
plt.xlabel('components', fontsize=15)
plt.ylabel('coefficient of determination', fontsize=15)
plt.ylim([-1, 1])
plt.legend(loc='lower right', prop={'size': 15})
plt.title('PCA components vs coefficient of determination (PA)')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(formatter_func))
plt.savefig('../result/pca_year_PA_score.eps')
plt.show()
