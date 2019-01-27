import sys
import xlrd
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
# from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# NFLIS Data
book = xlrd.open_workbook('MCM_NFLIS_Data.xlsx')
sheet = book.sheet_by_name('Data')
rows = sheet.nrows
columns = sheet.ncols
print(rows, columns)

dict1 = {}
for i in range(1, sheet.nrows):
    if sheet.cell(i, 5).value in dict1.keys():
        continue
    dict1[int(sheet.cell(i, 5).value)] = sheet.cell(i, 8).value
y = pd.DataFrame(list(dict1.items()), columns=['A', 'B'])

y = y.sort_values(by='A', axis=0, ascending=True)
col1 = y.iloc[:, 0]
arrs1 = col1.values

col2 = y.iloc[:, 1]
arrs2 = col2.values
# print(len(arrs2))

df = pd.read_csv('ACS_10_5YR_DP02_with_ann.csv', header=[0, 1])
df.columns = df.columns.get_level_values(0)
df = df.filter(regex='^(HC01|GEO)', axis=1)
# rownum = len(df)
for e in df['GEO.id2']:
    if float(e) not in arrs1:
        df = df[~df['GEO.id2'].isin([e])]
# D = df.shape[1]
# N = len(df)
# print(D)
# print(N)
# df = df.drop([df.columns[[0, 1, 2]]], axis=1, inplace=True)
df = df.filter(regex='^(HC01)', axis=1)
df = df.astype(float)
X = df.values

X = preprocessing.normalize(X, norm='l2')

pca = PCA(n_components=10)

model = pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.components_)
A = pca.transform(X)

sys.exit(0)

#
#
diabetes_X = A

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = arrs2[:-20]
diabetes_y_test = arrs2[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
#
plt.show()