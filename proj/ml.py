import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('../data/MachineLearningWithPython/Notebooks/data/pima-data.csv')
#print(df.shape)
#print(df.head(5))

#print(df.tail(5))

#print(df.isnull().values.any())




def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    range_series = range(len(corr.columns))
    print(list(range_series))
    plt.xticks(range_series, corr.columns)
    plt.yticks(range_series, corr.columns)
    plt.show()

#skin and thickness correlate 1:1
del df['skin']

#plot_corr(df, 10)
#print(df.corr())

diabetes_map = {True: 1, False: 0}

df['diabetes'] = df['diabetes'].map(diabetes_map)
print(df['diabetes'])

num_true = len(df.loc[df['diabetes'] == 1])
num_false = len(df.loc[df['diabetes'] == 0])
print(num_true, num_false)



