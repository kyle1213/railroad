import pandas as pd

a = pd.read_csv('./data/answer_sample.csv')
print(list(a.keys()))
print(a['Distance'].head())
print(a["Distance"].values)
for d in a["Distance"].values:
    print([d].append(1))