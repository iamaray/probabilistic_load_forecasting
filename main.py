from preprocessing import readtoFiltered, preprocess

df, train_loader, test_loader = preprocess(
    'data/upto_latest_actual.csv', variates=[])

print(df.head())

for x, y in train_loader:
    print(x.shape, y.shape)