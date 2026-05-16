import pandas as pd

for fname in ['train.csv', 'val.csv', 'test.csv']:
    df = pd.read_csv(fname)
    df = df[['text', 'label']]
    df = df.dropna()
    df.to_csv(fname, index=False)
    print(f'{fname}: {len(df)} rows, columns: {df.columns.tolist()}')