import pandas as pd

# Reload the full dataset and redo the split properly
train = pd.read_csv('train.csv')
val   = pd.read_csv('val.csv')
test  = pd.read_csv('test.csv')

# Combine everything back
all_df = pd.concat([train, val, test], ignore_index=True)
all_df = all_df.drop_duplicates(subset='text').reset_index(drop=True)
all_df = all_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Proper 70/15/15 split
n = len(all_df)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

train = all_df[:train_end]
val   = all_df[train_end:val_end]
test  = all_df[val_end:]

# Balance each split
def balance(df):
    b = df[df['label']==1]
    n = df[df['label']==0]
    m = min(len(b), len(n))
    return pd.concat([
        b.sample(n=m, random_state=42),
        n.sample(n=m, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

train = balance(train)
val   = balance(val)
test  = balance(test)

train.to_csv('train.csv', index=False)
val.to_csv('val.csv',     index=False)
test.to_csv('test.csv',   index=False)

total = len(train)+len(val)+len(test)
print(f'Train: {len(train)} | 0:{(train.label==0).sum()} 1:{(train.label==1).sum()}')
print(f'Val:   {len(val)}   | 0:{(val.label==0).sum()} 1:{(val.label==1).sum()}')
print(f'Test:  {len(test)}  | 0:{(test.label==0).sum()} 1:{(test.label==1).sum()}')
print(f'Total: {total}')