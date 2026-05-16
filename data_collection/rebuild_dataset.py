import pandas as pd
import random

# Load clean data
df = pd.read_csv('raw_dataset_clean.csv')

bully = df[df['label'] == 1].copy()
normal = df[df['label'] == 0].copy()

# Aggressively augment bully to get 800 samples
augmented = []
prefixes = ['omg ', 'seriously ', 'honestly ', 'lol ', 'wtf ', 'like ', 'bro ', 'ngl ']
suffixes = [' lol', ' smh', ' fr', ' ngl', ' tbh', ' istg']

for _, row in bully.iterrows():
    text = str(row['text'])
    augmented.append({'text': text.upper(), 'label': 1})
    augmented.append({'text': random.choice(prefixes) + text, 'label': 1})
    augmented.append({'text': text + random.choice(suffixes), 'label': 1})
    augmented.append({'text': random.choice(prefixes) + text + random.choice(suffixes), 'label': 1})

aug_df = pd.DataFrame(augmented)
bully_clean = bully[['text','label']]
bully_full = pd.concat([bully_clean, aug_df], ignore_index=True)

# Sample 800 from each
bully_final  = bully_full.sample(n=800, random_state=42, replace=True)
normal_final = normal.sample(n=800, random_state=42, replace=True)

final = pd.concat([bully_final, normal_final], ignore_index=True)
final = final.sample(frac=1, random_state=42).reset_index(drop=True)
final = final.drop_duplicates(subset='text').reset_index(drop=True)

# 70/15/15 split
n = len(final)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

train = final[:train_end]
val   = final[train_end:val_end]
test  = final[val_end:]

# Balance each
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
print('Done.')