import pandas as pd

df = pd.read_csv('raw_dataset.csv')

# Remove rows with URLs, image embeds, very short text
df = df[~df['text'].str.contains('https://', na=False)]
df = df[~df['text'].str.contains('!\[gif\]', na=False)]
df = df[~df['text'].str.contains('!\[img\]', na=False)]
df = df[df['text'].str.len() > 20]
df = df.dropna(subset=['text'])

print(f"After cleaning: {len(df)}")
print(f"Cyberbullying (1): {df['label'].sum()}")
print(f"Normal (0): {len(df) - df['label'].sum()}")

df.to_csv('raw_dataset_clean.csv', index=False)
print("Saved to raw_dataset_clean.csv")