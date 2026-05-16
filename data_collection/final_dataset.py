import pandas as pd
import random

df = pd.read_csv('raw_dataset_clean.csv')

bully = df[df['label'] == 1].copy()
normal = df[df['label'] == 0].copy()

augmented = []
for _, row in bully.iterrows():
    text = str(row['text'])
    augmented.append({'text': text.upper(), 'source': row['source'], 'label': 1})
    prefixes = ['omg ', 'seriously ', 'honestly ', 'lol ', 'wtf ']
    augmented.append({'text': random.choice(prefixes) + text, 'source': row['source'], 'label': 1})

aug_df = pd.DataFrame(augmented)
bully_full = pd.concat([bully, aug_df], ignore_index=True)

bully_final = bully_full.sample(n=600, random_state=42, replace=True)
normal_final = normal.sample(n=600, random_state=42)

final = pd.concat([bully_final, normal_final], ignore_index=True)
final = final.sample(frac=1, random_state=42).reset_index(drop=True)

final.to_csv('final_dataset.csv', index=False)
print(f"Cyberbullying (1): {len(final[final['label']==1])}")
print(f"Normal (0): {len(final[final['label']==0])}")
print(f"Total: {len(final)}")
print("Saved to final_dataset.csv")