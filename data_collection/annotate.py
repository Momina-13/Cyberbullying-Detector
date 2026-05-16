import pandas as pd
import re

df = pd.read_csv('raw_dataset.csv')

# Strong cyberbullying keywords
bully_keywords = [
    'kill yourself', 'kys', 'you are ugly', 'ur ugly', 'nobody likes you',
    'you are stupid', 'ur stupid', 'idiot', 'moron', 'retard', 'loser',
    'fat', 'ugly', 'dumb', 'pathetic', 'worthless', 'useless', 'trash',
    'shut up', 'go die', 'hate you', 'nobody cares', 'freak', 'weirdo',
    'disgusting', 'gross', 'eww', 'ew', 'cringe', 'embarrassing',
    'stupid', 'idiot', 'dumb', 'clown', 'joke', 'laughing at you',
    'harass', 'bully', 'attack', 'threat', 'abuse', 'humiliate'
]

def auto_label(text):
    text_lower = str(text).lower()
    for keyword in bully_keywords:
        if keyword in text_lower:
            return 1
    return 0

df['label'] = df['text'].apply(auto_label)

# Check balance
print(f"Total: {len(df)}")
print(f"Cyberbullying (1): {df['label'].sum()}")
print(f"Normal (0): {len(df) - df['label'].sum()}")

df.to_csv('raw_dataset.csv', index=False)
print("Labels saved to raw_dataset.csv")