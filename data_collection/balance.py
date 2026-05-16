import pandas as pd

df = pd.read_csv('raw_dataset.csv')

bully = df[df['label'] == 1]
normal = df[df['label'] == 0]

# Sample normal down to match bully count
normal_sampled = normal.sample(n=len(bully), random_state=42)

balanced = pd.concat([bully, normal_sampled], ignore_index=True)
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

balanced.to_csv('balanced_dataset.csv', index=False)

print(f"Cyberbullying (1): {len(balanced[balanced['label'] == 1])}")
print(f"Normal (0): {len(balanced[balanced['label'] == 0])}")
print(f"Total: {len(balanced)}")
print("Saved to balanced_dataset.csv")