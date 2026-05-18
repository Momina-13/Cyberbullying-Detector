"""
Member 3 — Final step: Compare all 3 models
Run AFTER train_lr.py, train_lstm.py, train_bert.py are all done.
Run: python compare_models.py
"""

import pandas as pd
import os

results_dir = '../saved_models'
files = {
    'Logistic Regression': 'lr_results.csv',
    'LSTM':                'lstm_results.csv',
    'BERT':                'bert_results.csv',
}

all_results = []
for model_name, fname in files.items():
    path = os.path.join(results_dir, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_results.append(df)
    else:
        print(f"WARNING: {fname} not found. Run the training script first.")

if all_results:
    comparison = pd.concat(all_results, ignore_index=True)
    comparison = comparison[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]

    print("\n" + "="*65)
    print("         FINAL MODEL COMPARISON TABLE")
    print("="*65)
    print(comparison.to_string(index=False))
    print("="*65)

    comparison.to_csv('../saved_models/final_comparison.csv', index=False)
    print("\nSaved to saved_models/final_comparison.csv")
    print("Share this table with Member 1 for the presentation slides.")
