import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ensemble_meta_data.csv")

plt.figure(figsize=(12, 5))

# Plot 1: ESM vs Actual (Good)
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='actual_pic50', y='esm_pred', alpha=0.1)
plt.plot([0, 10], [0, 10], 'r--')
plt.title(f"Sequence Model (R2={df[['actual_pic50', 'esm_pred']].corr().iloc[0,1]**2:.3f})")

# Plot 2: GNN vs Actual (Bad)
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='actual_pic50', y='gnn_pred', alpha=0.1, color='orange')
plt.plot([0, 10], [0, 10], 'r--')
plt.title(f"Structure Model (R2={df[['actual_pic50', 'gnn_pred']].corr().iloc[0,1]**2:.3f})")

plt.savefig("model_comparison.png")
print("Saved comparison plot to model_comparison.png")