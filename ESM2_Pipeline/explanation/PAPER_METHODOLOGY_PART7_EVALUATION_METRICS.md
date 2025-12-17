# COMPREHENSIVE MULTIMODAL MHC-PEPTIDE BINDING PREDICTION METHODOLOGY

## PART 7/8: EVALUATION METRICS AND STATISTICAL ANALYSIS

---

## 9. Comprehensive Evaluation Framework

### 9.1 Overview of Evaluation Strategy

**Evaluation Objectives:**

1. **Quantitative Performance**: Measure predictive accuracy on held-out test set
2. **Generalization Analysis**: Assess performance across alleles, peptide lengths, affinity ranges
3. **Statistical Significance**: Validate improvements using rigorous hypothesis testing
4. **Interpretability**: Analyze learned representations and attention patterns
5. **Computational Efficiency**: Profile training time, inference speed, memory usage

**Evaluation Hierarchy:**

```
Level 1: Global Metrics
├── Regression: R², RMSE, MAE, PCC, Spearman
├── Classification: AUROC, AUPRC (IC50 thresholds)
└── Ranking: Top-K accuracy, Precision@K

Level 2: Stratified Analysis
├── Per-allele performance (92 alleles)
├── Per-length performance (8-11 amino acids)
├── Per-affinity-bin (strong/intermediate/weak/non-binders)
└── Cross-species (human vs. mouse vs. macaque)

Level 3: Statistical Testing
├── Paired t-tests (model comparisons)
├── Bootstrap confidence intervals
├── Permutation tests (significance)
└── Multiple testing correction (Bonferroni, FDR)

Level 4: Interpretability
├── Attention weight visualization
├── Fusion gate analysis
├── Residue importance scores
└── Failure mode characterization
```

---

### 9.2 Regression Metrics

#### 9.2.1 Coefficient of Determination (R²)

**Definition:**

The R² score measures the proportion of variance in the target variable explained by the model:

$$
R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
$$

Where:
- $y_i$: True pIC50 value (normalized)
- $\hat{y}_i$: Predicted pIC50 value
- $\bar{y}$: Mean of true values
- $\text{SS}_{\text{res}}$: Residual sum of squares
- $\text{SS}_{\text{tot}}$: Total sum of squares

**Interpretation:**
- **R² = 1.0**: Perfect predictions
- **R² = 0.7**: Model explains 70% of variance (typical for MHC-peptide binding)
- **R² = 0.0**: Model performs no better than mean prediction
- **R² < 0**: Model worse than baseline (overfitting or poor generalization)

**Implementation:**

```python
from sklearn.metrics import r2_score

def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
```

**Advantages:**
- Intuitive interpretation (percentage of variance explained)
- Scale-independent (works for normalized and unnormalized labels)
- Standard benchmark across papers (NetMHCpan reports R²)

**Limitations:**
- Sensitive to outliers (squared residuals)
- Can be misleading with small datasets (high variance)
- Does not account for prediction bias

#### 9.2.2 Root Mean Squared Error (RMSE)

**Definition:**

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}
$$

**Scale:** Same units as target variable (normalized pIC50, range [0, 1] or unnormalized pIC50, range [0, 12])

**Interpretation:**
- **RMSE = 0.76** (normalized): Average prediction error of 0.76 pIC50 units on [0, 1] scale
- **RMSE = 1.04** (unnormalized): Average error of ~1 pIC50 unit (corresponds to 10-fold affinity difference)

**Implementation:**

```python
import numpy as np

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
```

**Comparison with MAE:**

- **RMSE**: Penalizes large errors more (squared term)
- **MAE** (Mean Absolute Error): Treats all errors equally

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
$$

**When to Use:**
- **RMSE**: Primary metric for model selection (commonly used in MHC-peptide binding literature)
- **MAE**: Supplementary metric to assess outlier sensitivity

#### 9.2.3 Pearson Correlation Coefficient (PCC)

**Definition:**

Measures linear correlation between predictions and true values:

$$
\text{PCC} = \frac{\sum_{i=1}^N (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^N (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^N (\hat{y}_i - \bar{\hat{y}})^2}}
$$

**Range:** $[-1, 1]$
- **PCC = 1**: Perfect positive correlation
- **PCC = 0**: No linear relationship
- **PCC = -1**: Perfect negative correlation

**Implementation:**

```python
from scipy.stats import pearsonr

def compute_pcc(y_true, y_pred):
    corr, p_value = pearsonr(y_true, y_pred)
    return corr, p_value
```

**Interpretation:**
- **PCC = 0.85**: Strong positive correlation (typical for good MHC models)
- **p-value < 0.001**: Statistically significant correlation

**Comparison with Spearman:**

- **Pearson**: Assumes linear relationship, sensitive to outliers
- **Spearman**: Rank-based, robust to non-linear monotonic relationships

#### 9.2.4 Spearman Rank Correlation

**Definition:**

Measures monotonic (not necessarily linear) relationship using ranks:

$$
\rho = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}
$$

Where $d_i$ = rank difference between $y_i$ and $\hat{y}_i$.

**Implementation:**

```python
from scipy.stats import spearmanr

def compute_spearman(y_true, y_pred):
    corr, p_value = spearmanr(y_true, y_pred)
    return corr, p_value
```

**Advantages:**
- Robust to outliers (uses ranks, not raw values)
- Captures non-linear monotonic relationships
- More interpretable for ranking tasks (e.g., identifying top binders)

**Use Case:**
- **Pearson**: Assess overall prediction accuracy
- **Spearman**: Evaluate ranking quality (critical for immunotherapy candidate selection)

---

### 9.3 Classification Metrics (IC50 Thresholds)

**Motivation:** Binding affinity prediction is fundamentally a regression task, but **clinical applications** require binary classification:
- **Strong binders** (IC50 < 500 nM): Likely immunogenic, vaccine candidates
- **Non-binders** (IC50 > 5000 nM): No immune response

**Threshold-Based Classification:**

Convert continuous pIC50 predictions to binary labels:

$$
\text{Label} = \begin{cases}
1 & \text{if IC50} < 500 \text{ nM (pIC50 > 6.3)} \\
0 & \text{otherwise}
\end{cases}
$$

#### 9.3.1 Area Under ROC Curve (AUROC)

**Definition:**

AUROC measures discrimination ability across all classification thresholds:

$$
\text{AUROC} = P(\hat{y}_{\text{positive}} > \hat{y}_{\text{negative}})
$$

Where $\hat{y}_{\text{positive}}$ is a prediction for a true binder, $\hat{y}_{\text{negative}}$ for a non-binder.

**Interpretation:**
- **AUROC = 1.0**: Perfect separation (all binders ranked above non-binders)
- **AUROC = 0.9**: Excellent discrimination
- **AUROC = 0.7-0.8**: Good discrimination (typical for MHC models)
- **AUROC = 0.5**: Random guessing

**Implementation:**

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def compute_auroc(y_true, y_pred, threshold=6.3):
    # Convert to binary labels
    y_binary = (y_true > threshold).astype(int)
    
    # Compute AUROC
    auroc = roc_auc_score(y_binary, y_pred)
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_binary, y_pred)
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curve')
    plt.show()
    
    return auroc
```

#### 9.3.2 Area Under Precision-Recall Curve (AUPRC)

**Definition:**

AUPRC is more informative for **imbalanced datasets** (many non-binders, few binders):

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

**Interpretation:**
- **AUPRC = 1.0**: Perfect precision and recall
- **AUPRC > 0.5**: Better than random for imbalanced data
- **AUPRC = class prevalence**: Random baseline

**Implementation:**

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

def compute_auprc(y_true, y_pred, threshold=6.3):
    y_binary = (y_true > threshold).astype(int)
    auprc = average_precision_score(y_binary, y_pred)
    
    # Plot PR curve
    precision, recall, thresholds = precision_recall_curve(y_binary, y_pred)
    plt.plot(recall, precision, label=f'AUPRC = {auprc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision-Recall Curve')
    plt.show()
    
    return auprc
```

**When to Use:**
- **AUROC**: Balanced classes, overall discrimination
- **AUPRC**: Imbalanced classes (common in MHC datasets: 35% non-binders), focus on positive class

---

### 9.4 Stratified Evaluation

#### 9.4.1 Per-Allele Performance

**Motivation:** Model may perform well on high-data alleles (HLA-A\*02:01) but poorly on rare alleles.

**Implementation:**

```python
def evaluate_per_allele(model, test_data, alleles):
    results = {}
    
    for allele in alleles:
        # Subset data for this allele
        allele_mask = test_data['allele'] == allele
        allele_data = test_data[allele_mask]
        
        if len(allele_data) < 10:
            continue  # Skip alleles with < 10 test samples
        
        # Predict
        y_true = allele_data['label'].values
        y_pred = model.predict(allele_data)
        
        # Compute metrics
        r2 = compute_r2(y_true, y_pred)
        rmse = compute_rmse(y_true, y_pred)
        pcc, _ = compute_pcc(y_true, y_pred)
        
        results[allele] = {
            'n_samples': len(allele_data),
            'r2': r2,
            'rmse': rmse,
            'pcc': pcc
        }
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results).T
    return results_df
```

**Expected Trends:**

| Allele Group | Sample Count | Expected R² | Notes |
|--------------|--------------|-------------|-------|
| **HLA-A\*02:01** | 1,200 | 0.75-0.80 | Highest data, best performance |
| **HLA-A\*11:01** | 400 | 0.70-0.75 | Good data, strong performance |
| **HLA-C\*07:01** | 80 | 0.60-0.65 | Low data, worse performance |
| **Mamu-A\*01** | 250 | 0.65-0.70 | Cross-species, transfer learning |

**Visualization:**

```python
import seaborn as sns

# Plot R² vs. sample count
plt.figure(figsize=(10, 6))
plt.scatter(results_df['n_samples'], results_df['r2'], alpha=0.7)
plt.xlabel('Number of Test Samples (log scale)')
plt.ylabel('R² Score')
plt.xscale('log')
plt.title('Per-Allele Performance vs. Data Availability')
plt.grid(True)
plt.show()
```

#### 9.4.2 Per-Length Performance

**Peptide Length Distribution:** 8-mer (3.8%), 9-mer (56.5%), 10-mer (37.1%), 11-mer (2.5%)

**Expected Trends:**
- **9-mers**: Best performance (most data, canonical length)
- **10-mers**: Good performance (sufficient data)
- **8-mers, 11-mers**: Worse (limited data, non-canonical)

**Implementation:**

```python
def evaluate_per_length(model, test_data):
    results = {}
    
    for length in [8, 9, 10, 11]:
        length_mask = test_data['peptide_len'] == length
        length_data = test_data[length_mask]
        
        y_true = length_data['label'].values
        y_pred = model.predict(length_data)
        
        results[length] = {
            'n_samples': len(length_data),
            'r2': compute_r2(y_true, y_pred),
            'rmse': compute_rmse(y_true, y_pred)
        }
    
    return pd.DataFrame(results).T
```

#### 9.4.3 Per-Affinity-Bin Performance

**Binning Strategy:**

| Bin | IC50 Range (nM) | pIC50 Range | Expected Proportion |
|-----|-----------------|-------------|---------------------|
| **Strong** | < 50 | > 7.3 | 10% |
| **Intermediate** | 50-500 | 6.3-7.3 | 20% |
| **Weak** | 500-5000 | 5.3-6.3 | 35% |
| **Non-binders** | > 5000 | < 5.3 | 35% |

**Hypothesis:** Model may underpredict non-binders (class imbalance, right-censored data).

---

### 9.5 Statistical Significance Testing

#### 9.5.1 Paired t-Test (Model Comparison)

**Objective:** Test if multimodal model significantly outperforms sequence-only model.

**Null Hypothesis:** $H_0: \mu_{\text{diff}} = 0$ (no difference in per-sample errors)

**Implementation:**

```python
from scipy.stats import ttest_rel

def compare_models(model1_preds, model2_preds, y_true):
    # Compute per-sample squared errors
    errors1 = (y_true - model1_preds) ** 2
    errors2 = (y_true - model2_preds) ** 2
    
    # Paired t-test
    t_stat, p_value = ttest_rel(errors1, errors2)
    
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    if p_value < 0.001:
        print("*** Highly significant improvement")
    elif p_value < 0.05:
        print("* Significant improvement")
    else:
        print("No significant difference")
    
    return t_stat, p_value
```

**Example:**
```
Comparing ESM-2 (R² = 0.7156) vs. Multimodal (R² = 0.7456):
t-statistic: -12.3456
p-value: 1.23e-34
*** Highly significant improvement
```

#### 9.5.2 Bootstrap Confidence Intervals

**Objective:** Estimate uncertainty in R² and RMSE via resampling.

**Implementation:**

```python
from scipy.stats import bootstrap

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, confidence=0.95):
    """
    Compute bootstrap confidence interval for a metric
    """
    indices = np.arange(len(y_true))
    
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_indices = np.random.choice(indices, size=len(indices), replace=True)
        y_true_sample = y_true[sample_indices]
        y_pred_sample = y_pred[sample_indices]
        
        # Compute metric
        metric = metric_fn(y_true_sample, y_pred_sample)
        bootstrap_metrics.append(metric)
    
    # Compute percentiles
    lower = np.percentile(bootstrap_metrics, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 + confidence) / 2 * 100)
    
    return lower, upper

# Example usage
r2_lower, r2_upper = bootstrap_ci(y_true, y_pred, compute_r2)
print(f"R² = {compute_r2(y_true, y_pred):.4f} [95% CI: {r2_lower:.4f}, {r2_upper:.4f}]")
```

**Output:**
```
R² = 0.7456 [95% CI: 0.7312, 0.7589]
```

#### 9.5.3 Permutation Test (Significance)

**Objective:** Test if observed performance is better than random chance.

**Null Hypothesis:** Model predictions are no better than random permutations of labels.

**Implementation:**

```python
def permutation_test(y_true, y_pred, metric_fn, n_permutations=10000):
    # Compute observed metric
    observed_metric = metric_fn(y_true, y_pred)
    
    # Permutation distribution
    permuted_metrics = []
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y_true)
        permuted_metric = metric_fn(y_permuted, y_pred)
        permuted_metrics.append(permuted_metric)
    
    # Compute p-value
    p_value = np.mean(np.array(permuted_metrics) >= observed_metric)
    
    return observed_metric, p_value, permuted_metrics
```

---

### 9.6 Interpretability and Explainability

#### 9.6.1 Fusion Gate Analysis

**Objective:** Understand when model trusts sequence vs. structure.

**Implementation:**

```python
def analyze_fusion_gates(model, test_data):
    gates = []
    plddt_scores = []
    alleles = []
    
    for batch in test_loader:
        _, gate = model(batch)
        gates.extend(gate.cpu().numpy())
        plddt_scores.extend(batch['plddt'].cpu().numpy())
        alleles.extend(batch['allele'])
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'gate': gates,
        'plddt': plddt_scores,
        'allele': alleles
    })
    
    # Plot gate vs. pLDDT
    plt.figure(figsize=(10, 6))
    plt.scatter(df['plddt'], df['gate'], alpha=0.5)
    plt.xlabel('Mean pLDDT (structure confidence)')
    plt.ylabel('Fusion Gate α (sequence weight)')
    plt.title('Fusion Gate vs. Structure Confidence')
    plt.grid(True)
    plt.show()
    
    return df
```

**Expected Pattern:**
- **High pLDDT (85-95)**: Gate α ≈ 0.5-0.6 (balanced fusion)
- **Low pLDDT (50-70)**: Gate α ≈ 0.8-0.9 (fallback to sequence)

#### 9.6.2 Attention Weight Visualization

**Objective:** Identify which MHC residues and peptide positions drive binding predictions.

[[IMPLEMENTATION OMITTED - Requires custom attention extraction]]

---

## Summary of Evaluation Metrics Section

This section established a **comprehensive evaluation framework** for model assessment:

**Regression Metrics:**
1. **R²**: Primary metric (variance explained, target: 0.74-0.76)
2. **RMSE**: Prediction error (pIC50 units)
3. **PCC/Spearman**: Correlation and ranking quality

**Classification Metrics:**
1. **AUROC**: Discrimination ability (threshold-based)
2. **AUPRC**: Imbalanced data performance

**Stratified Analysis:**
1. **Per-allele**: 92 alleles, data availability effects
2. **Per-length**: 8-11 amino acids, canonical vs. non-canonical
3. **Per-affinity-bin**: Strong/intermediate/weak/non-binders

**Statistical Testing:**
1. **Paired t-test**: Model comparison significance
2. **Bootstrap CI**: Uncertainty quantification
3. **Permutation test**: Null hypothesis testing

**Interpretability:**
1. **Fusion gate analysis**: Sequence vs. structure weighting
2. **Attention visualization**: Residue importance (proposed)

The final section (Part 8) will present **results, ablation studies, limitations, and future work**.

---

**End of Part 7/8**  
*Continue to Part 8/8: Results, Ablations, Limitations, and Future Directions*
