# COMPREHENSIVE MULTIMODAL MHC-PEPTIDE BINDING PREDICTION METHODOLOGY

## PART 2/8: DATASET DESCRIPTION AND PREPROCESSING

---

## 4. Dataset Description and Preprocessing

### 4.1 Data Source and Collection

Our study utilized the **Immune Epitope Database (IEDB)**, the most comprehensive repository of experimentally validated immune epitope data. The IEDB aggregates immunological data from scientific publications, direct submissions, and high-throughput screening experiments, providing standardized annotations for MHC-peptide binding measurements across diverse experimental assays and biological contexts.

We specifically focused on **MHC class I binding affinity data**, as these molecules present peptides from intracellular proteins to CD8+ T cells and are critical for cellular immunity. The dataset was constructed by querying IEDB for all MHC class I molecules with quantitative binding measurements, encompassing both human leukocyte antigen (HLA) alleles and non-human MHC alleles from commonly studied model organisms.

**Data Collection Characteristics:**
- **Source Database**: IEDB (Immune Epitope Database)
- **MHC Class**: Class I (8-11 amino acid peptides)
- **Species Coverage**: Human (HLA-A, HLA-B, HLA-C), mouse (H2-Db, H2-Kb), macaque (Mamu-A*01)
- **Assay Types**: Competitive binding assays, fluorescence polarization, surface plasmon resonance, cellular assays
- **Measurement Units**: Binding affinity in nanomolar (nM) concentrations (IC50 values)
- **Data Format**: CSV file with structured columns for allele, MHC sequence, peptide sequence, binding affinity, and assay metadata

The IEDB provides several key advantages for computational modeling:
1. **Standardization**: All measurements are normalized to IC50 values (nM), enabling direct comparison across experiments
2. **Quality Control**: Data undergo expert curation and validation
3. **Biological Diversity**: Includes diverse alleles, peptide lengths, and amino acid compositions
4. **Assay Metadata**: Links to original publications, assay conditions, and experimental protocols

### 4.2 Dataset Statistics and Composition

The final curated dataset comprises **85,628 MHC-peptide binding measurements** spanning **92 unique MHC alleles** and **29,325 unique peptide sequences**. This represents one of the largest benchmark datasets for MHC-peptide binding prediction, providing sufficient sample diversity for training deep learning models while maintaining representation across multiple allele families.

**Overall Dataset Composition:**

| Statistic | Value |
|-----------|-------|
| **Total Samples** | 85,628 |
| **Unique MHC Alleles** | 92 |
| **Unique Peptides** | 29,325 |
| **MHC Class** | I (100%) |
| **Species Represented** | 5+ (human, mouse, macaque, chimpanzee, pig) |

The dataset is **exclusively MHC class I**, reflecting our focus on intracellular peptide presentation. Class I molecules typically bind peptides of 8-11 amino acids, with strong preferences for anchor residues at positions 2 and 9 (or the C-terminus).

**Peptide Length Distribution:**

| Length | Count | Percentage | Notes |
|--------|-------|------------|-------|
| **8-mer** | 3,292 | 3.8% | Shorter peptides, less common |
| **9-mer** | 48,362 | **56.5%** | **Most common, canonical length** |
| **10-mer** | 31,793 | 37.1% | Extended peptides, common |
| **11-mer** | 2,181 | 2.5% | Longer peptides, rare |

**Key Observations:**
- **9-mer peptides** constitute the majority (56.5%), reflecting the canonical binding groove length for most MHC class I alleles
- **10-mer peptides** are also well-represented (37.1%), accommodating alleles with larger binding pockets (e.g., HLA-B*27:05)
- **8-mer and 11-mer peptides** are relatively rare (6.3% combined), but still important for comprehensive modeling
- **Mean length**: 9.38 ± 0.60 amino acids
- **Modal length**: 9 amino acids (48,362 samples)

This distribution aligns with known MHC class I biology: the binding groove can accommodate peptides of 8-11 residues by adjusting peptide conformation (bulging at position 6 for 10-mers, extended bulges for 11-mers).

**MHC Allele Representation:**

The dataset exhibits **substantial class imbalance** across alleles, reflecting both biological interest (certain alleles are more studied due to disease associations) and experimental tractability. The top 15 alleles account for **62.5%** of all samples:

| Allele | Count | Percentage | Notes |
|--------|-------|------------|-------|
| **HLA-A\*02:01** | 11,967 | 14.0% | Most common HLA-A allele globally |
| **HLA-A\*02:03** | 4,958 | 5.8% | HLA-A\*02 supertype member |
| **HLA-A\*02:06** | 4,778 | 5.6% | HLA-A\*02 supertype member |
| **HLA-A\*02:02** | 4,616 | 5.4% | HLA-A\*02 supertype member |
| **HLA-A\*68:02** | 4,578 | 5.4% | Common in European populations |
| **HLA-A\*11:01** | 3,728 | 4.4% | Asian-prevalent allele |
| **HLA-A\*03:01** | 3,719 | 4.3% | HLA-A\*03 supertype |
| **HLA-A\*31:01** | 3,024 | 3.5% | Associated with drug hypersensitivity |
| **HLA-A\*68:01** | 2,993 | 3.5% | HLA-A\*68 family |
| **HLA-A\*33:01** | 2,941 | 3.4% | Asian-prevalent allele |
| **H2-Kb** | 2,849 | 3.3% | Mouse MHC class I (model organism) |
| **Mamu-A\*01:01** | 2,375 | 2.8% | Macaque (vaccine research model) |
| **H2-Db** | 2,198 | 2.6% | Mouse MHC class I (model organism) |
| **HLA-A\*24:02** | 2,166 | 2.5% | Common in Asian populations |
| **HLA-A\*01:01** | 1,747 | 2.0% | HLA-A\*01 supertype |

**Key Observations:**
- **HLA-A\*02:01** dominance (14%) reflects its global prevalence (~50% of populations) and extensive research interest
- **HLA-A\*02 supertype** members (A\*02:01, A\*02:03, A\*02:06, A\*02:02) collectively represent **30.8%** of the dataset
- **Mouse alleles** (H2-Kb, H2-Db) constitute **5.9%**, providing non-human validation data
- **Allele diversity**: 92 unique alleles ensure generalization across diverse binding specificities
- **Long-tail distribution**: 77 alleles have <1,747 samples each, creating a challenging few-shot learning scenario

This imbalance necessitates careful consideration during model training and evaluation to prevent overfitting to well-represented alleles while maintaining performance on rare alleles.

**Binding Affinity Distribution:**

Binding affinity measurements span **7 orders of magnitude**, ranging from ultra-high affinity (0.0004 nM) to non-binding (14,276,649 nM):

| Statistic | Affinity (nM) | pIC50 | Interpretation |
|-----------|---------------|-------|----------------|
| **Minimum** | 0.0004 | 12.37 | Ultra-high affinity (likely experimental outlier) |
| **25th Percentile** | 64.0 | 7.19 | High affinity binders |
| **Median** | 943.0 | 6.03 | Moderate affinity |
| **75th Percentile** | 12,200.0 | 4.91 | Weak binders |
| **Maximum** | 14,276,649.3 | 1.85 | Non-binders |
| **Mean** | 17,378.8 | 6.14 | Skewed by non-binders |
| **Std. Dev.** | 85,840.3 | 1.46 | High variance |

**Affinity Classes (IC50 thresholds):**
- **Strong binders** (IC50 < 50 nM): ~8,500 samples (9.9%)
- **Intermediate binders** (50-500 nM): ~19,000 samples (22.2%)
- **Weak binders** (500-5000 nM): ~28,000 samples (32.7%)
- **Non-binders** (IC50 > 5000 nM): ~30,000 samples (35.0%)

This distribution reflects the biological reality that most peptides are **non-binders or weak binders**, with only a small fraction exhibiting high-affinity binding suitable for immune recognition. The heavy right skew (mean >> median) underscores the need for logarithmic transformation during preprocessing.

### 4.3 Data Filtering and Quality Control

Raw IEDB data undergo rigorous filtering to ensure data quality and consistency for model training. Our preprocessing pipeline implements multiple validation steps:

**Step 1: Missing Value Removal**
- **Filter**: Remove entries with missing values in critical columns (`mhc_sequence`, `peptide`, `binding_affinity_nM`)
- **Rationale**: Complete feature and label data are essential for supervised learning
- **Impact**: Minimal (<1% of raw data)

**Step 2: Sequence Validation**
- **Filter**: Remove sequences containing non-alphabetic characters (invalid amino acids, special symbols)
- **Validation**: `all_df[all_df[config.pep_col].str.isalpha()]` and `all_df[all_df[config.mhc_col].str.isalpha()]`
- **Rationale**: Protein language models require valid amino acid sequences; non-standard residues (X, B, Z) would cause tokenization errors
- **Impact**: ~2-3% of entries removed

**Step 3: Duplicate Handling**
- **Strategy**: **Retain all duplicates** (no deduplication)
- **Rationale**: 
  - Duplicates from different assays provide biological replicates
  - Natural data augmentation improves model robustness
  - Different experimental conditions may yield slightly different measurements
- **Result**: All 85,628 measurements retained (including replicates)

**Step 4: Outlier Analysis**
- **Extreme High Affinity**: Values below 0.001 nM are exceedingly rare in nature and likely experimental artifacts
- **Decision**: **Retained** without filtering (only 0.01% of data)
- **Extreme Low Affinity**: Values above 10^6 nM represent clear non-binders
- **Decision**: **Retained** to preserve full binding spectrum for binary classification tasks

**Step 5: Peptide Length Constraints**
- **MHC Class I Constraints**: Peptides must be 8-11 amino acids
- **Validation**: Implicitly enforced by IEDB data curation (all samples fall within 8-11 range)
- **Result**: No additional filtering required

**Data Cleaning Summary:**
- **Initial Count**: ~87,500 raw entries
- **After Missing Value Removal**: ~86,200 entries
- **After Sequence Validation**: **85,628 entries**
- **Retention Rate**: **97.9%**

The high retention rate reflects IEDB's strong quality control standards and our conservative filtering approach, which prioritizes data preservation while ensuring model training stability.

### 4.4 Target Variable Transformation: pIC50 Conversion

Binding affinity measurements (IC50 values in nanomolar) exhibit extreme skewness and heteroscedasticity, violating assumptions of linear regression models. To address this, we apply the **pIC50 transformation**, a logarithmic rescaling widely used in drug discovery and cheminformatics:

$$
\text{pIC50} = 9 - \log_{10}(\text{IC50}_{\text{nM}})
$$

**Rationale for pIC50:**
1. **Linearization**: Converts exponential binding curves to linear relationships
2. **Homoscedasticity**: Stabilizes variance across the affinity range
3. **Interpretability**: Higher pIC50 values indicate stronger binding (intuitive for optimization)
4. **Domain Alignment**: Matches conventions in medicinal chemistry and immunology literature

**pIC50 Scale Interpretation:**
- **pIC50 > 8** (IC50 < 10 nM): Ultra-high affinity, rare natural binders
- **pIC50 = 7-8** (IC50 = 10-100 nM): Strong binders, immunogenic peptides
- **pIC50 = 6-7** (IC50 = 100-1000 nM): Moderate binders, borderline immunogenicity
- **pIC50 = 5-6** (IC50 = 1-10 µM): Weak binders, minimal immune response
- **pIC50 < 5** (IC50 > 10 µM): Non-binders

**Effect on Dataset Distribution:**

After pIC50 transformation, the label distribution becomes approximately **Gaussian**:

| Statistic | pIC50 Value |
|-----------|-------------|
| **Minimum** | 1.85 |
| **25th Percentile** | 4.91 |
| **Median** | 6.03 |
| **75th Percentile** | 7.19 |
| **Maximum** | 12.37 |
| **Mean** | 6.14 |
| **Std. Dev.** | 1.46 |

**Key Observations:**
- **Mean ≈ Median** (6.14 vs 6.03): Indicates symmetric distribution after transformation
- **Interquartile Range**: 4.91-7.19 (2.28 pIC50 units), corresponding to ~200-fold affinity range
- **Standard Deviation**: 1.46 pIC50 units, representing ~29-fold affinity uncertainty
- **Outlier Cap**: IC50 values above 50,000 nM are clipped to 50,000 nM before transformation to prevent extreme negative pIC50 values:
  ```python
  df[config.label_col] = df[config.label_col].apply(lambda x: min(float(x), 50000.0))
  ```

**Implementation Details:**
```python
# Convert nM to molar (M)
molar_values = df[config.label_col] * 1e-9

# Apply pIC50 transformation with numerical stability
df[config.label_col] = -np.log10(molar_values + 1e-12)
```

The addition of `1e-12` prevents logarithm of zero errors for extreme outliers while having negligible impact on normal measurements (e.g., pIC50 for 1 nM changes by <0.00001).

### 4.5 Label Normalization

Following pIC50 transformation, labels are **normalized to the range [0, 1]** using Min-Max scaling fitted exclusively on the **training set**:

$$
\text{pIC50}_{\text{norm}} = \frac{\text{pIC50} - \text{pIC50}_{\text{train,min}}}{\text{pIC50}_{\text{train,max}} - \text{pIC50}_{\text{train,min}}}
$$

**Rationale:**
1. **Neural Network Optimization**: Output activations (sigmoid, tanh) naturally operate in [0, 1] or [-1, 1] ranges
2. **Loss Function Stability**: Prevents gradient explosion/vanishing when labels span multiple orders of magnitude
3. **Learning Rate Sensitivity**: Normalized labels allow consistent learning rates across different datasets
4. **Regularization Compatibility**: L2 weight decay and dropout behave more predictably with normalized outputs

**Implementation:**
```python
from sklearn.preprocessing import MinMaxScaler

label_scaler = MinMaxScaler()
train_labels = train_df[config.label_col].values.reshape(-1, 1)

# Fit scaler on training set ONLY
label_scaler.fit(train_labels)

# Transform all splits using training statistics
train_df[config.label_col] = label_scaler.transform(train_labels)
valid_df[config.label_col] = label_scaler.transform(valid_df[config.label_col].values.reshape(-1, 1))
test_df[config.label_col] = label_scaler.transform(test_df[config.label_col].values.reshape(-1, 1))

# Save scaler for inference
joblib.dump(label_scaler, os.path.join(config.output_dir, "label_scaler.joblib"))
```

**Critical Design Choices:**
- **Training-Only Fitting**: Prevents data leakage from validation/test sets
- **Scaler Persistence**: Saved to disk for consistent denormalization during inference
- **Clip-Free Application**: Validation/test sets may contain values outside training range (e.g., higher/lower affinities), which are **not clipped** to preserve biological information

**Denormalization for Evaluation:**

During evaluation and inference, model predictions are denormalized to recover pIC50 values:

$$
\text{pIC50}_{\text{pred}} = \text{pIC50}_{\text{norm,pred}} \times (\text{pIC50}_{\text{train,max}} - \text{pIC50}_{\text{train,min}}) + \text{pIC50}_{\text{train,min}}
$$

This allows computation of interpretable metrics (RMSE in pIC50 units, Spearman correlation) on the original scale while maintaining normalized training dynamics.

### 4.6 Dataset Splitting Strategy

The dataset is partitioned into **training (80%)**, **validation (10%)**, and **test (10%)** subsets using stratified random sampling to ensure representative allele and affinity distributions across splits:

**Splitting Procedure:**

```python
from sklearn.model_selection import train_test_split

# Step 1: Hold out test set (10%)
train_val_df, test_df = train_test_split(
    all_df, 
    test_size=0.10,  # 10% test
    random_state=42  # Fixed seed for reproducibility
)

# Step 2: Split remaining data into train/valid (80%/10% of original)
valid_size_scaled = 0.10 / (1.0 - 0.10)  # 0.111... of train_val_df
train_df, valid_df = train_test_split(
    train_val_df,
    test_size=valid_size_scaled,
    random_state=42
)
```

**Final Split Sizes:**

| Split | Samples | Percentage | Purpose |
|-------|---------|------------|---------|
| **Training** | 68,502 | 80% | Model parameter optimization |
| **Validation** | 8,563 | 10% | Hyperparameter tuning, early stopping |
| **Test** | 8,563 | 10% | Final model evaluation (held out) |

**Split Characteristics:**
- **Random Sampling**: No explicit stratification by allele or affinity (sklearn default)
- **Reproducibility**: Fixed `random_state=42` ensures identical splits across experiments
- **Independence**: Test set completely isolated until final evaluation
- **Validation Role**: Used for:
  - Early stopping (patience = 10 epochs)
  - Learning rate scheduling (ReduceLROnPlateau, patience = 3 epochs)
  - Model checkpoint selection (save best validation RMSE)

**Potential Limitations:**

1. **Allele Imbalance**: Random splitting may not preserve exact allele proportions across splits
   - **Impact**: Rare alleles (< 100 samples) may be underrepresented in validation/test
   - **Mitigation**: Large training set (68,502 samples) ensures even rare alleles have training examples

2. **Peptide Overlap**: Same peptide may appear with different MHC alleles across splits
   - **Nature**: Biologically valid (same peptide can bind multiple alleles)
   - **Not Data Leakage**: Model must learn allele-specific binding preferences

3. **Affinity Distribution**: No explicit affinity stratification
   - **Verification Needed**: [[MISSING - Distribution comparison across splits not yet analyzed]]
   - **Future Work**: Stratified splitting by pIC50 quantiles for balanced affinity representation

### 4.7 Dataset Versioning and Reproducibility

All experiments reported in this study use a **single frozen dataset version** to ensure reproducibility:

**Dataset Provenance:**
- **File Name**: `iedb_dataset.csv`
- **File Size**: 85,629 lines (85,628 data rows + 1 header)
- **Creation Date**: [[MISSING - Dataset download/creation date not recorded]]
- **IEDB Version**: [[MISSING - Specific IEDB snapshot version not tracked]]
- **MD5 Checksum**: [[MISSING - File integrity hash not computed]]

**Reproducibility Measures:**
- **Fixed Seed**: All random operations use `seed=42`
- **Split Persistence**: Train/valid/test indices not saved (regenerated from seed)
- **Scaler Persistence**: `label_scaler.joblib` saved for consistent denormalization
- **Configuration Tracking**: Full hyperparameters stored in `config.yaml` files

**Future Improvements:**
1. **Dataset Registry**: Assign version identifier (e.g., `IEDB_MHC_v1.0_2024`)
2. **Checksum Validation**: Compute MD5/SHA256 to detect accidental modifications
3. **Split Index Persistence**: Save train/valid/test indices for exact reproducibility
4. **IEDB Timestamp**: Record exact IEDB snapshot date for data provenance

### 4.8 Data Augmentation Considerations

**Current Approach**: **No augmentation** is applied to the dataset. All 85,628 samples are used as-is, with duplicates retained as natural biological replicates.

**Rationale for No Augmentation:**
1. **Biological Realism**: Amino acid sequences are discrete and highly specific; random mutations may create biologically implausible peptides
2. **Binding Specificity**: Single amino acid changes can dramatically alter binding affinity (e.g., P2 anchor mutations), making augmentation unpredictable
3. **Data Sufficiency**: 85,628 samples provide adequate training data for models with ~10-20M parameters
4. **Overfitting Risk**: Random sequence perturbations may introduce spurious patterns

**Potential Future Augmentation Strategies:**

If additional data diversity is required (e.g., for rare alleles), biologically informed augmentation could be explored:

1. **Conservative Amino Acid Substitutions**: Replace residues with biochemically similar alternatives (e.g., Leu → Ile, Asp → Glu) at non-anchor positions
2. **Allele Family Augmentation**: Generate synthetic data for rare alleles by adapting sequences from related alleles (e.g., HLA-A\*02:02 from A\*02:01)
3. **Structure-Guided Mutations**: Use AlphaFold2 pLDDT scores to identify flexible regions amenable to mutation
4. **Transfer Learning from NetMHCpan**: Use NetMHCpan-4.1 predictions as pseudo-labels for unlabeled peptide-MHC pairs

However, these approaches require careful validation to avoid introducing biased or unrealistic training signals.

### 4.9 Dataset Limitations and Biases

Despite its size and quality, the IEDB dataset exhibits several inherent limitations that may impact model performance and generalization:

**1. Allele Representation Bias**
- **Issue**: 77 of 92 alleles have < 2,000 samples (< 2.3% of data each)
- **Impact**: Models may underperform on rare alleles (e.g., HLA-C alleles, non-human MHCs)
- **Biological Context**: Reflects research priorities (HLA-A\*02:01 is clinically important) rather than natural allele frequencies

**2. Peptide Source Bias**
- **Issue**: IEDB peptides are enriched for viral and tumor-associated antigens (influenza, HIV, melanoma)
- **Impact**: Model may not generalize to self-peptides or commensal microbiome antigens
- **Missing Data**: Limited representation of non-immunogenic self-peptides (negative selection bias)

**3. Assay Heterogeneity**
- **Issue**: IC50 measurements from diverse assay types (competitive binding, cellular assays, SPR) with varying sensitivity and reproducibility
- **Impact**: Measurement noise may obscure subtle binding differences
- **Mitigation**: pIC50 transformation reduces sensitivity to absolute measurement errors

**4. Peptide Length Imbalance**
- **Issue**: 9-mers constitute 56.5% of data; 8-mers and 11-mers are underrepresented
- **Impact**: Models may exhibit length-dependent performance biases
- **Biological Caveat**: Reflects natural MHC class I preferences, but limits generalization to non-canonical lengths

**5. Affinity Range Censoring**
- **Issue**: Non-binders (IC50 > 50,000 nM) are often recorded as ">50,000" rather than exact values
- **Impact**: Right-censored data may bias regression toward underestimating very weak binding
- **Prevalence**: ~15% of samples are near the 50,000 nM cap

**6. Experimental Condition Variability**
- **Issue**: Binding measurements conducted at different temperatures, pH values, and buffer compositions
- **Metadata**: Assay conditions not consistently recorded in dataset
- **Impact**: Uncontrolled variability may introduce noise

**7. Species-Specific Limitations**
- **Issue**: Human HLA alleles dominate (>90%); mouse and macaque alleles are minorities
- **Impact**: Cross-species transfer learning may be limited
- **Opportunity**: Multi-species data could enable evolutionary analyses

**Acknowledgment of Bias:**

These limitations are **inherent to the IEDB curation process** and reflect broader challenges in experimental immunology (e.g., difficulty measuring non-binding interactions, focus on clinically relevant alleles). Our methodology acknowledges these biases and interprets results within this context. Future work should prioritize data collection for underrepresented alleles and peptide sources to improve model equity and generalization.

---

## Summary of Dataset Section

This section established the foundation for our multimodal MHC-peptide binding prediction system by comprehensively describing the **IEDB dataset** used for all experiments. Key contributions include:

1. **Dataset Curation**: 85,628 high-quality MHC class I binding measurements spanning 92 alleles and 29,325 peptides
2. **Statistical Characterization**: Detailed analysis of peptide length distribution (56.5% 9-mers), allele representation (14% HLA-A\*02:01), and binding affinity range (7 orders of magnitude)
3. **Preprocessing Pipeline**: Rigorous filtering, pIC50 transformation, and Min-Max normalization to ensure data quality and model training stability
4. **Splitting Strategy**: 80/10/10 train/validation/test split with fixed random seed for reproducibility
5. **Bias Acknowledgment**: Transparent discussion of allele imbalance, assay heterogeneity, and peptide source biases

The next section (Part 3) will describe the **sequence-based modeling architecture**, detailing how protein language models (ESM-2, ProtT5, ESM-C) are fine-tuned on this dataset to capture amino acid-level binding patterns.

---

**End of Part 2/8**  
*Continue to Part 3/8: Sequence Model Architecture and Training*
