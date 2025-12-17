# Structure-Based MHC-Peptide Binding Prediction: Methodology Part 2
## Graph Featurization and Geometric Features

---

## 1. Introduction

This document specifies the **graph featurization strategy** for protein structure representation. While Part 1 documented the basic implementation (21-dimensional one-hot amino acid encoding + 3D coordinates), this part details:

1. **Current features** (implemented)
2. **Advanced features** (partially implemented / `[INFERRED]`)
3. **State-of-the-art geometric features** (`[TODO]`)
4. **Theoretical justifications** for each feature type

The goal is to design a rich, informative graph representation that captures both **chemical properties** and **geometric relationships** relevant to MHC-peptide binding.

---

## 2. Node Features: Current Implementation

### 2.1 Amino Acid One-Hot Encoding (IMPLEMENTED)

**Dimension**: 21 (20 standard amino acids + unknown)

```python
# From build_graphs.py
AA_VOCAB = {
    'ALA': 0,  'CYS': 1,  'ASP': 2,  'GLU': 3,  'PHE': 4,
    'GLY': 5,  'HIS': 6,  'ILE': 7,  'LYS': 8,  'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
    'X': 20  # Unknown/modified residues
}

def encode_residue(residue_name: str) -> np.ndarray:
    idx = AA_VOCAB.get(residue_name, 20)
    encoding = np.zeros(21, dtype=np.float32)
    encoding[idx] = 1.0
    return encoding
```

**Properties**:
- **Categorical representation**: Each amino acid as distinct category
- **No similarity encoding**: GLY and ALA treated as equally different (despite structural similarity)
- **Interpretable**: Direct mapping to biochemistry

**Advantages**:
- Simple, standard representation
- No learned embedding bias
- Compatible with GNN aggregation functions

**Limitations**:
- **No chemical information**: Ignores hydrophobicity, charge, size
- **Sparse**: 21-dim vector with single non-zero entry
- **Fixed**: Cannot adapt to rare/modified amino acids

### 2.2 3D Coordinates (IMPLEMENTED)

**Dimension**: 3 (Cartesian x, y, z)

```python
positions = np.array([
    [atom['x'], atom['y'], atom['z']]
    for atom in ca_atoms
], dtype=np.float32)  # Shape: (N, 3)
```

**Usage in GNN**:
- Stored as `Data.pos` attribute
- Not directly concatenated with node features
- Used for:
  1. **Edge construction**: Distance calculations
  2. **Geometric GNNs**: Invariant/equivariant operations (EGNN, GVP-GNN)
  3. **Visualization**: Structure rendering

**Coordinate frame considerations**:
- **AlphaFold2 output**: Arbitrary global frame
- **Alignment**: No canonical orientation
- **Implication**: GNN must be **rotation-invariant** or **rotation-equivariant**

`[DESIGN DECISION]`: Use E(3)-equivariant GNNs (EGNN, TFN) to naturally handle rotations.

---

## 3. Advanced Node Features (`[TODO]` / `[INFERRED]`)

### 3.1 Biochemical Properties

**Rationale**: Amino acids with similar chemical properties often substitute functionally, even if sequence-different.

#### 3.1.1 Physicochemical Descriptors (7 dimensions)

`[INFERRED]` - Based on standard protein bioinformatics:

```python
def get_physicochemical_features(aa: str) -> np.ndarray:
    """
    Returns 7-dimensional property vector:
    [hydrophobicity, charge, polarity, size, aromaticity, 
     h_bond_donors, h_bond_acceptors]
    """
    
    # Kyte-Doolittle hydrophobicity scale (normalized to [0,1])
    HYDROPHOBICITY = {
        'ILE': 0.92, 'VAL': 0.88, 'LEU': 0.84, 'PHE': 0.77,
        'CYS': 0.73, 'MET': 0.69, 'ALA': 0.62, 'GLY': 0.12,
        'THR': 0.08, 'SER': 0.08, 'TRP': 0.08, 'TYR': 0.04,
        'PRO': 0.00, 'HIS': 0.00, 'GLU': 0.00, 'GLN': 0.00,
        'ASP': 0.00, 'ASN': 0.00, 'LYS': 0.00, 'ARG': 0.00
    }
    
    # Charge at pH 7.4
    CHARGE = {
        'ARG': +1, 'LYS': +1, 'HIS': +0.5,  # Basic
        'ASP': -1, 'GLU': -1,                # Acidic
        # Rest: 0 (neutral)
    }
    
    # Polarity: 1 = polar, 0 = nonpolar
    POLAR = set(['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'])
    
    # Volume (normalized by largest, TRP)
    VOLUME = {
        'GLY': 60.1, 'ALA': 88.6, 'SER': 89.0, 'CYS': 108.5,
        'ASP': 111.1, 'PRO': 112.7, 'ASN': 114.1, 'THR': 116.1,
        'GLU': 138.4, 'VAL': 140.0, 'GLN': 143.8, 'HIS': 153.2,
        'MET': 162.9, 'ILE': 166.7, 'LEU': 166.7, 'LYS': 168.6,
        'ARG': 173.4, 'PHE': 189.9, 'TYR': 193.6, 'TRP': 227.8
    }
    
    # Aromaticity
    AROMATIC = set(['PHE', 'TYR', 'TRP', 'HIS'])
    
    # Hydrogen bond capacity
    H_DONOR = {
        'ARG': 4, 'LYS': 2, 'TRP': 1, 'HIS': 1, 'ASN': 2, 
        'GLN': 2, 'SER': 1, 'THR': 1, 'TYR': 1, 'CYS': 1
    }
    
    H_ACCEPTOR = {
        'ASP': 2, 'GLU': 2, 'ASN': 2, 'GLN': 2, 'SER': 1,
        'THR': 1, 'TYR': 1, 'HIS': 1, 'MET': 1, 'CYS': 1
    }
    
    features = np.array([
        HYDROPHOBICITY.get(aa, 0.5),
        CHARGE.get(aa, 0.0),
        1.0 if aa in POLAR else 0.0,
        VOLUME.get(aa, 100) / 227.8,  # Normalize by TRP
        1.0 if aa in AROMATIC else 0.0,
        H_DONOR.get(aa, 0) / 4.0,      # Normalize by ARG
        H_ACCEPTOR.get(aa, 0) / 2.0    # Normalize by ASP/GLU
    ], dtype=np.float32)
    
    return features
```

**References**:
- Kyte & Doolittle (1982): Hydrophobicity scale
- Zimmerman et al. (1968): Amino acid volumes
- Fauchere & Pliska (1983): Octanol-water partition coefficients

#### 3.1.2 Combined Feature Vector

`[TODO]` - Implement in `build_graphs.py`:

```python
def create_node_features(residue_name: str) -> np.ndarray:
    """
    Combined feature vector (28 dimensions):
    - One-hot encoding: 21 dims
    - Physicochemical: 7 dims
    """
    one_hot = encode_residue(residue_name)  # 21 dims
    physchem = get_physicochemical_features(residue_name)  # 7 dims
    
    return np.concatenate([one_hot, physchem], axis=-1)  # 28 dims total
```

**Justification**:
- **Complementary information**: One-hot captures identity, physicochemical captures similarity
- **Learned weighting**: GNN can learn to weight identity vs. properties for binding prediction
- **Generalization**: Similar amino acids have similar property vectors → better transfer

### 3.2 Positional Encoding

**Problem**: Graph structure lacks inherent node ordering (unlike sequences). Proteins have sequential structure (N→C terminus).

#### 3.2.1 Residue Index Encoding (2 dimensions)

`[INFERRED]` - Sinusoidal positional encoding (inspired by Transformers):

```python
def get_positional_encoding(residue_idx: int, max_len: int = 500) -> np.ndarray:
    """
    Sinusoidal position encoding:
    PE(pos, 0) = sin(pos / 10000^(0/d))
    PE(pos, 1) = cos(pos / 10000^(0/d))
    """
    pos = residue_idx / max_len  # Normalize to [0, 1]
    
    pe = np.array([
        np.sin(pos * np.pi),
        np.cos(pos * np.pi)
    ], dtype=np.float32)
    
    return pe
```

**Alternative**: Learned embeddings (requires training)

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int = 500, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(max_len, embed_dim)
    
    def forward(self, residue_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(residue_indices)
```

**Use cases**:
- Distinguish N-terminal vs. C-terminal residues
- Capture sequential patterns (alpha helices, beta sheets)
- Improve attention mechanisms (sequence-structure cross-attention)

#### 3.2.2 Secondary Structure (3 dimensions)

`[TODO]` - Requires DSSP or AlphaFold2 secondary structure prediction:

```python
def get_secondary_structure(pdb_file: str) -> List[str]:
    """
    Run DSSP to assign secondary structure
    
    Returns:
        List of single-letter codes: ['H', 'E', 'C', ...]
        H: alpha helix, E: beta strand, C: coil
    """
    from Bio.PDB import DSSP
    
    # Parse PDB
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    
    # Run DSSP
    dssp = DSSP(model, pdb_file)
    
    # Extract secondary structure
    ss_list = [dssp[key][2] for key in dssp.keys()]
    return ss_list

def encode_secondary_structure(ss: str) -> np.ndarray:
    """
    One-hot encode secondary structure:
    [helix, sheet, coil]
    """
    encoding = np.zeros(3, dtype=np.float32)
    if ss in ['H', 'G', 'I']:  # Helices
        encoding[0] = 1.0
    elif ss in ['E', 'B']:     # Sheets
        encoding[1] = 1.0
    else:                      # Coil/loop
        encoding[2] = 1.0
    return encoding
```

**Biological relevance**:
- **Binding grooves**: Often formed by beta sheets or helix-loop-helix motifs
- **Flexibility**: Loops (coils) more flexible → important for induced fit
- **Structural context**: Neighboring residues in helices interact differently than in sheets

**AlphaFold2 alternative**:
- AlphaFold2 outputs per-residue pLDDT (confidence)
- Can be used as proxy for rigidity: high pLDDT = rigid core, low pLDDT = flexible loop

```python
def extract_plddt_from_pdb(pdb_file: str) -> np.ndarray:
    """pLDDT stored in B-factor column"""
    plddts = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                plddt = float(line[60:66])  # B-factor column
                plddts.append(plddt / 100.0)  # Normalize to [0, 1]
    return np.array(plddts, dtype=np.float32)
```

### 3.3 Full Node Feature Vector (Summary)

`[INFERRED]` - Proposed final node features (33 dimensions):

| Feature Type | Dimensions | Status | Description |
|--------------|------------|--------|-------------|
| One-hot AA | 21 | ✅ Implemented | Amino acid identity |
| Physicochemical | 7 | `[TODO]` | Hydrophobicity, charge, polarity, size, aromaticity, H-bonds |
| Positional | 2 | `[TODO]` | Sinusoidal encoding of residue index |
| Secondary structure | 3 | `[TODO]` | Helix, sheet, coil |
| **Total** | **33** | - | - |

**Implementation roadmap**:
```python
# In build_graphs.py
def create_full_node_features(
    residue_name: str,
    residue_idx: int,
    secondary_structure: str,
    max_len: int = 500
) -> np.ndarray:
    """
    Returns 33-dimensional feature vector
    """
    # Identity (21 dims)
    one_hot = encode_residue(residue_name)
    
    # Physicochemical (7 dims)
    physchem = get_physicochemical_features(residue_name)
    
    # Positional (2 dims)
    pos_enc = get_positional_encoding(residue_idx, max_len)
    
    # Secondary structure (3 dims)
    ss_enc = encode_secondary_structure(secondary_structure)
    
    return np.concatenate([one_hot, physchem, pos_enc, ss_enc])  # 33 dims
```

---

## 4. Edge Features: Current Implementation

### 4.1 Euclidean Distance (IMPLEMENTED)

**Dimension**: 1 (scalar)

```python
# From build_graphs.py
def build_edges(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = []
    edge_distances = []
    
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            
            if dist <= self.distance_threshold:  # 10 Å
                edges.append([i, j])
                edges.append([j, i])  # Undirected
                edge_distances.extend([dist, dist])
    
    return np.array(edges).T, np.array(edge_distances)
```

**Properties**:
- **Scalar value**: Single number per edge
- **Raw distance**: Not normalized or transformed
- **Symmetric**: dist(i→j) = dist(j→i)

**Usage in GNNs**:
```python
# Typical GNN message passing
edge_attr = edge_distances.unsqueeze(-1)  # (E, 1)
messages = MLP(torch.cat([node_features[edge_index[0]], 
                          node_features[edge_index[1]],
                          edge_attr], dim=-1))
```

**Limitations**:
- **No angular information**: Distance alone doesn't capture relative orientation
- **No bond type**: All edges treated equally (no sequential vs. spatial distinction)

---

## 5. Advanced Edge Features (`[TODO]` / `[INFERRED]`)

### 5.1 Radial Basis Function (RBF) Expansion

**Rationale**: GNNs struggle with raw continuous values. RBF expansion converts distances to learnable features.

#### 5.1.1 Gaussian RBF

`[INFERRED]` - Standard in geometric deep learning (SchNet, DimeNet):

```python
def gaussian_rbf(distances: torch.Tensor, num_basis: int = 16, 
                 cutoff: float = 10.0) -> torch.Tensor:
    """
    Expand distances using Gaussian RBF:
    RBF_k(d) = exp(- (d - μ_k)^2 / 2σ^2)
    
    Args:
        distances: (E,) tensor of edge distances
        num_basis: Number of RBF centers
        cutoff: Maximum distance
    
    Returns:
        (E, num_basis) expanded features
    """
    # Centers: evenly spaced from 0 to cutoff
    centers = torch.linspace(0, cutoff, num_basis)  # (num_basis,)
    
    # Width: uniform spacing
    sigma = (cutoff / num_basis) / 2.0
    
    # Expand distances
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)  # (E, num_basis)
    rbf = torch.exp(-0.5 * (diff / sigma) ** 2)
    
    return rbf  # (E, num_basis)
```

**Visualization** (for 16 basis functions, 10Å cutoff):
```
Distance 2Å:  [0.1, 0.9, 0.3, 0.0, ...]  ← Peak at μ ≈ 2Å
Distance 5Å:  [0.0, 0.1, 0.4, 0.8, ...]  ← Peak at μ ≈ 5Å
Distance 9Å:  [0.0, 0.0, 0.0, 0.1, ...]  ← Peak at μ ≈ 9Å
```

**Advantages**:
- **Learnable**: GNN can learn distance-dependent interactions
- **Smooth**: Continuous, differentiable representation
- **Informative**: Multi-scale distance encoding

#### 5.1.2 Cosine Cutoff Function

`[INFERRED]` - Smooth edge weighting (prevents discontinuity at cutoff):

```python
def cosine_cutoff(distances: torch.Tensor, cutoff: float = 10.0) -> torch.Tensor:
    """
    Smooth cutoff function:
    f(d) = 0.5 * (cos(π * d / cutoff) + 1)  if d < cutoff
           0                                 if d >= cutoff
    """
    cutoff_values = torch.where(
        distances < cutoff,
        0.5 * (torch.cos(torch.pi * distances / cutoff) + 1),
        torch.zeros_like(distances)
    )
    return cutoff_values  # (E,)
```

**Usage**:
```python
# Apply to RBF features
rbf_features = gaussian_rbf(distances, num_basis=16)  # (E, 16)
cutoff_weights = cosine_cutoff(distances)             # (E,)

edge_features = rbf_features * cutoff_weights.unsqueeze(-1)  # (E, 16)
```

**Effect**: Edges near cutoff (9-10Å) have reduced influence → smoother graph boundary.

### 5.2 Geometric Edge Features

#### 5.2.1 Unit Direction Vectors (3 dimensions)

`[INFERRED]` - Encode edge directionality:

```python
def compute_direction_vectors(
    positions: torch.Tensor,       # (N, 3)
    edge_index: torch.Tensor       # (2, E)
) -> torch.Tensor:
    """
    Compute normalized direction vectors for edges
    
    Returns:
        (E, 3) unit vectors
    """
    src, dst = edge_index
    
    # Vector from src to dst
    vectors = positions[dst] - positions[src]  # (E, 3)
    
    # Normalize
    distances = torch.norm(vectors, dim=-1, keepdim=True)  # (E, 1)
    unit_vectors = vectors / (distances + 1e-8)  # (E, 3)
    
    return unit_vectors
```

**Use in E(3)-equivariant GNNs**:
- EGNN uses vectors for directional message passing
- Maintains equivariance under rotations: R(v) = Rv for rotation matrix R

#### 5.2.2 Bond Angles (1 dimension per triplet)

`[TODO]` - Require triplet enumeration:

```python
def compute_bond_angles(
    positions: torch.Tensor,       # (N, 3)
    edge_index: torch.Tensor       # (2, E)
) -> Dict[Tuple[int, int, int], float]:
    """
    Compute angles for triplets (i, j, k) where:
    - i-j edge exists
    - j-k edge exists
    
    Angle at node j between edges i-j and j-k
    """
    angles = {}
    
    # Build adjacency list
    adj = defaultdict(list)
    for src, dst in edge_index.T:
        adj[src.item()].append(dst.item())
    
    # Enumerate triplets
    for j in adj.keys():
        neighbors = adj[j]
        for i in neighbors:
            for k in neighbors:
                if i < k:  # Avoid duplicates
                    # Vectors j→i and j→k
                    vec_ji = positions[i] - positions[j]
                    vec_jk = positions[k] - positions[j]
                    
                    # Angle via dot product
                    cos_angle = torch.dot(vec_ji, vec_jk) / (
                        torch.norm(vec_ji) * torch.norm(vec_jk) + 1e-8
                    )
                    angle = torch.acos(torch.clamp(cos_angle, -1, 1))
                    
                    angles[(i, j, k)] = angle.item()
    
    return angles
```

**Biological significance**:
- **Backbone angles**: Phi (φ), Psi (ψ), Omega (ω) define protein backbone
- **Side chain rotamers**: Chi (χ) angles determine side chain conformations
- **Local geometry**: Angles constrain 3D structure more than distances alone

**Implementation challenge**: Triplet features don't fit standard edge-based GNN message passing → require specialized architectures (e.g., DimeNet, SphereNet).

#### 5.2.3 Torsion Angles (1 dimension per quartet)

`[TODO]` - Dihedral angles for quadruplets:

```python
def compute_torsion_angle(p1, p2, p3, p4) -> float:
    """
    Compute dihedral angle for four points
    Angle between planes (p1,p2,p3) and (p2,p3,p4)
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Normal vectors to planes
    n1 = torch.cross(b1, b2)
    n2 = torch.cross(b2, b3)
    
    # Torsion angle
    cos_angle = torch.dot(n1, n2) / (torch.norm(n1) * torch.norm(n2) + 1e-8)
    angle = torch.acos(torch.clamp(cos_angle, -1, 1))
    
    return angle
```

**Applications**:
- **Ramachandran plot**: Phi-psi angles characterize backbone conformations
- **Rotamer libraries**: Chi angles define side chain orientations
- **Binding interface**: Torsional flexibility affects induced fit

### 5.3 Edge Type Classification (1 dimension, categorical)

`[INFERRED]` - Distinguish different interaction types:

```python
def classify_edge_type(
    residue_i: str, residue_j: str, distance: float
) -> str:
    """
    Classify edge based on distance and residue types
    
    Types:
    - 'close_contact': < 5Å (direct interaction)
    - 'medium_range': 5-8Å (indirect interaction)
    - 'long_range': 8-10Å (context)
    - 'charged_interaction': Oppositely charged, < 8Å
    - 'hydrophobic': Both hydrophobic, < 6Å
    """
    # Charge-based classification
    POSITIVE = set(['ARG', 'LYS', 'HIS'])
    NEGATIVE = set(['ASP', 'GLU'])
    HYDROPHOBIC = set(['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET'])
    
    if distance < 5.0:
        return 'close_contact'
    elif residue_i in POSITIVE and residue_j in NEGATIVE and distance < 8.0:
        return 'charged_interaction'
    elif residue_i in HYDROPHOBIC and residue_j in HYDROPHOBIC and distance < 6.0:
        return 'hydrophobic'
    elif distance < 8.0:
        return 'medium_range'
    else:
        return 'long_range'

def encode_edge_type(edge_type: str) -> np.ndarray:
    """One-hot encode edge type (5 dimensions)"""
    types = ['close_contact', 'charged_interaction', 'hydrophobic', 
             'medium_range', 'long_range']
    encoding = np.zeros(5, dtype=np.float32)
    encoding[types.index(edge_type)] = 1.0
    return encoding
```

**GNN usage**:
- **Heterogeneous GNNs**: Different message passing functions per edge type
- **Attention weighting**: Attend more to charged interactions near binding site
- **Interpretability**: Identify which interaction types drive predictions

### 5.4 Full Edge Feature Vector (Summary)

`[INFERRED]` - Proposed final edge features (25 dimensions):

| Feature Type | Dimensions | Status | Description |
|--------------|------------|--------|-------------|
| RBF expansion | 16 | `[TODO]` | Gaussian basis functions (0-10Å) |
| Cutoff weight | 1 | `[TODO]` | Cosine cutoff function |
| Direction vector | 3 | `[TODO]` | Normalized (src→dst) unit vector |
| Edge type | 5 | `[TODO]` | Close/charged/hydrophobic/medium/long |
| **Total** | **25** | - | - |

**Note**: Bond angles and torsions not included in edge features (require specialized GNN architectures like DimeNet).

---

## 6. Alternative Featurization Strategies

### 6.1 Learned Embeddings

`[ALTERNATIVE]` - Replace one-hot with learned embeddings:

```python
class NodeEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 21, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, aa_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aa_indices: (N,) tensor of amino acid indices
        Returns:
            (N, embed_dim) embeddings
        """
        return self.embedding(aa_indices)
```

**Advantages**:
- **Compact**: 64 dims vs. 21 dims (one-hot)
- **Learned similarity**: Similar amino acids have similar embeddings
- **Flexible**: Can be pre-trained on large protein datasets

**Disadvantages**:
- **Requires training data**: No biochemical grounding
- **Harder to interpret**: Embedding dimensions lack clear meaning
- **Overfitting risk**: May learn dataset-specific patterns

### 6.2 ESM-2 Residue Embeddings

`[INFERRED]` - Use sequence model embeddings as node features:

```python
def extract_esm2_residue_embeddings(
    sequence: str,
    esm2_model,
    tokenizer
) -> torch.Tensor:
    """
    Extract per-residue embeddings from ESM-2
    
    Returns:
        (seq_len, 1280) for ESM-2 650M
        (seq_len, 2560) for ESM-2 3B
    """
    # Tokenize
    tokens = tokenizer(sequence, return_tensors='pt')
    
    # Forward pass
    with torch.no_grad():
        outputs = esm2_model(**tokens, output_hidden_states=True)
    
    # Extract last layer (layer 33 for 650M)
    residue_embeddings = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    
    # Remove batch dim and special tokens
    embeddings = residue_embeddings[0, 1:-1, :]  # (seq_len, hidden_dim)
    
    return embeddings
```

**Integration with GNN**:
```python
# Node features = ESM-2 embeddings (1280 dims)
node_features = extract_esm2_residue_embeddings(sequence, esm2_model, tokenizer)

# Build graph with these features
graph = Data(
    x=node_features,           # (N, 1280) instead of (N, 21)
    pos=positions,             # (N, 3)
    edge_index=edges,          # (2, E)
    edge_attr=edge_features    # (E, 25)
)
```

**Advantages**:
- **Rich representations**: Evolutionary + sequence context
- **Pre-trained**: No need to train amino acid embeddings
- **Consistency**: Same embeddings used in sequence model

**Disadvantages**:
- **High dimensionality**: 1280 dims → more GNN parameters
- **Sequence-only bias**: ESM-2 trained without structure info
- **Computational cost**: ESM-2 forward pass for every graph

`[DESIGN DECISION]`: Hybrid approach:
1. Use one-hot + physicochemical for initial GNN layers (grounded features)
2. Fuse ESM-2 embeddings at multimodal fusion stage (see Part 4)

---

## 7. Graph Construction Variants

### 7.1 Sequential + Spatial Edges

`[TODO]` - Add explicit backbone connectivity:

```python
def build_hybrid_edges(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two edge types:
    1. Sequential: Consecutive residues (i, i+1)
    2. Spatial: Distance-based (< 10Å)
    """
    N = len(positions)
    edges = []
    edge_types = []  # 0 = sequential, 1 = spatial
    
    # Sequential edges
    for i in range(N - 1):
        edges.append([i, i+1])
        edges.append([i+1, i])
        edge_types.extend([0, 0])
    
    # Spatial edges (excluding sequential)
    for i in range(N):
        for j in range(i + 2, N):  # Skip i+1 (already added)
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= 10.0:
                edges.append([i, j])
                edges.append([j, i])
                edge_types.extend([1, 1])
    
    return np.array(edges).T, np.array(edge_types)
```

**Heterogeneous GNN**:
```python
# Different message passing per edge type
def message_passing(node_features, edge_index, edge_types):
    sequential_edges = edge_index[:, edge_types == 0]
    spatial_edges = edge_index[:, edge_types == 1]
    
    # Process separately
    msg_seq = gnn_layer_seq(node_features, sequential_edges)
    msg_spatial = gnn_layer_spatial(node_features, spatial_edges)
    
    # Combine
    return msg_seq + msg_spatial
```

### 7.2 K-Nearest Neighbors (KNN) Graphs

`[ALTERNATIVE]` - Fixed node degree:

```python
from sklearn.neighbors import NearestNeighbors

def build_knn_graph(positions: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Connect each node to k nearest neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip self
            edges.append([i, j])
    
    return np.array(edges).T
```

**Advantages**:
- **Uniform degree**: Every node has exactly k edges → balanced information flow
- **Adaptive**: Automatically adjusts to local density

**Disadvantages**:
- **Non-symmetric**: j in k-NN(i) doesn't guarantee i in k-NN(j)
- **Hyper parameter**: Choice of k affects performance

### 7.3 Radius Graph with Adaptive Cutoff

`[TODO]` - Per-residue cutoff based on local density:

```python
def build_adaptive_radius_graph(
    positions: np.ndarray,
    base_radius: float = 10.0,
    density_factor: float = 0.5
) -> np.ndarray:
    """
    Adaptive cutoff: larger in sparse regions, smaller in dense regions
    """
    from scipy.spatial import distance_matrix
    
    dist_matrix = distance_matrix(positions, positions)
    
    edges = []
    for i in range(len(positions)):
        # Local density: median distance to 5 nearest neighbors
        sorted_dists = np.sort(dist_matrix[i])[1:6]  # Exclude self
        local_density = np.median(sorted_dists)
        
        # Adaptive cutoff
        cutoff = base_radius * (1 + density_factor * (local_density / base_radius))
        
        # Add edges within cutoff
        neighbors = np.where(dist_matrix[i] < cutoff)[0]
        for j in neighbors:
            if i != j:
                edges.append([i, j])
    
    return np.array(edges).T
```

**Rationale**: Binding interfaces often have high local density → smaller cutoff prevents over-connectivity.

---

## 8. Benchmark Featurization Schemes

### 8.1 Comparison of Feature Dimensionalities

| Scheme | Node Dims | Edge Dims | Reference |
|--------|-----------|-----------|-----------|
| **Current (basic)** | 21 | 1 | Our implementation |
| **Proposed (full)** | 33 | 25 | This document |
| SchNet | - | 50 (RBF) | Schütt et al., 2017 |
| DimeNet | - | 128 (SBF) | Gasteiger et al., 2020 |
| GVP-GNN | 100 (scalar + vector) | 16 (scalar) | Jing et al., 2021 |
| AlphaFold2 (pair repr.) | - | 128 | Jumper et al., 2021 |

**SBF** (Spherical Bessel Functions): Angular + radial features for triplets (DimeNet)

**GVP** (Geometric Vector Perceptrons): Combines scalar features + 3D vector features

### 8.2 Ablation Study Design (`[FUTURE WORK]`)

**Proposed experiments**:

| Experiment | Node Features | Edge Features | Expected Performance |
|------------|---------------|---------------|----------------------|
| Baseline | One-hot (21) | Distance (1) | Moderate |
| + Physicochemical | One-hot + physchem (28) | Distance (1) | +5% improvement |
| + RBF edges | One-hot + physchem (28) | RBF (16) | +8% improvement |
| + Pos + SS | Full (33) | RBF (16) | +10% improvement |
| + Edge types | Full (33) | Full (25) | +12% improvement |
| + ESM-2 node embeddings | ESM-2 (1280) | Full (25) | +15% improvement (?) |

**Evaluation metric**: Test set R² on binding affinity prediction

---

## 9. Implementation Roadmap

### 9.1 Phase 1: Enhanced Basic Features (`[HIGH PRIORITY]`)

**Tasks**:
1. Implement physicochemical descriptors (7 dims)
2. Add RBF expansion for edge distances (16 dims)
3. Integrate into `build_graphs.py`

**Estimated effort**: 1-2 days

**Code changes**:
```python
# build_graphs.py
class PDBToGraph:
    def __init__(self, 
                 distance_threshold: float = 10.0,
                 use_rbf: bool = True,
                 rbf_basis: int = 16):
        self.distance_threshold = distance_threshold
        self.use_rbf = use_rbf
        self.rbf_basis = rbf_basis
    
    def create_node_features(self, residue_name: str) -> np.ndarray:
        one_hot = encode_residue(residue_name)
        physchem = get_physicochemical_features(residue_name)
        return np.concatenate([one_hot, physchem])  # 28 dims
    
    def create_edge_features(self, distances: np.ndarray) -> np.ndarray:
        if self.use_rbf:
            return gaussian_rbf(distances, num_basis=self.rbf_basis)
        else:
            return distances.reshape(-1, 1)
```

### 9.2 Phase 2: Structural Features (`[MEDIUM PRIORITY]`)

**Tasks**:
1. Integrate DSSP or AlphaFold2 secondary structure
2. Add positional encoding
3. Add unit direction vectors for edges

**Estimated effort**: 3-5 days (requires DSSP installation/integration)

### 9.3 Phase 3: Advanced Features (`[LOW PRIORITY]`)

**Tasks**:
1. Edge type classification
2. Bond angle computation (triplet features)
3. Torsion angle computation (quartet features)
4. Adaptive graph construction

**Estimated effort**: 1-2 weeks (requires architecture changes for angles)

---

## 10. Conclusion

This document specified comprehensive graph featurization strategies:

**Implemented**:
- ✅ One-hot amino acid encoding (21 dims)
- ✅ 3D Cartesian coordinates
- ✅ Euclidean distance edges (1 dim)

**Proposed (`[TODO]`)**:
- 🔄 Physicochemical properties (7 dims)
- 🔄 Positional encoding (2 dims)
- 🔄 Secondary structure (3 dims)
- 🔄 RBF distance expansion (16 dims)
- 🔄 Edge directionality (3 dims)
- 🔄 Edge type classification (5 dims)

**Advanced (`[FUTURE]`)**:
- ⏳ Bond angles (triplet features)
- ⏳ Torsion angles (quartet features)
- ⏳ ESM-2 residue embeddings (1280 dims)
- ⏳ Adaptive graph construction

**Key takeaways**:
1. **Minimal features suffice initially**: One-hot + distance can work (GNN learns representations)
2. **Domain knowledge helps**: Physicochemical properties accelerate learning
3. **Geometric features crucial**: RBF + angles capture 3D constraints
4. **Scalability matters**: 33-dim nodes vs. 1280-dim ESM-2 embeddings (computational trade-off)

**Next**: Part 3 details GNN architectures to process these graph representations.

---

**Document metadata**:
- **Version**: 1.0
- **Status**: Part 2 of 5 (Graph Featurization)
- **Word count**: ~6,800 words
- **Code examples**: 20+
