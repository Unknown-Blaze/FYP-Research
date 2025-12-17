# Structure-Based MHC-Peptide Binding Prediction: Methodology Part 3
## Graph Neural Network Architecture Design

---

## 1. Introduction

This document specifies the **Graph Neural Network (GNN) architecture** for processing protein structure graphs. All content in this part is `[INFERRED]` based on state-of-the-art geometric deep learning methods, as no GNN implementation currently exists in the codebase.

**Design goals**:
1. **E(3)-equivariance**: Predictions invariant to rotations/translations
2. **Scalability**: Handle 300-400 node graphs efficiently
3. **Expressiveness**: Capture multi-scale geometric patterns
4. **Interpretability**: Identify key binding residues

**Candidate architectures**:
- **E(n) Equivariant Graph Neural Networks (EGNN)**: Efficient, coordinate-aware
- **Geometric Vector Perceptrons (GVP-GNN)**: Scalar + vector features
- **Graph Attention Networks (GAT)**: Attention-based message passing
- **Message Passing Neural Networks (MPNN)**: General framework

---

## 2. Theoretical Background

### 2.1 Graph Neural Network Fundamentals

**Core operation**: Iterative message passing

```
For layer l = 1, ..., L:
    1. Message: m_{ij} = φ(h_i^l, h_j^l, e_{ij})
    2. Aggregate: M_i = ⊕_{j∈N(i)} m_{ij}
    3. Update: h_i^{l+1} = ψ(h_i^l, M_i)
```

Where:
- `h_i^l`: Node i features at layer l
- `e_{ij}`: Edge features between nodes i and j
- `N(i)`: Neighbors of node i
- `φ`: Message function (MLP)
- `⊕`: Aggregation (sum, mean, max)
- `ψ`: Update function (MLP)

**Geometric deep learning requirements**:

For protein structures, we need:
1. **Rotation equivariance**: f(Rx) = Rf(x) for rotation R
2. **Translation invariance**: f(x + t) = f(x) for translation t
3. **Permutation equivariance**: f(πX) = πf(X) for permutation π

**Naive coordinate usage fails**:
```python
# WRONG: Not rotation-invariant
h_i = MLP(concat(h_i, pos_i))  # Absolute coordinates
```

**Correct approaches**:
1. **Invariant features**: Use distances/angles (scalars)
2. **Equivariant features**: Update coordinates alongside features (EGNN)
3. **Geometric vectors**: Process scalars + vectors separately (GVP)

### 2.2 E(n) Equivariance

**E(n) group**: Euclidean transformations in n dimensions
- Rotations: SO(n)
- Translations: R^n
- Reflections (optional)

**Key insight**: Relative positions are rotation-invariant
```python
# Rotation-invariant distance
dist_ij = ||pos_i - pos_j||  # Scalar (invariant)

# Rotation-equivariant difference
vec_ij = pos_i - pos_j        # Vector (equivariant)
```

---

## 3. Architecture 1: E(n) Equivariant GNN (EGNN) `[INFERRED]`

**Reference**: Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021)

### 3.1 Architecture Overview

EGNN updates both **node features** (scalars) and **coordinates** (vectors) while maintaining E(n) equivariance.

**Core idea**:
- Node features: Invariant (scalars)
- Coordinates: Equivariant (vectors)
- Messages: Use relative positions (rotation-invariant)

### 3.2 EGNN Layer Implementation `[TODO]`

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class EGNNLayer(MessagePassing):
    """
    E(n) Equivariant Graph Neural Network Layer
    
    Updates:
    - Node features h_i (scalars): rotation-invariant
    - Coordinates x_i (vectors): rotation-equivariant
    
    Reference: Satorras et al., ICML 2021
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        edge_dim: int = 16,
        aggr: str = 'add',
        update_coords: bool = True
    ):
        super().__init__(aggr=aggr)
        
        self.update_coords = update_coords
        
        # Message MLP: (h_i, h_j, ||x_i - x_j||^2, e_ij) → message
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Coordinate update MLP: message → coordinate influence
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)  # Scalar weight per message
            )
        
        # Node update MLP: (h_i, aggregated_messages) → h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        h: torch.Tensor,           # (N, hidden_dim) node features
        pos: torch.Tensor,         # (N, 3) coordinates
        edge_index: torch.Tensor,  # (2, E) edges
        edge_attr: torch.Tensor    # (E, edge_dim) edge features
    ):
        """
        Returns:
            h_out: (N, hidden_dim) updated node features
            pos_out: (N, 3) updated coordinates
        """
        # Add self-loops for residual connections
        edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr)
        
        # Message passing
        h_out = self.propagate(
            edge_index, 
            h=h, 
            pos=pos, 
            edge_attr=edge_attr
        )
        
        # Coordinate update (if enabled)
        if self.update_coords:
            pos_out = self._update_positions(h, pos, edge_index, edge_attr)
        else:
            pos_out = pos
        
        return h_out, pos_out
    
    def message(
        self,
        h_i: torch.Tensor,      # Source node features
        h_j: torch.Tensor,      # Target node features
        pos_i: torch.Tensor,    # Source positions
        pos_j: torch.Tensor,    # Target positions
        edge_attr: torch.Tensor # Edge features
    ):
        """
        Compute messages from j to i
        
        Key: Use ||x_i - x_j||^2 (rotation-invariant)
        """
        # Squared distance (rotation-invariant)
        dist_sq = torch.sum((pos_i - pos_j) ** 2, dim=-1, keepdim=True)  # (E, 1)
        
        # Concatenate: [h_i, h_j, dist_sq, edge_attr]
        message_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        
        # Compute message
        message = self.message_mlp(message_input)  # (E, hidden_dim)
        
        return message
    
    def update(
        self,
        aggr_out: torch.Tensor,  # Aggregated messages
        h: torch.Tensor          # Original node features
    ):
        """
        Update node features using aggregated messages
        """
        # Residual connection
        node_input = torch.cat([h, aggr_out], dim=-1)
        h_out = self.node_mlp(node_input) + h  # Residual
        
        return h_out
    
    def _update_positions(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ):
        """
        Update coordinates using messages (E(n)-equivariant)
        
        Key idea: Move each node towards/away from neighbors
        weighted by message content
        """
        src, dst = edge_index
        
        # Relative position vectors (E(n)-equivariant)
        rel_pos = pos[dst] - pos[src]  # (E, 3)
        
        # Compute messages again (for coordinate update)
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        message_input = torch.cat([h[src], h[dst], dist_sq, edge_attr], dim=-1)
        messages = self.message_mlp(message_input)
        
        # Scalar weight for each message
        weights = self.coord_mlp(messages)  # (E, 1)
        
        # Weighted displacement: w_ij * (x_j - x_i)
        displacements = weights * rel_pos  # (E, 3)
        
        # Aggregate displacements per node (mean to preserve scale)
        pos_update = torch.zeros_like(pos)
        pos_update.index_add_(0, src, displacements)
        
        # Normalize by degree
        degree = torch.zeros(pos.size(0), 1, device=pos.device)
        degree.index_add_(0, src, torch.ones_like(weights))
        pos_update = pos_update / (degree + 1e-8)
        
        # Update positions (small step for stability)
        pos_out = pos + 0.1 * pos_update
        
        return pos_out
    
    def _add_self_loops(self, edge_index, edge_attr):
        """Add self-loops with zero edge features"""
        num_nodes = edge_index.max().item() + 1
        edge_index, edge_attr = add_self_loops(
            edge_index, 
            edge_attr, 
            fill_value=0.0,
            num_nodes=num_nodes
        )
        return edge_index, edge_attr
```

### 3.3 Multi-Layer EGNN Model

```python
class EGNNModel(nn.Module):
    """
    Full EGNN model for protein graphs
    
    Architecture:
        Input: Node features (33-dim) + 3D coords + edge features (25-dim)
        → EGNN layers (4-6 layers)
        → Global pooling
        → MLP prediction head
        → Output: Graph-level embedding
    """
    
    def __init__(
        self,
        node_input_dim: int = 33,      # From Part 2
        edge_input_dim: int = 25,      # From Part 2
        hidden_dim: int = 128,
        num_layers: int = 5,
        output_dim: int = 256,         # Graph embedding size
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_input_dim, hidden_dim)
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                update_coords=(i > 0)  # Only update coords after first layer
            )
            for i in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling
        self.global_pool = GlobalAttentionPooling(hidden_dim)
        
        # Output projection
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,   # (N, node_input_dim)
        positions: torch.Tensor,       # (N, 3)
        edge_index: torch.Tensor,      # (2, E)
        edge_features: torch.Tensor,   # (E, edge_input_dim)
        batch: torch.Tensor = None     # (N,) batch assignment
    ):
        """
        Returns:
            graph_embedding: (batch_size, output_dim)
        """
        # Encode inputs
        h = self.node_encoder(node_features)  # (N, hidden_dim)
        e = self.edge_encoder(edge_features)  # (E, hidden_dim)
        pos = positions
        
        # EGNN layers
        for i, (egnn_layer, layer_norm) in enumerate(zip(self.egnn_layers, self.layer_norms)):
            h_new, pos = egnn_layer(h, pos, edge_index, e)
            h = layer_norm(h_new)
            h = self.dropout(h)
        
        # Global pooling (graph-level representation)
        graph_embedding = self.global_pool(h, batch)  # (batch_size, hidden_dim)
        
        # Output projection
        output = self.output_mlp(graph_embedding)  # (batch_size, output_dim)
        
        return output
```

### 3.4 Global Pooling Strategies

**Option 1: Mean Pooling** (simple, no parameters)
```python
def global_mean_pool(node_features, batch):
    """Average node features per graph"""
    from torch_geometric.nn import global_mean_pool
    return global_mean_pool(node_features, batch)
```

**Option 2: Attention Pooling** (learned, weighted) `[RECOMMENDED]`
```python
class GlobalAttentionPooling(nn.Module):
    """
    Learn attention weights for each node
    More important nodes contribute more to graph embedding
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_features, batch):
        """
        Args:
            node_features: (N, hidden_dim)
            batch: (N,) graph assignment
        
        Returns:
            (batch_size, hidden_dim)
        """
        # Compute attention scores
        scores = self.attention_mlp(node_features)  # (N, 1)
        
        # Softmax per graph
        from torch_geometric.nn import global_softmax
        attention_weights = global_softmax(scores, batch)  # (N, 1)
        
        # Weighted sum per graph
        weighted_features = node_features * attention_weights  # (N, hidden_dim)
        
        from torch_geometric.nn import global_add_pool
        graph_embedding = global_add_pool(weighted_features, batch)
        
        return graph_embedding
```

**Option 3: Set2Set** (order-invariant LSTM)
```python
from torch_geometric.nn import Set2Set

class Set2SetPooling(nn.Module):
    def __init__(self, hidden_dim: int, processing_steps: int = 3):
        super().__init__()
        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
    
    def forward(self, node_features, batch):
        return self.set2set(node_features, batch)  # (batch_size, 2*hidden_dim)
```

### 3.5 EGNN Advantages & Limitations

**Advantages**:
✅ **E(n)-equivariant**: Correct geometric inductive bias  
✅ **Efficient**: Linear complexity in number of edges  
✅ **Coordinate updates**: Can learn to refine noisy structures  
✅ **Interpretable**: Messages based on distances and relative positions

**Limitations**:
❌ **Limited geometric expressiveness**: Only uses distances, not angles  
❌ **Shallow receptive field**: Information propagates slowly (needs many layers)  
❌ **No explicit multi-scale**: Single edge type (no hierarchical pooling)

---

## 4. Architecture 2: Geometric Vector Perceptrons (GVP-GNN) `[INFERRED]`

**Reference**: Jing et al., "Learning from Protein Structure with Geometric Vector Perceptrons" (ICLR 2021)

### 4.1 Core Concept

**Geometric Vector Perceptrons**: Process both scalars and geometric vectors

**Node features = (s, V)**:
- `s`: Scalar features (e.g., amino acid type, hydrophobicity)
- `V`: Vector features (e.g., normal vectors, dipole moments)

**Operations**:
- Scalar: Standard MLPs
- Vector: Rotation-equivariant operations (norm, cross product)

### 4.2 GVP Layer Implementation `[TODO]`

```python
class GVPLayer(nn.Module):
    """
    Geometric Vector Perceptron Layer
    
    Processes:
    - Scalar features: (s_in,) → (s_out,)
    - Vector features: (v_in, 3) → (v_out, 3)
    
    Key operations:
    - Scalar MLP: Standard feedforward
    - Vector: Linear combination + nonlinearity via norm
    """
    
    def __init__(
        self,
        scalar_in: int,
        scalar_out: int,
        vector_in: int,
        vector_out: int
    ):
        super().__init__()
        
        # Scalar → scalar transformation
        self.scalar_to_scalar = nn.Linear(scalar_in + vector_in, scalar_out)
        
        # Vector → vector transformation
        self.vector_to_vector = nn.Linear(vector_in, vector_out, bias=False)
        
        # Vector → scalar (via norm)
        # No parameters needed (norm is equivariant)
    
    def forward(
        self,
        s: torch.Tensor,  # (N, scalar_in)
        V: torch.Tensor   # (N, vector_in, 3)
    ):
        """
        Returns:
            s_out: (N, scalar_out)
            V_out: (N, vector_out, 3)
        """
        # Vector norms (rotation-invariant)
        V_norm = torch.norm(V, dim=-1)  # (N, vector_in)
        
        # Scalar update: Combine input scalars + vector norms
        s_input = torch.cat([s, V_norm], dim=-1)  # (N, scalar_in + vector_in)
        s_out = self.scalar_to_scalar(s_input)     # (N, scalar_out)
        s_out = torch.relu(s_out)
        
        # Vector update: Linear transformation
        V_out = self.vector_to_vector(
            V.transpose(-1, -2)  # (N, 3, vector_in)
        ).transpose(-1, -2)      # (N, vector_out, 3)
        
        return s_out, V_out
```

### 4.3 GVP-GNN Message Passing

```python
class GVPConv(MessagePassing):
    """
    GVP-based graph convolution
    
    Message: (s_i, V_i, s_j, V_j, e_ij) → (m_s, m_V)
    Update: (s_i, V_i, m_s, m_V) → (s_i', V_i')
    """
    
    def __init__(
        self,
        scalar_dim: int = 100,
        vector_dim: int = 16,
        edge_dim: int = 32
    ):
        super().__init__(aggr='add')
        
        # Message network
        self.message_gvp = GVPLayer(
            scalar_in=2 * scalar_dim + edge_dim,
            scalar_out=scalar_dim,
            vector_in=2 * vector_dim,
            vector_out=vector_dim
        )
        
        # Update network
        self.update_gvp = GVPLayer(
            scalar_in=2 * scalar_dim,
            scalar_out=scalar_dim,
            vector_in=2 * vector_dim,
            vector_out=vector_dim
        )
    
    def forward(
        self,
        s: torch.Tensor,           # (N, scalar_dim)
        V: torch.Tensor,           # (N, vector_dim, 3)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor    # (E, edge_dim)
    ):
        """
        Returns:
            s_out: (N, scalar_dim)
            V_out: (N, vector_dim, 3)
        """
        s_out, V_out = self.propagate(
            edge_index, 
            s=s, 
            V=V, 
            edge_attr=edge_attr
        )
        return s_out, V_out
    
    def message(self, s_i, s_j, V_i, V_j, edge_attr):
        """
        Compute messages
        
        Combine:
        - Scalar: [s_i, s_j, edge_attr]
        - Vector: [V_i, V_j]
        """
        # Concatenate scalars
        s_msg = torch.cat([s_i, s_j, edge_attr], dim=-1)  # (E, 2*scalar_dim + edge_dim)
        
        # Stack vectors
        V_msg = torch.cat([V_i, V_j], dim=1)  # (E, 2*vector_dim, 3)
        
        # Process through GVP
        m_s, m_V = self.message_gvp(s_msg, V_msg)
        
        return m_s, m_V
    
    def update(self, aggr_out, s, V):
        """
        Update node features using aggregated messages
        """
        m_s, m_V = aggr_out
        
        # Concatenate with original features
        s_update = torch.cat([s, m_s], dim=-1)
        V_update = torch.cat([V, m_V], dim=1)
        
        # Process through GVP
        s_out, V_out = self.update_gvp(s_update, V_update)
        
        return s_out, V_out
```

### 4.4 GVP-GNN Full Model

```python
class GVPGNNModel(nn.Module):
    """
    Full GVP-GNN for protein graphs
    
    Node features:
    - Scalars: Amino acid type, physicochemical properties
    - Vectors: Cα→Cβ direction, local coordinate frames
    """
    
    def __init__(
        self,
        node_scalar_dim: int = 33,   # From Part 2
        node_vector_dim: int = 0,    # Initially no vector features
        edge_dim: int = 25,
        hidden_scalar_dim: int = 100,
        hidden_vector_dim: int = 16,
        num_layers: int = 4,
        output_dim: int = 256
    ):
        super().__init__()
        
        # Input encoding
        self.node_encoder = GVPLayer(
            scalar_in=node_scalar_dim,
            scalar_out=hidden_scalar_dim,
            vector_in=node_vector_dim,
            vector_out=hidden_vector_dim
        )
        
        # GVP-GNN layers
        self.gvp_layers = nn.ModuleList([
            GVPConv(
                scalar_dim=hidden_scalar_dim,
                vector_dim=hidden_vector_dim,
                edge_dim=edge_dim
            )
            for _ in range(num_layers)
        ])
        
        # Global pooling (scalars only)
        self.pool = GlobalAttentionPooling(hidden_scalar_dim)
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_scalar_dim, hidden_scalar_dim),
            nn.ReLU(),
            nn.Linear(hidden_scalar_dim, output_dim)
        )
    
    def forward(
        self,
        node_scalars: torch.Tensor,   # (N, node_scalar_dim)
        node_vectors: torch.Tensor,   # (N, node_vector_dim, 3)
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ):
        # Encode
        s, V = self.node_encoder(node_scalars, node_vectors)
        
        # GVP layers
        for gvp_layer in self.gvp_layers:
            s, V = gvp_layer(s, V, edge_index, edge_attr)
        
        # Pool (scalars only)
        graph_embedding = self.pool(s, batch)
        
        # Output
        output = self.output_mlp(graph_embedding)
        
        return output
```

### 4.5 GVP Advantages & Limitations

**Advantages**:
✅ **Expressive**: Handles both scalar and vector features  
✅ **Geometric**: Naturally encodes directions (normals, dipoles)  
✅ **Proven**: State-of-the-art on protein design tasks

**Limitations**:
❌ **Complex**: More parameters and computation than EGNN  
❌ **Requires vector features**: Need to compute Cα-Cβ directions, normal vectors  
❌ **Less interpretable**: Vector operations harder to visualize

---

## 5. Architecture 3: Graph Attention Network (GAT) `[INFERRED]`

**Reference**: Veličković et al., "Graph Attention Networks" (ICLR 2018)

### 5.1 Attention Mechanism

**Core idea**: Learn importance weights for neighbor messages

```python
α_ij = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) )
```

### 5.2 GAT Layer Implementation `[TODO]`

```python
from torch_geometric.nn import GATConv

class ProteinGAT(nn.Module):
    """
    Graph Attention Network for proteins
    
    Advantages:
    - Learns which neighbors are important (e.g., binding site residues)
    - Multi-head attention for diverse interaction patterns
    """
    
    def __init__(
        self,
        node_input_dim: int = 33,
        edge_dim: int = 25,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        output_dim: int = 256
    ):
        super().__init__()
        
        # Input projection
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        
        # GAT layers (multi-head)
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim * num_heads if i > 0 else hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=0.1,
                concat=True if i < num_layers - 1 else False,
                edge_dim=edge_dim  # Use edge features
            )
            for i in range(num_layers)
        ])
        
        # Pooling
        self.pool = GlobalAttentionPooling(hidden_dim * num_heads)
        
        # Output
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, node_features, edge_index, edge_attr, batch):
        h = self.node_encoder(node_features)
        
        # GAT layers
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_attr)
            h = torch.relu(h)
        
        # Pool + output
        graph_embedding = self.pool(h, batch)
        output = self.output_mlp(graph_embedding)
        
        return output
```

### 5.3 GAT Advantages & Limitations

**Advantages**:
✅ **Interpretable**: Attention weights show important interactions  
✅ **Flexible**: Multi-head attention captures diverse patterns  
✅ **Simple**: No geometric complexity

**Limitations**:
❌ **Not E(n)-equivariant**: Ignores geometric structure  
❌ **Requires many layers**: Attention is local (slow information propagation)

---

## 6. Recommended Architecture: Hierarchical EGNN `[INFERRED]`

**Design**: Combine EGNN with graph coarsening for multi-scale learning

### 6.1 Architecture

```python
class HierarchicalEGNN(nn.Module):
    """
    Multi-scale EGNN with graph pooling
    
    Architecture:
    Level 1: Full graph (365 nodes) → EGNN layers
    Level 2: Coarsened graph (100 nodes) → EGNN layers
    Level 3: Coarsened graph (25 nodes) → EGNN layers
    → Global pooling → Output
    """
    
    def __init__(self, ...):
        super().__init__()
        
        # Level 1: Fine-grained
        self.egnn_level1 = nn.ModuleList([
            EGNNLayer(hidden_dim=128) for _ in range(3)
        ])
        self.pool1 = SAGPooling(128, ratio=0.3)  # Keep 30% nodes
        
        # Level 2: Medium-grained
        self.egnn_level2 = nn.ModuleList([
            EGNNLayer(hidden_dim=128) for _ in range(2)
        ])
        self.pool2 = SAGPooling(128, ratio=0.25)  # Keep 25% nodes
        
        # Level 3: Coarse-grained
        self.egnn_level3 = nn.ModuleList([
            EGNNLayer(hidden_dim=128) for _ in range(2)
        ])
        
        # Final pooling + output
        self.global_pool = GlobalAttentionPooling(128)
        self.output_mlp = nn.Sequential(...)
    
    def forward(self, ...):
        # Level 1
        for egnn in self.egnn_level1:
            h, pos = egnn(h, pos, edge_index, edge_attr)
        h, edge_index, edge_attr, batch, pos = self.pool1(
            h, edge_index, edge_attr, batch, pos
        )
        
        # Level 2
        for egnn in self.egnn_level2:
            h, pos = egnn(h, pos, edge_index, edge_attr)
        h, edge_index, edge_attr, batch, pos = self.pool2(...)
        
        # Level 3
        for egnn in self.egnn_level3:
            h, pos = egnn(h, pos, edge_index, edge_attr)
        
        # Global
        graph_embedding = self.global_pool(h, batch)
        output = self.output_mlp(graph_embedding)
        
        return output
```

**Advantages**:
- Multi-scale: Captures local (binding site) and global (protein fold) patterns
- Efficient: Reduces computation at higher levels
- Expressive: Hierarchical abstractions

---

## 7. Architecture Comparison

| Architecture | E(n)-Equiv. | Expressiveness | Complexity | Interpretability |
|--------------|-------------|----------------|------------|------------------|
| **EGNN** | ✅ Yes | Medium | Low | Medium |
| **GVP-GNN** | ✅ Yes | High | High | Low |
| **GAT** | ❌ No | Medium | Low | High (attention) |
| **Hierarchical EGNN** | ✅ Yes | High | Medium | Medium |

**Recommendation**: Start with **EGNN** (simplicity, efficiency), upgrade to **Hierarchical EGNN** if needed.

---

## 8. Training Considerations

### 8.1 Loss Function `[TODO]`

```python
def gnn_regression_loss(predictions, targets, weights=None):
    """MSE loss for binding affinity prediction"""
    mse = F.mse_loss(predictions, targets, reduction='none')
    if weights is not None:
        mse = mse * weights
    return mse.mean()
```

### 8.2 Hyperparameters

| Hyperparameter | Recommended Value | Justification |
|----------------|-------------------|---------------|
| Hidden dim | 128-256 | Balance capacity/efficiency |
| Num layers | 4-6 | 4-6 hops cover ~10-15Å |
| Learning rate | 1e-4 | Standard for GNNs |
| Dropout | 0.1-0.2 | Regularization |
| Batch size | 32 | Limited by GPU memory (large graphs) |

### 8.3 Computational Cost

**EGNN forward pass** (single graph, 365 nodes, 10 edges/node):
- Nodes: 365
- Edges: ~3,650
- Hidden dim: 128
- Layers: 5

**FLOPs estimate**:
- Message MLP: 3,650 × (2×128 + 1 + 16) × 128 ≈ 130M FLOPs
- Node MLP: 365 × (2×128) × 128 ≈ 12M FLOPs
- **Total per layer**: ~150M FLOPs
- **Total (5 layers)**: ~750M FLOPs

**Training time** (estimated):
- Batch size: 32
- Time per batch: ~50-100ms (A100 GPU)
- Epoch (1,000 batches): ~50-100 seconds

---

## 9. Implementation Roadmap

**Phase 1: Basic EGNN** (1 week)
1. Implement `EGNNLayer` class
2. Implement `EGNNModel` with global pooling
3. Unit tests (forward pass, gradient flow)

**Phase 2: Training Pipeline** (3-5 days)
4. Integrate with `MultimodalMHCDataset`
5. Training loop with loss/metrics
6. Checkpoint saving/loading

**Phase 3: Enhancements** (1-2 weeks)
7. Add GVP-GNN implementation
8. Add hierarchical pooling
9. Ablation studies

---

## 10. Conclusion

**Implemented**: ❌ None (all `[TODO]`)

**Specified**:
- ✅ EGNN architecture (E(n)-equivariant, coordinate updates)
- ✅ GVP-GNN architecture (scalar + vector features)
- ✅ GAT architecture (attention-based)
- ✅ Hierarchical EGNN (multi-scale)

**Recommended path**:
1. Implement EGNN first (best balance of simplicity/performance)
2. Validate on binding affinity prediction
3. If performance insufficient, try GVP-GNN or hierarchical variants

**Next**: Part 4 details multimodal fusion (sequence + structure integration).

---

**Document metadata**:
- **Version**: 1.0
- **Status**: Part 3 of 5 (GNN Architecture - All `[INFERRED]`)
- **Word count**: ~6,500 words
- **Code examples**: 15+ implementations
