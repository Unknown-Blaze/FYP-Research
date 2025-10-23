from graphviz import Digraph

# ============================================================
# Multimodal (Sequence + Structure) Fine-tuning Pipeline
# ============================================================
dot = Digraph("Multimodal_ESM2_AlphaFold_Pipeline", format="png")
# Reduced ranksep from 1.5 to 0.8 to decrease horizontal spacing
dot.attr(rankdir="LR", splines="ortho", concentrate="true", bgcolor="white", ranksep="0.8", nodesep="1.2")
# Added global node attributes for bigger text (fontsize) and bigger boxes (height)
dot.attr('node', fontsize='22', height='1.2')


# ============================================================
# 🧬 Sequence Branch
# ============================================================

# MHC and Peptide Inputs
# Increased width
dot.node("mhc_seq", "MHC Sequence", shape="box", style="filled", fillcolor="#f4cccc", width="3.0")
dot.node("pep_seq", "Peptide Sequence", shape="box", style="filled", fillcolor="#c9daf8", width="3.0")

# Tokenizers
dot.node("mhc_tok", "EsmTokenizer", shape="ellipse", style="filled", fillcolor="#d9ead3")
dot.node("pep_tok", "EsmTokenizer", shape="ellipse", style="filled", fillcolor="#d9ead3")

# Encoders
dot.node("mhc_enc", "ESM3 Encoder\n(hidden dim = 1280)", shape="box3d", style="filled", fillcolor="#ffe599")
dot.node("pep_enc", "ESM3 Encoder\n(hidden dim = 1280)", shape="box3d", style="filled", fillcolor="#ffe599")

# Pooling
dot.node("mhc_pool", "Mean Pooling\n→ (1×1280)", shape="parallelogram", style="filled", fillcolor="#fff2cc")
dot.node("pep_pool", "Mean Pooling\n→ (1×1280)", shape="parallelogram", style="filled", fillcolor="#fff2cc")

# Cross Attention (MHC ↔ Peptide)
dot.node("cross1", "Cross Attention\n[Embedding₁ ↔ Embedding₂]\n→ Fused Sequence Embedding₃ (1×1280)", 
         shape="ellipse", style="filled", fillcolor="#ead1dc")

# ============================================================
# 🧩 Structure Branch
# ============================================================

# Combine MHC + Peptide for structure prediction
# Increased width
dot.node("combine", "Combine MHC + Peptide\n(Complex Input)", shape="box", style="filled", fillcolor="#fce5cd", width="3.3")

# AlphaFold2 Fine-tuning (Structure Prediction)
# Increased width
dot.node("alphafold", "AlphaFold2 Fine-tune\n(Predict MHC–Peptide Structure)", shape="box3d", style="filled", fillcolor="#d9ead3", width="3.8")

# Graph Mamba Encoder
dot.node("graph_mamba", "Graph Mamba Encoder\n→ Structural Embedding₄ (1×1024)", shape="box3d", style="filled", fillcolor="#b6d7a8")

# ============================================================
# 🔗 Fusion & Regression Head
# ============================================================

# Cross Attention Fusion (Sequence ↔ Structure)
dot.node("cross2", "Cross Attention\n[Embedding₃ ↔ Embedding₄]\n→ Unified Embedding₅", 
         shape="ellipse", style="filled", fillcolor="#d9d2e9")

# Regression Head
dot.node("reg_head", "Regression Head\n(FC + Dropout + ReLU)\n→ pIC₅₀ (1×1)", 
         shape="rect", style="filled,bold", fillcolor="#cfe2f3")

# Mathematical Conversion
# Added a specific fontsize to keep this note box from being too large
dot.node("eq", """Mathematical Conversion:
pIC₅₀ = −log₁₀(IC₅₀ [M])
pIC₅₀_norm = (pIC₅₀ − pIC₅₀_min) / (pIC₅₀_max − pIC₅₀_min)
IC₅₀ = 10⁻ᵖᴵᶜ⁵⁰
pIC₅₀ = pIC₅₀_norm × (pIC₅₀_max − pIC₅₀_min) + pIC₅₀_min
""",
    shape="note", style="filled", fillcolor="#f9cb9c", fontsize='16'
)

# ============================================================
# 🧩 Connections
# ============================================================

# --- Sequence Path ---
dot.edge("mhc_seq", "mhc_tok")
dot.edge("pep_seq", "pep_tok")
dot.edge("mhc_tok", "mhc_enc")
dot.edge("pep_tok", "pep_enc")
dot.edge("mhc_enc", "mhc_pool")
dot.edge("pep_enc", "pep_pool")
dot.edge("mhc_pool", "cross1")
dot.edge("pep_pool", "cross1")

# --- Structure Path ---
dot.edge("mhc_seq", "combine")
dot.edge("pep_seq", "combine")
dot.edge("combine", "alphafold")
dot.edge("alphafold", "graph_mamba")

# --- Fusion ---
dot.edge("cross1", "cross2")
dot.edge("graph_mamba", "cross2")

# --- Output ---
dot.edge("cross2", "reg_head")
dot.edge("reg_head", "eq")

# ============================================================
# ✅ Render Diagram
# ============================================================
dot.render("multimodal_seq_struct_pipeline", cleanup=True)
print("✅ Saved as multimodal_seq_struct_pipeline.png")