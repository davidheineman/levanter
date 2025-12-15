# muP and CompleteP Implementation Summary

Search for `Begin (muP|CompleteP) code`!!

## Overview

The implementation adds two parameterization schemes to nanoGPT:
- **muP**: Enables hyperparameter transfer across model widths
- **CompleteP**: Extends muP to also handle depth scaling

---

## muP Code Blocks

### 1. Attention Scaling (lines 64-66)

**Location**: `CausalSelfAttention.forward()`

```python
### Begin muP code ###
attention_scale = 1.0 / k.size(-1)
### End muP code ###
```

**Summary**: Changes attention scaling from `1/sqrt(d_k)` to `1/d_k`. This prevents attention entropy from growing with width, ensuring consistent attention patterns across model scales.

---

### 2. Hidden Weight Initialization (lines 169-175)

**Location**: `GPT.__init__()` (parameter initialization loop)

```python
### Begin muP code ###
# Adjust hidden weight initialization variance by 1 / mup_width_multiplier
if pn.endswith('c_attn.weight') or pn.endswith('c_fc.weight'):
    torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))
elif pn.endswith('c_proj.weight'):
    torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))
### End muP code ###
```

**Summary**: Scales initialization variance of hidden layer weights by `1/mup_width_multiplier`. This ensures that the magnitude of activations remains consistent regardless of model width.

---

### 3. Input Embedding Scaling (lines 214-216)

**Location**: `GPT.forward()`

```python
### Begin muP code ###
x *= self.config.mup_input_alpha
### End muP code ###
```

**Summary**: Applies an optional tunable multiplier (`mup_input_alpha`) to the input embeddings. This provides a hyperparameter for controlling the scale of inputs into the transformer blocks.

---

### 4. Output Logit Scaling (lines 224-227)

**Location**: `GPT.forward()` (before computing loss)

```python
### Begin muP code ###
# Scaling `x` instead of `logits` allows coord check to log change
x *= self.config.mup_output_alpha / self.config.mup_width_multiplier
### End muP code ###
```

**Summary**: Scales the final hidden states before the language model head by `mup_output_alpha / mup_width_multiplier`. The `1/width_multiplier` factor ensures logits have consistent scale across widths, while `mup_output_alpha` is a tunable hyperparameter.

---

### 5. Optimizer Parameter Groups with Width-Scaled LR (lines 365-393)

**Location**: `GPT.configure_optimizers()`

```python
### Begin muP code ###
optim_groups = [
    {
        'params': emb_params,
        'weight_decay': weight_decay,
        'lr_scale': 1.0,
    },
    {
        'params': hidden_ln_params,
        'weight_decay': 0.0,
        'lr_scale': 1.0,
    },
    {
        'params': hidden_weight_params,
        'weight_decay': weight_decay,
        'lr_scale': width_lr_scaling,  # = 1 / mup_width_multiplier
    },
    {
        'params': hidden_bias_params,
        'weight_decay': 0.0,
        'lr_scale': 1.0,
    },
    {
        'params': final_ln_params,
        'weight_decay': 0.0,
        'lr_scale': 1.0,
    },
]
### End muP code ###
```

**Summary**: Sets up parameter groups with different learning rate scales:
- **Embedding params**: Standard LR (scale 1.0)
- **Hidden layer weights** (`c_attn`, `c_fc`, `c_proj`): LR scaled by `1/width_multiplier`
- **Other params** (biases, LayerNorms): Standard LR (scale 1.0)

This ensures that hidden weight updates remain O(1) regardless of width, enabling hyperparameter transfer.

---

## CompleteP Code Blocks

### 1. Residual Branch Scaling (lines 116-119)

**Location**: `Block.forward()`

```python
### Begin CompleteP code ###
x = x + self.residual_scaling * self.attn(self.ln_1(x))
x = x + self.residual_scaling * self.mlp(self.ln_2(x))
### End CompleteP code ###
```

**Where `residual_scaling` is computed in `Block.__init__()` (line 113)**:
```python
self.residual_scaling = 1/(config.depth_multiplier ** config.depth_alpha_exp) if config.depth_alpha_enabled else 1.0
```

**Summary**: Scales each residual branch by `1/(depth_multiplier^depth_alpha_exp)`. This prevents activation magnitudes from growing with depth. The `depth_alpha_exp` parameter (typically in range [0.5, 1]) controls the strength of this scaling.

---

### 2. Optimizer with Depth-Scaled LR and Epsilon (lines 334-363)

**Location**: `GPT.configure_optimizers()`

```python
### Begin CompleteP code ###
adam_eps *= (1 / self.config.mup_width_multiplier) * (self.config.depth_multiplier ** (-1 * self.config.depth_alpha_exp))
optim_groups = [
    {
        'params': emb_params,
        'weight_decay': weight_decay,
        'lr_scale': 1.0,
    },
    {
        'params': hidden_ln_params,
        'weight_decay': 0.0,
        'lr_scale': depth_lr_scaling,  # = depth_multiplier^(depth_alpha_exp - 1)
    },
    {
        'params': hidden_weight_params,
        'weight_decay': weight_decay / width_lr_scaling,
        'lr_scale': width_lr_scaling * depth_lr_scaling,
    },
    {
        'params': hidden_bias_params,
        'weight_decay': 0.0,
        'lr_scale': depth_lr_scaling,
    },
    {
        'params': final_ln_params,
        'weight_decay': 0.0,
        'lr_scale': 1.0,
    },
]
### End CompleteP code ###
```

**Summary**: Extends the muP optimizer configuration for depth scaling:
- **Adam epsilon**: Scaled by `(1/width_multiplier) * (depth_multiplier^(-depth_alpha_exp))`
- **Hidden LayerNorm params**: LR scaled by `depth_multiplier^(depth_alpha_exp - 1)`
- **Hidden weight params**: LR scaled by both width and depth factors
- **Hidden bias params**: LR scaled by depth factor
- **Weight decay for hidden weights**: Scaled by `width_multiplier` to compensate for LR scaling

This ensures consistent optimization dynamics when scaling both width and depth.

---

## Configuration Parameters

From `GPTConfig` (lines 122-140):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mup_enabled` | `False` | Master switch for muP |
| `mup_disable_attention_scaling` | `False` | Disable muP attention scaling |
| `mup_disable_hidden_lr_scaling` | `False` | Disable muP hidden LR scaling |
| `mup_width_multiplier` | `1` | `width / base_width` (base typically 256) |
| `mup_input_alpha` | `1` | Tunable input embedding multiplier |
| `mup_output_alpha` | `1` | Tunable output logit multiplier |
| `depth_alpha_enabled` | `False` | Enable CompleteP depth scaling |
| `depth_multiplier` | `1.0` | `depth / base_depth` |
| `depth_alpha_exp` | `1.0` | Controls residual scaling strength [0.5, 1] |

---

## Key Insights

1. **muP enables width transfer**: By scaling initializations, learning rates, and attention, optimal hyperparameters found on a small model transfer to larger widths.

2. **CompleteP adds depth transfer**: The residual branch scaling and LR corrections allow hyperparameters to also transfer across depths.

3. **The implementation is surgical**: Only a few targeted modifications are needed—the core transformer architecture remains unchanged.

4. **Scaling factors are parameterized**: `mup_width_multiplier` and `depth_multiplier` allow flexible scaling relative to a base model size.

## muP Summary from Eleuther repo

| Parameterization | SP | **μP** | Code |
|------------------|----|----|----|
| Embedding Init. Var. | $σ_{base}^2$ | $σ_{base}^2$ |    |
| Embedding LR | $η_{base}$ | $η_{base}$ |    |
| Embedding Fwd. | $x W_{\text{emb}}$ | $\mathbf{α_{input}} · x W_{\text{emb}}$ |  [Code](https://github.com/EleutherAI/nanoGPT-mup/blob/bcadbc3c7a44138525eca8a799764afba7dca2b3/model.py#L208)  |
| Hidden Init. Var. | $σ_{base}^2$ | $σ_{base}^2 / \mathbf{m_d}$ |  [Code](https://github.com/EleutherAI/nanoGPT-mup/blob/bcadbc3c7a44138525eca8a799764afba7dca2b3/model.py#L163-L169)  |
| Hidden LR (Adam) | $η_{base}$ | $η_{base} / \mathbf{m_d}$ |  [Code](https://github.com/EleutherAI/nanoGPT-mup/blob/bcadbc3c7a44138525eca8a799764afba7dca2b3/model.py#L306-L329)  |
| Output Logit Fwd. | $x W_{\text{emb}}^\top$ | $\mathbf{α_{output}} · x W_{\text{emb}}^\top / \mathbf{m_d}$ |  [Code](https://github.com/EleutherAI/nanoGPT-mup/blob/bcadbc3c7a44138525eca8a799764afba7dca2b3/model.py#L219)  |
| Attention logits | $Q^\top K / \sqrt{d_{\text{head}}}$ | $Q^\top K / \mathbf{d_{\text{head}}}$ |  [Code](https://github.com/EleutherAI/nanoGPT-mup/blob/bcadbc3c7a44138525eca8a799764afba7dca2b3/model.py#L65)  |
