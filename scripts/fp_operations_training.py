MODEL = "1.7B"

if MODEL == "30B":
    from flops_config_30b import *
elif MODEL == "1.7B":
    from flops_config_17b import *
elif MODEL == "Minimax2.7":
    from flops_config_minimax27 import *


num_standard_attention_layers = num_layers
num_linear_attention_layers = 0

# - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
#       backward wgrad [weight gradient], backward dgrad [data gradient]).
forward_backward_expansion_factor = 3
# - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
fma_expansion_factor = 2
# - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
# - 2x (SwiGLU disabled): h->ffn_h GEMM and ffn_h->h GEMM are stacked.
ffn_expansion_factor = 3 if swiglu else 2


# Training FLOPs Computation 
query_projection_size = kv_channels * num_attention_heads
key_projection_size = kv_channels * num_query_groups
value_projection_size = kv_channels * num_query_groups
gate_projection_size = query_projection_size if attention_output_gate else 0
standard_self_attn_term = (
    forward_backward_expansion_factor
    * fma_expansion_factor
    * (
        ## qkv proj
        hidden_size
        * (
            query_projection_size
            + key_projection_size
            + value_projection_size
            + gate_projection_size
        )
    ## core attention
    + query_projection_size
    * seq_length
    / 2  # causal mask (only half of the mask is non-zero)
    * 2  # QK^T and (QK^T)V
    ## out proj
    + query_projection_size
    * hidden_size
)
)

linear_self_attn_term = 0
num_linear_attention_layers = 0
self_attn_term = (
            linear_self_attn_term * num_linear_attention_layers
            + standard_self_attn_term * num_standard_attention_layers
        )

print (f"Self attention term: {self_attn_term/float(1024**4)}")

total_floating_point_operations = (
    batch_size
    * seq_length
    * (
        # MLP
        forward_backward_expansion_factor
        * fma_expansion_factor
        * hidden_size
        * (
            # dense layer (deepseek v2, v3 style)
            (ffn_hidden_size * ffn_expansion_factor)
            * num_dense_layers
            # routed experts
            + (moe_ffn_hidden_size * num_experts_routed_to * ffn_expansion_factor)
            * num_moe_layers
            # Shared Experts.
            + (shared_expert_ffn_hidden_size * ffn_expansion_factor)
            * num_moe_layers
        )
        # Self Attention
        + self_attn_term
        # MTP norms and proj
        + forward_backward_expansion_factor
        * fma_expansion_factor
        * mtp_num_layers
        * (
            # MTP eh norm + final nrom
            3 * hidden_size
            # MTH eh proj
            + 2 * hidden_size * hidden_size
        )
        # Logit.
        + forward_backward_expansion_factor
        * fma_expansion_factor
        * hidden_size
        * padded_vocab_size
        * (mtp_num_layers + 1)  # MTP + final logit
    )
)

print (f"Total floating point operations: {total_floating_point_operations/float(1024**4)}")