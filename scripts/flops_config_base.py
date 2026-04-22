"""Base configuration class for FLOPs calculation."""


class Config:
    """Base configuration object to mimic args from training.py"""

    def __init__(
        self,
        # Model architecture parameters
        num_layers=28,
        hidden_size=2048,
        ffn_hidden_size=0,
        moe_ffn_hidden_size=0,
        num_experts_routed_to=0,
        num_moe_layers=0,
        num_dense_layers=0,
        mtp_num_layers=0,
        padded_vocab_size=151936,
        shared_expert_ffn_hidden_size=0,
        num_query_groups=8,
        num_attention_heads=16,
        kv_channels=128,
        attention_output_gate=False,
        swiglu=True,
        # Advanced features
        multi_latent_attention=False,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_head_dim=None,
        qk_pos_emb_head_dim=None,
        v_head_dim=None,
        experimental_attention_variant=None,
        linear_attention_freq=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        linear_num_key_heads=None,
        linear_num_value_heads=None,
        linear_conv_kernel_dim=None,
        is_hybrid_model=False,
        hybrid_override_pattern=None,
        hybrid_attention_ratio=0.0,
        hybrid_mlp_ratio=0.0,
        mamba_state_dim=None,
        mamba_head_dim=None,
        mamba_num_groups=None,
        mamba_num_heads=None,
        moe_latent_size=None,
        num_experts=None,
    ):
        """Initialize configuration with default or provided values."""
        # Runtime parameters (set externally)
        self.batch_size = None
        self.seq_length = None

        # Basic architecture config
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.num_experts_routed_to = num_experts_routed_to
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.kv_channels = kv_channels
        self.padded_vocab_size = padded_vocab_size
        self.swiglu = swiglu
        self.attention_output_gate = attention_output_gate

        # Derived values
        self.query_projection_size = kv_channels * num_attention_heads
        self.query_projection_to_hidden_size_ratio = self.query_projection_size / hidden_size

        # Layer counts
        self.num_moe_layers = num_moe_layers
        self.num_dense_layers = num_dense_layers
        self.mtp_num_layers = mtp_num_layers if mtp_num_layers > 0 else None

        # MoE configuration
        self.num_experts = num_experts
        self.moe_router_topk = num_experts_routed_to
        self.moe_shared_expert_intermediate_size = (
            shared_expert_ffn_hidden_size if shared_expert_ffn_hidden_size > 0 else None
        )

        # MoE layer pattern - can be int or list
        if num_moe_layers > 0:
            if num_moe_layers == num_layers:
                self.moe_layer_freq = 1
            elif num_dense_layers > 0:
                # Create pattern: 1 for MoE layer, 0 for dense
                self.moe_layer_freq = [1] * num_moe_layers + [0] * num_dense_layers
            else:
                self.moe_layer_freq = 1
        else:
            self.moe_layer_freq = None

        # Group Query Attention
        self.group_query_attention = num_query_groups < num_attention_heads

        # Multi-latent attention (MLA) - DeepSeek style
        self.multi_latent_attention = multi_latent_attention
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.v_head_dim = v_head_dim

        # Linear attention variants
        self.experimental_attention_variant = experimental_attention_variant
        self.linear_attention_freq = linear_attention_freq
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        # Hybrid model (Mamba + Attention)
        self.is_hybrid_model = is_hybrid_model
        self.hybrid_override_pattern = hybrid_override_pattern
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.mamba_state_dim = mamba_state_dim
        self.mamba_head_dim = mamba_head_dim
        self.mamba_num_groups = mamba_num_groups
        self.mamba_num_heads = mamba_num_heads
        self.moe_latent_size = moe_latent_size

    def __repr__(self):
        """Return string representation of config."""
        return (
            f"Config(model_size={self.num_layers}L-{self.hidden_size}H, "
            f"batch={self.batch_size}, seq_len={self.seq_length})"
        )
