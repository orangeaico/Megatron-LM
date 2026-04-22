"""Configuration for Moonshot Kimi-K2.6 model."""

from flops_config_base import Config


class ConfigKimi26(Config):
    """Moonshot Kimi-K2.6 MoE model configuration.

    Based on: https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json

    This model uses Multi-Latent Attention (MLA) similar to DeepSeek:
    - 61 layers with MLA
    - 384 routed experts + 1 shared expert
    - 8 experts per token
    - 262K context window
    """

    def __init__(self):
        super().__init__(
            num_layers=61,
            hidden_size=7168,
            ffn_hidden_size=18432,  # intermediate_size for dense layers
            moe_ffn_hidden_size=2048,  # moe_intermediate_size
            num_experts_routed_to=8,
            num_moe_layers=61,  # All layers are MoE (moe_layer_freq=1)
            num_dense_layers=0,
            mtp_num_layers=0,
            padded_vocab_size=163840,
            shared_expert_ffn_hidden_size=2048,  # Assuming same as moe_intermediate_size
            num_query_groups=64,  # Same as num_key_value_heads
            num_attention_heads=64,
            kv_channels=128,  # Derived from v_head_dim
            attention_output_gate=False,
            swiglu=True,  # hidden_act="silu" indicates SwiGLU
            num_experts=384,
            # Multi-latent attention (MLA) - DeepSeek style
            multi_latent_attention=True,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_head_dim=128,  # qk_nope_head_dim
            qk_pos_emb_head_dim=64,  # qk_rope_head_dim
            v_head_dim=128,
        )
