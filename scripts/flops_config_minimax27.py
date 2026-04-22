"""Configuration for Minimax 2.7B model."""

from flops_config_base import Config


class ConfigMinimax27(Config):
    """Minimax 2.7B MoE model configuration."""

    def __init__(self):
        super().__init__(
            num_layers=62,
            hidden_size=3072,
            ffn_hidden_size=0,
            moe_ffn_hidden_size=1536,
            num_experts_routed_to=8,
            num_moe_layers=62,
            num_dense_layers=0,
            mtp_num_layers=0,
            padded_vocab_size=200064,
            shared_expert_ffn_hidden_size=0,
            num_query_groups=8,
            num_attention_heads=48,
            kv_channels=128,
            attention_output_gate=False,
            swiglu=True,
            num_experts=None,
        )
