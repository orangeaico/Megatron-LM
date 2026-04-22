"""Configuration for 1.7B model."""

from flops_config_base import Config


class Config17B(Config):
    """1.7B dense model configuration."""

    def __init__(self):
        super().__init__(
            num_layers=28,
            hidden_size=2048,
            ffn_hidden_size=6144,
            moe_ffn_hidden_size=0,
            num_experts_routed_to=0,
            num_moe_layers=0,
            num_dense_layers=28,
            mtp_num_layers=0,
            padded_vocab_size=151936,
            shared_expert_ffn_hidden_size=0,
            num_query_groups=8,
            num_attention_heads=16,
            kv_channels=128,
            attention_output_gate=False,
            swiglu=True,
            num_experts=None,
        )
