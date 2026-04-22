"""Configuration for 30B model."""

from flops_config_base import Config


class Config30B(Config):
    """30B MoE model configuration."""

    def __init__(self):
        super().__init__(
            num_layers=48,
            hidden_size=2048,
            ffn_hidden_size=0,
            moe_ffn_hidden_size=768,
            num_experts_routed_to=8,
            num_moe_layers=48,
            num_dense_layers=0,
            mtp_num_layers=0,
            padded_vocab_size=151936,
            shared_expert_ffn_hidden_size=0,
            num_query_groups=4,
            num_attention_heads=32,
            kv_channels=128,
            attention_output_gate=False,
            swiglu=True,
            num_experts=256,
        )
