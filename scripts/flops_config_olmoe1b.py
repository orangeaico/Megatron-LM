"""Configuration for AllenAI OLMoE-1B-7B-0924 model."""

from flops_config_base import Config


class ConfigOLMoE1B(Config):
    """AllenAI OLMoE-1B-7B-0924 MoE model configuration.

    Based on: https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json

    This is a smaller MoE model:
    - 16 layers
    - 64 experts with 8 experts per token
    - 1B total parameters, 7B activated
    - 4K context window
    """

    def __init__(self):
        super().__init__(
            num_layers=16,
            hidden_size=2048,
            ffn_hidden_size=0,
            moe_ffn_hidden_size=1024,  # intermediate_size
            num_experts_routed_to=8,
            num_moe_layers=16,  # All layers are MoE
            num_dense_layers=0,
            mtp_num_layers=0,
            padded_vocab_size=50304,
            shared_expert_ffn_hidden_size=0,  # No shared experts
            num_query_groups=16,  # Same as num_key_value_heads
            num_attention_heads=16,
            kv_channels=128,  # Derived from hidden_size / num_attention_heads
            attention_output_gate=False,
            swiglu=True,  # hidden_act="silu" indicates SwiGLU
            num_experts=64,
        )
