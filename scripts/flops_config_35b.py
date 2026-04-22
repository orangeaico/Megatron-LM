"""Configuration for Qwen3.6-35B-A3B model."""

from flops_config_base import Config


class Config35B(Config):
    """Qwen3.6-35B-A3B MoE model configuration.

    Based on: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/main/config.json

    This is a hybrid model with:
    - 30 linear attention layers (gated_delta_net)
    - 10 standard attention layers
    - Total: 40 layers + 1 MTP layer
    """

    def __init__(self):
        super().__init__(
            num_layers=40,
            hidden_size=2048,
            ffn_hidden_size=0,
            moe_ffn_hidden_size=512,
            num_experts_routed_to=8,
            num_moe_layers=40,
            num_dense_layers=0,
            mtp_num_layers=1,
            padded_vocab_size=248320,
            shared_expert_ffn_hidden_size=512,
            num_query_groups=2,
            num_attention_heads=16,
            kv_channels=256,
            attention_output_gate=True,
            swiglu=True,
            num_experts=256,
            # Linear attention (gated_delta_net) configuration
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=4,  # Every 4th layer is standard attention (10 standard, 30 linear)
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_conv_kernel_dim=4,
        )
