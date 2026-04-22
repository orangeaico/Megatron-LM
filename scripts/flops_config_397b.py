"""Configuration for Qwen3.5-397B-A17B model."""

from flops_config_base import Config


class Config397B(Config):
    """Qwen3.5-397B-A17B MoE model configuration.

    Based on: https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json

    This is a hybrid model with:
    - 45 linear attention layers (gated_delta_net)
    - 15 standard attention layers (every 4th layer)
    - Total: 60 layers + 1 MTP layer
    - 512 experts with 10 experts per token
    """

    def __init__(self):
        super().__init__(
            num_layers=60,
            hidden_size=4096,
            ffn_hidden_size=0,
            moe_ffn_hidden_size=1024,
            num_experts_routed_to=10,
            num_moe_layers=60,
            num_dense_layers=0,
            mtp_num_layers=1,
            padded_vocab_size=248320,
            shared_expert_ffn_hidden_size=1024,
            num_query_groups=2,
            num_attention_heads=32,
            kv_channels=256,
            attention_output_gate=True,
            swiglu=True,
            num_experts=512,
            # Linear attention (gated_delta_net) configuration
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=4,  # Every 4th layer is standard attention (15 standard, 45 linear)
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=64,
            linear_conv_kernel_dim=4,
        )
