"""Configuration for Qwen3.6-27B-FP8 model."""

from flops_config_base import Config


class ConfigQwen27B(Config):
    """Qwen3.6-27B-FP8 dense model configuration.

    Based on: https://huggingface.co/Qwen/Qwen3.6-27B-FP8/blob/main/config.json

    This is a dense (non-MoE) model with linear attention:
    - 48 linear attention layers (gated_delta_net)
    - 16 standard attention layers (every 4th layer)
    - Total: 64 layers
    - No MoE
    """

    def __init__(self):
        super().__init__(
            num_layers=64,
            hidden_size=5120,
            ffn_hidden_size=17408,  # intermediate_size for dense layers
            moe_ffn_hidden_size=0,  # Not an MoE model
            num_experts_routed_to=0,
            num_moe_layers=0,  # Dense model, no MoE
            num_dense_layers=64,  # All layers are dense
            mtp_num_layers=0,
            padded_vocab_size=248320,
            shared_expert_ffn_hidden_size=0,
            num_query_groups=4,  # num_key_value_heads
            num_attention_heads=24,
            kv_channels=256,  # head_dim
            attention_output_gate=True,
            swiglu=True,  # hidden_act="silu"
            num_experts=None,  # Not MoE
            # Linear attention (gated_delta_net) configuration
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=4,  # Every 4th layer is standard attention (16 standard, 48 linear)
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_conv_kernel_dim=4,
        )
