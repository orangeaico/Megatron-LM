"""Configuration for ZAI GLM-5.2 model."""

from flops_config_base import Config


class ConfigGLM52(Config):
    """ZAI GLM-5.2 (glm_moe_dsa) MoE model configuration.

    Based on: https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json

    This model uses Multi-Latent Attention (MLA), DeepSeek style:
    - 78 layers (first 3 dense, remaining 75 MoE; first_k_dense_replace=3)
    - 256 routed experts + 1 shared expert, top-8 routing
    - 1 Multi-Token Prediction (MTP) layer (num_nextn_predict_layers=1)
    - 1M context window (max_position_embeddings=1048576)

    DeepSeek Sparse Attention (DSA): the main MLA attention attends to only the
    top-k (index_topk=2048) selected keys per query on every layer. The selection
    is produced by a lightning indexer that scores all preceding tokens; the
    indexer is computed on the 21 "full" layers (indexer_types: layers 0,1,2 then
    every 4th via index_topk_freq=4) and shared by the remaining 57 layers.
    """

    def __init__(self):
        first_k_dense_replace = 3
        num_layers = 78
        num_dense_layers = first_k_dense_replace
        num_moe_layers = num_layers - num_dense_layers

        # Number of layers that compute the lightning indexer ("full" entries in
        # `indexer_types`): the first 3 layers plus every 4th layer thereafter.
        num_indexer_layers = first_k_dense_replace + len(
            range(first_k_dense_replace * 2, num_layers, 4)
        )  # == 21

        super().__init__(
            num_layers=num_layers,
            hidden_size=6144,
            ffn_hidden_size=12288,  # intermediate_size for dense layers
            moe_ffn_hidden_size=2048,  # moe_intermediate_size
            num_experts_routed_to=8,  # num_experts_per_tok
            num_moe_layers=num_moe_layers,
            num_dense_layers=num_dense_layers,
            mtp_num_layers=1,  # num_nextn_predict_layers
            padded_vocab_size=154880,  # vocab_size
            shared_expert_ffn_hidden_size=2048,  # n_shared_experts * moe_intermediate_size
            num_query_groups=64,  # num_key_value_heads
            num_attention_heads=64,
            kv_channels=192,  # head_dim (unused on the MLA path)
            attention_output_gate=False,
            swiglu=True,  # hidden_act="silu" indicates SwiGLU
            num_experts=256,  # n_routed_experts
            # Multi-latent attention (MLA) - DeepSeek style
            multi_latent_attention=True,
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_head_dim=192,  # qk_nope_head_dim
            qk_pos_emb_head_dim=64,  # qk_rope_head_dim
            v_head_dim=256,
            # DeepSeek Sparse Attention (DSA) / lightning indexer
            deepseek_sparse_attention=True,
            dsa_index_topk=2048,  # index_topk
            dsa_index_n_heads=32,  # index_n_heads
            dsa_index_head_dim=128,  # index_head_dim
            dsa_num_indexer_layers=num_indexer_layers,  # "full" indexer_types entries
        )

        # Dense layers are the FIRST `first_k_dense_replace` layers, so the last
        # layer (and thus the MTP layer) is MoE. Override the base-class pattern,
        # which places dense layers last, so `last_layer_is_moe` is computed as 1.
        self.moe_layer_freq = [0] * num_dense_layers + [1] * num_moe_layers
