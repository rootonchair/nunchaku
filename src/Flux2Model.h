#pragma once

#include "FluxModel.h"

// FLUX.2 differs from FLUX.1:
// - Modulation is computed externally (Python) and passed as tensors, not computed per-block
// - MLP uses SwiGLU (silu_and_mul) instead of GEGLU
// - Block counts and dimensions are configurable (not hard-coded)
// - Gated residual pattern: residual += gate * update

class Flux2JointTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM                     = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;

    Flux2JointTransformerBlock(int dim,
                               int num_attention_heads,
                               int attention_head_dim,
                               int mlp_ratio,
                               bool use_fp4,
                               Tensor::ScalarType dtype,
                               Device device);

    // mod_img, mod_txt: pre-computed modulation tensors [B, 6*dim] each
    std::tuple<Tensor, Tensor> forward(Tensor hidden_states,
                                       Tensor encoder_hidden_states,
                                       Tensor mod_img,
                                       Tensor mod_txt,
                                       Tensor rotary_emb_img,
                                       Tensor rotary_emb_txt);

public:
    const int dim;
    const int dim_head;
    const int num_heads;
    const int mlp_hidden_dim;

    AttentionImpl attnImpl = AttentionImpl::FlashAttention2;
    std::function<Tensor(Tensor)> custom_attn_func;

private:
    LayerNorm norm1, norm1_context, norm2, norm2_context;
    GEMM qkv_proj, qkv_proj_context;
    RMSNorm norm_q, norm_k, norm_added_q, norm_added_k;
    Attention attn;
    GEMM out_proj, out_proj_context;
    GEMM mlp_fc1, mlp_fc2;
    GEMM mlp_context_fc1, mlp_context_fc2;
};

class Flux2SingleTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM                     = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;

    Flux2SingleTransformerBlock(int dim,
                                int num_attention_heads,
                                int attention_head_dim,
                                int mlp_ratio,
                                bool use_fp4,
                                Tensor::ScalarType dtype,
                                Device device);

    // mod: pre-computed modulation tensor [B, 3*dim]
    Tensor forward(Tensor hidden_states, Tensor mod, Tensor rotary_emb);

public:
    const int dim;
    const int dim_head;
    const int num_heads;
    const int mlp_hidden_dim;

    AttentionImpl attnImpl = AttentionImpl::FlashAttention2;
    std::function<Tensor(Tensor)> custom_attn_func;

private:
    LayerNorm norm;
    GEMM qkv_proj;
    GEMM mlp_fc1, mlp_fc2;
    RMSNorm norm_q, norm_k;
    Attention attn;
    GEMM out_proj;
};

class Flux2Model : public Module {
public:
    Flux2Model(int num_layers,
               int num_single_layers,
               int dim,
               int num_attention_heads,
               int attention_head_dim,
               int mlp_ratio,
               bool use_fp4,
               bool offload,
               Tensor::ScalarType dtype,
               Device device);

    Tensor forward(Tensor hidden_states,
                   Tensor encoder_hidden_states,
                   Tensor mod_img,
                   Tensor mod_txt,
                   Tensor mod_single,
                   Tensor rotary_emb_img,
                   Tensor rotary_emb_txt,
                   Tensor rotary_emb_single);

    void setAttentionImpl(AttentionImpl impl, std::function<Tensor(Tensor)>);

public:
    const int dim;
    const Tensor::ScalarType dtype;

    std::vector<std::unique_ptr<Flux2JointTransformerBlock>> transformer_blocks;
    std::vector<std::unique_ptr<Flux2SingleTransformerBlock>> single_transformer_blocks;

    bool isOffloadEnabled() const {
        return offload;
    }

private:
    bool offload;
};
