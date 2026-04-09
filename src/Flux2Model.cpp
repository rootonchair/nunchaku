#include "Flux2Model.h"
#include "kernels/misc_kernels.h"
#include "kernels/activation_kernels.h"
#include "kernels/zgemm/zgemm.h"
#include "flash_api.h"
#include "activation.h"
#include <nvtx3/nvToolsExt.h>

using spdlog::fmt_lib::format;
using namespace nunchaku;

// SwiGLU MLP: fc1 outputs 2*hidden_dim, split and gate via silu_and_mul, then fc2
static Tensor forward_mlp_swiglu(GEMM_W4A4 &fc1, GEMM_W4A4 &fc2, Tensor x) {
    Tensor fc1_out = std::get<Tensor>(fc1.forward(x, GEMM_W4A4::FuseOptions::EMPTY, nullptr));
    // fc1_out: [B, S, 2*mlp_hidden_dim]
    int d          = fc1_out.shape[-1] / 2;
    Tensor swiglu  = Tensor::allocate({fc1_out.shape[0], fc1_out.shape[1], d}, fc1_out.scalar_type(), fc1_out.device());
    silu_and_mul(swiglu, fc1_out);
    return std::get<Tensor>(fc2.forward(swiglu, GEMM_W4A4::FuseOptions::EMPTY, nullptr));
}

static Tensor forward_fc(GEMM_W4A4 &fc, Tensor x) {
    return fc.forward(x);
}

// ============================================================================
// Flux2JointTransformerBlock
// ============================================================================

Flux2JointTransformerBlock::Flux2JointTransformerBlock(int dim,
                                                       int num_attention_heads,
                                                       int attention_head_dim,
                                                       int mlp_ratio,
                                                       bool use_fp4,
                                                       Tensor::ScalarType dtype,
                                                       Device device)
    : dim(dim), dim_head(attention_head_dim / num_attention_heads), num_heads(num_attention_heads),
      mlp_hidden_dim(dim * mlp_ratio),
      // LayerNorms (no affine shift, eps=1e-6)
      norm1(dim, 1e-6, false, dtype, device), norm1_context(dim, 1e-6, false, dtype, device),
      norm2(dim, 1e-6, false, dtype, device), norm2_context(dim, 1e-6, false, dtype, device),
      // QKV projections
      qkv_proj(dim, dim * 3, true, use_fp4, dtype, device),
      qkv_proj_context(dim, dim * 3, true, use_fp4, dtype, device),
      // RMSNorm for Q/K
      norm_q(dim_head, 1e-6, false, dtype, device), norm_k(dim_head, 1e-6, false, dtype, device),
      norm_added_q(dim_head, 1e-6, false, dtype, device), norm_added_k(dim_head, 1e-6, false, dtype, device),
      // Attention
      attn(num_attention_heads, attention_head_dim / num_attention_heads, device),
      // Output projections
      out_proj(dim, dim, true, use_fp4, dtype, device), out_proj_context(dim, dim, true, use_fp4, dtype, device),
      // SwiGLU MLP: fc1 outputs 2*mlp_hidden_dim for gating
      mlp_fc1(dim, mlp_hidden_dim * 2, true, use_fp4, dtype, device),
      mlp_fc2(mlp_hidden_dim, dim, true, use_fp4, dtype, device),
      mlp_context_fc1(dim, mlp_hidden_dim * 2, true, use_fp4, dtype, device),
      mlp_context_fc2(mlp_hidden_dim, dim, true, use_fp4, dtype, device) {
    registerChildren(norm1, "norm1")(norm1_context, "norm1_context")(norm2, "norm2")(norm2_context, "norm2_context")(
        qkv_proj, "qkv_proj")(qkv_proj_context, "qkv_proj_context")(norm_q, "norm_q")(norm_k, "norm_k")(
        norm_added_q, "norm_added_q")(norm_added_k, "norm_added_k")(attn, "attn")(out_proj, "out_proj")(
        out_proj_context, "out_proj_context")(mlp_fc1, "mlp_fc1")(mlp_fc2, "mlp_fc2")(mlp_context_fc1,
                                                                                       "mlp_context_fc1")(
        mlp_context_fc2, "mlp_context_fc2");
}

std::tuple<Tensor, Tensor> Flux2JointTransformerBlock::forward(Tensor hidden_states,
                                                               Tensor encoder_hidden_states,
                                                               Tensor mod_img,
                                                               Tensor mod_txt,
                                                               Tensor rotary_emb_img,
                                                               Tensor rotary_emb_txt) {
    nvtxRangePushA("Flux2JointTransformerBlock");

    const int batch_size     = hidden_states.shape[0];
    const int num_tokens_img = hidden_states.shape[1];
    const int num_tokens_txt = encoder_hidden_states.shape[1];

    // Split modulation: each mod is [B, 6*dim] → 2 sets of (shift, scale, gate)
    auto &&[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = kernels::split_mod<6>(mod_img);
    auto &&[c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp] =
        kernels::split_mod<6>(mod_txt);

    // Norm + modulate image stream: (1 + scale) * norm + shift
    Tensor norm_hidden_states = norm1.forward(hidden_states);
    kernels::mul_add_batch(norm_hidden_states, scale_msa, true, 1.0, shift_msa, true);

    // Norm + modulate text stream: (1 + scale) * norm + shift
    Tensor norm_encoder_hidden_states = norm1_context.forward(encoder_hidden_states);
    kernels::mul_add_batch(norm_encoder_hidden_states, c_scale_msa, true, 1.0, c_shift_msa, true);

    // QKV projection + attention
    int num_tokens_img_pad = 0, num_tokens_txt_pad = 0;
    Tensor raw_attn_output;

    auto stream = getCurrentCUDAStream();

    if (attnImpl == AttentionImpl::NunchakuFP16) {
        num_tokens_img_pad = ceilDiv(num_tokens_img, 256) * 256;
        num_tokens_txt_pad = ceilDiv(num_tokens_txt, 256) * 256;

        Tensor concat_q = Tensor::allocate({batch_size, num_heads, num_tokens_img_pad + num_tokens_txt_pad, dim_head},
                                           Tensor::FP16,
                                           norm_hidden_states.device());
        Tensor concat_k = Tensor::empty_like(concat_q);
        Tensor concat_v = Tensor::empty_like(concat_q);

        for (int i = 0; i < batch_size; i++) {
            auto sliceImg = [&](Tensor x) { return x.slice(0, i, i + 1).slice(2, 0, num_tokens_img_pad); };
            auto sliceTxt = [&](Tensor x) {
                return x.slice(0, i, i + 1).slice(2, num_tokens_img_pad, num_tokens_img_pad + num_tokens_txt_pad);
            };

            qkv_proj.forward(norm_hidden_states.slice(0, i, i + 1),
                             {},
                             {},
                             norm_q.weight,
                             norm_k.weight,
                             rotary_emb_img,
                             sliceImg(concat_q),
                             sliceImg(concat_k),
                             sliceImg(concat_v),
                             num_tokens_img);

            qkv_proj_context.forward(norm_encoder_hidden_states.slice(0, i, i + 1),
                                     {},
                                     {},
                                     norm_added_q.weight,
                                     norm_added_k.weight,
                                     rotary_emb_txt,
                                     sliceTxt(concat_q),
                                     sliceTxt(concat_k),
                                     sliceTxt(concat_v),
                                     num_tokens_txt);
        }

        raw_attn_output = Tensor::allocate({batch_size, num_tokens_img_pad + num_tokens_txt_pad, num_heads * dim_head},
                                           norm_hidden_states.scalar_type(),
                                           norm_hidden_states.device());
        kernels::attention_fp16(concat_q, concat_k, concat_v, raw_attn_output, pow(dim_head, (-0.5)));
        raw_attn_output =
            raw_attn_output.view({batch_size, num_tokens_img_pad + num_tokens_txt_pad, num_heads, dim_head});
    } else {
        // FlashAttention2 or Custom
        num_tokens_img_pad = num_tokens_img;
        num_tokens_txt_pad = num_tokens_txt;

        Tensor concat = Tensor::allocate({batch_size, num_tokens_img + num_tokens_txt, dim * 3},
                                         norm_hidden_states.scalar_type(),
                                         norm_hidden_states.device());
        for (int i = 0; i < batch_size; i++) {
            Tensor qkv         = concat.slice(0, i, i + 1).slice(1, 0, num_tokens_img);
            Tensor qkv_context = concat.slice(0, i, i + 1).slice(1, num_tokens_img, num_tokens_img + num_tokens_txt);

            qkv_proj.forward(
                norm_hidden_states.slice(0, i, i + 1), qkv, {}, norm_q.weight, norm_k.weight, rotary_emb_img);
            qkv_proj_context.forward(norm_encoder_hidden_states.slice(0, i, i + 1),
                                     qkv_context,
                                     {},
                                     norm_added_q.weight,
                                     norm_added_k.weight,
                                     rotary_emb_txt);
        }

        if (attnImpl == AttentionImpl::Custom) {
            raw_attn_output =
                custom_attn_func(concat.view({batch_size, num_tokens_img + num_tokens_txt, 3, num_heads, dim_head}));
        } else {
            raw_attn_output = attn.forward(concat);
        }
        raw_attn_output = raw_attn_output.view({batch_size, num_tokens_img + num_tokens_txt, num_heads, dim_head});
    }

    // Split attention output → image + context
    // Image output projection + gated residual
    {
        Tensor raw_attn_img;
        if (batch_size == 1) {
            raw_attn_img =
                raw_attn_output.slice(1, 0, num_tokens_img).reshape({batch_size, num_tokens_img, num_heads * dim_head});
        } else {
            raw_attn_img = Tensor::allocate(
                {batch_size, num_tokens_img, num_heads * dim_head}, raw_attn_output.scalar_type(), raw_attn_output.device());
            checkCUDA(cudaMemcpy2DAsync(raw_attn_img.data_ptr(),
                                        num_tokens_img * num_heads * dim_head * raw_attn_img.scalar_size(),
                                        raw_attn_output.data_ptr(),
                                        (num_tokens_img_pad + num_tokens_txt_pad) * num_heads * dim_head *
                                            raw_attn_output.scalar_size(),
                                        num_tokens_img * num_heads * dim_head * raw_attn_img.scalar_size(),
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
        }

        Tensor attn_output = forward_fc(out_proj, raw_attn_img);
        // Gated residual: hidden_states = hidden_states + gate_msa * attn_output
        kernels::mul_add_batch(attn_output, gate_msa, true, 0.0, hidden_states, true);
        hidden_states = std::move(attn_output);

        // MLP: norm → modulate → SwiGLU
        Tensor norm_h = norm2.forward(hidden_states);
        kernels::mul_add_batch(norm_h, scale_mlp, true, 1.0, shift_mlp, true);
        Tensor ff_output = forward_mlp_swiglu(mlp_fc1, mlp_fc2, norm_h);
        // Gated residual: hidden_states = hidden_states + gate_mlp * ff_output
        kernels::mul_add_batch(ff_output, gate_mlp, true, 0.0, hidden_states, true);
        hidden_states = std::move(ff_output);
    }

    // Context output projection + gated residual
    {
        Tensor raw_attn_ctx;
        if (batch_size == 1) {
            raw_attn_ctx = raw_attn_output.slice(1, num_tokens_img_pad, num_tokens_img_pad + num_tokens_txt)
                               .reshape({batch_size, num_tokens_txt, num_heads * dim_head});
        } else {
            raw_attn_ctx = Tensor::allocate(
                {batch_size, num_tokens_txt, num_heads * dim_head}, raw_attn_output.scalar_type(), raw_attn_output.device());
            checkCUDA(cudaMemcpy2DAsync(
                raw_attn_ctx.data_ptr(),
                num_tokens_txt * num_heads * dim_head * raw_attn_ctx.scalar_size(),
                raw_attn_output.data_ptr<char>() +
                    num_tokens_img_pad * num_heads * dim_head * raw_attn_output.scalar_size(),
                (num_tokens_img_pad + num_tokens_txt_pad) * num_heads * dim_head * raw_attn_output.scalar_size(),
                num_tokens_txt * num_heads * dim_head * raw_attn_ctx.scalar_size(),
                batch_size,
                cudaMemcpyDeviceToDevice,
                stream));
        }

        Tensor attn_output = forward_fc(out_proj_context, raw_attn_ctx);
        kernels::mul_add_batch(attn_output, c_gate_msa, true, 0.0, encoder_hidden_states, true);
        encoder_hidden_states = std::move(attn_output);

        Tensor norm_h = norm2_context.forward(encoder_hidden_states);
        kernels::mul_add_batch(norm_h, c_scale_mlp, true, 1.0, c_shift_mlp, true);
        Tensor ff_output = forward_mlp_swiglu(mlp_context_fc1, mlp_context_fc2, norm_h);
        kernels::mul_add_batch(ff_output, c_gate_mlp, true, 0.0, encoder_hidden_states, true);
        encoder_hidden_states = std::move(ff_output);

        // Note: FP16 clamping for encoder_hidden_states is handled in Python
        // when needed (BF16 has sufficient range and is the default dtype)
    }

    nvtxRangePop();
    return {hidden_states, encoder_hidden_states};
}

// ============================================================================
// Flux2SingleTransformerBlock
// ============================================================================

Flux2SingleTransformerBlock::Flux2SingleTransformerBlock(int dim,
                                                         int num_attention_heads,
                                                         int attention_head_dim,
                                                         int mlp_ratio,
                                                         bool use_fp4,
                                                         Tensor::ScalarType dtype,
                                                         Device device)
    : dim(dim), dim_head(attention_head_dim / num_attention_heads), num_heads(num_attention_heads),
      mlp_hidden_dim(dim * mlp_ratio),
      norm(dim, 1e-6, false, dtype, device),
      qkv_proj(dim, dim * 3, true, use_fp4, dtype, device),
      // SwiGLU MLP: fc1 outputs 2*mlp_hidden_dim for gating
      mlp_fc1(dim, mlp_hidden_dim * 2, true, use_fp4, dtype, device),
      mlp_fc2(mlp_hidden_dim, dim, true, use_fp4, dtype, device),
      norm_q(dim_head, 1e-6, false, dtype, device), norm_k(dim_head, 1e-6, false, dtype, device),
      attn(num_attention_heads, attention_head_dim / num_attention_heads, device),
      out_proj(dim, dim, true, use_fp4, dtype, device) {
    registerChildren(norm, "norm")(qkv_proj, "qkv_proj")(mlp_fc1, "mlp_fc1")(mlp_fc2, "mlp_fc2")(norm_q, "norm_q")(
        norm_k, "norm_k")(attn, "attn")(out_proj, "out_proj");
}

Tensor Flux2SingleTransformerBlock::forward(Tensor hidden_states, Tensor mod, Tensor rotary_emb) {
    nvtxRangePushA("Flux2SingleTransformerBlock");

    const int batch_size = hidden_states.shape[0];
    const int num_tokens = hidden_states.shape[1];

    Tensor residual = hidden_states;

    // Modulation: split mod [B, 3*dim] → (shift, scale, gate)
    auto &&[shift, scale, gate] = kernels::split_mod<3>(mod);

    // Norm + modulate: (1 + scale) * norm + shift
    Tensor norm_hidden_states = norm.forward(hidden_states);
    kernels::mul_add_batch(norm_hidden_states, scale, true, 1.0, shift, true);

    // Attention
    Tensor attn_output;

    if (attnImpl == AttentionImpl::NunchakuFP16) {
        const int num_tokens_pad = ceilDiv(num_tokens, 256) * 256;

        Tensor q = Tensor::allocate(
            {batch_size, num_heads, num_tokens_pad, dim_head}, Tensor::FP16, norm_hidden_states.device());
        Tensor k = Tensor::empty_like(q);
        Tensor v = Tensor::empty_like(q);

        for (int i = 0; i < batch_size; i++) {
            qkv_proj.forward(norm_hidden_states.slice(0, i, i + 1),
                             {},
                             {},
                             norm_q.weight,
                             norm_k.weight,
                             rotary_emb,
                             q.slice(0, i, i + 1),
                             k.slice(0, i, i + 1),
                             v.slice(0, i, i + 1),
                             num_tokens);
        }

        Tensor o = Tensor::allocate(
            {batch_size, num_tokens_pad, num_heads * dim_head}, norm_hidden_states.scalar_type(), norm_hidden_states.device());
        kernels::attention_fp16(q, k, v, o, pow(dim_head, (-0.5)));

        if (batch_size == 1 || num_tokens_pad == num_tokens) {
            attn_output = o.slice(1, 0, num_tokens);
        } else {
            attn_output =
                Tensor::allocate({batch_size, num_tokens, num_heads * dim_head}, o.scalar_type(), o.device());
            checkCUDA(cudaMemcpy2DAsync(attn_output.data_ptr(),
                                        attn_output.stride(0) * attn_output.scalar_size(),
                                        o.data_ptr(),
                                        o.stride(0) * o.scalar_size(),
                                        attn_output.stride(0) * attn_output.scalar_size(),
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        getCurrentCUDAStream()));
        }
    } else {
        Tensor qkv = Tensor::allocate(
            {batch_size, num_tokens, dim * 3}, norm_hidden_states.scalar_type(), norm_hidden_states.device());
        for (int i = 0; i < batch_size; i++) {
            qkv_proj.forward(norm_hidden_states.slice(0, i, i + 1),
                             qkv.slice(0, i, i + 1),
                             {},
                             norm_q.weight,
                             norm_k.weight,
                             rotary_emb);
        }

        if (attnImpl == AttentionImpl::Custom) {
            attn_output = custom_attn_func(qkv.view({batch_size, num_tokens, 3, num_heads, dim / num_heads}));
        } else {
            attn_output = attn.forward(qkv);
        }
        attn_output = attn_output.reshape({batch_size, num_tokens, num_heads * dim_head});
    }

    // Output projection
    attn_output = forward_fc(out_proj, attn_output);

    // SwiGLU MLP (parallel with attention, applied to original norm_hidden_states)
    Tensor ff_output = forward_mlp_swiglu(mlp_fc1, mlp_fc2, norm_hidden_states);

    // Combine: attn_output + ff_output
    hidden_states = kernels::add(attn_output, ff_output);

    // Gated residual: result = residual + gate * (attn + mlp)
    kernels::mul_add_batch(hidden_states, gate, true, 0.0, residual, true);

    // Note: FP16 clamping for hidden_states is handled in Python
    // when needed (BF16 has sufficient range and is the default dtype)

    nvtxRangePop();
    return hidden_states;
}

// ============================================================================
// Flux2Model
// ============================================================================

Flux2Model::Flux2Model(int num_layers,
                       int num_single_layers,
                       int dim,
                       int num_attention_heads,
                       int attention_head_dim,
                       int mlp_ratio,
                       bool use_fp4,
                       bool offload,
                       Tensor::ScalarType dtype,
                       Device device)
    : dim(dim), dtype(dtype), offload(offload) {
    CUDADeviceContext model_construction_ctx(device.idx);

    for (int i = 0; i < num_layers; i++) {
        transformer_blocks.push_back(std::make_unique<Flux2JointTransformerBlock>(
            dim, num_attention_heads, attention_head_dim, mlp_ratio, use_fp4, dtype, device));
        registerChildren(*transformer_blocks.back(), format("transformer_blocks.{}", i));
        if (offload && i > 0) {
            transformer_blocks.back()->setLazyLoad(true);
            transformer_blocks.back()->releaseLazyParams();
        }
    }
    for (int i = 0; i < num_single_layers; i++) {
        single_transformer_blocks.push_back(std::make_unique<Flux2SingleTransformerBlock>(
            dim, num_attention_heads, attention_head_dim, mlp_ratio, use_fp4, dtype, device));
        registerChildren(*single_transformer_blocks.back(), format("single_transformer_blocks.{}", i));
        if (offload) {
            single_transformer_blocks.back()->setLazyLoad(true);
            single_transformer_blocks.back()->releaseLazyParams();
        }
    }
}

Tensor Flux2Model::forward(Tensor hidden_states,
                           Tensor encoder_hidden_states,
                           Tensor mod_img,
                           Tensor mod_txt,
                           Tensor mod_single,
                           Tensor rotary_emb_img,
                           Tensor rotary_emb_txt,
                           Tensor rotary_emb_single) {
    const int batch_size = hidden_states.shape[0];
    const int txt_tokens = encoder_hidden_states.shape[1];
    const int img_tokens = hidden_states.shape[1];

    const int numLayers = transformer_blocks.size() + single_transformer_blocks.size();

    Tensor concat;

    auto compute = [&](int layer) {
        if (size_t(layer) < transformer_blocks.size()) {
            auto &block = transformer_blocks.at(layer);
            std::tie(hidden_states, encoder_hidden_states) =
                block->forward(hidden_states, encoder_hidden_states, mod_img, mod_txt, rotary_emb_img, rotary_emb_txt);
        } else {
            if (size_t(layer) == transformer_blocks.size()) {
                // Concatenate: [encoder (txt), hidden (img)] — txt first
                concat = Tensor::allocate({batch_size, txt_tokens + img_tokens, dim}, dtype, hidden_states.device());
                for (int i = 0; i < batch_size; i++) {
                    concat.slice(0, i, i + 1).slice(1, 0, txt_tokens).copy_(encoder_hidden_states.slice(0, i, i + 1));
                    concat.slice(0, i, i + 1)
                        .slice(1, txt_tokens, txt_tokens + img_tokens)
                        .copy_(hidden_states.slice(0, i, i + 1));
                }
                hidden_states         = concat;
                encoder_hidden_states = {};
            }

            auto &block   = single_transformer_blocks.at(layer - transformer_blocks.size());
            hidden_states = block->forward(hidden_states, mod_single, rotary_emb_single);
        }
    };

    auto load = [&](int layer) {
        if (size_t(layer) < transformer_blocks.size()) {
            transformer_blocks.at(layer)->loadLazyParams();
        } else {
            single_transformer_blocks.at(layer - transformer_blocks.size())->loadLazyParams();
        }
    };

    auto unload = [&](int layer) {
        if (size_t(layer) < transformer_blocks.size()) {
            transformer_blocks.at(layer)->releaseLazyParams();
        } else {
            single_transformer_blocks.at(layer - transformer_blocks.size())->releaseLazyParams();
        }
    };

    LayerOffloadHelper helper(this->offload, numLayers, compute, load, unload);
    helper.run();

    return hidden_states;
}

void Flux2Model::setAttentionImpl(AttentionImpl impl, std::function<Tensor(Tensor)> func) {
    for (auto &block : transformer_blocks) {
        block->attnImpl        = impl;
        block->custom_attn_func = func;
    }
    for (auto &block : single_transformer_blocks) {
        block->attnImpl        = impl;
        block->custom_attn_func = func;
    }
    if (impl == AttentionImpl::NunchakuFP16) {
        Attention::setForceFP16(this, true);
    }
}
