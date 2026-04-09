#pragma once

#include "interop/torch.h"
#include "Flux2Model.h"
#include "Serialization.h"
#include "debug.h"
#include "Linear.h"
#include "module.h"

class QuantizedFlux2Model : public ModuleWrapper<Flux2Model> {
public:
    void init(int num_layers,
              int num_single_layers,
              int dim,
              int num_attention_heads,
              int attention_head_dim,
              int mlp_ratio,
              bool use_fp4,
              bool offload,
              bool bf16,
              int8_t deviceId) {
        spdlog::info("Initializing QuantizedFlux2Model on device {} ({} double + {} single blocks, dim={})",
                     deviceId,
                     num_layers,
                     num_single_layers,
                     dim);
        if (!bf16) {
            spdlog::info("Use FP16 model");
        }
        if (offload) {
            spdlog::info("Layer offloading enabled");
        }
        ModuleWrapper::init(deviceId);

        CUDADeviceContext ctx(this->deviceId);
        net = std::make_unique<Flux2Model>(num_layers,
                                           num_single_layers,
                                           dim,
                                           num_attention_heads,
                                           attention_head_dim,
                                           mlp_ratio,
                                           use_fp4,
                                           offload,
                                           bf16 ? Tensor::BF16 : Tensor::FP16,
                                           Device::cuda((int)deviceId));
    }

    bool isBF16() {
        checkModel();
        return net->dtype == Tensor::BF16;
    }

    torch::Tensor forward(torch::Tensor hidden_states,
                          torch::Tensor encoder_hidden_states,
                          torch::Tensor mod_img,
                          torch::Tensor mod_txt,
                          torch::Tensor mod_single,
                          torch::Tensor rotary_emb_img,
                          torch::Tensor rotary_emb_txt,
                          torch::Tensor rotary_emb_single) {
        checkModel();
        CUDADeviceContext ctx(deviceId);

        hidden_states         = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        mod_img               = mod_img.contiguous();
        mod_txt               = mod_txt.contiguous();
        mod_single            = mod_single.contiguous();
        rotary_emb_img        = rotary_emb_img.contiguous();
        rotary_emb_txt        = rotary_emb_txt.contiguous();
        rotary_emb_single     = rotary_emb_single.contiguous();

        Tensor result = net->forward(from_torch(hidden_states),
                                     from_torch(encoder_hidden_states),
                                     from_torch(mod_img),
                                     from_torch(mod_txt),
                                     from_torch(mod_single),
                                     from_torch(rotary_emb_img),
                                     from_torch(rotary_emb_txt),
                                     from_torch(rotary_emb_single));

        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();
        return output;
    }

    void setLoraScale(int skipRanks, float scale) {
        if (skipRanks % 16 != 0) {
            throw std::invalid_argument("skipRanks must be multiples of 16");
        }

        CUDADeviceContext ctx(deviceId);
        spdlog::info("Set lora scale to {} (skip {} ranks)", scale, skipRanks);

        net->traverse([&](Module *module) {
            if (auto *m = dynamic_cast<GEMV_AWQ *>(module)) {
                m->lora_scale = scale;
            } else if (auto *m = dynamic_cast<GEMM_W4A4 *>(module)) {
                for (int i = 0; i < skipRanks / 16; i++) {
                    m->lora_scales[i] = 1.0f;
                }
                for (int i = skipRanks / 16; i < (int)m->lora_scales.size(); i++) {
                    m->lora_scales[i] = scale;
                }
            }
        });
    }

    void setAttentionImpl(std::string name, pybind11::function attn_func) {
        if (name.empty() || name == "default") {
            name = "flashattn2";
        }

        spdlog::info("Set attention implementation to {}", name);

        if (name == "flashattn2") {
            net->setAttentionImpl(AttentionImpl::FlashAttention2, nullptr);
        } else if (name == "nunchaku-fp16") {
            net->setAttentionImpl(AttentionImpl::NunchakuFP16, nullptr);
        } else if (name == "custom") {
            pybind11::object f = attn_func;
            net->setAttentionImpl(AttentionImpl::Custom, [f](Tensor qkv) -> Tensor {
                torch::Tensor torch_qkv = to_torch(qkv, true);
                pybind11::object result  = f(torch_qkv);
                torch::Tensor output     = result.cast<torch::Tensor>();
                return from_torch(output);
            });
        } else {
            throw std::invalid_argument(spdlog::fmt_lib::format("Invalid attention implementation {}", name));
        }
    }
};
