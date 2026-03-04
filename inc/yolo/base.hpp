#pragma once
#include "corekit/cuda/core.hpp"
#include "corekit/cuda/image.hpp"
#include "corekit/cuda/model.hpp"
#include "yolo/types.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace yolo {

using namespace corekit::cuda;

class Base : public Model {

  public:
    using Ptr = std::shared_ptr<Base>;

    struct Settings {
        Path engine;
        Path config;
    };

    Base(const Settings& settings)
        : Model(settings.engine)
        , config(settings.config)
        , netsize(make_uint2(NET_WIDTH, NET_HEIGHT))
    {
        resized_cache = Image3U(netsize);
        padded_cache  = Image3U(netsize);
        tensor_cache  = Image3F(netsize);
    }

    yolo::Result process(const Image3U& in, Image3U& out, float conf = 0.25f)
    {
        Transform tf = {
            .imgsize = in.getSize(),
            .netsize = netsize,
        };

        Letterbox letter = {
            .in   = in,
            .out  = out,
            .tf   = tf,
            .conf = conf,
            .dets = MAX_DETS,
        };

        check_cuda();
        if (out.ptr() != in.ptr()) {
            cudaMemcpyAsync(out.ptr(), in.ptr(), in.get_bytes(), cudaMemcpyDeviceToDevice, stream);
        }

        this->preprocess(letter);
        this->exec();
        this->postprocess(letter);

        yolo::Result result(letter.dets);
        cudaMemcpyAsync(
            result.data(), storage.d_boxes.ptr(), letter.dets * sizeof(BoundingBox), cudaMemcpyDeviceToHost, stream);

        check_cuda();
        return result;
    }

  protected:
    void preprocess(Letterbox& letter)
    {
        const IOBinding& b_in    = inputs.front();
        const uint2      imgsize = letter.in.getSize();

        const float scaledX = 1.0f * netsize.x / imgsize.x;
        const float scaledY = 1.0f * netsize.y / imgsize.y;
        letter.tf.scaling   = std::min<float>(scaledX, scaledY);
        const uint scaledW  = (imgsize.x * letter.tf.scaling + 0.5f);
        const uint scaledH  = (imgsize.y * letter.tf.scaling + 0.5f);
        const uint paddedW  = (netsize.x - scaledW) / 2;
        const uint paddedH  = (netsize.y - scaledH) / 2;

        const uint2 scaled = make_uint2(scaledW, scaledH);
        letter.tf.padding  = make_uint2(paddedW, paddedH);

        letter.tf.imgsize = imgsize;
        letter.tf.netsize = netsize;

        letter.in.resize_into(resized_cache, scaled);
        resized_cache.pad_into(padded_cache, make_uchar3(114, 114, 114));
        padded_cache.chnflip_into(tensor_cache);

        cudaMemcpyAsync(b_in.ptr, tensor_cache.ptr(), tensor_cache.get_bytes(), cudaMemcpyDeviceToDevice, stream);
    }

    virtual void postprocess(Letterbox& letter) = 0;

    virtual bool prepare() override
    {
        bool success = Model::prepare();

        if (success) {
            storage = Storage::load(config, stream);
        }

        return true;
    }

    Path    config;
    Storage storage;
    uint2   netsize;
    Image3U resized_cache;
    Image3U padded_cache;
    Image3F tensor_cache;

}; // namespace yolo

} // namespace yolo
