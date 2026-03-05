#include "corekit/cuda/core.hpp"
#include "corekit/utils/color.hpp"
#include "yolo/bbox.hpp"
#include "yolo/types.hpp"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>
#include <stdexcept>
#include <sys/types.h>
#include <vector_functions.h>
#include <vector_types.h>

namespace yolo {

namespace {
    __device__ __forceinline__ uint2 upscale(float x, float y, const Transform& tf)
    {
        x = (x - tf.padding.x) / tf.scaling;
        y = (y - tf.padding.y) / tf.scaling;

        return make_uint2(x, y);
    }

    __device__ __forceinline__ void xyxy2ccwh(uint2 p0, uint2 p1, uint2& center, uint2& shape)
    {
        center.x = (p0.x + p1.x) / 2.0f;
        center.y = (p0.y + p1.y) / 2.0f;
        shape.x  = (p1.x - p0.x);
        shape.y  = (p1.y - p0.y);
    }

    __device__ __forceinline__ bool extractBox(
        const float* base, const bool* blacklist, Letterbox* letbox, BoundingBox* bbox)
    {
        bbox->score = base[4];
        bbox->clsid = base[5];

        if (bbox->score < letbox->conf || blacklist[bbox->clsid]) {
            return false;
        }

        float x0 = base[0];
        float y0 = base[1];
        float x1 = base[2];
        float y1 = base[3];

        uint2 p0 = upscale(x0, y0, letbox->tf);
        uint2 p1 = upscale(x1, y1, letbox->tf);

        xyxy2ccwh(p0, p1, bbox->pos, bbox->dim);

        return true;
    }
} // namespace

__global__ void buildBBoxKernel(
    const float*  d_mout,   // Raw detection data (x0, y0, x1, y1, conf, class_id)
    uint*         d_count,  // Atomic counter for valid detections
    Letterbox*    d_letbox, // Letterbox info for coordinate transformation
    DeviceStorage mem       // Memory manager for intermediate buffers
)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= d_letbox->dets) {
        return;
    }

    BoundingBox bbox;

    if (extractBox(&d_mout[idx * 6], mem.d_blacklist, d_letbox, &bbox)) {

        // Atomically increment counter and store bbox
        uint write_idx = atomicAdd(d_count, 1);

        if (write_idx >= d_letbox->dets) {
            return;
        }

        mem.d_boxes[write_idx] = bbox; // Store basic bbox info (pos, dim, conf, clsid)
    }
}

__global__ void extractBBoxPartsKernel(
    DeviceStorage mem,  // Memory manager for intermediate buffers
    uint          count // Number of valid boxes
)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count) {
        return;
    }

    const BoundingBox& bbox = mem.d_boxes[idx];

    // Object rectangle
    Rect& boxrect  = mem.d_rects[2 * idx + 0];
    Rect& txtrect  = mem.d_rects[2 * idx + 1];
    Text& txtlabel = mem.d_texts[1 * idx + 0];

    const uint txt_size = DEF_TEXT_SIZE;

    uint2  pos  = bbox.pos;
    uint2  dim  = bbox.dim;
    Color  rgb  = mem.d_palette[bbox.clsid];
    uchar4 rgba = make_uchar4(255 * rgb.r, 255 * rgb.g, 255 * rgb.b, 255);

    rgba.w            = 64;
    boxrect.center    = pos;
    boxrect.shape     = dim;
    boxrect.border    = rgba;
    boxrect.fill      = rgba;
    boxrect.thickness = 0;

    pos.y -= (dim.y + txt_size) / 2.0f; // Position above the box
    dim.y = txt_size;                   // Height for text background

    rgba.w            = 127;
    txtrect.center    = pos;
    txtrect.shape     = dim;
    txtrect.border    = rgba;
    txtrect.fill      = rgba;
    txtrect.thickness = 0;

    pos.x -= dim.x / 2.0f;    // Center text background on box
    pos.y -= txt_size / 2.0f; // Position above the box
    rgba = make_uchar4(255, 255, 255, 255);

    txtlabel.pos   = pos;
    txtlabel.color = rgba;

    const char* base = &mem.d_classes[bbox.clsid * MAX_TEXT_LEN];
    for (uint i = 0; i < MAX_TEXT_LEN; i++) {
        txtlabel.msg[i] = base[i];
    }
}

/**
 * GPU-accelerated detection post-processing
 * Converts raw model output to drawable bounding boxes
 */
uint BBox::buildBBox(
    const float* d_out,  // Output array of net
    Letterbox&   letbox, // Letterbox info for coordinate transformation
    Storage&     mem,    // Memory manager for intermediate buffers
    cudaStream_t stream  // CUDA stream for asynchronous execution
)
{
    check_cuda();

    if (!d_out) {
        throw std::invalid_argument("Null pointer passed to buildBBox");
    }

    check_cuda(cudaMemsetAsync(mem.d_count.ptr(), 0, sizeof(uint), stream));
    check_cuda(cudaMemcpyAsync(mem.d_letbox.ptr(), &letbox, sizeof(Letterbox), cudaMemcpyHostToDevice, stream));

    int blockSize = 256;
    int gridSize  = (letbox.dets + blockSize - 1) / blockSize;

    buildBBoxKernel<<<gridSize, blockSize, 0, stream>>>(d_out, mem.d_count.ptr(), mem.d_letbox.ptr(), mem.device());
    check_cuda();

    letbox.dets = 0; // Reset count on host before copying back
    check_cuda(cudaMemcpyAsync(&letbox.dets, mem.d_count.ptr(), sizeof(uint), cudaMemcpyDeviceToHost, stream));
    check_cuda();

    return letbox.dets;
}

/**
 * Render bounding boxes to image
 */
void BBox::drawBBox(
    Image3U&     d_img, // Image data on device
    const uint   count, // Number of valid boxes found
    Storage&     mem,   // Memory manager for intermediate buffers
    cudaStream_t stream // CUDA stream for asynchronous execution
)
{
    check_cuda();

    if (count == 0) {
        return; // Nothing to draw
    }

    int blockSize = 256;
    int gridSize  = (count + blockSize - 1) / blockSize;

    extractBBoxPartsKernel<<<gridSize, blockSize, 0, stream>>>(mem.device(), count);
    check_cuda();

    // Draw rectangles directly from device memory
    drawRect(d_img, mem.d_rects.ptr(), 2 * count, stream);
    drawText(d_img, mem.d_texts.ptr(), 1 * count, mem.d_txtfont, stream);

    check_cuda();
}

} // namespace yolo