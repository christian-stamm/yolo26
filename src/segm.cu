#include "corekit/cuda/core.hpp"
#include "corekit/utils/color.hpp"
#include "yolo/bbox.hpp"
#include "yolo/segm.hpp"
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

__global__ void buildSegmKernel(
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

    if (extractBox(&d_mout[idx * 38], mem.d_blacklist, d_letbox, &bbox)) {

        // Atomically increment counter and store bbox
        uint write_idx = atomicAdd(d_count, 1);

        if (write_idx >= d_letbox->dets) {
            return;
        }

        mem.d_boxes[write_idx] = bbox; // Store basic bbox info (pos, dim, conf, clsid)

        const float* m_base = &d_mout[idx * 38 + 6];
        Coeff&       d_base = mem.d_coeffs[write_idx];

        // Store mask coefficients (32 values)
        for (int i = 0; i < MASK_COEFFS; ++i) {
            d_base[i] = m_base[i];
        }
    }
}

__global__ void buildMaskKernel(
    const float*  d_mout, // Mask prototypes [MASK_COEFFS][MASK_HEIGHT][MASK_WIDTH]
    uint          count,  // Number of valid detections
    DeviceStorage mem     // Memory manager for intermediate buffers
)
{
    // 2D grid: blockIdx.x = detection index, blockIdx.y = pixel block index
    const uint det_idx = blockIdx.x;
    const uint pix_idx = blockIdx.y * blockDim.x + threadIdx.x;

    const uint total_pixels = MASK_WIDTH * MASK_HEIGHT;

    if (det_idx >= count || pix_idx >= total_pixels) {
        return;
    }

    // Compute linear combination of mask prototypes for this pixel
    float result = 0.0f;

#pragma unroll
    for (int i = 0; i < MASK_COEFFS; ++i) {
        const float  coeff = mem.d_coeffs[det_idx][i];
        const float* proto = &d_mout[i * total_pixels];
        result += coeff * proto[pix_idx];
    }

    mem.d_proto[det_idx][pix_idx] = result;
}

__global__ void overlayMaskKernel(
    uchar*        d_img,    // Image data on device
    uint2         img_size, // Image dimensions
    Letterbox*    d_letbox, // Letterbox info for coordinate transformation
    uint          count,    // Number of valid detections
    DeviceStorage mem       // Memory manager for intermediate buffers
)
{
    // 2D grid: blockIdx.x = detection index, remaining dims = pixel index
    const uint det_idx = blockIdx.x;
    const uint pix_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (det_idx >= count) {
        return;
    }

    const BoundingBox& bbox = mem.d_boxes[det_idx];

    // bbox is already in original image space
    // Use floating point for proper centroid-to-corner conversion (avoids integer truncation)
    const float x0_f = bbox.pos.x - bbox.dim.x * 0.5f;
    const float y0_f = bbox.pos.y - bbox.dim.y * 0.5f;
    const float x1_f = bbox.pos.x + bbox.dim.x * 0.5f;
    const float y1_f = bbox.pos.y + bbox.dim.y * 0.5f;

    const int x0 = (int)x0_f;
    const int y0 = (int)y0_f;
    const int x1 = (int)x1_f;
    const int y1 = (int)y1_f;

    const int bbox_width   = x1 - x0;
    const int bbox_height  = y1 - y0;
    const int total_pixels = bbox_width * bbox_height;

    if (pix_idx >= total_pixels || bbox_width <= 0 || bbox_height <= 0) {
        return;
    }

    // Convert linear pixel index to 2D coordinates within bbox
    const int local_x = pix_idx % bbox_width;
    const int local_y = pix_idx / bbox_width;

    // Calculate global image coordinates
    const int img_x = x0 + local_x;
    const int img_y = y0 + local_y;

    // Bounds check for image dimensions
    if (img_x < 0 || img_x >= img_size.x || img_y < 0 || img_y >= img_size.y) {
        return;
    }

    // Convert image coordinates back to network space (640x640)
    // Reverse of: x_img = (x_net - padding.x) / scaling
    // So: x_net = (x_img * scaling) + padding.x
    const float net_x = img_x * d_letbox->tf.scaling + d_letbox->tf.padding.x;
    const float net_y = img_y * d_letbox->tf.scaling + d_letbox->tf.padding.y;

    // Clamp to network bounds
    if (net_x < 0 || net_x >= d_letbox->tf.netsize.x || net_y < 0 || net_y >= d_letbox->tf.netsize.y) {
        return;
    }

    // Map from network space (640x640) to mask prototype space (160x160)
    // Mask is 4x downsampled from network output
    const float mask_x = (net_x / d_letbox->tf.netsize.x) * MASK_WIDTH;
    const float mask_y = (net_y / d_letbox->tf.netsize.y) * MASK_HEIGHT;

    // Bilinear interpolation coordinates
    const int mx0 = (int)mask_x;
    const int my0 = (int)mask_y;
    const int mx1 = min(mx0 + 1, MASK_WIDTH - 1);
    const int my1 = min(my0 + 1, MASK_HEIGHT - 1);

    // Clamp mx0/my0 to valid range
    const int mx0_clamped = max(0, mx0);
    const int my0_clamped = max(0, my0);

    const float fx = mask_x - mx0;
    const float fy = mask_y - my0;

    // Sample mask prototype at 4 corners with proper bounds checking
    const float* proto = mem.d_proto[det_idx];

    float v00 = (mx0 >= 0 && my0 >= 0) ? proto[my0_clamped * MASK_WIDTH + mx0_clamped] : 0.0f;
    float v10 = (mx1 < MASK_WIDTH && my0 >= 0) ? proto[my0_clamped * MASK_WIDTH + mx1] : 0.0f;
    float v01 = (mx0 >= 0 && my1 < MASK_HEIGHT) ? proto[my1 * MASK_WIDTH + mx0_clamped] : 0.0f;
    float v11 = (mx1 < MASK_WIDTH && my1 < MASK_HEIGHT) ? proto[my1 * MASK_WIDTH + mx1] : 0.0f;

    // Bilinear interpolation
    const float v0       = v00 * (1.0f - fx) + v10 * fx;
    const float v1       = v01 * (1.0f - fx) + v11 * fx;
    const float mask_val = v0 * (1.0f - fy) + v1 * fy;

    // Apply sigmoid activation to get mask probability [0, 1]
    const float alpha = 1.0f / (1.0f + expf(-mask_val));

    // Skip pixels with very low alpha (optimization)
    if (alpha < 0.1f) {
        return;
    }

    // Get color for this class from palette
    const Color& color = mem.d_palette[bbox.clsid];

    // Convert color from [0,1] to [0,255]
    const uchar3 color_u8 =
        make_uchar3((uchar)(color.r * 255.0f), (uchar)(color.g * 255.0f), (uchar)(color.b * 255.0f));

    // Blend with image using alpha
    const int pixel_idx = (img_y * img_size.x + img_x) * 3;

    d_img[pixel_idx + 0] = (uchar)(d_img[pixel_idx + 0] * (1.0f - alpha * 0.5f) + color_u8.x * alpha * 0.5f);
    d_img[pixel_idx + 1] = (uchar)(d_img[pixel_idx + 1] * (1.0f - alpha * 0.5f) + color_u8.y * alpha * 0.5f);
    d_img[pixel_idx + 2] = (uchar)(d_img[pixel_idx + 2] * (1.0f - alpha * 0.5f) + color_u8.z * alpha * 0.5f);
}

uint Segm::buildSegm(
    const float* d_out,  // Output array of net [x0, y0, x1, y1, conf, class, 32 x coeff] * MAX_DETS
    Letterbox&   letbox, // Letterbox info for coordinate transformation
    Storage      mem,    // Memory manager for intermediate buffers
    cudaStream_t stream  // CUDA stream for asynchronous execution
)
{

    check_cuda(cudaMemsetAsync(mem.d_count.ptr(), 0, sizeof(uint), stream));
    check_cuda(cudaMemcpyAsync(mem.d_letbox.ptr(), &letbox, sizeof(Letterbox), cudaMemcpyHostToDevice, stream));

    int blockSize = 256;
    int gridSize  = (letbox.dets + blockSize - 1) / blockSize;

    buildSegmKernel<<<gridSize, blockSize, 0, stream>>>(d_out, mem.d_count.ptr(), mem.d_letbox.ptr(), mem.device());
    check_cuda();

    // Copy back the count of valid detections
    letbox.dets = 0; // Reset count on host before copying back
    check_cuda(cudaMemcpyAsync(&letbox.dets, mem.d_count.ptr(), sizeof(uint), cudaMemcpyDeviceToHost, stream));
    check_cuda();

    return letbox.dets;
}

void Segm::drawSegm(
    Image3U&     d_img,  // Image data on device
    const float* d_mout, // mask protos
    Letterbox&   letbox, // Letterbox info for coordinate transformation
    Storage      mem,    // Memory manager containing boxes and masks
    uint         count,  // Number of valid boxes found
    cudaStream_t stream  // CUDA stream for asynchronous execution
)
{
    check_cuda();

    if (count == 0) {
        return;
    }

    // Clear proto buffers
    cudaMemsetAsync(mem.d_proto.ptr(), 0, MAX_DETS * sizeof(Proto), stream);

    const uint total_pixels = MASK_WIDTH * MASK_HEIGHT;
    const int  blockSize    = 256;

    // Step 1: Build masks - 2D grid: x-dimension for detections, y-dimension for pixels
    // Total threads: count × 25,600 (one thread per output pixel)
    dim3 maskGridSize(count, (total_pixels + blockSize - 1) / blockSize);
    buildMaskKernel<<<maskGridSize, blockSize, 0, stream>>>(d_mout, count, mem.device());

    check_cuda(cudaMemcpyAsync(mem.d_letbox.ptr(), &letbox, sizeof(Letterbox), cudaMemcpyHostToDevice, stream));

    // Step 2: Overlay masks on image - 2D grid: x-dimension for detections, y-dimension for pixels
    // Launch enough threads to cover the full image dimensions, not just a max bbox size
    // Each thread processes one potential pixel in the bbox
    const uint2 img_size         = d_img.getSize();
    const uint  max_image_pixels = img_size.x * img_size.y;
    dim3        overlayGridSize(count, (max_image_pixels + blockSize - 1) / blockSize);
    overlayMaskKernel<<<overlayGridSize, blockSize, 0, stream>>>(
        d_img.ptr(), img_size, mem.d_letbox.ptr(), count, mem.device());

    check_cuda();
}

} // namespace yolo
