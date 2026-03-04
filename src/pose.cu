#include "corekit/cuda/core.hpp"
#include "corekit/utils/color.hpp"
#include "yolo/bbox.hpp"
#include "yolo/pose.hpp"
#include "yolo/types.hpp"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>
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

    __device__ __forceinline__ bool extractBox(const float* base, Letterbox* letbox, BoundingBox* bbox)
    {
        bbox->score = base[4];
        bbox->clsid = base[5];

        if (bbox->score < letbox->conf) {
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

__device__ __forceinline__ uchar4 rgb2col(Color rgb)
{
    return make_uchar4(255 * rgb.r, 255 * rgb.g, 255 * rgb.b, 255 * rgb.a);
}

__global__ void buildPoseKernel(
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

    if (extractBox(&d_mout[idx * 57], d_letbox, &bbox)) {

        // Atomically increment counter and store bbox
        uint write_idx = atomicAdd(d_count, 1);

        if (write_idx >= d_letbox->dets) {
            return;
        }

        mem.d_boxes[write_idx] = bbox; // Store basic bbox info (pos, dim, conf, clsid)

        const float* kp_base = &d_mout[idx * 57 + 6];
        Skeleton*    skelet  = mem.d_skelets + write_idx;

        for (int i = 0; i < NUM_KPS; i++) {
            float kpx = kp_base[i * 3 + 0];
            float kpy = kp_base[i * 3 + 1];
            float kps = kp_base[i * 3 + 2];

            (*skelet)[i].pos = upscale(kpx, kpy, d_letbox->tf);
            (*skelet)[i].vis = d_letbox->conf <= kps;
        }
    }
}

__global__ void extractPosePartsKernel(
    uint          count,         // Number of valid boxes
    uint*         d_bone_count,  // Atomic counter for bones
    uint*         d_joint_count, // Atomic counter for joints
    DeviceStorage mem            // Memory manager for intermediate buffers
)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count) {
        return;
    }

    const Skeleton* skelet = mem.d_skelets + idx;

    for (int i = 0; i < NUM_KPS; i++) {
        const Keypoint& kp = (*skelet)[i];

        if (!kp.vis) {
            continue; // Skip invisible joints
        }

        const uchar4 color = rgb2col(mem.d_palette[kp.colID]);

        uint write_idx = atomicAdd(d_joint_count, 1);

        Circle& circ   = mem.d_circles[write_idx];
        circ.center    = kp.pos;
        circ.radius    = 10;
        circ.fill      = color;
        circ.border    = color;
        circ.thickness = 0;
    }

    for (int i = 0; i < NUM_BNS; i++) {
        const Bone& bone = mem.d_bones[i];

        if (bone.origin >= NUM_KPS || bone.target >= NUM_KPS) {
            continue; // Skip invalid bone definitions
        }

        const Keypoint& kp0 = (*skelet)[bone.origin];
        const Keypoint& kp1 = (*skelet)[bone.target];

        const uchar4 c0 = rgb2col(mem.d_palette[kp0.colID]);
        const uchar4 c1 = rgb2col(mem.d_palette[kp1.colID]);

        if (kp0.vis && kp1.vis) {

            uint write_idx = atomicAdd(d_bone_count, 1);

            Line& line     = mem.d_lines[write_idx];
            line.origin    = kp0.pos;
            line.target    = kp1.pos;
            line.p0_col    = c0;
            line.p1_col    = c1;
            line.thickness = 3;
        }
    }
}

uint Pose::buildPose(
    const float* d_out,  // Output array of net
    Letterbox&   letbox, // Letterbox info for coordinate transformation
    Storage&     mem,    // Memory manager for intermediate buffers
    cudaStream_t stream  // CUDA stream for asynchronous execution
)
{
    if (!d_out) {
        throw std::invalid_argument("Null pointer passed to buildPose");
    }

    check_cuda(cudaMemsetAsync(mem.d_count.ptr(), 0, sizeof(uint), stream));
    check_cuda(cudaMemcpyAsync(mem.d_letbox.ptr(), &letbox, sizeof(Letterbox), cudaMemcpyHostToDevice, stream));

    int blockSize = 256;
    int gridSize  = (letbox.dets + blockSize - 1) / blockSize;

    buildPoseKernel<<<gridSize, blockSize, 0, stream>>>(d_out, mem.d_count.ptr(), mem.d_letbox.ptr(), mem.device());

    letbox.dets = 0; // Reset count on host before copying back
    check_cuda(cudaMemcpyAsync(&letbox.dets, mem.d_count.ptr(), sizeof(uint), cudaMemcpyDeviceToHost, stream));
    check_cuda();

    return letbox.dets;
}

void Pose::drawPose(
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

    check_cuda(cudaMemsetAsync(mem.d_bone_count.ptr(), 0, sizeof(uint), stream));
    check_cuda(cudaMemsetAsync(mem.d_joint_count.ptr(), 0, sizeof(uint), stream));

    int blockSize = 256;
    int gridSize  = (count + blockSize - 1) / blockSize;

    extractPosePartsKernel<<<gridSize, blockSize, 0, stream>>>(
        count, mem.d_bone_count.ptr(), mem.d_joint_count.ptr(), mem.device());

    uint h_bone_count  = 0;
    uint h_joint_count = 0;
    cudaMemcpyAsync(&h_bone_count, mem.d_bone_count.ptr(), sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_joint_count, mem.d_joint_count.ptr(), sizeof(uint), cudaMemcpyDeviceToHost, stream);

    yolo::BBox::drawBBox(d_img, count, mem, stream);
    drawLine(d_img, mem.d_lines.ptr(), h_bone_count, stream);
    drawCircle(d_img, mem.d_circles.ptr(), h_joint_count, stream);

    check_cuda();
}

} // namespace yolo
