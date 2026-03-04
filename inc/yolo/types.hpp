#pragma once
#include "corekit/cuda/core.hpp"
#include "corekit/cuda/draw.hpp"
#include "corekit/cuda/image.hpp"
#include "corekit/types.hpp"
#include "corekit/utils/color.hpp"
#include "corekit/utils/filemgr.hpp"
#include "yolo/types.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <vector_functions.h>
#include <vector_types.h>

namespace yolo {

using namespace corekit::types;
using namespace corekit::utils;
using namespace corekit::cuda;

constexpr uint NET_WIDTH     = 640;
constexpr uint NET_HEIGHT    = 640;
constexpr uint MAX_DETS      = 300;
constexpr uint NUM_CLS       = 80;
constexpr uint NUM_KPS       = 17;
constexpr uint NUM_BNS       = 18;
constexpr uint MASK_COEFFS   = 32;
constexpr uint MASK_WIDTH    = 160;
constexpr uint MASK_HEIGHT   = 160;
constexpr uint MAX_PRIMITVES = 1024;

struct Transform {
    uint2 imgsize;
    uint2 netsize;
    uint2 padding;
    float scaling;
};

struct Letterbox {
    const Image3U& in;
    Image3U&       out;
    Transform      tf;
    float          conf;
    uint           dets;
};

struct BoundingBox {
    uint2 pos;
    uint2 dim;
    uint  clsid;
    float score;
};

struct Keypoint {
    uint  colID;
    uint2 pos;
    bool  vis;
};

struct Bone {
    uint origin;
    uint target;
};

using Proto = float[MASK_WIDTH * MASK_HEIGHT];
using Coeff = float[MASK_COEFFS];

using Skeleton = Keypoint[NUM_KPS];
using Result   = std::vector<BoundingBox>;

// Device-friendly view with raw pointers (for kernel usage)
struct DeviceStorage {
    Letterbox*   d_letbox;
    BoundingBox* d_boxes;
    Skeleton*    d_skelets;
    Bone*        d_bones;
    Proto*       d_proto;
    Coeff*       d_coeffs;
    Color*       d_palette;
    char*        d_classes;
    Font*        d_txtfont;
    Line*        d_lines;
    Rect*        d_rects;
    Text*        d_texts;
    Circle*      d_circles;
    uint*        d_count;
    uint*        d_bone_count;
    uint*        d_joint_count;
};

struct Storage {

    static Storage load(const Path& cfgFile, cudaStream_t stream)
    {
        JsonMap config = File::loadJson(cfgFile);

        Storage mem;

        std::vector<std::string> classes(NUM_CLS);
        std::vector<Keypoint>    keypts(NUM_KPS);
        std::vector<Bone>        bones;
        std::vector<Color>       palette;

        for (auto& [key, name] : config.at("classes").items()) {
            uint classId     = std::stoi(key);
            classes[classId] = name.get<std::string>();
        }

        for (auto& keypt : config.at("keypoints")) {
            const uint kpt_id = keypt.at("kptid").get<uint>();
            const uint col_id = keypt.at("color").get<uint>();

            keypts[kpt_id].colID = ((float)(col_id) / 15.0) * NUM_CLS;
        }

        for (auto& bone : config.at("bones")) {
            const uint origin = bone.at("origin").get<uint>();
            const uint target = bone.at("target").get<uint>();

            bones.push_back({origin, target});
        }

        palette = Color::sample(classes.size());

        mem.d_boxes   = NvMem<BoundingBox>(MAX_DETS);
        mem.d_letbox  = NvMem<Letterbox>(1);
        mem.d_skelets = NvMem<Keypoint>(MAX_DETS * NUM_KPS);               // Flatten array
        mem.d_proto   = NvMem<float>(MAX_DETS * MASK_WIDTH * MASK_HEIGHT); // Flatten array
        mem.d_coeffs  = NvMem<float>(MAX_DETS * MASK_COEFFS);              // Flatten array
        mem.d_bones   = NvMem<Bone>(bones.size());
        mem.d_palette = NvMem<Color>(palette.size());
        mem.d_classes = NvMem<char>(classes.size() * MAX_TEXT_LEN);
        mem.d_lines   = NvMem<Line>(MAX_PRIMITVES);
        mem.d_rects   = NvMem<Rect>(MAX_PRIMITVES);
        mem.d_texts   = NvMem<Text>(MAX_PRIMITVES);
        mem.d_circles = NvMem<Circle>(MAX_PRIMITVES);

        // Counters for atomic operations
        mem.d_count       = NvMem<uint>(1);
        mem.d_bone_count  = NvMem<uint>(1);
        mem.d_joint_count = NvMem<uint>(1);

        mem.d_txtfont = Font::loadFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", DEF_TEXT_SIZE);

        cudaMemcpyAsync(mem.d_bones.ptr(), bones.data(), bones.size() * sizeof(Bone), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(
            mem.d_palette.ptr(), palette.data(), palette.size() * sizeof(Color), cudaMemcpyHostToDevice, stream);

        for (uint cls = 0; cls < classes.size(); cls++) {
            const Name&  name = classes.at(cls);
            char*        base = mem.d_classes.ptr() + cls * MAX_TEXT_LEN;
            const size_t size = std::min<size_t>(name.size(), MAX_TEXT_LEN);
            cudaMemcpyAsync(base, name.c_str(), size * sizeof(char), cudaMemcpyHostToDevice, stream);
        }

        for (uint det = 0; det < MAX_DETS; det++) {
            cudaMemcpyAsync(
                &mem.d_skelets.ptr()[det * NUM_KPS], keypts.data(), sizeof(Skeleton), cudaMemcpyHostToDevice, stream);
        }

        return mem;
    };

    static void free(Storage& mem, cudaStream_t stream)
    {
        Font::freeFont(mem.d_txtfont);
    }

    // Get device-friendly view with raw pointers
    DeviceStorage device() const
    {
        return DeviceStorage{
            d_letbox.ptr(),
            d_boxes.ptr(),
            reinterpret_cast<Skeleton*>(d_skelets.ptr()),
            d_bones.ptr(),
            reinterpret_cast<Proto*>(d_proto.ptr()),
            reinterpret_cast<Coeff*>(d_coeffs.ptr()),
            d_palette.ptr(),
            d_classes.ptr(),
            d_txtfont,
            d_lines.ptr(),
            d_rects.ptr(),
            d_texts.ptr(),
            d_circles.ptr(),
            d_count.ptr(),
            d_bone_count.ptr(),
            d_joint_count.ptr()};
    }

    NvMem<Letterbox>   d_letbox;
    NvMem<BoundingBox> d_boxes;
    NvMem<Keypoint>    d_skelets; // Flattened: MAX_DETS * NUM_KPS
    NvMem<Bone>        d_bones;
    NvMem<float>       d_proto;  // Flattened: MAX_DETS * MASK_WIDTH * MASK_HEIGHT
    NvMem<float>       d_coeffs; // Flattened: MAX_DETS * MASK_COEFFS
    NvMem<Color>       d_palette;
    NvMem<char>        d_classes;
    Font*              d_txtfont;
    NvMem<Line>        d_lines;
    NvMem<Rect>        d_rects;
    NvMem<Text>        d_texts;
    NvMem<Circle>      d_circles;

    // Atomic counters
    NvMem<uint> d_count;
    NvMem<uint> d_bone_count;
    NvMem<uint> d_joint_count;
};

} // namespace yolo