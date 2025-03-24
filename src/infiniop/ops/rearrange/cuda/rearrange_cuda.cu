#include "../../../devices/cuda/cuda_common.cuh"
#include "../../../tensor.h"
#include "rearrange_cuda.cuh"
#include "rearrange_kernel.cuh"
#include <algorithm>
#include <cmath>
#include <memory>
#include <nvrtc.h>
#include <stdint.h>
#include <vector>

namespace op::rearrange::cuda {

// 定义CUDA_CHECK宏
#define CUDA_CHECK(API)                                                   \
    do {                                                                  \
        cudaError_t err = (API);                                          \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto dtype = y_desc->dtype();
    auto ndim = y_desc->ndim();

    CHECK_API_OR(x_desc->dtype(), dtype, return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(x_desc->ndim(), ndim, return INFINI_STATUS_BAD_TENSOR_SHAPE);

    for (size_t i = 0; i < ndim; ++i) {
        std::cout << x_desc->shape()[i] << " ";
        std::cout << y_desc->shape()[i] << " ";
        CHECK_API_OR(x_desc->shape()[i], y_desc->shape()[i], return INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    // 保存临时vector对象
    auto y_shape = y_desc->shape();
    auto y_strides = y_desc->strides();
    auto x_strides = x_desc->strides();

    std::cout << "y_strides: ";
    for (size_t i = 0; i < ndim; ++i) {
        std::cout << y_strides[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "x_strides: ";
    for (size_t i = 0; i < ndim; ++i) {
        std::cout << x_strides[i] << " ";
    }
    std::cout << std::endl;

    auto meta = utils::RearrangeMeta::create(
        y_shape.data(),
        y_strides.data(),
        x_strides.data(),
        ndim,
        infiniSizeOf(dtype));

    std::cout << "DEBUG: meta信息:" << std::endl;
    if (meta) {
        std::cout << "  ndim: " << meta->ndim() << std::endl;
        std::cout << "  unit: " << meta->unit() << std::endl;
        std::cout << "  count: " << meta->count() << std::endl;

        if (meta->ndim() > 0) {
            std::cout << "  idx_strides: ";
            for (size_t i = 0; i < meta->ndim(); ++i) {
                std::cout << meta->idx_strides()[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "  dst_strides: ";
            for (size_t i = 0; i < meta->ndim(); ++i) {
                std::cout << meta->dst_strides()[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "  src_strides: ";
            for (size_t i = 0; i < meta->ndim(); ++i) {
                std::cout << meta->src_strides()[i] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "  meta为空" << std::endl;
    }

    if (!meta) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(
        std::move(*meta),
        new Opaque{reinterpret_cast<device::cuda::Handle *>(handle)->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// 维度信息结构
struct Dim {
    size_t len;
    int src_stride;
    int dst_stride;
};

// 分割维度结构
struct SplitDim {
    size_t choose_idx;
    size_t num_per_block;
    size_t num_per_grid;
    int array_struct_idx_block;
    int array_struct_idx_grid;
    size_t dim_len;
};

// 根据元数据准备计算参数
RearrangeParams prepareRearrangeParams(const utils::RearrangeMeta &original_meta) {
    RearrangeParams params;

    // 获取更适合GPU处理的单元大小，这里使用2的幂次方
    auto meta_opt = original_meta.distribute_unit({32, 16, 8, 4, 2, 1});

    // 如果找不到合适的单元大小，直接panic
    if (!meta_opt.has_value()) {
        throw std::runtime_error("无法找到合适的单元大小");
    }
    const utils::RearrangeMeta &meta = meta_opt.value();

    // 获取维度信息
    const size_t ndim = meta.ndim();
    const size_t unit = meta.unit();

    // 特殊情况：无维度，只需要简单复制
    if (ndim == 0) {
        params.block_dim = 0;
        params.block_len_total = 1;
        params.block_len = {1};
        params.src_block_stride = {0};
        params.dst_block_stride = {0};
        params.grid_len = {1};
        params.src_grid_stride = {0};
        params.dst_grid_stride = {0};
        params.unit_size = unit;
        return params;
    }

    // 从元数据中提取必要的信息
    const ptrdiff_t *idx_strides = meta.idx_strides();
    const ptrdiff_t *dst_strides = meta.dst_strides();
    const ptrdiff_t *src_strides = meta.src_strides();

    // 准备维度信息
    std::vector<Dim> dims;
    std::vector<size_t> shape;
    dims.reserve(ndim);
    shape.reserve(ndim);

    auto prev_idx_stride = meta.count();
    for (size_t i = 0; i < ndim; ++i) {
        size_t len = prev_idx_stride / idx_strides[i];
        shape.push_back(len);
        dims.push_back({len, static_cast<int>(src_strides[i]), static_cast<int>(dst_strides[i])});
        prev_idx_stride = idx_strides[i];
    }

    // 计算src_strides的降序排序索引，类似于Rust版本中的src_strides_desc_idx
    std::vector<size_t> src_strides_desc_idx(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        src_strides_desc_idx[i] = i;
    }
    std::sort(src_strides_desc_idx.begin(), src_strides_desc_idx.end(),
              [&dims](size_t a, size_t b) {
                  return std::abs(dims[a].src_stride) > std::abs(dims[b].src_stride);
              });

    // 根据最大线程数选择block和grid维度
    const size_t block_size = 256; // 与Rust版本保持一致
    std::vector<bool> block_dim_choose(ndim, false);

    // 初始化计数器
    size_t block_elements = 1;
    size_t block_src_elements = 1;
    size_t block_dst_elements = 1;
    size_t src_choose_idx = ndim;
    size_t dst_choose_idx = ndim;

    // 用于存储分割维度信息
    std::vector<SplitDim> split_dims;

    // 维度选择循环
    while (src_choose_idx > 0 && dst_choose_idx > 0) {
        // 获取当前需要处理的维度索引
        size_t src_idx = src_strides_desc_idx[src_choose_idx - 1];
        size_t dst_idx = dst_choose_idx - 1;

        if (src_idx == dst_idx) {
            // 源和目标维度相同，可以一起处理
            size_t idx = src_idx;
            size_t len = shape[idx];

            // 检查是否可以将此维度完全添加到block中
            if (block_elements * len <= block_size) {
                // 选择此维度
                block_dim_choose[idx] = true;
                block_elements *= len;
                block_src_elements *= len;
                block_dst_elements *= len;
                src_choose_idx--;
                dst_choose_idx--;
            } else {
                // 需要分割此维度
                size_t num_per_block = block_size / block_elements;

                // 确保num_per_block > 0且len >= num_per_block
                if (num_per_block > 0 && len >= num_per_block && num_per_block > 1) {
                    size_t num_per_grid = (len + num_per_block - 1) / num_per_block; // 向上取整

                    SplitDim split_dim = {
                        idx,           // choose_idx
                        num_per_block, // num_per_block
                        num_per_grid,  // num_per_grid
                        0,             // array_struct_idx_block (待更新)
                        0,             // array_struct_idx_grid (待更新)
                        len            // 原始维度长度
                    };
                    split_dims.push_back(split_dim);
                }
                break;
            }
        } else {
            // 源和目标维度不同，需要分别处理
            // 计算块比例
            double src_div_dst = static_cast<double>(block_src_elements) / block_dst_elements;
            double src_num_per_block = std::sqrt(block_size / (double)block_elements / src_div_dst);
            double dst_num_per_block = src_num_per_block * src_div_dst;

            size_t src_current_dim_len = shape[src_idx];
            size_t dst_current_dim_len = shape[dst_idx];

            if (static_cast<double>(src_current_dim_len) < src_num_per_block) {
                // 源维度可以完全添加到block
                block_dim_choose[src_idx] = true;
                block_elements *= src_current_dim_len;
                block_src_elements *= src_current_dim_len;
                src_choose_idx--;
            } else if (static_cast<double>(dst_current_dim_len) < dst_num_per_block) {
                // 目标维度可以完全添加到block
                block_dim_choose[dst_idx] = true;
                block_elements *= dst_current_dim_len;
                block_dst_elements *= dst_current_dim_len;
                dst_choose_idx--;
            } else {
                // 需要分割源和目标维度
                size_t src_num_per_block_int = static_cast<size_t>(std::floor(src_num_per_block));
                size_t dst_num_per_block_int = static_cast<size_t>(std::floor(dst_num_per_block));

                // 计算网格尺寸
                size_t src_num_per_grid = (src_current_dim_len + src_num_per_block_int - 1) / src_num_per_block_int; // 向上取整
                size_t dst_num_per_grid = (dst_current_dim_len + dst_num_per_block_int - 1) / dst_num_per_block_int; // 向上取整

                // 处理源维度
                if (src_num_per_block_int > 1) {
                    if (src_num_per_grid == 1) {
                        // 可以完全放入块
                        block_dim_choose[src_idx] = true;
                        block_elements *= src_current_dim_len;
                        block_src_elements *= src_current_dim_len;
                        src_choose_idx--;
                    } else {
                        // 需要分割
                        SplitDim split_dim = {
                            src_idx,               // choose_idx
                            src_num_per_block_int, // num_per_block
                            src_num_per_grid,      // num_per_grid
                            0,                     // array_struct_idx_block (待更新)
                            0,                     // array_struct_idx_grid (待更新)
                            src_current_dim_len    // 原始维度长度
                        };
                        split_dims.push_back(split_dim);
                    }
                }

                // 处理目标维度
                if (dst_num_per_block_int > 1) {
                    if (dst_num_per_grid == 1) {
                        // 可以完全放入块
                        block_dim_choose[dst_idx] = true;
                        block_elements *= dst_current_dim_len;
                        block_dst_elements *= dst_current_dim_len;
                        dst_choose_idx--;
                    } else {
                        // 需要分割
                        SplitDim split_dim = {
                            dst_idx,               // choose_idx
                            dst_num_per_block_int, // num_per_block
                            dst_num_per_grid,      // num_per_grid
                            0,                     // array_struct_idx_block (待更新)
                            0,                     // array_struct_idx_grid (待更新)
                            dst_current_dim_len    // 原始维度长度
                        };
                        split_dims.push_back(split_dim);
                    }
                }

                break;
            }
        }
    }

    // 准备block维度相关参数
    size_t block_dim = 0;
    size_t block_len_total = 1;

    std::vector<int> block_len;
    std::vector<int> src_block_stride;
    std::vector<int> dst_block_stride;

    std::vector<int> grid_len;
    std::vector<int> src_grid_stride;
    std::vector<int> dst_grid_stride;

    // 处理block维度，填充block_len和block_stride
    for (size_t i = 0; i < ndim; ++i) {
        if (block_dim_choose[i]) {
            block_len.push_back(shape[i]);
            src_block_stride.push_back(dims[i].src_stride);
            dst_block_stride.push_back(dims[i].dst_stride);
            block_dim += 1;
            block_len_total *= shape[i];
        }

        // 处理分割维度的block部分
        for (size_t j = 0; j < split_dims.size(); ++j) {
            if (i == split_dims[j].choose_idx) {
                block_len.push_back(split_dims[j].num_per_block);
                src_block_stride.push_back(dims[i].src_stride);
                dst_block_stride.push_back(dims[i].dst_stride);
                split_dims[j].array_struct_idx_block = block_dim;
                block_dim += 1;
                block_len_total *= split_dims[j].num_per_block;
            }
        }
    }

    // 处理grid维度，填充grid_len和grid_stride
    for (size_t i = 0; i < ndim; ++i) {
        if (!block_dim_choose[i]) {
            bool is_split = false;

            // 检查是否是分割维度
            for (size_t j = 0; j < split_dims.size(); ++j) {
                if (i == split_dims[j].choose_idx) {
                    is_split = true;
                    grid_len.push_back(split_dims[j].num_per_grid);
                    src_grid_stride.push_back(dims[i].src_stride * split_dims[j].num_per_block);
                    dst_grid_stride.push_back(dims[i].dst_stride * split_dims[j].num_per_block);
                    split_dims[j].array_struct_idx_grid = grid_len.size() - 1;
                }
            }

            // 如果不是分割维度，则作为完整的grid维度
            if (!is_split) {
                grid_len.push_back(shape[i]);
                src_grid_stride.push_back(dims[i].src_stride);
                dst_grid_stride.push_back(dims[i].dst_stride);
            }
        }
    }

    // 如果grid_len为空，添加一个默认值
    if (grid_len.empty()) {
        grid_len.push_back(1);
        src_grid_stride.push_back(0);
        dst_grid_stride.push_back(0);
    }

    // 处理约束条件 - 使用与Rust版本相似的逻辑
    std::vector<Constrains<int>> constrains;

    // 限制最多处理2个约束条件
    for (size_t i = 0; i < split_dims.size(); ++i) {
        if (split_dims[i].dim_len % split_dims[i].num_per_block == 0) {
            continue;
        }
        Constrains<int> constrain;
        constrain.grid_idx = split_dims[i].array_struct_idx_grid;
        constrain.block_idx = split_dims[i].array_struct_idx_block;
        constrain.grid_div_block = split_dims[i].num_per_block;
        constrain.total_len = split_dims[i].dim_len;
        constrains.push_back(constrain);
    }

    // 设置参数
    params.block_dim = block_dim;
    params.block_len_total = block_len_total;
    params.block_len = block_len;
    params.src_block_stride = src_block_stride;
    params.dst_block_stride = dst_block_stride;
    params.grid_len = grid_len;
    params.src_grid_stride = src_grid_stride;
    params.dst_grid_stride = dst_grid_stride;
    params.constrains = constrains;
    params.unit_size = unit;

    return params;
}

// 带约束的内核启动模板函数
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    void *y,
    const void *x,
    unsigned int grid_size,
    const RearrangeParams &params,
    size_t unit_size,
    cudaStream_t stream) {

    std::cout << "DEBUG: launchKernel<" << BLOCK_SIZE << "> 开始执行" << std::endl;
    std::cout << "DEBUG: grid_size = " << grid_size << ", unit_size = " << unit_size << std::endl;

    // 获取内核函数
    RearrangeParams params_copy = params; // 创建一个非const副本
    std::cout << "DEBUG: 开始获取rearrange内核函数" << std::endl;
    void *kernel_func = get_rearrange_kernel(params_copy);

    if (kernel_func == nullptr) {
        std::cerr << "ERROR: 无法获取内核函数，可能是参数不匹配" << std::endl;
        return INFINI_STATUS_BAD_PARAM;
    }
    std::cout << "DEBUG: 成功获取内核函数: " << kernel_func << std::endl;

    // 创建非const的临时变量
    unsigned int block_dim = params.block_dim;
    unsigned int block_len_total = params.block_len_total;

    std::cout << "DEBUG: 准备内核参数" << std::endl;
    std::cout << "  block_dim = " << block_dim << std::endl;
    std::cout << "  block_len_total = " << block_len_total << std::endl;
    std::cout << "  block_len = [";
    for (size_t i = 0; i < params.block_len.size(); i++) {
        std::cout << params.block_len[i];
        if (i < params.block_len.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "  src_block_stride = [";
    for (size_t i = 0; i < params.src_block_stride.size(); i++) {
        std::cout << params.src_block_stride[i];
        if (i < params.src_block_stride.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "  dst_block_stride = [";
    for (size_t i = 0; i < params.dst_block_stride.size(); i++) {
        std::cout << params.dst_block_stride[i];
        if (i < params.dst_block_stride.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "  grid_len = [";
    for (size_t i = 0; i < params.grid_len.size(); i++) {
        std::cout << params.grid_len[i];
        if (i < params.grid_len.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "  src_grid_stride = [";
    for (size_t i = 0; i < params.src_grid_stride.size(); i++) {
        std::cout << params.src_grid_stride[i];
        if (i < params.src_grid_stride.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "  dst_grid_stride = [";
    for (size_t i = 0; i < params.dst_grid_stride.size(); i++) {
        std::cout << params.dst_grid_stride[i];
        if (i < params.dst_grid_stride.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // 检查向量尺寸是否合理
    if (params.block_len.size() < block_dim || params.src_block_stride.size() < block_dim || params.dst_block_stride.size() < block_dim) {
        std::cerr << "ERROR: block维度参数vector大小不足" << std::endl;
        return INFINI_STATUS_BAD_PARAM;
    }

    if (params.grid_len.empty() || params.src_grid_stride.empty() || params.dst_grid_stride.empty()) {
        std::cerr << "ERROR: grid维度参数vector为空" << std::endl;
        return INFINI_STATUS_BAD_PARAM;
    }

    const Constrains<int> *constrains_data;
    auto empty_constrains = Constrains<int>();
    if (params.constrains.empty()) {
        constrains_data = &empty_constrains;
    } else {
        constrains_data = params.constrains.data();
    }

    void *args[]
        = {
            &y, &x,
            &block_dim,
            &block_len_total,
            const_cast<void *>(static_cast<const void *>(params.block_len.data())),
            const_cast<void *>(static_cast<const void *>(params.src_block_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.dst_block_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.grid_len.data())),
            const_cast<void *>(static_cast<const void *>(params.src_grid_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.dst_grid_stride.data())),
            const_cast<void *>(static_cast<const void *>(constrains_data))};

    std::cout << "DEBUG: 内核参数准备完成，开始启动内核" << std::endl;
    std::cout << "DEBUG: gridDim = " << grid_size << ", blockDim = " << BLOCK_SIZE << std::endl;

    try {
        CUDA_CHECK(cudaLaunchKernel(
            kernel_func,
            grid_size, BLOCK_SIZE,
            args, 0, stream));
        std::cout << "DEBUG: 内核启动命令已发送" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "EXCEPTION: 启动内核时发生异常: " << e.what() << std::endl;
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 添加调试信息 - 输入参数
    std::cout << "DEBUG: calculate() 开始执行" << std::endl;
    std::cout << "DEBUG: 输入指针 - y: " << y << ", x: " << x << ", stream: " << stream << std::endl;
    std::cout << "DEBUG: 元数据 - ndim: " << _meta.ndim() << ", unit: " << _meta.unit() << std::endl;

    if (_meta.ndim() > 0) {
        std::cout << "DEBUG: 步长信息:" << std::endl;
        std::cout << "  idx_strides: ";
        for (size_t i = 0; i < _meta.ndim(); ++i) {
            std::cout << _meta.idx_strides()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "  dst_strides: ";
        for (size_t i = 0; i < _meta.ndim(); ++i) {
            std::cout << _meta.dst_strides()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "  src_strides: ";
        for (size_t i = 0; i < _meta.ndim(); ++i) {
            std::cout << _meta.src_strides()[i] << " ";
        }
        std::cout << std::endl;
    }

    // 如果没有维度，直接进行内存拷贝
    if (_meta.ndim() == 0) {
        std::cout << "DEBUG: 没有维度，执行简单内存拷贝, 大小: " << _meta.unit() << std::endl;
        CUDA_CHECK(cudaMemcpyAsync(y, x, _meta.unit(), cudaMemcpyDeviceToDevice, cuda_stream));
        std::cout << "DEBUG: 内存拷贝完成" << std::endl;
        return INFINI_STATUS_SUCCESS;
    }

    // 准备参数
    std::cout << "DEBUG: 开始准备RearrangeParams参数" << std::endl;
    RearrangeParams params = prepareRearrangeParams(_meta);

    // 打印参数信息
    std::cout << "DEBUG: RearrangeParams准备完成:" << std::endl;
    std::cout << "  block_dim: " << params.block_dim << std::endl;
    std::cout << "  block_len_total: " << params.block_len_total << std::endl;

    std::cout << "  block_len: ";
    for (size_t i = 0; i < params.block_len.size(); ++i) {
        std::cout << params.block_len[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  src_block_stride: ";
    for (size_t i = 0; i < params.src_block_stride.size(); ++i) {
        std::cout << params.src_block_stride[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  dst_block_stride: ";
    for (size_t i = 0; i < params.dst_block_stride.size(); ++i) {
        std::cout << params.dst_block_stride[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  grid_len: ";
    for (size_t i = 0; i < params.grid_len.size(); ++i) {
        std::cout << params.grid_len[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  src_grid_stride: ";
    for (size_t i = 0; i < params.src_grid_stride.size(); ++i) {
        std::cout << params.src_grid_stride[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  dst_grid_stride: ";
    for (size_t i = 0; i < params.dst_grid_stride.size(); ++i) {
        std::cout << params.dst_grid_stride[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  约束数量: " << params.constrains.size() << std::endl;
    for (size_t i = 0; i < params.constrains.size(); ++i) {
        std::cout << "    约束 " << i << ": "
                  << "grid_idx=" << params.constrains[i].grid_idx
                  << ", block_idx=" << params.constrains[i].block_idx
                  << ", grid_div_block=" << params.constrains[i].grid_div_block
                  << ", total_len=" << params.constrains[i].total_len << std::endl;
    }

    // 计算grid大小
    unsigned int grid_size = 1;
    std::cout << "DEBUG: 计算grid大小" << std::endl;
    for (size_t i = 0; i < params.grid_len.size(); ++i) {
        std::cout << "  grid_len[" << i << "] = " << params.grid_len[i] << std::endl;
        grid_size *= params.grid_len[i];
    }
    std::cout << "DEBUG: 计算得到的grid_size = " << grid_size << std::endl;

    // 检查grid大小是否为0
    if (grid_size == 0) {
        std::cerr << "ERROR: grid_size为0，无法启动内核" << std::endl;
        return INFINI_STATUS_BAD_PARAM;
    }

    // 获取设备属性
    int max_threads = _opaque->internal->maxThreadsPerBlock();
    std::cout << "DEBUG: 设备最大线程数: " << max_threads << std::endl;

    // 根据设备属性选择合适的内核
    infiniStatus_t status = INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    if (max_threads >= 1024) {
        std::cout << "DEBUG: 选择启动1024线程内核" << std::endl;
        status = launchKernel<1024>(y, x, grid_size, params, _meta.unit(), cuda_stream);
    } else if (max_threads >= 512) {
        std::cout << "DEBUG: 选择启动512线程内核" << std::endl;
        status = launchKernel<512>(y, x, grid_size, params, _meta.unit(), cuda_stream);
    } else if (max_threads >= 256) {
        std::cout << "DEBUG: 选择启动256线程内核" << std::endl;
        status = launchKernel<256>(y, x, grid_size, params, _meta.unit(), cuda_stream);
    } else {
        std::cerr << "ERROR: 设备不支持所需的线程数" << std::endl;
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    // 检查内核启动状态
    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "ERROR: 内核启动失败，错误码: " << status << std::endl;
    } else {
        std::cout << "DEBUG: 内核启动成功" << std::endl;

        // 检查CUDA错误
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(cuda_err) << std::endl;
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }

    std::cout << "DEBUG: calculate()执行完毕，返回状态: " << status << std::endl;
    return status;
}

} // namespace op::rearrange::cuda