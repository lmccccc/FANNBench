/**
 * @file hnswlib.h
 * @brief Header file for Hierarchical Navigable Small World (HNSW) library.
 */

#pragma once

#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
// 定义使用SSE指令集
// #define USE_SSE
#ifdef __AVX__
// 定义使用AVX指令集
// #define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>

#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32))) ///< 对齐宏定义（GCC）
#else
#define PORTABLE_ALIGN32 __declspec(align(32)) ///< 对齐宏定义（MSVC）
#endif
#endif

#include <string.h>

#include <iostream>
#include <queue>
#include <vector>

namespace base_hnsw
{

    /**
     * @brief 类型定义：标签类型为size_t
     */
    typedef size_t labeltype;

    /**
     * @brief 模板类：用于比较pair类型的元素
     *
     * @tparam T 要比较的pair类型
     */
    template <typename T>
    class pairGreater
    {
    public:
        /**
         * @brief 比较两个pair元素的第一个成员是否大于
         *
         * @param p1 第一个pair元素
         * @param p2 第二个pair元素
         * @return true 如果第一个元素大于第二个元素
         * @return false 否则
         */
        bool operator()(const T &p1, const T &p2) { return p1.first > p2.first; }
    };

    /**
     * @brief 写入基本数据类型到二进制流
     *
     * @tparam T 基本数据类型
     * @param out 输出流
     * @param podRef 数据引用
     */
    template <typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef)
    {
        out.write((char *)&podRef, sizeof(T));
    }

    /**
     * @brief 从二进制流读取基本数据类型
     *
     * @tparam T 基本数据类型
     * @param in 输入流
     * @param podRef 数据引用
     */
    template <typename T>
    static void readBinaryPOD(std::istream &in, T &podRef)
    {
        in.read((char *)&podRef, sizeof(T));
    }

    /**
     * @brief 距离计算函数类型定义
     *
     * @tparam MTYPE 计算结果的数据类型
     */
    template <typename MTYPE>
    using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);

    /**
     * @brief 空间接口类模板
     *
     * @tparam MTYPE 计算结果的数据类型
     */
    template <typename MTYPE>
    class SpaceInterface
    {
    public:
        /**
         * @brief 获取数据大小
         *
         * @return size_t 数据大小
         */
        virtual size_t get_data_size() = 0;

        /**
         * @brief 获取距离计算函数
         *
         * @return DISTFUNC<MTYPE> 距离计算函数指针
         */
        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        /**
         * @brief 获取距离计算函数参数
         *
         * @return void* 函数参数
         */
        virtual void *get_dist_func_param() = 0;

        virtual ~SpaceInterface() {} ///< 析构函数
    };

    /**
     * @brief 算法接口类模板
     *
     * @tparam dist_t 距离数据类型
     */
    template <typename dist_t>
    class AlgorithmInterface
    {
    public:
        /**
         * @brief 链接邻居节点
         *
         * @param data_point 数据点
         * @param label 标签
         * @param neighbors 邻居列表
         */
        void linkNeighbors(const void *data_point, labeltype label, vector<int> neighbors) {}

        /**
         * @brief 添加邻居点
         *
         * @param data_point 数据点
         * @param label 标签
         * @param level 层级
         */
        void addNeighborPoint(const void *data_point, labeltype label, int level) {}

        /**
         * @brief 添加数据点
         *
         * @param datapoint 数据点
         * @param label 标签
         */
        virtual void addPoint(const void *datapoint, labeltype label) = 0;

        /**
         * @brief K近邻搜索
         *
         * @param query 查询数据
         * @param k 近邻数量
         * @return std::priority_queue<std::pair<dist_t, labeltype>> 结果队列
         */
        virtual std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(const void *, size_t) const = 0;

        /**
         * @brief K近邻搜索，返回按距离递增排序的结果
         *
         * @param query_data 查询数据
         * @param k 近邻数量
         * @return std::vector<std::pair<dist_t, labeltype>> 结果向量
         */
        virtual std::vector<std::pair<dist_t, labeltype>> searchKnnCloserFirst(const void *query_data, size_t k) const;

        /**
         * @brief 将索引保存至指定位置
         *
         * @param location 文件路径
         */
        virtual void saveIndex(const std::string &location) = 0;

        virtual ~AlgorithmInterface() {} ///< 析构函数
    };

    /**
     * @brief 实现K近邻搜索，返回按距离递增排序的结果
     *
     * @tparam dist_t 距离数据类型
     * @param query_data 查询数据
     * @param k 近邻数量
     * @return std::vector<std::pair<dist_t, labeltype>> 结果向量
     */
    template <typename dist_t>
    std::vector<std::pair<dist_t, labeltype>> AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void *query_data, size_t k) const
    {
        std::vector<std::pair<dist_t, labeltype>> result;

        // 执行K近邻搜索并获取结果（默认按距离递减排序）
        auto ret = searchKnn(query_data, k);

        // 反转结果顺序以实现按距离递增排序
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty())
        {
            result[--sz] = ret.top();
            ret.pop();
        }

        return result;
    }

} // namespace base_hnsw

#include "bruteforce.h"
#include "hnswalg.h"
#include "space_ip.h"
#include "space_l2.h"
