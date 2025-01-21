/**
 * @file utils.h
 * @brief 提供了一系列实用工具函数。
 */

#pragma once

#include <assert.h>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#elif __APPLE__
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#endif

// 使用标准库中的命名空间元素
using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::make_pair;
using std::pair;
using std::string;
using std::vector;

/**
 * 计算两个向量之间的欧几里得距离。
 *
 * @param lhs 左侧向量。
 * @param rhs 右侧向量。
 * @param startDim 起始维度。
 * @param lensDim 向量长度。
 * @return 欧几里得距离。
 */
float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs,
                        const int &startDim, int lensDim);

/**
 * 简化版计算两个向量之间的欧几里得距离。
 *
 * @param lhs 左侧向量。
 * @param rhs 右侧向量。
 * @return 欧几里得距离。
 */
float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs);

/**
 * 计算两个向量之间的欧几里得距离平方。
 *
 * @param lhs 左侧向量。
 * @param rhs 右侧向量。
 * @return 欧几里得距离平方。
 */
float EuclideanDistanceSquare(const vector<float> &lhs,
                              const vector<float> &rhs);

/**
 * 积累时间差值。
 *
 * @param t2 结束时间。
 * @param t1 开始时间。
 * @param val_time 时间差值。
 */
void AccumulateTime(timeval &t2, timeval &t1, double &val_time);

/**
 * 计算并记录时间差值。
 *
 * @param t1 开始时间。
 * @param t2 结束时间。
 * @param val_time 时间差值。
 */
void CountTime(timeval &t1, timeval &t2, double &val_time);

/**
 * 返回两次时间测量的时间差值。
 *
 * @param t1 开始时间。
 * @param t2 结束时间。
 * @return 时间差值。
 */
double CountTime(timeval &t1, timeval &t2);

// the same to sort_indexes
template <typename T>
std::vector<std::size_t> sort_permutation(const std::vector<T> &vec)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
              [&](std::size_t i, std::size_t j)
              { return vec[i] < vec[j]; });
    return p;
}

// apply permutation
template <typename T>
void apply_permutation_in_place(std::vector<T> &vec,
                                const std::vector<std::size_t> &p)
{
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (done[i])
        {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j)
        {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

template <typename T>
vector<int> sort_indexes(const vector<T> &v)
{
    // initialize original index locations
    vector<int> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2)
                { return v[i1] < v[i2]; });

    return idx;
}

template <typename T>
vector<int> sort_indexes(const vector<T> &v, const int begin_bias,
                         const int end_bias)
{
    // initialize original index locations
    vector<int> idx(end_bias - begin_bias);
    iota(idx.begin() + begin_bias, idx.begin() + end_bias, 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin() + begin_bias, idx.begin() + end_bias,
                [&v](size_t i1, size_t i2)
                { return v[i1] < v[i2]; });

    return idx;
}

template <typename T>
void print_set(const vector<T> &v)
{
    if (v.size() == 0)
    {
        cout << "ERROR: EMPTY VECTOR!" << endl;
        return;
    }
    cout << "vertex in set: {";
    for (size_t i = 0; i < v.size() - 1; i++)
    {
        cout << v[i] << ", ";
    }
    cout << v.back() << "}" << endl;
}

void logTime(timeval &begin, timeval &end, const string &log);

/**
 * 计算精度。
 *
 * @param truth 真实结果。
 * @param pred 预测结果。
 * @return 精度值。
 */
double countPrecision(const vector<int> &truth, const vector<int> &pred);

/**
 * 计算近似比率。
 *
 * @param raw_data 原始数据集。
 * @param truth 真实结果。
 * @param pred 预测结果。
 * @param query 查询点。
 * @return 近似比率。
 */
double countApproximationRatio(const vector<vector<float>> &raw_data,
                               const vector<int> &truth,
                               const vector<int> &pred,
                               const vector<float> &query);

/**
 * 打印内存使用情况。
 */
void print_memory();

/**
 * 记录当前内存使用情况。
 *
 * @param memory 当前内存使用量引用。
 */
void record_memory(long long &memory);

#define _INT_MAX 2147483640

/**
 * 贪婪算法寻找最近邻。
 *
 * @param dpts 数据点集。
 * @param query 查询点。
 * @param k_smallest 寻找的最小数量。
 * @return 最近邻点的索引列表。
 */
vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query, const int k_smallest);

// void evaluateKNNG(const vector<vector<int>> &gt,
//                   const vector<vector<int>> &knng, const int K, double
//                   &recall, double &precision);

void rangeGreedy(const vector<vector<float>> &nodes, const int k_smallest,
                 const int l_bound, const int r_bound);

void greedyNearest(const int query_pos, const vector<vector<float>> &dpts,
                   const int k_smallest, const int l_bound, const int r_bound);

vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query, const int l_bound,
                          const int r_bound, const int k_smallest);

void heuristicPrune(const vector<vector<float>> &nodes,
                    vector<pair<int, float>> &top_candidates, const size_t M);

vector<int> str2vec(const string str);