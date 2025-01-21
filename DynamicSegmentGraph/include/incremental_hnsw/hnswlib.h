/**
 * @file hnswlib.h
 * @brief Incremental Hierarchical Navigable Small World Graphs library header.
 */

#pragma once

#ifndef HNSW_INCRE_
#define HNSW_INCRE_

#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
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
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

#include <iostream>
#include <queue>
#include <string.h>
#include <vector>

using std::vector;

namespace hnswlib_incre
{

    /**
     * @typedef labeltype
     * @brief Type definition for labels used to identify data points.
     */
    typedef size_t labeltype;

    /**
     * @class pairGreater<T>
     * @brief A comparator class that compares pairs based on their first element.
     *
     * @tparam T The type of the elements being compared.
     */
    template <typename T>
    class pairGreater
    {
    public:
        bool operator()(const T &p1, const T &p2) { return p1.first > p2.first; }
    };

    template <typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef)
    {
        out.write((char *)&podRef, sizeof(T));
    }

    template <typename T>
    static void readBinaryPOD(std::istream &in, T &podRef)
    {
        in.read((char *)&podRef, sizeof(T));
    }

    template <typename MTYPE>
    using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);

    /**
     * @class SpaceInterface<MTYPE>
     * @brief Interface for space-specific operations and distance calculations.
     *
     * @tparam MTYPE Metric type used for distance calculations.
     */
    template <typename MTYPE>
    class SpaceInterface
    {
    public:
        virtual size_t get_data_size() = 0;
        virtual DISTFUNC<MTYPE> get_dist_func() = 0;
        virtual void *get_dist_func_param() = 0;
        virtual ~SpaceInterface() {}
    };

    /**
     * @class AlgorithmInterface<dist_t>
     * @brief Interface for algorithms implementing hierarchical graph-based search.
     *
     * @tparam dist_t Distance type used for comparisons.
     */
    template <typename dist_t>
    class AlgorithmInterface
    {
    public:
        void linkNeighbors(const void *data_point, labeltype label,
                           vector<int> neighbors) {}
        void addNeighborPoint(const void *data_point, labeltype label, int level) {}
        virtual void addPoint(const void *datapoint, labeltype label) = 0;
        virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnnEF(const void *, size_t, const int lbound, const int rbound, const int K_query, const bool fixed_ef) const = 0;

        // Return k nearest neighbor in the order of closer fist
        virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void *query_data, size_t k, const int lbound,
                             const int rbound) const;

        virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void *query_data, size_t k, const int lbound,
                             const int rbound, const bool fixed_ef) const;

        virtual void saveIndex(const std::string &location) = 0;
        virtual ~AlgorithmInterface() {}
    };

    template <typename dist_t>
    std::vector<std::pair<dist_t, labeltype>>
    AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void *query_data,
                                                     size_t k, const int lbound,
                                                     const int rbound) const
    {
        std::vector<std::pair<dist_t, labeltype>> result;

        // here searchKnn returns the result in the order of further first
        auto ret = searchKnnEF(query_data, k, lbound, rbound, (int)k, false);
        {
            size_t sz = ret.size();
            result.resize(sz);
            while (!ret.empty())
            {
                result[--sz] = ret.top();
                ret.pop();
            }
        }

        return result;
    }

    template <typename dist_t>
    std::vector<std::pair<dist_t, labeltype>>
    AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void *query_data,
                                                     size_t k, const int lbound,
                                                     const int rbound, const bool fixed_ef) const
    {
        std::vector<std::pair<dist_t, labeltype>> result;

        // here searchKnn returns the result in the order of further first
        auto ret = searchKnnEF(query_data, k, lbound, rbound, (int)k, fixed_ef);
        {
            size_t sz = ret.size();
            result.resize(sz);
            while (!ret.empty())
            {
                result[--sz] = ret.top();
                ret.pop();
            }
        }

        return result;
    }

} // namespace hnswlib_incre

#include "space_ip.h"
#include "space_l2.h"
#include "hnswalg.h"

#endif