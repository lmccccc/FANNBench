#ifndef SORTEDARRAY_H
#define SORTEDARRAY_H

#include <vector>
#include <algorithm>
#include <limits>

// 虚类 SortedArrayBase
class SortedArrayBase
{
protected:
    unsigned max_size;

public:
    std::vector<unsigned> sorted_arr;

    explicit SortedArrayBase(unsigned _max_size) : max_size(_max_size) {}

    virtual ~SortedArrayBase() = default;

    virtual bool addPoint(unsigned point_id, unsigned &ret_rank) = 0;

    virtual unsigned getBackBound() = 0;

    virtual unsigned getSpecifiedBound(unsigned rank) = 0;

    void clear() { sorted_arr.clear(); }
};

// 维护最小值数组的类 MinSortedArray
class MinSortedArray : public SortedArrayBase
{
public:
    MinSortedArray(unsigned _max_size) : SortedArrayBase(_max_size) {}

    bool addPoint(unsigned point_id, unsigned &ret_rank) override
    {
        // 获取当前数组的最大值
        unsigned min_id = std::numeric_limits<unsigned>::max();
        if (!sorted_arr.empty())
        {
            min_id = sorted_arr.front();
        }

        if (sorted_arr.size() < max_size || point_id <= min_id)
        {
            // 确定插入位置
            auto it = std::lower_bound(sorted_arr.begin(), sorted_arr.end(), point_id);

            // 计算插入后的位置（即排名第几）
            ret_rank = std::distance(sorted_arr.begin(), it) + 1;

            // 如果数组未满或者新点小于等于当前最小点，则插入新点
            sorted_arr.insert(it, point_id);

            // 如果数组超过最大容量，则删除最后一个元素（最大值）
            if (sorted_arr.size() > max_size)
            {
                sorted_arr.pop_back();
            }

            // 成功插入新点
            return true;
        }

        // 无法添加新点
        ret_rank = std::numeric_limits<unsigned>::max();
        return false;
    }

    // For right part, boundary should be minus 1
    unsigned getBackBound()
    {
        return sorted_arr.back() - 1;
    }

    unsigned getSpecifiedBound(unsigned rank)
    {
        if (rank > sorted_arr.size())
            return getBackBound();
        return sorted_arr[rank - 1] - 1;
    }
};

// 维护最大值数组的类 MaxSortedArray
class MaxSortedArray : public SortedArrayBase
{
public:
    MaxSortedArray(unsigned _max_size) : SortedArrayBase(_max_size) {}

    bool addPoint(unsigned point_id, unsigned &ret_rank) override
    {
        // 获取当前数组的最大值
        unsigned max_id = std::numeric_limits<unsigned>::min();
        if (!sorted_arr.empty())
        {
            max_id = sorted_arr.back();
        }

        if (sorted_arr.size() < max_size || point_id >= max_id)
        {
            // 自定义比较函数以降序排列
            auto comp_desc = [](unsigned x, unsigned y)
            { return x > y; };

            // 确定插入位置（降序排列）
            auto it = std::upper_bound(sorted_arr.begin(), sorted_arr.end(), point_id, comp_desc);

            // 计算插入后的位置（即排名第几）
            ret_rank = std::distance(sorted_arr.begin(), it) + 1;

            // 如果数组未满或者新点大于等于当前最大值，则插入新点
            sorted_arr.insert(it, point_id);

            // 如果数组超过最大容量，则删除最后一个元素（最小值）
            if (sorted_arr.size() > max_size)
            {
                sorted_arr.pop_back();
            }

            // 成功插入新点
            return true;
        }

        // 无法添加新点
        ret_rank = std::numeric_limits<unsigned>::max();
        return false;
    }

    // For left part, boundary should be add 1
    unsigned getBackBound()
    {
        return sorted_arr.back() == 0 ? 0 : sorted_arr.back() + 1;
    }

    unsigned getSpecifiedBound(unsigned rank)
    {
        if (rank >= sorted_arr.size())
            return getBackBound();
        return sorted_arr[rank - 1] + 1;
    }
};

#endif // SORTEDARRAY_H
