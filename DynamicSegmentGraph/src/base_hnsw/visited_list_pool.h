/**
 * @file visited_list_pool.h
 * @brief 定义了管理VisitedList实例的线程安全池类
 */

#pragma once

#include <string.h>
#include <mutex>
#include <deque>

namespace base_hnsw
{

    /**
     * @brief 存储访问过的元素列表
     *
     * 这个类用于存储一组被访问过的元素，其中每个元素用vl_type表示。
     */
    typedef unsigned short int vl_type;

    class VisitedList
    {
    public:
        /**
         * @brief 当前元素索引
         */
        vl_type curV;

        /**
         * @brief 元素数组
         */
        vl_type *mass;

        /**
         * @brief 元素数量
         */
        unsigned int numelements;

        VisitedList(int numelements1)
        {
            curV = -1;
            numelements = numelements1;
            mass = new vl_type[numelements];
        }

        void reset()
        {
            curV++;
            if (curV == 0)
            {
                memset(mass, 0, sizeof(vl_type) * numelements);
                curV++;
            }
        };

        ~VisitedList() { delete[] mass; }
    };
    ///////////////////////////////////////////////////////////
    //
    // Class for multi-threaded pool-management of VisitedLists
    //
    /////////////////////////////////////////////////////////

    class VisitedListPool
    {
    private:
        /**
         * @brief 可用的VisitedList实例队列
         */
        std::deque<VisitedList *> pool;

        /**
         * @brief 锁保护池操作
         */
        std::mutex poolguard;

        /**
         * @brief 每个VisitedList实例的元素数量
         */
        int numelements;

    public:
        /**
         * @brief 构造函数
         * @param initmaxpools 初始最大池大小
         * @param numelements1 每个VisitedList实例的元素数量
         */
        VisitedListPool(int initmaxpools, int numelements1)
        {
            numelements = numelements1;
            for (int i = 0; i < initmaxpools; i++)
                pool.push_front(new VisitedList(numelements));
        }

        /**
         * @brief 获取一个空闲的VisitedList实例
         * @return 返回一个空闲的VisitedList指针
         */
        VisitedList *getFreeVisitedList()
        {
            VisitedList *rez;
            {
                std::unique_lock<std::mutex> lock(poolguard);
                if (pool.size() > 0)
                {
                    rez = pool.front();
                    pool.pop_front();
                }
                else
                {
                    rez = new VisitedList(numelements);
                }
            }
            rez->reset();
            return rez;
        };

        /**
         * @brief 释放一个VisitedList实例回池
         * @param vl 需要释放的VisitedList指针
         */
        void releaseVisitedList(VisitedList *vl)
        {
            std::unique_lock<std::mutex> lock(poolguard);
            pool.push_front(vl);
        };

        /**
         * @brief 析构函数
         */
        ~VisitedListPool()
        {
            while (pool.size())
            {
                VisitedList *rez = pool.front();
                pool.pop_front();
                delete rez;
            }
        };
    };
} // namespace hnswlib_compose
