#pragma once
#include <algorithm>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace base_hnsw
{
    template <typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t>
    {
    public:
        BruteforceSearch(SpaceInterface<dist_t> *s) {}
        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
        {
            loadIndex(location, s);
        }

        BruteforceSearch(SpaceInterface<dist_t> *s, size_t maxElements)
        {
            maxelements_ = maxElements;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *)malloc(maxElements * size_per_element_);
            if (data_ == nullptr)
                std::runtime_error(
                    "Not enough memory: BruteforceSearch failed to allocate data");
            cur_element_count = 0;
        }

        ~BruteforceSearch() { free(data_); }

        /**
         * @brief 数据存储区指针，用于保存元素数据
         */
        char *data_;

        /**
         * @brief 最大可容纳元素数量
         */
        size_t maxelements_;

        /**
         * @brief 当前已存储的元素数量
         */
        size_t cur_element_count;

        /**
         * @brief 每个元素所占的大小（单位：字节）
         */
        size_t size_per_element_;

        /**
         * @brief 整体数据占用的空间大小
         */
        size_t data_size_;

        /**
         * @brief 距离计算函数模板，用于计算两个元素之间的距离
         */
        DISTFUNC<dist_t> fstdistfunc_;

        /**
         * @brief 距离计算函数的附加参数
         */
        void *dist_func_param_;

        /**
         * @brief 线程互斥锁，保证多线程环境下数据操作的安全性
         */
        std::mutex index_lock;

        /**
         * @brief 外部标签到内部索引的映射表，便于快速查找
         */
        std::unordered_map<labeltype, size_t> dict_external_to_internal;

        /**
         * 向索引中添加一个数据点及其标签。
         *
         * @param datapoint 指向要添加的数据点的指针。
         * @param label 数据点对应的外部标签。
         */
        void addPoint(const void *datapoint, labeltype label)
        {
            int idx;
            {
                // 加锁以保护共享资源访问
                std::unique_lock<std::mutex> lock(index_lock);

                // 查找是否已存在该外部标签到内部索引的映射
                auto search = dict_external_to_internal.find(label);
                if (search != dict_external_to_internal.end())
                {
                    idx = search->second;
                }
                else
                {
                    // 如果当前元素数量达到最大限制，则抛出异常
                    if (cur_element_count >= maxelements_)
                    {
                        throw std::runtime_error("The number of elements exceeds the specified limit\n");
                    }
                    // 分配新的内部索引并更新映射关系
                    idx = cur_element_count;
                    dict_external_to_internal[label] = idx;
                    cur_element_count++;
                }
            }
            // 将标签存储在数据结构中的特定位置
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
            // 复制数据点至数据结构中对应的位置
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
        };

        /**
         * 从索引中移除指定标签的数据点。
         *
         * @param cur_external 要移除的数据点的外部标签。
         */
        void removePoint(labeltype cur_external)
        {
            // 获取待删除数据点的当前内部索引
            size_t cur_c = dict_external_to_internal[cur_external];

            // 移除外部标签到内部索引的映射
            dict_external_to_internal.erase(cur_external);

            // 获取最后一个数据点的标签
            labeltype label = *((labeltype *)(data_ + size_per_element_ * (cur_element_count - 1) + data_size_));
            // 更新最后一个数据点的外部标签到内部索引的映射
            dict_external_to_internal[label] = cur_c;

            // 将最后一个数据点移动到待删除数据点的位置
            memcpy(data_ + size_per_element_ * cur_c, data_ + size_per_element_ * (cur_element_count - 1), data_size_ + sizeof(labeltype));

            // 减少当前元素计数
            cur_element_count--;
        }

        /**
         * @brief 搜索最近邻点
         *
         * 此函数用于搜索数据集中距离给定查询点最近的k个点。
         *
         * @param query_data 查询点的数据指针
         * @param k 需要返回的最近邻点的数量
         * @return 返回一个优先队列，其中包含k个最近邻点的距离及其标签
         */
        std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
            const void *query_data, size_t k) const
        {
            // 初始化存储结果的优先队列
            std::priority_queue<std::pair<dist_t, labeltype>> topResults;

            // 如果当前元素数量为零，则直接返回空的结果集
            if (cur_element_count == 0)
                return topResults;

            // 计算前k个元素到查询点的距离并加入结果队列
            for (int i = 0; i < k; i++)
            {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i,
                                           dist_func_param_);

                // 获取当前元素的标签并创建距离标签对
                topResults.push(std::make_pair(dist,
                                               *(reinterpret_cast<labeltype *>(data_ + size_per_element_ * i + data_size_))));
            }

            // 获取队列顶部元素的距离作为参考值
            dist_t lastdist = topResults.top().first;

            // 继续计算剩余元素到查询点的距离
            for (int i = k; i < cur_element_count; i++)
            {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i,
                                           dist_func_param_);

                // 如果新计算的距离小于等于队列中的最大距离
                if (dist <= lastdist)
                {
                    // 将新的距离标签对加入队列
                    topResults.push(std::make_pair(dist,
                                                   *(reinterpret_cast<labeltype *>(data_ + size_per_element_ * i + data_size_))));

                    // 如果队列大小超过k，则移除最大的距离
                    if (topResults.size() > k)
                        topResults.pop();

                    // 更新队列顶部元素的距离作为新的参考值
                    lastdist = topResults.top().first;
                }
            }

            // 返回最终结果
            return topResults;
        };

        void saveIndex(const std::string &location)
        {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, maxelements_ * size_per_element_);

            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s)
        {
            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *)malloc(maxelements_ * size_per_element_);
            if (data_ == nullptr)
                std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate data");

            input.read(data_, maxelements_ * size_per_element_);

            input.close();
        }
    };
} // namespace hnswlib_compose
