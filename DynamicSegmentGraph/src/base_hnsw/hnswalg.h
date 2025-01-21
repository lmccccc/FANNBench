#pragma once

#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <list>
#include <random>
#include <unordered_set>

#include "hnswlib.h"
#include "visited_list_pool.h"

using std::vector;

// ef_max_ -> original ef_construction_ for search neighbors before pruning
// ef_for_pruning -> top in ef_max, actual ef_construction

namespace base_hnsw
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template <typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t>
    {
    public:
        /**
         * 定义HierarchicalNSW类的最大更新元素锁数量常量。
         */
        static const tableint max_update_element_locks = 65536;

        /**
         * 构造一个HierarchicalNSW对象，用于近似最近邻搜索算法。
         *
         * @param s 空间接口指针，提供距离计算方法。
         */
        HierarchicalNSW(SpaceInterface<dist_t> *s) {}

        /**
         * 构造一个HierarchicalNSW对象并从指定位置加载索引。
         *
         * @param s 空间接口指针，提供距离计算方法。
         * @param location 存储索引的位置字符串。
         * @param nmslib 是否使用NMSLIB库，默认为false。
         * @param max_elements 最大元素数量，默认为0（无限制）。
         */
        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location,
                        bool nmslib = false, size_t max_elements = 0)
        {
            loadIndex(location, s, max_elements);
        }

        /**
         * 构造一个HierarchicalNSW对象，初始化参数以构建新的索引结构。
         *
         * @param s 空间接口指针，提供距离计算方法。
         * @param max_elements 预期的最大元素数量。
         * @param M 初始连接列表大小，默认为16。
         * @param ef_construction 建立索引时的效率因子，默认为200。
         * @param random_seed 随机数种子，默认为100。
         */
        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16,
                        size_t ef_construction = 200, size_t random_seed = 100)
            : link_list_locks_(max_elements),
              link_list_update_locks_(max_update_element_locks),
              element_levels_(max_elements)
        {
            max_elements_ = max_elements;

            has_deletions_ = false;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_;
            ef_basic_construction_ =
                std::max(ef_construction, M_); // 或原始的ef_construction值
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = (maxM0_) * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ =
                size_links_level0_ + data_size_ + sizeof(labeltype); // 在最底下一层里原来HNSW里面vector的数据是和他的领结列表放在一起的
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ =
                (char *)malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("内存不足");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            // 对第一个节点的特殊处理初始化
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error(
                    "内存不足：HierarchicalNSW分配链接列表失败");
            size_links_per_element_ =
                maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;

            is_recursively_add = false;
        }

        struct CompareByFirst
        {
            constexpr bool operator()(
                std::pair<dist_t, tableint> const &a,
                std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW()
        {
            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++)
            {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        // for range hnsw

        bool is_recursively_add;
        size_t recursive_limitation = 25;
        bool is_two_way = true;
        size_t range_add_type = 0;
        // 0: original(x), 1: single-way, 2: two-way-simple 3: two-way-round, 4:
        // two-way-pair-windows
        size_t ef_max_;
        size_t ef_basic_construction_;

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;

        double mult_, revSize_;
        int maxlevel_;

        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at
        // same time. Note: Locks for additions can also be used to prevent this
        // race condition if the querying of KNN is not exposed along with
        // update/inserts i.e multithread insert/update/query in parallel.

        // 线程锁向量，用于链接列表更新操作的互斥访问
        std::vector<std::mutex> link_list_update_locks_;
        // 入口节点标识符
        tableint enterpoint_node_;

        // 第0级链表大小
        size_t size_links_level0_;
        // 数据偏移量，分别对应第0级数据和链表数据
        size_t offsetData_, offsetLevel0_;

        // 分配给第0级数据的内存指针
        char *data_level0_memory_;
        // 链接列表数组指针
        char **linkLists_;
        // 存储element 有多少layer的向量
        std::vector<int> element_levels_;

        // 单个数据项的大小
        size_t data_size_;

        // 标记是否发生过删除操作
        bool has_deletions_;

        // 标签偏移量
        size_t label_offset_;
        // 距离计算函数
        DISTFUNC<dist_t> fstdistfunc_;
        // 距离计算函数参数
        void *dist_func_param_;
        // 标签查找映射表
        std::unordered_map<labeltype, tableint> label_lookup_;

        // 默认随机数引擎，用于层级生成
        std::default_random_engine level_generator_;
        // 默认随机数引擎，用于更新概率生成
        std::default_random_engine update_probability_generator_;

        /**
         * 根据内部ID获取外部标签值。
         *
         * @param internal_id 内部标识符
         * @return 返回对应的外部标签值
         */
        inline labeltype getExternalLabel(tableint internal_id) const
        {
            labeltype return_label;
            memcpy(&return_label,
                   (data_level0_memory_ + internal_id * size_data_per_element_ +
                    label_offset_),
                   sizeof(labeltype));
            return return_label;
        }

        /**
         * 设置指定内部ID的外部标签值。
         *
         * @param internal_id 内部标识符
         * @param label 要设置的标签值
         */
        inline void setExternalLabel(tableint internal_id, labeltype label) const
        {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ +
                    label_offset_),
                   &label, sizeof(labeltype));
        }

        /**
         * 获取指向特定内部ID的外部标签指针。
         *
         * @param internal_id 内部标识符
         * @return 指向对应外部标签的指针
         */
        inline labeltype *getExternalLabeLp(tableint internal_id) const
        {
            return (labeltype *)(data_level0_memory_ +
                                 internal_id * size_data_per_element_ + label_offset_);
        }

        /**
         * 根据内部ID获取数据指针。
         *
         * @param internal_id 内部标识符
         * @return 数据的起始地址
         */
        inline char *getDataByInternalId(tableint internal_id) const
        {
            return (data_level0_memory_ + internal_id * size_data_per_element_ +
                    offsetData_);
        }

        inline char *getDataByLabel(labeltype label) const
        {   
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            return getDataByInternalId(label_c);
        }

        inline tableint getInnerIdByLabel(labeltype label) const
        {   
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            return label_c;
        }
        /**
         * 随机生成一个层级。
         *
         * @param reverse_size 反转大小参数
         * @return 生成的随机层级
         */
        int getRandomLevel(double reverse_size)
        {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int)r;
        }

        /**
         * @file hnswalg.h
         * @brief 搜索基础层第0级的实现方法
         */

        /**
         * @brief 在HNSW算法的基础层搜索指定层级上的最近邻点
         *
         * 此函数用于从HNSW索引结构的基础层开始，在给定的层级上执行近似最近邻搜索。
         *
         * @param ep_id 起始节点的内部ID
         * @param data_point 查询数据点
         * @param layer 执行搜索的层级
         * @return 返回一个优先队列，其中包含距离和对应的节点ID对，按距离排序
         */
        virtual std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
        searchBaseLayerLevel0(tableint ep_id, const void *data_point, int layer)
        {
            // 获取空闲的访问列表
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // 初始化候选集和顶点集合
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                candidateSet;

            // 设置扩展因子（EF）
            size_t ef_construction = layer ? ef_basic_construction_ : ef_max_;

            // 计算起始点的距离并初始化候选集
            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                           dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }

            // 标记起始点为已访问
            visited_array[ep_id] = visited_array_tag;

            // 主循环：遍历候选集直到为空
            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound)
                {
                    break;
                }
                candidateSet.pop();

                // 获取当前节点信息
                tableint curNodeNum = curr_el_pair.second;

                // 加锁以保证线程安全
                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                // 获取链接列表数据
                int *data;
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                }

                // 处理链接列表中的每个元素
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                // 预取指令优化性能
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                    // 预取指令优化性能
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    // 如果节点已被访问，则跳过
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;

                    // 计算查询点到候选点的距离
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    // 更新候选集和顶点集合
                    if (top_candidates.size() < ef_construction || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        // 预取指令优化性能
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                                     _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }

            // 释放访问列表资源
            visited_list_pool_->releaseVisitedList(vl);

            // 返回最终候选集
            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                candidateSet;

            size_t ef_construction = layer ? ef_basic_construction_ : ef_max_;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                           dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound)
                {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum *
                           // size_links_per_element0_);
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer
                    //                    - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                                     _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        /**
         * @brief 计算距离计算次数的原子计数器
         */
        mutable std::atomic<long> metric_distance_computions;
        /**
         * @brief 计算跳转次数的原子计数器
         */
        mutable std::atomic<long> metric_hops;

        /**
         * @brief 搜索基础层中的节点
         *
         * 此模板函数用于搜索基础层结构中的最近邻节点，根据给定的数据点和扩展因子(ef)，
         * 返回一个优先队列，其中包含候选节点的距离及其ID。
         *
         * @param ep_id 起始探索节点的ID。
         * @param data_point 数据点指针，用于比较距离。
         * @param ef 扩展因子，控制返回结果的数量。
         * @return std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
         *         包含距离和节点ID的优先队列，按距离排序。
         */
        template <bool has_deletions, bool collect_metrics = false>
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const
        {

            // 获取空闲访问列表
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // 初始化两个优先队列：顶部候选者和候选集
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                top_candidates, candidate_set;

            // 计算下界距离值
            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                           dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            // 标记起始节点为已访问
            visited_array[ep_id] = visited_array_tag;

            // 主循环处理候选节点
            while (!candidate_set.empty())
            {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                if ((-current_node_pair.first) > lowerBound)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);

                // 收集度量信息（可选）
                if (collect_metrics)
                {
                    metric_hops++;
                    metric_distance_computions += size;
                }

                // 预取优化（SSE指令集支持）
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ +
                                 (*(data + 1)) * size_data_per_element_ + offsetData_,
                             _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                // 遍历链接列表
                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ +
                                     (*(data + j + 1)) * size_data_per_element_ +
                                     offsetData_,
                                 _MM_HINT_T0);
#endif

                    // 如果当前候选节点未被访问过，则进一步处理
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        // 更新候选集和顶部候选者
                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);

#ifdef USE_SSE
                            _mm_prefetch(
                                data_level0_memory_ +
                                    candidate_set.top().second * size_data_per_element_ +
                                    offsetLevel0_,
                                _MM_HINT_T0);
#endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            // 释放访问列表资源并返回顶部候选者集合
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        /**
         * 根据启发式方法获取邻居节点（版本2）???
         *
         * @param top_candidates 优先队列，存储候选节点的距离和标识
         * @param M 邻居数量上限
         */
        void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> &top_candidates,
            const size_t M)
        {
            // 如果候选节点的数量小于M，则直接返回
            if (top_candidates.size() < M)
            {
                return;
            }

            // 创建一个新的优先队列用于存储最近邻点
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;

            // 创建一个返回列表以保存最终选择的邻居
            std::vector<std::pair<dist_t, tableint>> return_list;

            // 将top_candidates中的元素转换并存入queue_closest
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first,
                                      top_candidates.top().second);
                top_candidates.pop();
            }

            // 循环处理直到queue_closest为空
            while (queue_closest.size())
            {
                // 当返回列表大小达到M时，提前结束循环
                if (return_list.size() >= M)
                    break;

                // 获取当前队列顶部的元素
                std::pair<dist_t, tableint> current_pair = queue_closest.top();

                // 计算距离到查询点的实际值
                dist_t dist_to_query = -current_pair.first;

                // 移除队列顶部元素
                queue_closest.pop();

                // 判断当前元素是否满足条件
                bool good = true;

                // 检查当前元素与其他已选元素之间的距离
                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist = fstdistfunc_(
                        getDataByInternalId(second_pair.second),
                        getDataByInternalId(current_pair.second),
                        dist_func_param_);

                    // 如果发现更近的点，则标记为不满足条件
                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }

                // 如果满足条件，则将该元素加入返回列表
                if (good)
                {
                    return_list.push_back(current_pair);
                }
            }

            // 将处理后的元素重新放入top_candidates
            for (std::pair<dist_t, tableint> current_pair : return_list)
            {
                top_candidates.emplace(-current_pair.first, current_pair.second);
            }
        }

        linklistsizeint *get_linklist0(tableint internal_id) const
        {
            return (linklistsizeint *)(data_level0_memory_ +
                                       internal_id * size_data_per_element_ +
                                       offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id,
                                       char *data_level0_memory_) const
        {
            return (linklistsizeint *)(data_level0_memory_ +
                                       internal_id * size_data_per_element_ +
                                       offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const
        {
            return (linklistsizeint *)(linkLists_[internal_id] +
                                       (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id,
                                               int level) const
        {
            return level == 0 ? get_linklist0(internal_id)
                              : get_linklist(internal_id, level);
        };

        /**
         * @brief 连接新元素到现有图结构中的指定层级
         *
         * 此函数负责将一个数据点连接至当前层级的其他节点，
         * 并根据层级的不同调整最大邻居数量。
         *
         * @param data_point 数据点指针
         * @param cur_c 当前处理的节点ID
         * @param top_candidates 候选邻近节点队列
         * @param level 当前层级
         * @param isUpdate 是否为更新操作而非插入
         * @return tableint 返回下一个最近的入口点ID
         */
        tableint mutuallyConnectNewElement(
            const void *data_point, tableint cur_c,
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> &top_candidates,
            int level, bool isUpdate)
        {
            // 根据层级确定最大邻居数
            size_t Mcurmax = level ? maxM_ : maxM0_;

            // 获取候选邻居并限制其数量不超过M_
            getNeighborsByHeuristic2(top_candidates, M_);

            if (top_candidates.size() > M_)
                throw std::runtime_error("候选者不应超过M_的数量");

            // 准备存储选定的邻居列表
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);

            // 将优先级队列中的邻居转移到向量中
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            // 确定下一个最近的入口点
            tableint next_closest_entry_point = selectedNeighbors.back();

            // 更新当前节点的链接列表
            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                // 检查新插入元素的链表是否为空（仅首次插入）
                if (*ll_cur && !isUpdate)
                    throw std::runtime_error("新插入的元素应有空白的链接列表");

                // 设置链接列表大小
                setListCount(ll_cur, selectedNeighbors.size());

                // 更新链接列表的数据部分
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("可能的内存损坏");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("尝试在一个不存在的层级上建立链接");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            // 遍历选定的邻居以更新它们的链接列表
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
            {
                // 锁定目标节点的互斥锁
                std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                // 获取目标链接列表的大小
                size_t sz_link_list_other = getListCount(ll_other);

                // 检查链接列表大小是否合理
                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("sz_link_list_other的值不正确");

                // 检查是否尝试自连
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("尝试将元素与其自身相连");

                // 检查层级一致性
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("尝试在一个不存在的层级上建立链接");

                // 获取链接列表的数据部分
                tableint *data = (tableint *)(ll_other + 1);

                // 判断当前节点是否已存在于目标节点的邻居列表中
                bool is_cur_c_present = false;
                if (isUpdate)
                {
                    for (size_t j = 0; j < sz_link_list_other; j++)
                    {
                        if (data[j] == cur_c)
                        {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // 如果当前节点不在目标节点的邻居列表中，则进行相应的链接更新
                if (!is_cur_c_present)
                {
                    if (sz_link_list_other < Mcurmax)
                    {
                        // 直接添加当前节点到目标节点的邻居列表末尾
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    }
                    else
                    {
                        // 找出最弱的链接以便替换
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);

                        // 使用优先级队列找到最佳的Mcurmax个邻居
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]), dist_func_param_), data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0)
                        {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx); // ??? seems can be optimized here, because we have known that the orignal neighbor list of ll_other is pruned, then the part of those farther than inserted element can be pruned, only need to update the part of those farther than inserted element
                                                      // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
                        getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            // 返回下一个最近的入口点
            return next_closest_entry_point;
        }

        /**
         * @brief 连接新元素到层级0中的最近邻节点
         *
         * 此方法用于将一个数据点连接至层级0中的最近邻节点，
         * 并更新候选列表以反映这一变化。
         *
         * @param data_point 数据点指针
         * @param cur_c 当前计数器值
         * @param top_candidates 候选节点优先队列
         * @param level 当前处理的层级
         * @param isUpdate 是否为更新操作
         * @return tableint 返回操作结果
         */
        virtual tableint mutuallyConnectNewElementLevel0(
            const void *data_point, tableint cur_c,
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> &top_candidates,
            int level, bool isUpdate)
        {
            return 0; // ??? why return 0
        }

        /**
         * @brief 全局互斥锁
         *
         * 确保多线程环境下数据的一致性和完整性。
         */
        std::mutex global;

        size_t ef_;

        /**
         * @brief 设置搜索效率因子
         *
         * 设置当前对象的搜索效率因子（ef），影响搜索性能。
         *
         * @param ef 效率因子大小
         */
        void setEf(size_t ef) { ef_ = ef; }

        /**
         * @brief 内部执行k近邻搜索
         *
         * 根据查询数据执行内部k近邻搜索算法，返回最接近的数据点集合。
         *
         * @param query_data 查询数据指针
         * @param k 需要查找的邻居数量
         * @return std::priority_queue<std::pair<dist_t, tableint>> 包含距离和索引的优先队列
         */
        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(
            void *query_data, int k)
        {
            // 初始化候选节点优先队列
            std::priority_queue<std::pair<dist_t, tableint>> top_candidates;

            // 如果数据库为空，则直接返回空候选集
            if (cur_element_count == 0)
                return top_candidates;

            // 从入口节点开始搜索
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(
                query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            // 其实是先用最简单的greedy方法选一个新的entry point
            // 自上而下遍历所有层级
            for (size_t level = maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    // 获取当前层级的链接列表
                    int *data = (int *)get_linklist(currObj, level);
                    int size = getListCount(data);

                    // 遍历链接列表中的每个节点
                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];

                        // 检查候选节点的有效性
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");

                        // 计算查询数据与候选节点的距离
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                                dist_func_param_);

                        // 更新当前最近邻节点
                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            // 根据是否包含删除标记选择不同的搜索策略
            if (has_deletions_)
            {
                std::priority_queue<std::pair<dist_t, tableint>> top_candidates1 =
                    searchBaseLayerST<true>(currObj, query_data, ef_);
                top_candidates.swap(top_candidates1);
            }
            else
            {
                std::priority_queue<std::pair<dist_t, tableint>> top_candidates1 =
                    searchBaseLayerST<false>(currObj, query_data, ef_);
                top_candidates.swap(top_candidates1);
            }

            // 调整候选集大小至k
            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements)
        {
            if (new_max_elements < cur_element_count)
                throw std::runtime_error(
                    "Cannot resize, max element is less than the current number of "
                    "elements");

            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);

            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char *data_level0_memory_new = (char *)realloc(
                data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error(
                    "Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new =
                (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error(
                    "Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location)
        {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_basic_construction_);

            output.write(data_level0_memory_,
                         cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++)
            {
                unsigned int linkListSize =
                    element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i]
                                           : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                       size_t max_elements_i = 0)
        {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if (max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_basic_construction_);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos = input.tellg();

            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_, input.cur);
            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (input.tellg() < 0 || input.tellg() >= total_filesize)
                {
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0)
                {
                    input.seekg(linkListSize, input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if (input.tellg() != total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos, input.beg);

            data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ =
                maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks)
                .swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++)
            {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0)
                {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                }
                else
                {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *)malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error(
                            "Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            has_deletions_ = false;

            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (isMarkedDeleted(i))
                    has_deletions_ = true;
            }

            input.close();

            return;
        }

        template <typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label)
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char *data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *)dist_func_param_);
            std::vector<data_t> data;
            data_t *data_ptr = (data_t *)data_ptrv;
            for (int i = 0; i < dim; i++)
            {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
        //        static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the
         * current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            has_deletions_ = true;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end())
            {
                throw std::runtime_error("Label not found");
            }
            markDeletedInternal(search->second);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the
         * mark, whereas maxM0_ has to be limited to the lower 24 bits, however,
         * still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId)
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur |= DELETE_MARK;
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId)
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked
         * deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint *ptr) const
        {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint *ptr, unsigned short int size) const
        {
            *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label)
        {
            addPoint(data_point, label, -1);
        }

        /**
         * 更新现有点的数据并重新构建其邻居关系。
         *
         * @param dataPoint 新数据点的特征向量指针。
         * @param internalId 数据点的内部ID。
         * @param updateNeighborProbability 更新邻居的概率阈值。
         */
        void updatePoint(const void *dataPoint, tableint internalId,
                         float updateNeighborProbability)
        {
            // 更新与现有点关联的特征向量为新的向量
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // 如果要更新的点是入口点且图仅包含单个元素，则直接返回。
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++)
            {
                /**
                 * @brief 执行层次图中的节点更新操作，包括候选集和邻居集的构建。
                 *
                 * 此段代码用于从给定层的指定内部ID开始，通过一跳和二跳连接，
                 * 构建候选节点集合（sCand）和可能更新的邻居集合（sNeigh）。
                 */

                // 声明一个无序集合存储候选节点
                std::unordered_set<tableint> sCand;
                // 声明一个无序集合存储可能更新的邻居节点
                std::unordered_set<tableint> sNeigh;

                // 获取当前节点的一级连接列表
                std::vector<tableint> listOneHop =
                    getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0) // 如果一级连接为空，则跳过此节点
                    continue;

                // 将当前节点加入候选集
                sCand.insert(internalId);

                // 遍历一级连接列表
                for (auto &&elOneHop : listOneHop)
                {
                    // 将一级连接节点加入候选集
                    sCand.insert(elOneHop);

                    // 根据概率决定是否将该节点作为邻居更新
                    if (distribution(update_probability_generator_) >
                        updateNeighborProbability)
                        continue; // 若不满足条件，则跳过

                    // 添加到可能更新的邻居集中
                    sNeigh.insert(elOneHop);

                    // 获取二级连接列表
                    std::vector<tableint> listTwoHop =
                        getConnectionsWithLock(elOneHop, layer);

                    // 遍历二级连接列表并将其加入候选集
                    for (auto &&elTwoHop : listTwoHop)
                    {
                        sCand.insert(elTwoHop);
                    }
                }

                // now sNeigh is only a subset of sCand
                for (auto &&neigh : sNeigh)
                {
                    /**
                     * 创建一个优先队列用于存储候选节点及其距离，
                     * 按照距离排序（最小堆）。
                     */
                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        candidates;

                    /**
                     * 计算候选集的大小，如果当前近邻不在候选集中，则保持原大小；
                     * 否则减去一，但保证至少有一个元素。？？？ 不理解为什么会有neigh不再sCand的情况
                     */
                    size_t size =
                        sCand.find(neigh) == sCand.end()
                            ? sCand.size()
                            : sCand.size() - 1;

                    /**
                     * 设置要保留的元素数量，取候选集大小和预设的最大边数之间的较小值。
                     */
                    size_t elementsToKeep = std::min(ef_basic_construction_, size);

                    /**
                     * 遍历所有候选节点计算其到当前选择的neigh的距离，
                     * 然后找出离的最近的elementsToKeep个点
                     */
                    for (auto &&cand : sCand)
                    {
                        if (cand == neigh)
                            continue;

                        dist_t distance =
                            fstdistfunc_(getDataByInternalId(neigh),
                                         getDataByInternalId(cand), dist_func_param_);

                        if (candidates.size() < elementsToKeep)
                        {
                            candidates.emplace(distance, cand);
                        }
                        else
                        {
                            if (distance < candidates.top().first)
                            {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    /**
                     * 根据启发式算法选择邻居并建立连接。
                     * 如果是第一层，则使用maxM0_作为最大边数；否则使用maxM_。
                     * 进一步筛选candidates
                     */
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    /**
                     * 获取锁以安全地更新链接列表。
                     */
                    {
                        std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);

                        /**
                         * 获取指定层级上的链接列表指针。
                         */
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);

                        /**
                         * 更新链接列表中的计数值。
                         */
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);

                        /**
                         * 将候选节点数据复制到链接列表中。
                         */
                        tableint *data = (tableint *)(ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++)
                        {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            // 修复因更新导致的连接问题。
            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel,
                                       maxLevelCopy);
        };

        /**
         * 修复更新数据点时的连接关系。
         *
         * @param dataPoint 数据点指针。
         * @param entryPointInternalId 入口节点的内部ID。
         * @param dataPointInternalId 数据点的内部ID。
         * @param dataPointLevel 数据点所在的层级。
         * @param maxLevel 最大层级。
         */
        void repairConnectionsForUpdate(const void *dataPoint,
                                        tableint entryPointInternalId,
                                        tableint dataPointInternalId,
                                        int dataPointLevel, int maxLevel)
        {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel)
            {
                // 计算当前对象到数据点的距离
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj),
                                              dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--)
                {
                    bool changed = true;
                    while (changed)
                    {
                        changed = false;
                        unsigned int *data;
                        // 锁定链表锁以保护访问
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj, level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++)
                        {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand),
                                                    dist_func_param_);
                            if (d < curdist)
                            {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--)
            {
                // 搜索基础层中的候选对象
                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    topCandidates = searchBaseLayer(currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    filteredTopCandidates;
                while (topCandidates.size() > 0)
                {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());
                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there
                // could be cases where `topCandidates` could just contains entry point
                // itself. To prevent self loops, the `topCandidates` is filtered and
                // thus can be empty. ？？？不懂
                if (filteredTopCandidates.size() > 0)
                {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted)
                    {
                        filteredTopCandidates.emplace(
                            fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId),
                                         dist_func_param_),
                            entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_basic_construction_)
                            filteredTopCandidates.pop();
                    }

                    if (level != 0)
                        currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId,
                                                            filteredTopCandidates, level, true);
                    else
                        currObj = mutuallyConnectNewElementLevel0(dataPoint, dataPointInternalId, filteredTopCandidates, level,
                                                                  true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) // need to copy the memory of linklist because of the lock
        {
            std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *)(data + 1);
            memcpy(result.data(), ll, size * sizeof(tableint));
            return result;
        };

        /**
         * @file hnswalg.h
         * @brief 链接新节点到HNSW图中的邻居节点
         *
         * 此函数用于将一个数据点链接至其在HNSW图中的邻居节点，
         * 根据给定的数据点、标签以及邻居列表执行操作。
         */

        void linkNeighbors(const void *data_point, labeltype label,
                           vector<int> neighbors)
        {
            /**
             * @brief 获取当前元素在表中的位置
             */
            tableint cur_c = label_lookup_[label];
            /**
             * @brief 确定当前元素有的层数量
             */
            int curlevel = element_levels_[cur_c];

            /**
             * @brief 如果当前元素不是只有一层0层，则分配内存以存储链接信息
             */
            if (curlevel)
            {
                linkLists_[cur_c] =
                    (char *)malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error(
                        "Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            /**
             * @brief 初始化当前对象为入口节点
             */
            tableint currObj = enterpoint_node_;
            int maxlevelcopy = maxlevel_;

            /**
             * @brief 对于从最小(curlevel 和 maxlevelcopy)到零的所有层级
             */
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
            {
                /**
                 * @brief 检查层级是否超出范围
                 */
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                /**
                 * @brief 创建优先队列以存储候选邻居及其距离
                 */
                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    top_candidates;

                /**
                 * @brief 计算数据点与当前对象之间的距离并加入候选队列
                 */
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(currObj),
                                           dist_func_param_);
                top_candidates.emplace(dist, currObj);

                /**
                 * @brief 在非零层级上连接新元素
                 */
                if (level != 0)
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, // ？？？ 这里的top_candidates  不是只有一个entry point吗
                                                        level, false);
                else
                {
                    /**
                     * @brief 在零层级上连接新元素
                     */
                    currObj = mutuallyConnectNewElementLevel0(data_point, cur_c,
                                                              top_candidates, level, false);
                }
            }
        }

        void addNeighborPoint(const void *data_point, labeltype label, int level)
        {
            tableint cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;

            size_t ef_construction = level ? ef_basic_construction_ : ef_max_;

            // Take update lock to prevent race conditions on an element with
            // insertion/update at the same time.
            std::unique_lock<std::mutex> lock_el_update(
                link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;
            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_,
                   0, size_data_per_element_);
            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            return;
            if ((signed)currObj != -1)
            {
                if (curlevel < maxlevelcopy)
                {
                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj),
                                                  dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand),
                                                        dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        top_candidates = searchBaseLayer(currObj, data_point, level);
                    if (epDeleted)
                    {
                        top_candidates.emplace(
                            fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy),
                                         dist_func_param_),
                            enterpoint_copy);
                        if (top_candidates.size() > ef_construction)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates,
                                                        level, false);
                }
            }
            else
            {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            // Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return;
        }

        /**
         * 添加一个点到数据结构中。
         *
         * @param data_point 数据点指针。
         * @param label 标签类型。
         * @param level 指定级别（可选）。
         * @return 返回内部标识符。
         */
        tableint addPoint(const void *data_point, labeltype label, int level)
        {
            tableint cur_c = 0;
            {
                // 检查是否已存在相同标签的元素，若存在，则更新而非创建新元素。
                std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end())
                {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock<std::mutex> lock_el_update(link_list_update_locks_[(
                        existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId))
                    {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);

                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_)
                {
                    throw std::runtime_error(
                        "The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // 获取更新锁以防止插入/更新同一元素时的竞争条件。
            std::unique_lock<std::mutex> lock_el_update(
                link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;

            // if 我要去更新最大的层高
            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_,
                   0, size_data_per_element_);

            // 初始化数据和标签
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            // 如果我的层数不在0层 我就得给这些层每一层分配领结列表的空间
            if (curlevel)
            {
                linkLists_[cur_c] =
                    (char *)malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error(
                        "Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1) // 只有在图为空的时候 entrypoint才是-1
            {
                if (curlevel < maxlevelcopy)
                {
                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj),
                                                  dist_func_param_);

                    // 贪心的 从entrypoint出发找到离自己最近的 （会遍历所有比自己最高的层还高的层）
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand),
                                                        dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    size_t ef_construction = level ? ef_basic_construction_ : ef_max_;

                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        top_candidates;
                    if (level == 0)
                    {
                        top_candidates = searchBaseLayerLevel0(currObj, data_point, level);
                    }
                    else
                    {
                        top_candidates = searchBaseLayer(currObj, data_point, level);
                    }

                    if (epDeleted)
                    {
                        top_candidates.emplace(
                            fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy),
                                         dist_func_param_),
                            enterpoint_copy);
                        if (top_candidates.size() > ef_construction)
                            top_candidates.pop();
                    }
                    if (level == 0)
                    {
                        currObj = mutuallyConnectNewElementLevel0(
                            data_point, cur_c, top_candidates, level, false);
                    }
                    else
                    {
                        currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates,
                                                            level, false);
                    }
                }
            }
            else
            {
                // 对于第一个元素不做任何操作
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            // 释放最大级别的锁
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        /**
         * 搜索最近邻节点
         *
         * @param query_data 查询数据指针
         * @param k 返回的近邻数量
         * @return 优先队列，包含距离和标签类型的配对
         */
        std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
            const void *query_data, size_t k) const
        {
            // 初始化结果优先队列
            std::priority_queue<std::pair<dist_t, labeltype>> result;

            // 如果当前元素计数为零，则直接返回空结果
            if (cur_element_count == 0)
                return result;

            // 设置起始点为入口节点
            tableint currObj = enterpoint_node_;
            // 计算查询数据到入口节点的距离
            dist_t curdist = fstdistfunc_(
                query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            // 遍历从最大层至第一层
            for (int level = maxlevel_; level > 0; level--)
            {
                bool changed = true;

                // 当前层级有变化时循环
                while (changed)
                {
                    changed = false;

                    // 获取链接列表数据
                    unsigned int *data = (unsigned int *)get_linklist(currObj, level);
                    // 获取列表大小
                    int size = getListCount(data);
                    // 更新度量统计信息
                    metric_hops++;
                    metric_distance_computions += size;

                    // 解析链接列表中的节点
                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        // 获取候选节点
                        tableint cand = datal[i];

                        // 检查候选节点的有效性
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");

                        // 计算查询数据到候选节点的距离
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                                dist_func_param_);

                        // 如果新距离小于当前最小距离，则更新之
                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            // 创建顶点候选优先队列
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                top_candidates;

            // 根据是否已删除状态选择搜索基底层的方法
            if (has_deletions_)
            {
                top_candidates =
                    searchBaseLayerST<true, true>(currObj, query_data, std::max(ef_, k));
            }
            else
            {
                top_candidates =
                    searchBaseLayerST<false, true>(currObj, query_data, std::max(ef_, k));
            }

            // 调整候选队列大小以满足k值要求
            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }

            // 将候选队列转换为结果队列
            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first,
                                                         getExternalLabel(rez.second)));
                top_candidates.pop();
            }

            // 返回最终结果
            return result;
        };

        /**
         * 检查图结构的完整性，验证所有节点的连接是否正确且无环。
         */
        void checkIntegrity()
        {
            // 初始化已检查的连接数为0
            int connections_checked = 0;

            // 创建一个向量以存储每个元素的入站连接数量
            std::vector<int> inbound_connections_num(cur_element_count, 0);

            // 遍历所有当前元素
            for (int i = 0; i < cur_element_count; i++)
            {
                // 对于每个元素，在其所在的所有层级上进行遍历
                for (int l = 0; l <= element_levels_[i]; l++)
                {
                    // 获取指定层级上的链接列表
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);

                    // 计算链接列表中的元素个数
                    int size = getListCount(ll_cur);

                    // 将链接列表数据转换为整型指针
                    tableint *data = (tableint *)(ll_cur + 1);

                    // 使用无序集合存储链接目标，用于去重检查
                    std::unordered_set<tableint> s;

                    // 遍历链接列表中的每一个元素
                    for (int j = 0; j < size; j++)
                    {
                        // 断言：链接的目标必须大于0且小于当前元素总数，且不能指向自身
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);

                        // 增加目标元素的入站连接计数
                        inbound_connections_num[data[j]]++;

                        // 插入链接目标到无序集合中
                        s.insert(data[j]);

                        // 更新已检查的连接数
                        connections_checked++;
                    }

                    // 断言：无序集合的大小应等于链接列表的实际大小（即没有重复）
                    assert(s.size() == size);
                }
            }

            // 如果当前元素总数超过1，则进一步检查入站连接的数量范围
            if (cur_element_count > 1)
            {
                // 初始化最小值和最大值为第一个元素的入站连接数
                int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];

                // 遍历所有元素的入站连接数
                for (int i = 0; i < cur_element_count; i++)
                {
                    // 断言：每个元素至少有一个入站连接
                    assert(inbound_connections_num[i] > 0);

                    // 更新最小值和最大值
                    min1 = std::min(inbound_connections_num[i], min1);
                    max1 = std::max(inbound_connections_num[i], max1);
                }

                // 输出最小和最大的入站连接数
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }

            // 输出完整性检查结果及已检查的总连接数
            std::cout << "integrity ok, checked " << connections_checked
                      << " connections\n";
        }
    };

} // namespace base_hnsw
