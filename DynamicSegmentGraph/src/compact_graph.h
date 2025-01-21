#include <algorithm>
#include <boost/functional/hash.hpp>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>
#include <random>
#include <unordered_map>
#include <fstream>
#include <omp.h>

#include "base_hnsw/hnswalg.h"
#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "utils.h"
using namespace base_hnsw;

namespace Compact {

void set_num_threads(int num_threads){
    omp_set_num_threads(num_threads);
}

template <typename dist_t>
struct CompressedPoint {
    // TODO: here it should be a tuple not a pair, there should be ll, lr, rl, rr
    CompressedPoint(unsigned _external_id, unsigned _ll, unsigned _lr, unsigned _rl, unsigned _rr) :
        external_id(_external_id), ll(_ll), lr(_lr), rl(_rl), rr(_rr) {
    }

    CompressedPoint() {
    }

    unsigned external_id;
    unsigned ll, lr, rl, rr;

    inline bool const if_in_compressed_range(const unsigned &query_L, const unsigned &query_R) const {
        return ((ll <= query_L && query_L <= lr) && (rl <= query_R && query_R <= rr));
    }

    bool operator<(const CompressedPoint &other) const {
        // return this->dist < other.dist;
        return this->external_id < other.external_id;
    }
};

template <typename dist_t>
struct DirectedPointNeighbors {
    vector<CompressedPoint<dist_t>> nns;
    vector<CompressedPoint<dist_t>> rev_nns;

    size_t countNeighbors() {
        return nns.size() + rev_nns.size();
    }
};

template <typename dist_t>
class CompactHNSW : public HierarchicalNSW<float> {
public:
    /**
     * Construct a 2D segment graph hierarchical nearest neighbor search tree (Hierarchical Navigable Small World graph) instance.
     *
     * @param index_params Index parameter configuration object, containing key parameters for the index construction process.
     * @param s Distance calculation space interface, used for performing distance measurement operations.
     * @param max_elements Maximum number of elements, i.e., the maximum number of data points the index can accommodate.
     * @param M Default connectivity, each node is connected to M other nodes by default.
     * @param ef_construction Expansion factor, query efficiency parameter used during construction.
     * @param random_seed Random seed, used to initialize the random number generator.
     */
    CompactHNSW(const BaseIndex::IndexParams &index_params,
                SpaceInterface<float> *s,
                size_t max_elements,
                size_t M = 16,
                size_t ef_construction = 200,
                size_t random_seed = 100) :
        HierarchicalNSW(s, max_elements, M, index_params.ef_construction, random_seed) {
        // Assign the passed index parameter pointer to the member variable
        params = &index_params;

        // Set the maximum expansion factor to the ef_max value in the index parameters
        ef_max_ = index_params.ef_max;
    }

    unsigned max_external_id_ = 0;
    unsigned min_external_id_ = std::numeric_limits<unsigned>::max();

    // log
    size_t forward_batch_nn_amount = 0;
    size_t backward_batch_theoratical_nn_amount = 0;
    size_t drop_points_ = 0;

    // size_t Mcurmax;

    // Pointer to a constant BaseIndex::IndexParams type, storing index parameters
    const BaseIndex::IndexParams *params;

    // Pointer to the segment graph neighbor list, representing edge information in the graph structure
    vector<DirectedPointNeighbors<dist_t>> *compact_graph;
    bool if_rebuild_HNSW = false;
    /**
     * Optimize the search process when building the HNSW graph, retaining more neighbor node information.
     * This is basically the original search, searching for the nearest in the current layer of the entire graph.
     * Perhaps it can be combined with (RNN-descent) to improve efficiency??? chaoji left it.
     *
     * @param ep_id Starting point ID
     * @param data_point Data point pointer
     * @param layer Current layer
     * @return Returns a priority queue containing distance and node ID pairs, sorted by distance.
     */
    virtual std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
    searchBaseLayerLevel0(tableint ep_id, const void *data_point, int layer) {
        // Get free visited list
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // Initialize candidate set and processing set
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidateSet;

        // Store the list of deleted adjacent nodes
        std::vector<pair<dist_t, tableint>> deleted_list;

        // Set the EF value during construction
        size_t ef_construction = ef_max_;

        // Calculate the lower bound of the starting point distance
        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                       dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        // Main loop: traverse the candidate set until it is empty
        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound) {
                break;
            }
            candidateSet.pop();

            // Process the current node
            tableint curNodeNum = curr_el_pair.second;
            std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

            // Get link list data based on the layer
            int *data;
            if (layer == 0) {
                data = (int *)get_linklist0(curNodeNum);
            } else {
                data = (int *)get_linklist(curNodeNum, layer);
            }
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);

#ifdef USE_SSE
            // Prefetch instructions to improve performance
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            // Traverse each element in the link list
            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                // Prefetch instructions to improve performance
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;

                // Calculate the distance from the candidate node to the target point
                char *currObj1 = (getDataByInternalId(candidate_id));
                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

                // Update the candidate set and visited nodes
                if (top_candidates.size() < ef_construction || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    // Prefetch instructions to improve performance
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                                 _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    // Record and remove nodes that exceed the EF limit
                    if (top_candidates.size() > ef_construction) {
                        deleted_list.emplace_back(top_candidates.top());
                        top_candidates.pop();
                    }

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }

        // Release visited list resources
        visited_list_pool_->releaseVisitedList(vl);

        // Re-add previously recorded deleted nodes to the candidate set
        for (auto deleted_candidate : deleted_list) {
            top_candidates.emplace(deleted_candidate);
        }

        return top_candidates;
    }

    // std::vector<tableint> selectedNeighbors;
    // std::vector<tableint> return_list;
    // unsigned iter_counter = 0;
    // bool complete = false;
    // tableint next_closest_entry_point;
    
    // void init_selectedNeighbors() {
    //     selectedNeighbors.clear();
    //     return_list.clear();
    //     iter_counter = 0;
    //     next_closest_entry_point = 0;
    //     complete = false;
    // }

    void get_selectedNeighbors(tableint passed_c, dist_t dist_to_query, const unsigned &Mcurmax, 
                                std::vector<tableint> &selectedNeighbors, std::vector<tableint> &return_list, unsigned &iter_counter,
                                bool &complete, tableint &next_closest_entry_point) {
        if (complete)
            return;

        if (return_list.size() >= Mcurmax || iter_counter >= ef_basic_construction_) {
            // The first batch, also use for original HNSW constructing
            next_closest_entry_point =
                return_list.front(); // TODO: check whether the nearest neighbor
            for (auto point : return_list) {
                selectedNeighbors.push_back((tableint)point);
            }

            return_list.clear(); // Clear the return list here
            iter_counter = 0;
            complete = true;
            return;
        }

        iter_counter++;
        bool good = true;

        // Check if it will be pruned by the return list
        for (auto point : return_list) {
            dist_t curdist = fstdistfunc_(getDataByInternalId(point),
                                          getDataByInternalId(passed_c),
                                          dist_func_param_);

            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }

        if (good) {
            return_list.emplace_back(passed_c);
        }
    }

    // // the internal ids of points sorted by distance from queue_closest
    // std::vector<pair<unsigned, dist_t>> sorted_cands;
    // // the tmp map for saving the result of dominationion pair
    // // to avoid the duplicate calculation
    // std::unordered_map<unsigned, bool> calculated_pair;
    // // TODO: we need to get the boundary of each nbr
    // std::vector<bool> if_nbr;
    // std::vector<unsigned> nbr_ll;
    // std::vector<unsigned> nbr_lr;
    // std::vector<unsigned> nbr_rl;
    // std::vector<unsigned> nbr_rr;

    void dfs(vector<unsigned> &prefix_idx, unsigned PIVOT_ID, unsigned L, unsigned R, unsigned lr, unsigned rl, size_t& Mcurmax, 
                std::vector<pair<unsigned, dist_t>> &sorted_cands, std::unordered_map<unsigned, bool>& calculated_pair, std::vector<bool> &if_nbr, std::vector<unsigned> &nbr_ll,
             std::vector<unsigned> &nbr_lr, std::vector<unsigned> &nbr_rl, std::vector<unsigned> &nbr_rr) {
        if (prefix_idx.size() == Mcurmax) {
            return;
        }

        unsigned st_idx = prefix_idx.empty() ? 0 : prefix_idx.back() + 1;
        for (auto i = st_idx; i < sorted_cands.size(); ++i) {
            unsigned cur_external_id = getExternalLabel(sorted_cands[i].first);
            auto cur_dist = sorted_cands[i].second;

            // TODO: how to quickly get the corresponding ID that locates in the range [L,R]
            if (cur_external_id <= R && cur_external_id >= L) {
                bool dominated_flag = false;
                // Iterate each nb in prefix_nbr and check its domination relationship with current closest nb
                for (auto const &pre_nb_idx : prefix_idx) {
                    unsigned encoded_pair = (pre_nb_idx << 16) + i;

                    // check whether it has been calculated
                    if (calculated_pair.find(encoded_pair) != calculated_pair.end()) {
                        const auto &domination_result = calculated_pair[encoded_pair];
                        if (domination_result == true) {
                            dominated_flag = true;
                            break;
                        }
                    } else {
                        // if not calculated, we need to calculate the result
                        dist_t tmp_dist = fstdistfunc_(getDataByInternalId(sorted_cands[i].first), getDataByInternalId(sorted_cands[pre_nb_idx].first), dist_func_param_);
                        auto domination_result = tmp_dist < cur_dist;
                        calculated_pair[encoded_pair] = domination_result;
                        if (domination_result) {
                            dominated_flag = true;
                            break;
                        }
                    }
                }

                if (dominated_flag)
                    continue;

                unsigned next_lr = (cur_external_id < PIVOT_ID) ? std::min(cur_external_id, lr) : lr;
                unsigned next_rl = (cur_external_id > PIVOT_ID) ? std::max(cur_external_id, rl) : rl;

                prefix_idx.push_back(i);

                // TODO: check whether this is good
                if (if_nbr[i] == false) {
                    if_nbr[i] = true;
                    nbr_ll[i] = L;
                    nbr_lr[i] = next_lr;
                    nbr_rl[i] = next_rl;
                    nbr_rr[i] = R;
                } else {
                    nbr_ll[i] = std::min(nbr_ll[i], L);
                    nbr_lr[i] = std::max(nbr_lr[i], next_lr);
                    nbr_rl[i] = std::min(nbr_rl[i], next_rl);
                    nbr_rr[i] = std::max(nbr_rr[i], R);
                }

                dfs(prefix_idx, PIVOT_ID, L, R, next_lr, next_rl, Mcurmax, sorted_cands, calculated_pair, if_nbr, nbr_ll, nbr_lr, nbr_rl, nbr_rr);

                prefix_idx.pop_back();

                if (cur_external_id < lr) {
                    L = cur_external_id + 1;
                } else if (cur_external_id > rl) {
                    R = cur_external_id - 1;
                } else {
                    break;
                }
            }
        }
    }

    void generate_compressed_neighbors(
        std::priority_queue<std::pair<dist_t, tableint>> &queue_closest,
        unsigned center_external_id,
        const unsigned &index_k,
        size_t& Mcurmax,
        std::vector<tableint> &selectedNeighbors,
        std::vector<tableint> &return_list,
        unsigned &iter_counter,
        bool &complete,
        tableint &next_closest_entry_point
        ) {
        if (queue_closest.size() == 0) {
            return;
        }
        // unsigned tmp_left_bound = min_external_id_ == 0 ? 0 : min_external_id_ - 1;                                                              // consider we have 0 as min external id
        // unsigned tmp_right_bound = max_external_id_ == std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : max_external_id_ + 1; // consider we have too maximum value as max external id

        // // TODO: Make sure left bound as 0 is perfect? no any bugs? if -1 that will be fined but if 0 I am not sure
        unsigned tmp_left_bound = 0;
        unsigned tmp_right_bound = max_elements_;

        std::vector<pair<unsigned, dist_t>> sorted_cands;
        // sorted_cands.clear();
        while (!queue_closest.empty()) {
            std::pair<dist_t, tableint> current_pair = queue_closest.top(); // 当前离我最近的点
            dist_t dist_to_query = -current_pair.first;
            sorted_cands.emplace_back(current_pair.second, dist_to_query); // 把queue_closest里的按照顺序塞进sorted_cands里面
            queue_closest.pop();
            get_selectedNeighbors(current_pair.second, dist_to_query, index_k, selectedNeighbors, return_list, iter_counter, complete, next_closest_entry_point);
        }

        // Not need to find compressed points, not need
        if (if_rebuild_HNSW == true) 
            return;

        // some initiliazation for some variables serving for dfs function
        vector<unsigned> prefix_idx;
        prefix_idx.reserve(Mcurmax);

        // TODO: we can shrink this memory that they do not need so much space we can integrate them into one data structure
        
        // the internal ids of points sorted by distance from queue_closest
        // the tmp map for saving the result of dominationion pair
        // to avoid the duplicate calculation
        std::unordered_map<unsigned, bool> calculated_pair;
        // TODO: we need to get the boundary of each nbr
        std::vector<bool> if_nbr;
        std::vector<unsigned> nbr_ll;
        std::vector<unsigned> nbr_lr;
        std::vector<unsigned> nbr_rl;
        std::vector<unsigned> nbr_rr;
        
        
        if_nbr.resize(sorted_cands.size());
        nbr_ll.resize(sorted_cands.size());
        nbr_lr.resize(sorted_cands.size());
        nbr_rl.resize(sorted_cands.size());
        nbr_rr.resize(sorted_cands.size());
        std::fill_n(if_nbr.begin(), if_nbr.size(), false);
        // calculated_pair.clear();

        // always choosing the range [0, max_elements] as dfs input
        // If choose current [min_element_external_id, max_element_exteranl_id] that will make the recall a little bit lower
        dfs(prefix_idx, center_external_id, tmp_left_bound, tmp_right_bound, center_external_id, center_external_id, Mcurmax, sorted_cands, calculated_pair, if_nbr, nbr_ll, nbr_lr, nbr_rl, nbr_rr);

        // generate the compressed point
        for (unsigned i = 0; i < if_nbr.size(); i++) {
            if (if_nbr[i]) {
                // TODO: each point is actually corresponding to a boundary we need to get the accurate positions
                unsigned tmp_external_id = getExternalLabel(sorted_cands[i].first);
                compact_graph->at(center_external_id).nns.emplace_back(tmp_external_id, nbr_ll[i], nbr_lr[i], nbr_rl[i], nbr_rr[i]);
            }
        }
        sort(compact_graph->at(center_external_id).nns.begin(), compact_graph->at(center_external_id).nns.end());
    }

    void gen_rev_neighbors(unsigned center_external_id) {
        auto &nns = compact_graph->at(center_external_id).nns;
        for (auto &point : nns) {
            auto rev_point_id = point.external_id;
            auto &rev_nns = compact_graph->at(rev_point_id).rev_nns;
            rev_nns.emplace_back(center_external_id, point.ll, point.lr, point.rl, point.rr);
        }
        return;
    }


    virtual tableint mutuallyConnectNewElementLevel0(
        const void *data_point, 
        tableint cur_c,        
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates, 
        int level,                                           
        bool isUpdate)                                       
    {
        size_t Mcurmax = maxM0_; 


        unsigned external_id = getExternalLabel(cur_c);

        if (external_id > max_external_id_)
            max_external_id_ = external_id;
        if (external_id < min_external_id_)
            min_external_id_ = external_id;

        std::vector<tableint> selectedNeighbors;
        std::vector<tableint> return_list;
        unsigned iter_counter = 0;
        bool complete = false;
        tableint next_closest_entry_point = 0;
        {
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            while (!top_candidates.empty()) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }


            
            generate_compressed_neighbors(queue_closest, external_id, (unsigned)Mcurmax, Mcurmax, selectedNeighbors, return_list, iter_counter, complete, next_closest_entry_point);

            // if (if_rebuild_HNSW == false) {
            //     gen_rev_neighbors(external_id);
            // }

            if (return_list.size()) // 这种情况是上面的while 跑完了 但是一个batch都没满 所以需要单独处理
            {
                // The first batch, also use for original HNSW constructing
                next_closest_entry_point = return_list.front();
                for (auto point : return_list) {
                    selectedNeighbors.push_back(point);
                }

                return_list.clear();
            }
        }

        {
            linklistsizeint *ll_cur;
            ll_cur = get_linklist0(cur_c);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error(
                    "The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *)(ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error(
                        "Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::mutex> lock(
                link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            ll_other = get_linklist0(selectedNeighbors[idx]);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error(
                    "Trying to make a link on a non-existent level");

            tableint *data = (tableint *)(ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of
            // `selectedNeighbors[idx]` then no need to modify any connections or
            // run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(
                        getDataByInternalId(cur_c),
                        getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            fstdistfunc_(getDataByInternalId(data[j]),
                                         getDataByInternalId(selectedNeighbors[idx]),
                                         dist_func_param_),
                            data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    setListCount(ll_other, indx);
                }
            }
        }

        return next_closest_entry_point;
    }
};

class IndexCompactGraph : public BaseIndex {
public:
    vector<DirectedPointNeighbors<float>> directed_indexed_arr;
    base_hnsw::DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;
    VisitedListPool *visited_list_pool_ = nullptr;
    IndexInfo *index_info = nullptr;
    const BaseIndex::IndexParams *index_params_;
    CompactHNSW<float> *hnsw;

    IndexCompactGraph(base_hnsw::SpaceInterface<float> *s,
                      const DataWrapper *data) :
        BaseIndex(data) {
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        index_info = new IndexInfo();
        index_info->index_version_type = "IndexCompactGraph";
    }

    void printOnebatch() {
        cout << "Print one batch" << endl;
        for (auto cp :
             directed_indexed_arr[data_wrapper->data_size / 2].nns) {
            cout << "[" << cp.external_id << "," << cp.ll << ","
                 << cp.rr << "], ";
        }
        cout << endl;
    }

    void countNeighbrs() {
        size_t max_nns_len = 0;
        // 如果有向图索引不为空，则开始处理
        if (!directed_indexed_arr.empty()) {
            // 遍历所有节点的前向邻居列表
            for (unsigned j = 0; j < directed_indexed_arr.size(); j++) {
                index_info->nodes_amount += directed_indexed_arr[j].countNeighbors();
                max_nns_len = std::max(max_nns_len, directed_indexed_arr[j].nns.size());
            }
        }

        // 计算平均前向邻居数
        index_info->avg_forward_nns = index_info->nodes_amount / static_cast<float>(data_wrapper->data_size);

        // 打印日志（如果启用）
        if (isLog) {
            cout << "Max. nns length of one point" << max_nns_len << endl;
            cout << "Sum of forward nn #: " << index_info->nodes_amount << endl;
            cout << "Avg. forward nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
            cout << "Avg. delta nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
        }
    }

    void processReverseNeighbors() {
        for (size_t i = 0; i < data_wrapper->data_size; ++i) {                                     // each point
            for (auto& point : this->directed_indexed_arr.at(i).nns) {                             // each point's nn
                auto rev_point_id = point.external_id;                                             // get nn's id
                auto &rev_nns = this->directed_indexed_arr.at(rev_point_id).rev_nns;              // get nn's reverse nn list
                rev_nns.emplace_back(i, point.ll, point.lr, point.rl, point.rr);                   // add point to nn's reverse nn list
            }
        }
    }


    void buildIndex(const IndexParams *index_params) override {
        cout << "Building Index using " << index_info->index_version_type << endl;
        timeval tt1, tt2;
        visited_list_pool_ =
            new base_hnsw::VisitedListPool(1, data_wrapper->data_size);

        index_params_ = index_params;
        // build HNSW
        L2Space space(data_wrapper->data_dim);
        hnsw = new CompactHNSW<float>(
            *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
            index_params->ef_construction, index_params->random_seed);

        directed_indexed_arr.clear();
        directed_indexed_arr.resize(data_wrapper->data_size);
        hnsw->compact_graph = &directed_indexed_arr;
        gettimeofday(&tt1, NULL);

        // random add points
        // Step 1: Generate a sequence 0, 1, ..., data_size - 1
        std::vector<size_t> permutation(data_wrapper->data_size);
        std::iota(permutation.begin(), permutation.end(), 0);

        // Step 2: Shuffle the sequence
        std::random_device rd;    // obtain a random number from hardware
        unsigned int seed = 2024; // fix the seed for debug
        // std::mt19937 g(rd());
        std::mt19937 g(seed); // seed the generator
        std::shuffle(permutation.begin(), permutation.end(), g);

        // Step 3: Traverse the shuffled sequence

        cout << "First point" << permutation[0] << endl;

        #pragma omp parallel for schedule(monotonic : dynamic)
        for (size_t i = 0; i < data_wrapper->data_size; ++i) {
            // if (i % 1000 == 0) {
            //     // get thread id
            //     int thread_id = omp_get_thread_num();
            //     cout << "Processing point " << i  << " at thread " << thread_id << " total thread number " << omp_get_num_threads() << endl;
            // }
            hnsw->addPoint(data_wrapper->nodes.at(permutation[i]).data(), permutation[i]);
        }
        // for (size_t i : permutation) {
        //     hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
        // }

        processReverseNeighbors();

        gettimeofday(&tt2, NULL);
        index_info->index_time = CountTime(tt1, tt2);

        cout << "All the forward batch nn #: " << hnsw->forward_batch_nn_amount << endl;
        cout << "Theoratical backward batch nn #: " << hnsw->backward_batch_theoratical_nn_amount << endl;
        // count neighbors number
        countNeighbrs();
    };

    void initForScabilityExp(const IndexParams *index_params, L2Space *space) {
        if(visited_list_pool_ == nullptr)
            visited_list_pool_ =
                new base_hnsw::VisitedListPool(1, data_wrapper->data_size);
        index_params_ = index_params;
        // build HNSW
        hnsw = new CompactHNSW<float>(
            *index_params, space, 2 * data_wrapper->data_size, index_params->K,
            index_params->ef_construction, index_params->random_seed);

        // directed_indexed_arr.clear();
        directed_indexed_arr.resize(data_wrapper->data_size);
        hnsw->compact_graph = &directed_indexed_arr;
    }

    void rebuild_batchInHNSW(vector<unsigned> &nodes_ids) {
        hnsw->if_rebuild_HNSW = true;
        timeval tt1, tt2;
        gettimeofday(&tt1, NULL);
        for (auto i : nodes_ids) {
            hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
        }

        gettimeofday(&tt2, NULL);
        index_info->index_time = CountTime(tt1, tt2);
        cout << "Reinsert for rebuilding a  " << nodes_ids.size() << " batch need" << index_info->index_time << endl;
        // count neighbors number
        countNeighbrs();
        hnsw->if_rebuild_HNSW = false;
    }

    void insert_batch(vector<unsigned> &nodes_ids) {
        timeval tt1, tt2;
        gettimeofday(&tt1, NULL);
        for (auto i : nodes_ids) {
            hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
        }

        gettimeofday(&tt2, NULL);
        index_info->index_time = CountTime(tt1, tt2);
        cout << "Insert a  " << nodes_ids.size() << " batch need" << index_info->index_time << endl;
        // count neighbors number
        countNeighbrs();
    }

    vector<unsigned> fetched_nns;

    vector<int> rangeFilteringSearchInRange(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) override {
        fetched_nns.reserve(100);
        fetched_nns.clear();

        // 时间测量变量初始化
        timeval tt1, tt2, tt3, tt4;

        // 初始化访问列表
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        float lower_bound = std::numeric_limits<float>::max(); 
        std::priority_queue<pair<float, int>> top_candidates;  
        std::priority_queue<pair<float, int>> candidate_set;   

        search_info->total_comparison = 0;
        search_info->internal_search_time = 0;
        search_info->pos_point_traverse_counter = 0;
        search_info->pos_point_used_counter = 0;
        search_info->neg_point_traverse_counter = 0;
        search_info->neg_point_used_counter = 0;
        search_info->cal_dist_time = 0;
        search_info->fetch_nns_time = 0;
        search_info->path_counter = 0;
        num_search_comparison = 0;

        // 初始化三个entry points
        {
            int lbound = query_bound.first;
            int interval = (query_bound.second - lbound) / 3;
            for (size_t i = 0; i < 3; i++) {
                int point = lbound + interval * i;
                float dist = EuclideanDistance(data_wrapper->nodes[point], query); 
                candidate_set.push(make_pair(-dist, point));                       
                visited_array[point] = visited_array_tag;                          
            }
        }
        gettimeofday(&tt3, NULL);


        size_t hop_counter = 0;
        float total_traversed_nn_amount = 0;
        float pos_point_traverse_counter = 0;
        float pos_point_used_counter = 0;
        float neg_point_traverse_counter = 0;
        float neg_point_used_counter = 0;

        while (!candidate_set.empty()) {
            std::pair<float, int> current_node_pair = candidate_set.top(); 
            int current_node_id = current_node_pair.second;

            if (-current_node_pair.first > lower_bound) 
            {
                break;
            }

#ifdef LOG_DEBUG_MODE
            cout << "current node: " << current_node_pair.second << "  -- "
                 << -current_node_pair.first << endl;
#endif

            hop_counter++;

            candidate_set.pop();

            // // only search when candidate point is inside the range
            // this can be commented because no way to do this
            if (current_node_id < query_bound.first || current_node_id > query_bound.second) {
                cout << "no satisfied range point" << endl;
                continue;
            }
            gettimeofday(&tt1, NULL);

            auto const &pos_edges = directed_indexed_arr[current_node_id].nns;
            auto const &neg_edges = directed_indexed_arr[current_node_id].rev_nns;
            // fetch nns first
            fetched_nns.clear();
            for (auto i = 0; i < pos_edges.size(); i++) {
                const unsigned &candidate_id = pos_edges[i].external_id;
                if (candidate_id < query_bound.first)
                    continue;
                if (candidate_id > query_bound.second) 
                    break;
                const auto &cp = pos_edges[i];
                if (!cp.if_in_compressed_range(query_bound.first, query_bound.second)) {
                    continue;
                }
                fetched_nns.emplace_back(candidate_id);
            }

            for (auto i = 0; i < neg_edges.size(); i++) {
                const unsigned &candidate_id = neg_edges[i].external_id;
                if (candidate_id < query_bound.first)
                    continue;
                if (candidate_id > query_bound.second) 
                    continue;
                auto &cp = neg_edges[i];
                if (!cp.if_in_compressed_range(query_bound.first, query_bound.second)) {
                    continue;
                }
                fetched_nns.emplace_back(candidate_id);
            }
            gettimeofday(&tt2, NULL);                              
            AccumulateTime(tt1, tt2, search_info->fetch_nns_time); 

            // now iterate fetched nn and calculate distance
            for (auto &candidate_id : fetched_nns) {
                if (!(visited_array[candidate_id] == visited_array_tag)) 
                {
                    visited_array[candidate_id] = visited_array_tag; 


                    gettimeofday(&tt1, NULL); 
                    float dist = fstdistfunc_(query.data(),
                                              data_wrapper->nodes[candidate_id].data(),
                                              dist_func_param_);

                    num_search_comparison++; // 更新比较次数
                    if (top_candidates.size() < search_params->search_ef || lower_bound > dist) {
                        candidate_set.push(make_pair(-dist, candidate_id)); 
                        top_candidates.push(make_pair(dist, candidate_id)); 
                        if (top_candidates.size() > search_params->search_ef) {
                            top_candidates.pop(); 
                        }
                        if (!top_candidates.empty()) {
                            lower_bound = top_candidates.top().first; 
                        }
                    }
                    gettimeofday(&tt2, NULL);                            
                    AccumulateTime(tt1, tt2, search_info->cal_dist_time);
                }
            }
            total_traversed_nn_amount += float(pos_edges.size()) + float(neg_edges.size());
        }

        // 构建结果列表
        vector<int> res;
        while (top_candidates.size() > search_params->query_K) {
            top_candidates.pop(); 
        }

        while (!top_candidates.empty()) {
            res.emplace_back(top_candidates.top().second); 
            top_candidates.pop();
        }
        search_info->total_comparison += num_search_comparison;  
        search_info->path_counter += hop_counter;
        search_info->pos_point_traverse_counter = pos_point_traverse_counter;
        search_info->pos_point_used_counter = pos_point_used_counter;
        search_info->neg_point_traverse_counter = neg_point_traverse_counter;
        search_info->neg_point_used_counter = neg_point_used_counter;
        search_info->total_traversed_nn_amount = total_traversed_nn_amount;

        // 释放资源和更新时间统计
        visited_list_pool_->releaseVisitedList(vl);
        gettimeofday(&tt4, NULL);
        CountTime(tt3, tt4, search_info->internal_search_time);
        return res; // 返回结果列表
    }

    vector<int> rangeFilteringSearchOutBound(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) override {
        return vector<int>();
    }

    // Save function to store the IndexCompactGraph to a file
    void save(const std::string &file_path) override {
        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file for saving index.");
        }

        // Save directed_indexed_arr
        size_t arr_size = directed_indexed_arr.size();
        out.write((char *)&arr_size, sizeof(arr_size));
        for (auto &neighbors : directed_indexed_arr) {
            size_t nns_size = neighbors.nns.size();
            out.write((char *)&nns_size, sizeof(nns_size));
            out.write((char *)neighbors.nns.data(), nns_size * sizeof(CompressedPoint<float>));

            size_t rev_nns_size = neighbors.rev_nns.size();
            out.write((char *)&rev_nns_size, sizeof(rev_nns_size));
            out.write((char *)neighbors.rev_nns.data(), rev_nns_size * sizeof(CompressedPoint<float>));
        }

        out.close();
    }

    // Load function to load the IndexCompactGraph from a file
    void load(const std::string &file_path) override {
        std::ifstream in(file_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open file for loading index.");
        }
        visited_list_pool_ = new base_hnsw::VisitedListPool(1, data_wrapper->data_size);
        // Load directed_indexed_arr
        size_t arr_size;
        in.read((char *)&arr_size, sizeof(arr_size));
        directed_indexed_arr.resize(arr_size);
        for (auto &neighbors : directed_indexed_arr) {
            size_t nns_size;
            in.read((char *)&nns_size, sizeof(nns_size));
            neighbors.nns.resize(nns_size);
            in.read((char *)neighbors.nns.data(), nns_size * sizeof(CompressedPoint<float>));

            size_t rev_nns_size;
            in.read((char *)&rev_nns_size, sizeof(rev_nns_size));
            neighbors.rev_nns.resize(rev_nns_size);
            in.read((char *)neighbors.rev_nns.data(), rev_nns_size * sizeof(CompressedPoint<float>));
        }

        in.close();

        // print out the basic neighbor amount of the loaded index
        countNeighbrs();
    }

    ~IndexCompactGraph() {
        delete hnsw;
        delete index_info;
        directed_indexed_arr.clear();
        delete visited_list_pool_;
    }
};
} // namespace Compact