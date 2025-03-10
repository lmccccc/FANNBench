#pragma once

#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <fstream>
#include <list>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "base.h"
#include "optimizer.h"
#include "visited_list_pool.h"

namespace hannlib
{
typedef unsigned int tableint;
typedef std::vector<bool> Bitmap;

enum SearchStrategy
{
  kHybridFiltering = 0,
  kPreFiltering    = 1,
  kPostFiltering   = 2,
  kCBO             = 3
};

//     a node: skiplist next | (linksize + links) * n
// a fat node: skiplist next | (linksize + links) * n | vector data | label |
// payload

struct NodePtr
{
 protected:
  char *ptr_;

 public:
  NodePtr(char *ptr) : ptr_(ptr){};

  inline tableint get_skiplist_next() const { return *((tableint *)ptr_); };

  inline void set_skiplist_next(tableint node_id)
  {
    *((tableint *)ptr_) = node_id;
  };

  inline const tableint *get_links(int size_per_slot, int slot_i) const
  {
    return (tableint *)(ptr_ + sizeof(tableint) + size_per_slot * slot_i);
  };

  inline tableint *get_mutable_links(int size_per_slot, int slot_i)
  {
    return (tableint *)(ptr_ + sizeof(tableint) + size_per_slot * slot_i);
  };

  inline tableint get_link_count(int size_per_slot, tableint slot_i) const
  {
    return *get_links(size_per_slot, slot_i);
  }

  inline void set_link_count(int size_per_slot, tableint slot_i, tableint count)
  {
    *get_mutable_links(size_per_slot, slot_i) = count;
  }

  void Clear(size_t size_node) { memset(ptr_, 0, size_node); }

  template <typename data_t>
  void PrintNode(std::ostream &stream, size_t num_slots,
                 size_t max_links_per_slot) const
  {
    size_t size_per_slot =
        sizeof(tableint) + max_links_per_slot * sizeof(tableint);
    // print skiplist link
    stream << get_skiplist_next() << "|" << std::endl;

    // print graph links
    for (int i = 0; i < num_slots; i++)
    {
      const tableint *links = get_links(size_per_slot, i);
      std::cout << links[0] << ":";
      ++links;
      for (int j = 0; j < max_links_per_slot - 1; j++)
      {
        std::cout << links[j] << ",";
      }
      std::cout << links[max_links_per_slot - 1] << "|" << std::endl;
    }
  }
};

struct FatNodePtr : public NodePtr
{
  FatNodePtr(char *ptr) : NodePtr(ptr) {}

  inline const void *get_data_ptr(size_t data_offset) const
  {
    return (void *)(ptr_ + data_offset);
  };

  inline void *get_mutable_data_ptr(size_t data_offset)
  {
    return (void *)(ptr_ + data_offset);
  };

  inline void set_data(size_t data_offset, const void *data, size_t data_size)
  {
    memcpy(get_mutable_data_ptr(data_offset), data, data_size);
  }

  inline labeltype get_label(size_t label_offset) const
  {
    labeltype return_label;
    memcpy(&return_label, ptr_ + label_offset, sizeof(labeltype));
    return return_label;
  };

  inline void set_label(size_t label_offset, labeltype label)
  {
    memcpy(ptr_ + label_offset, &label, sizeof(labeltype));
  };

  template <typename Payload>
  inline Payload get_payload(size_t payload_offset) const
  {
    Payload return_payload;
    memcpy(&return_payload, ptr_ + payload_offset, sizeof(Payload));
    return return_payload;
  };

  template <typename Payload>
  inline void set_payload(size_t payload_offset, Payload payload)
  {
    memcpy(ptr_ + payload_offset, &payload, sizeof(Payload));
  };

  template <typename data_t, typename Payload>
  void PrintFatNode(std::ostream &stream, size_t num_slots,
                    size_t max_links_per_slot, size_t data_dim,
                    size_t data_offset, size_t label_offset,
                    size_t payload_offset) const
  {
    PrintNode<data_t>(stream, num_slots, max_links_per_slot);

    // print data
    data_t *data = (data_t *)get_data_ptr(data_offset);
    for (int i = 0; i < data_dim - 1; i++)
    {
      std::cout << data[i] << ",";
    }
    std::cout << data[data_dim - 1] << "|" << std::endl;
    std::cout << get_payload<Payload>(payload_offset) << "|" << std::endl;
    std::cout << get_label(label_offset) << "|" << std::endl;
  }
};

}  // namespace hannlib

namespace hannlib
{
template <typename T>
std::ostream &debug(std::string prefix, const T &t)
{
  std::cout << prefix << ":" << t << "\n";
  return std::cout;
}

void PrintLockState(size_t cur_node, int slot, std::string action,
                    std::string lock_name, int lock_id)
{
  // std::ostringstream os;
  // os << "Node " << cur_node << ", slot " << slot << ": " << action << " "
  //    << lock_name << " lock " << lock_id << "\n";
  // std::cout << os.str();
}

/*  Memory layout:

    1. Vector data and graph links are stored together for level 0 by
   `data_level0_memory_`:

    -- data_level0_memory_:
    --     (links | vector data | label | payload) * n

    2. For other layers, only the graph links are stored by `link_lists_`:

    -- link_lists_:
    --     links * n
*/

template <typename dist_t, typename QueryExtension>
class HSIG : public HybridIndexInterface<dist_t, QueryExtension>
{
 public:
  using Payload      = typename QueryExtension::Payload;
  using PayloadQuery = typename QueryExtension::PayloadQuery;
  using Record       = std::pair<dist_t, labeltype>;

  struct CompareByFirst
  {
    constexpr bool operator()(Record const &a, Record const &b) const noexcept
    {
      return a.first < b.first;
    }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Constructors and destructors
  ///////////////////////////////////////////////////////////////////////////////

  HSIG(SpaceInterface<dist_t> *s) {}

  // HSIG(SpaceInterface<dist_t> *s, const std::string &location,
  //            bool nmslib = false, size_t max_elements = 0) {
  //   LoadIndex(location, s, max_elements);
  // }

  HSIG(SpaceInterface<dist_t> *s, const std::string &location,
       bool nmslib = false, size_t max_elements = 0)
  {
    LoadIndex(location, s, max_elements);


    // all layers
    // std::cout << "ele cnt:" <<cur_element_count_ <<std::endl;
    // std::cout << "max level:" << maxlevel << std::endl;
    // for (int i = 0; i < cur_element_count_; ++i){
    //   for (int level = 0; level < element_levels_[i]; level++){
    //     const Bitmap &bitmap = *global_link_bitmaps_[i][level];
    //     for (unsigned j = 0; j < bitmap.size(); j++)
    //     {
    //       if (!bitmap[j])
    //       {
    //         continue;
    //       }
    //       nbr_cnt++;
    //     }
    //   }
    // }

    // std::cout << "Total edge:" << nbr_cnt << std::endl;
    
    // double avg_deg = (double)nbr_cnt / cur_element_count_;
    // std::cout << "Average degree:" << avg_deg << std::endl;

    int nbr_cnt = 0;
    int slot_num = slot_ranges_.size();
    
    for (int slot_i = 0; slot_i < slot_num; ++slot_i){
      for (int i = 0; i < cur_element_count_; ++i){
        for (int level = 0; level < element_levels_[i]; level++){
          const tableint *links = GetLinks(i, level, slot_i);
          nbr_cnt += GetLinkCount(links);
        }
      }
    }

    std::cout << "Total edge:" << nbr_cnt << std::endl;
    
    double avg_deg = (double)nbr_cnt / cur_element_count_;
    std::cout << "Average degree:" << avg_deg << std::endl;
    

  }

  HSIG(SpaceInterface<dist_t> *s, SlotRanges slot_ranges, size_t max_elements,
       size_t max_links_per_slot = 8, size_t ef_construction = 200,
       size_t random_seed = 100)
      : element_levels_(max_elements),
        slot_ranges_(slot_ranges),
        link_list_locks_(max_elements),
        link_list_update_locks_(max_update_element_locks),
        global_slot_locks_(slot_ranges.size())
  {
    max_elements_ = max_elements;

    data_dim_        = s->get_dim();
    data_size_       = s->get_data_size();
    fstdistfunc_     = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    // Note that level 0 has twice the number of links as the other levels.
    num_segments_              = slot_ranges_.size();
    max_links_per_slot_        = max_links_per_slot;
    max_links_per_slot_level0_ = max_links_per_slot * 2;

    al_              = max_links_per_slot_;
    al_level0_       = max_links_per_slot_level0_;
    ef_construction_ = std::max(ef_construction, al_);
    ef_              = 10;

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    //     a node: skiplist next | (linksize + links) * n
    // a fat node: skiplist next | (linksize + links) * n
    //                       <-> | vector data | label | payload
    size_per_slot_level0_ = (max_links_per_slot_level0_ + 1) * sizeof(tableint);
    size_fat_node_level0_ =
        sizeof(tableint) +                     // skiplist link
        num_segments_ * size_per_slot_level0_  // graph links
        + data_size_                           // vector data
        + sizeof(labeltype)                    // label
        + sizeof(Payload);                     // payload

    data_offset_    = sizeof(tableint) + size_per_slot_level0_ * num_segments_;
    label_offset_   = data_offset_ + data_size_;
    payload_offset_ = label_offset_ + sizeof(labeltype);

    data_level0_memory_ = (char *)malloc(max_elements_ * size_fat_node_level0_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error("Not enough memory");

    cur_element_count_ = 0;

    visited_list_pool_ = new VisitedListPool(1, max_elements);

    // initializations for special treatment of the first node
    slot_enterpoint_nodes_ =
        (tableint *)malloc(sizeof(tableint) * num_segments_);
    for (unsigned i = 0; i < num_segments_; i++)
    {
      slot_enterpoint_nodes_[i] = -1;
    }

    global_enterpoint_node_ = -1;
    global_max_level_       = -1;

    slot_maxlevels_ = (int *)malloc(sizeof(int) * num_segments_);
    for (unsigned i = 0; i < num_segments_; i++)
    {
      slot_maxlevels_[i] = -1;
    }

    link_lists_ = (char **)malloc(sizeof(void *) * max_elements_);
    if (link_lists_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: HSIG failed to allocate linklists");
    size_per_slot_ =
        sizeof(tableint) +
        max_links_per_slot_ *
            sizeof(tableint);  // fix: max_links_per_slot -> max_links_per_slot_
    size_node_ = sizeof(tableint) + num_segments_ * size_per_slot_;

    global_link_bitmaps_ =
        (Bitmap ***)malloc(sizeof(Bitmap **) * max_elements_);
    if (global_link_bitmaps_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: HSIG failed to allocate global_link_bitmaps_");

    mult_     = 1 / log(1.0 * al_);
    rev_size_ = 1.0 / mult_;
  }

  ~HSIG()
  {
    free(data_level0_memory_);
    for (tableint i = 0; i < cur_element_count_; i++)
    {
      if (element_levels_[i] > 0) free(link_lists_[i]);
    }
    free(link_lists_);
    for (tableint i = 0; i < cur_element_count_; i++)
    {
      for (int j = 0; j < element_levels_[i]; j++)
      {
        delete global_link_bitmaps_[i][j];
      }
      free(global_link_bitmaps_[i]);
    }
    free(global_link_bitmaps_);

    free(slot_enterpoint_nodes_);
    free(slot_maxlevels_);
    delete visited_list_pool_;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Methods for index search
  ///////////////////////////////////////////////////////////////////////////////

  std::priority_queue<std::pair<dist_t, labeltype>> OptimizedHybridSearch(
      const void *query_data, size_t k, PayloadQuery payload_query, int64_t *comp_cnt)
  {
    // std::cout << "force_skiplist_search_=" << force_skiplist_search_ << "\n";
    switch (search_strategy_)
    {
      case SearchStrategy::kHybridFiltering:
        // std::cout << "HybridFilering\n";
        return HybridFiltering(query_data, k, payload_query, comp_cnt);
      case SearchStrategy::kPreFiltering:
        // std::cout << "PreFilering\n";
        return PreFiltering(query_data, k, payload_query, comp_cnt);
      case SearchStrategy::kPostFiltering:
        // std::cout << "PostFilering\n";
        return PostFiltering(query_data, k, payload_query, comp_cnt);
      default:
        break;
    }

    // // Cost-based optimization
    // double graph_search_cost;
    // int ef, al;
    // std::tie(graph_search_cost, ef, al) = optimizer_.EstimateGraphSearchCost(
    //     slot_ranges_, k, payload_query, target_recall_);

    // double skiplist_search_cost;
    // size_t cardinality;

    // std::tie(skiplist_search_cost, cardinality) =
    //     optimizer_.EstimateSkiplistSearchCost(payload_query);

    // // std::cout << "Query range: [" << payload_query.first << ", "
    // //           << payload_query.second << "], "
    // //           << "estimated cardinality: " << cardinality << ", "
    // //           << "graph search cost: " << graph_search_cost
    // //           << ", skiplist search cost: " << skiplist_search_cost << "\n";
    
    auto queryrange = payload_query.second - payload_query.first;
    float selectivity = (queryrange *1.0 / cur_element_count_) * 1.0;
    if (selectivity <= low_range_)
    {
      return PreFiltering(query_data, k, payload_query, comp_cnt);
    }
    else if (selectivity >= high_range_)
    {
      return PostFiltering(query_data, k, payload_query, comp_cnt);
    }
    else
    {
      return HybridFiltering(query_data, k, payload_query, comp_cnt);
    }
    

    // if (graph_search_cost < skiplist_search_cost)
    // {
    //   set_ef(ef);
    //   set_al(al);
    //   return HybridFiltering(query_data, k, payload_query);
    // }
    // else
    // {
    //   return PreFiltering(query_data, k, payload_query);
    // }
    return std::priority_queue<std::pair<dist_t, labeltype>>();
  }

  std::priority_queue<std::pair<dist_t, labeltype>> KnnSearch(
      const void *query_data, size_t k) const
  {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count_ == 0) return result;

    tableint curr_obj = global_enterpoint_node_;
    int maxlevel      = global_max_level_;

    // Number of activated links for per slot
    unsigned al_per_slot = std::floor((double)al_ / num_segments_);
    al_per_slot          = std::max(1u, al_per_slot);
    // std::cout << "al_per_slot: " << al_per_slot << "\n";

    dist_t curdist = fstdistfunc_(query_data, GetDataByInternalId(curr_obj),
                                  dist_func_param_);

    for (int level = maxlevel; level > 0; level--)
    {
      bool changed = true;
      while (changed)
      {
        changed = false;
        for (unsigned slot_i = 0; slot_i < num_segments_;
             slot_i++)  //找到curr_obj在level的每个slot中的最近邻
        {
          const tableint *links = GetLinks(
              curr_obj, level, slot_i);  // curr_obj在level的slot_i中的连接
          int size             = std::min(GetLinkCount(links), al_per_slot);
          const tableint *data = links + 1;

          for (int j = 0; j < size; j++)
          {
            tableint cand = data[j];
            if (cand < 0 || cand > max_elements_)
              throw std::runtime_error("cand error");
            dist_t d = fstdistfunc_(query_data, GetDataByInternalId(cand),
                                    dist_func_param_);

            if (d < curdist)
            {
              curdist  = d;
              curr_obj = cand;
              changed  = true;
            }
          }
        }
      }
    }  //找到在第一层的入口点

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_ef_results;
    SearchBaseLayer(
        top_ef_results, curr_obj, query_data,
        std::max(
            ef_,
            k));  //在第0层找knn（k为ef_和k中大的那一个）在所有的slot中进行查询

    while (top_ef_results.size() > k)
    {
      top_ef_results.pop();
    }

    while (top_ef_results.size() > 0)
    {
      std::pair<dist_t, tableint> rez = top_ef_results.top();
      result.push(std::pair<dist_t, labeltype>(
          rez.first, GetLabelByInternalId(rez.second)));
      top_ef_results.pop();
    }
    return result;
  }

  std::priority_queue<std::pair<dist_t, labeltype>> HybridFiltering(
      const void *query_data, size_t k, PayloadQuery payload_query, int64_t* comp_cnt=nullptr) const
  {
    // std::cout<<"--------HybridFiltering--------"<<std::endl;

    // std::cout << "Get slots\n";
    std::vector<unsigned int> activated_slots =
        QueryExtension::GetActivatedSlotIndices(
            payload_query,
            slot_ranges_);  //找到查询范围所在的slot集合，GetActivatedSlotIndices在scalar.h中

    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count_ == 0) return result;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_ef_results;
    SearchSlots(top_ef_results, activated_slots, query_data, k, payload_query, comp_cnt);

    while (top_ef_results.size() > k)
    {
      top_ef_results.pop();
    }

    while (top_ef_results.size() > 0)
    {
      std::pair<dist_t, tableint> rez = top_ef_results.top();
      result.push(std::pair<dist_t, labeltype>(
          rez.first, GetLabelByInternalId(rez.second)));
      top_ef_results.pop();
    }
    return result;
  }

  std::priority_queue<std::pair<dist_t, labeltype>> PreFiltering(
      const void *query_data, size_t k, PayloadQuery payload_query, int64_t* comp_cnt=nullptr) const
  {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count_ == 0) return result;

    tableint enterpoint_copy = global_enterpoint_node_;
    int maxlevelcopy         = global_max_level_;
    tableint cur_obj         = enterpoint_copy;

    if ((signed)enterpoint_copy == -1 || enterpoint_copy > cur_element_count_)
    {
      throw std::runtime_error(std::string("enterpoint error: ") +
                               std::to_string(enterpoint_copy));
    }

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        internal_results;

    Scalar left          = payload_query.first;
    bool is_head_skipped = false;
    // Find the last point whose scalar value < left in each level
    for (int level = maxlevelcopy; level >= 0; level--)
    {
      if (!is_head_skipped)
      {
        cur_obj = skiplist_heads_[level];
      }

      while (true)
      {
        tableint next = GetSkipListNext(cur_obj, level);
        if ((signed)next == -1)  // Reach the tail of linked list
        {
          break;
        }
        auto value = GetPayloadByInternalId(next);
        if (value < left)
        {
          is_head_skipped = true;
          cur_obj         = next;
        }
        else
        {
          break;
        }
      }
    }

    // Perform linear search in level 0
    {
      if (!is_head_skipped)
      {
        cur_obj = skiplist_heads_[0];
      }
      else
      {
        // cur_obj is the last point whose scalar value < left,
        // therefore cur_obj.next should have a scalar value >= left
        tableint next = GetSkipListNext(cur_obj, 0);
        if ((signed)next == -1)  // Reach the tail of linked list
        {
          return result;
        }
        cur_obj = next;
      }

      // std::cout << "Query range: [" << payload_query.first << ","
      //           << payload_query.second << "], find entry point in level0:
      //           id="
      //           << GetLabelByInternalId(cur_obj)
      //           << ", value=" << GetPayloadByInternalId(cur_obj)
      //           << ", is_head_skipped=" << is_head_skipped << "\n";

      while (true)
      {
        auto value = GetPayloadByInternalId(cur_obj);
        if (value >= payload_query.first && value <= payload_query.second)
        {
          
          if(comp_cnt!=nullptr) (*comp_cnt)++;
          dist_t curdist = fstdistfunc_(
              query_data, GetDataByInternalId(cur_obj), dist_func_param_);
          if (internal_results.size() < k ||
              curdist < internal_results.top().first)
          {
            internal_results.emplace(curdist, cur_obj);
          }

          if (internal_results.size() > k)
          {
            internal_results.pop();
          }
          tableint next = GetSkipListNext(cur_obj, 0);
          if ((signed)next == -1)  // Reach the tail of linked list
          {
            break;
          }
          cur_obj = next;
        }
        else
        {
          break;
        }
      }
    }

    while (internal_results.size() > k)
    {
      internal_results.pop();
    }

    while (internal_results.size() > 0)
    {
      std::pair<dist_t, tableint> rez = internal_results.top();
      result.push(std::pair<dist_t, labeltype>(
          rez.first, GetLabelByInternalId(rez.second)));
      internal_results.pop();
    }
    return result;
  }

  std::priority_queue<std::pair<dist_t, labeltype>> PostFiltering(
      const void *query_data, size_t k, PayloadQuery payload_query, int64_t* comp_cnt=nullptr) const
  {
    // std::cout << "=================================================\n";
    // std::cout << "PostFiltering: [" << payload_query.first << ","
    //           << payload_query.second << "], ef=" << ef_ << ", al=" << al_
    //           << "\n";
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count_ == 0) return result;

    tableint curr_obj = global_enterpoint_node_;
    int maxlevel      = global_max_level_;

    // std::cout << "Graph entry point: " << curr_obj << "\n";
    // std::cout << "al_per_slot: " << al_per_slot << "\n";
    if(comp_cnt!=nullptr) (*comp_cnt)++;
    dist_t curdist = fstdistfunc_(query_data, GetDataByInternalId(curr_obj),
                                  dist_func_param_);

    for (int level = maxlevel; level > 0; level--)
    {
      bool changed = true;
      while (changed)
      {
        changed = false;

        const tableint *linklist =
            GetLinks(curr_obj, level, 0);  // curr_obj在level的全部链表
        const Bitmap &bitmap = *global_link_bitmaps_[curr_obj][level];
        for (unsigned j = 0; j < bitmap.size(); j++)
        {
          if (!bitmap[j])
          {
            continue;
          }

          tableint cand = linklist[j];
          if (cand < 0 || cand > max_elements_)
            throw std::runtime_error("cand error");
          if(comp_cnt!=nullptr) (*comp_cnt)++;
          dist_t d = fstdistfunc_(query_data, GetDataByInternalId(cand),
                                  dist_func_param_);

          if (d < curdist)
          {
            curdist  = d;
            curr_obj = cand;
            changed  = true;
          }
        }
      }
    }  //找到在第一层的入口点

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_ef_results;
    SearchBaseLayer(
        top_ef_results, curr_obj, query_data,
        std::max(
            ef_,
            k), comp_cnt);  //在第0层找knn（k为ef_和k中大的那一个）在所有的slot中进行查询

    // std::cout << "Entry point in level 0: " << curr_obj << "\n";
    // std::cout << "Found " << top_ef_results.size() << " items\n";

    while (top_ef_results.size() > 0)
    {
      std::pair<dist_t, tableint> rez = top_ef_results.top();
      // std::cout << GetLabelByInternalId(rez.second) << ",";
      if (QueryExtension::IsPayloadQualified(GetPayloadByInternalId(rez.second),
                                             payload_query))
      {
        result.emplace(rez.first, GetLabelByInternalId(rez.second));
        if (result.size() > k)
        {
          result.pop();
        }
      }

      top_ef_results.pop();
    }
    // std::cout << "\n";
    return result;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Methods for index building
  ///////////////////////////////////////////////////////////////////////////////

  void Insert(const void *data_point, labeltype label, Payload payload)
  {
    Insert(data_point, label, payload, -1);
  }

  tableint Insert(const void *data_point, labeltype label, Payload payload,
                  int level)
  {
    tableint cur_c = 0;
    {
      // Checking if the element with the same label already exists
      // if so, throw a runtime exception
      std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);

      auto search = label_lookup_.find(label);
      if (search != label_lookup_.end())
      {
        throw std::runtime_error("The inserted element already exists");
      }

      if (cur_element_count_ >= max_elements_)
      {
        throw std::runtime_error(
            "The number of elements exceeds the specified limit");
      };

      cur_c = cur_element_count_;  // cur_c是data_point的id
      cur_element_count_++;
      label_lookup_[label] = cur_c;
    }

    // Take update lock to prevent race conditions on an element with
    // insertion or update at the same time.
    // std::unique_lock<std::mutex> lock_el_update(
    //     link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);

    // PrintLockState(cur_c, -1, "waiting", "links", cur_c);
    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    // PrintLockState(cur_c, -1, "got", "links", cur_c);

    int curlevel = GetRandomLevel(mult_);
    if (level > 0) curlevel = level;

    element_levels_[cur_c] = curlevel;  //每个数据所在的层

    FatNodePtr cur_fat_node =
        GetFatNodePtrLevel0(cur_c);  //该数据（节点）在大图的第0层的地址
    cur_fat_node.Clear(size_fat_node_level0_);  //将这块地址的内存区清空

    // Initialisation of the data, label, and payload
    cur_fat_node.set_label(label_offset_, label);
    cur_fat_node.set_payload(payload_offset_, payload);
    cur_fat_node.set_data(data_offset_, data_point,
                          data_size_);  //将data_point插入到第0层

    // PrintFatNode(cur_c);

    if (curlevel > 0)
    {
      link_lists_[cur_c] = (char *)malloc(size_node_ * curlevel +
                                          1);  // size_node_:所有slot的大小
      if (link_lists_[cur_c] == nullptr)
        throw std::runtime_error(
            "Not enough memory: Insert failed to allocate linklist");
      memset(link_lists_[cur_c], 0, size_node_ * curlevel + 1);
      // PrintNode(cur_c, curlevel);
    }

    unsigned cur_c_slot = QueryExtension::ComputeSlotIdx(
        payload, slot_ranges_);  //根据一维数值计算对应的slote的ID

    // Add skiplist connections
    {
      // This lock is used to prevent race contitions for
      // global_enterpoint_node_, global_max_level, and all skiplist links.
      std::unique_lock<std::mutex> templock(global_);

      tableint enterpoint_copy = global_enterpoint_node_;
      int maxlevelcopy         = global_max_level_;
      tableint cur_obj         = enterpoint_copy;

      if ((signed)enterpoint_copy != -1 && enterpoint_copy > cur_element_count_)
      {
        throw std::runtime_error(std::string("enterpoint error: ") +
                                 std::to_string(enterpoint_copy));
      }

      // The inserted object is not the first object
      if ((signed)cur_obj != -1)
      {
        bool is_head_skipped = false;
        // 找到最后一个值小于当前插入节点值的节点
        for (int level = global_max_level_; level >= curlevel; level--)
        {
          if (!is_head_skipped)
          {
            cur_obj = skiplist_heads_[level];
            // 头节点的值比当前插入节点的值小，那么它会被跳过，否则就不会被跳过
            auto value = GetPayloadByInternalId(cur_obj);
            if (value < payload)
            {
              is_head_skipped = true;
            }
          }

          while (true)
          {
            tableint next = GetSkipListNext(cur_obj, level);
            if ((signed)next == -1)  // Reach the tail of linked list
            {
              break;
            }
            auto value = GetPayloadByInternalId(next);
            if (value < payload)
            {
              is_head_skipped = true;
              cur_obj         = next;
            }
            else
            {
              break;
            }
          }
        }

        // Insert the new point to all layers < curlevel
        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
        {
          if (level > maxlevelcopy || level < 0)  // possible?
          {
            throw std::runtime_error("Level error");
          }

          if (!is_head_skipped)
          {
            cur_obj = skiplist_heads_[level];
            // 头节点的值比当前插入节点的值小，那么它会被跳过，否则就不会被跳过
            auto value = GetPayloadByInternalId(cur_obj);
            if (value < payload)
            {
              is_head_skipped = true;
            }
          }

          while (true)
          {
            tableint next = GetSkipListNext(cur_obj, level);
            if ((signed)next == -1)  // Reach the tail of linked list
            {
              break;
            }
            auto value = GetPayloadByInternalId(next);
            if (value < payload)
            {
              is_head_skipped = true;
              cur_obj         = next;
            }
            else
            {
              break;
            }
          }

          if (is_head_skipped)
          {
            tableint old_next = GetSkipListNext(cur_obj, level);
            SetSkipListNext(cur_obj, level, cur_c);
            SetSkipListNext(cur_c, level, old_next);
          }
          else
          {
            tableint old_next      = skiplist_heads_[level];
            skiplist_heads_[level] = cur_c;
            SetSkipListNext(cur_c, level, old_next);
          }
        }
      }
      else
      {
        // Do nothing for the first element
        global_enterpoint_node_ = cur_c;
        global_max_level_       = curlevel;
        skiplist_heads_.resize(curlevel + 1);
        for (int level = 0; level <= curlevel; level++)
        {
          skiplist_heads_[level] = cur_c;
          SetSkipListNext(cur_c, level, -1);
        }
      }

      // The current object level is larger than the maximum level, and
      // the inserted object is not the first object
      if (curlevel > maxlevelcopy && (signed)cur_obj != -1)
      {
        // Update the global entry point and max level
        global_enterpoint_node_ = cur_c;
        global_max_level_       = curlevel;

        skiplist_heads_.resize(curlevel + 1);
        for (int level = maxlevelcopy + 1; level <= curlevel; level++)
        {
          skiplist_heads_[level] = cur_c;
          SetSkipListNext(cur_c, level, -1);
        }
      }
    }

    // Add connections for every slot
    for (tableint slot_i = 0; slot_i < num_segments_; slot_i++)
    {
      // PrintLockState(cur_c, slot_i, "waiting", "global", -1);
      std::unique_lock<std::mutex> templock(global_slot_locks_[slot_i]);
      // PrintLockState(cur_c, slot_i, "got", "global", -1);

      tableint enterpoint_copy =
          slot_enterpoint_nodes_[slot_i];  // slot_i中的入口点
      int maxlevelcopy =
          slot_maxlevels_[slot_i];  // slot_i中入口点所在的层次，即最高层次
      tableint cur_obj = enterpoint_copy;

      // The current object level is not larger than the maximum level, and
      // the inserted object is not the first object of slot_i
      if (curlevel <= maxlevelcopy && (signed)cur_obj != -1)
      {
        templock.unlock();
        // PrintLockState(cur_c, slot_i, "release", "global", -1);
      }

      if ((signed)cur_obj != -1)  // slot_i中已经有入口点
      {
        cur_fat_node = GetFatNodePtrLevel0(cur_obj);  //第0层

        if (curlevel < maxlevelcopy)
        {
          // Perform nn search in layers > curlevel
          dist_t curdist = fstdistfunc_(
              data_point, cur_fat_node.get_data_ptr(data_offset_),
              dist_func_param_);  // data_point是当前的插入点，和slot_i中的入口点计算距离
          for (
              int level = maxlevelcopy; level > curlevel;
              level--)  // curlevel相当于插入点在大图上的层次，在所有的slot中该节点都在curlevel这一层
          {  //如果data_point的范围不在这个slot_i中呢？还会被插入到slot_i中吗？(存放的不是数据本身，而是在该slot_i上和当前插入点的连接)
            //在这里应该是添加的data_point在这个slot_i中的邻节点连接
            //在slot中找最近邻
            bool changed = true;
            while (
                changed)  //从当前最近邻的连接中找到最近邻，然后再通过该最近邻的连接找
            {
              changed = false;
              // PrintLockState(cur_c, slot_i, "waiting", "links", cur_obj);
              std::unique_lock<std::mutex> lock(link_list_locks_[cur_obj]);
              // PrintLockState(cur_c, slot_i, "waiting", "links", cur_obj);

              const tableint *links =
                  GetLinks(cur_obj, level,
                           slot_i);  //返回的是在slot_i中入口点存放link的指针
              tableint link_count = GetLinkCount(
                  links);  // links中第一个是sizeof(tableint)，应该是目前的数量，后面是具体的数据
              const tableint *data = links + 1;  //加1跳过数量，直接取数据

              for (tableint i = 0; i < link_count;
                   i++)  //对于当前的最近邻cur_obj（第一次是入口点）的所有连接
              {
                tableint cand = data[i];

                // Do not include the new inserted point itself as its kNN
                if (cand == cur_c) continue;

                if (cand < 0 || cand > max_elements_)
                  throw std::runtime_error("cand error");
                dist_t d = fstdistfunc_(data_point, GetDataByInternalId(cand),
                                        dist_func_param_);
                if (d < curdist)
                {
                  curdist = d;
                  cur_obj = cand;  // cur_obj是当前的最近邻
                  changed = true;  //更新了最近邻
                }
              }
            }
          }
        }

        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
        {
          if (level > maxlevelcopy || level < 0)  // possible?
            throw std::runtime_error("Level error");

          std::priority_queue<std::pair<dist_t, tableint>,
                              std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirst>
              top_candidates = SearchLayerSlotForInsertion(  // level层的结果集
                  cur_obj, data_point, cur_c, level,
                  slot_i);  // cur_obj:data_point的最近邻，data_point:插入点，cur_c:插入点的id

          if (!top_candidates.empty())
          {
            cur_obj =
                MutuallyConnectNewElement(  // cur_obj是next_closet_point，相当于下一层的入口点
                    slot_i, data_point, cur_c, cur_c_slot, top_candidates,
                    level);  // slot_i和cur_c_slot中不一定一样，是data_point在slot_i中的最近邻集合
          }
        }
      }
      else
      {
        /* There are no points in slot_i now. */

        // Do nothing for the first element of slot_i
        if (slot_i == cur_c_slot)
        {
          slot_enterpoint_nodes_[slot_i] = cur_c;
          slot_maxlevels_[slot_i]        = curlevel;
        }
      }

      if (curlevel > maxlevelcopy && slot_i == cur_c_slot)
      {
        slot_enterpoint_nodes_[slot_i] = cur_c;
        slot_maxlevels_[slot_i]        = curlevel;
      }

      // PrintLockState(cur_c, slot_i, "release", "global", -1);
    }

    // Compute bitmap for overall graph links
    PruneGlobalLinks(cur_c, curlevel);
    return cur_c;
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Methods for save/load index
  ///////////////////////////////////////////////////////////////////////////////

  void SaveIndex(const std::string &location)
  {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;
    WriteBinaryPOD(output, ef_construction_);
    WriteBinaryPOD(output, max_links_per_slot_);
    WriteBinaryPOD(output, max_links_per_slot_level0_);
    WriteBinaryPOD(output, size_per_slot_level0_);
    WriteBinaryPOD(output, num_segments_);
    WriteBinaryVector(output, slot_ranges_);
    WriteBinaryPOD(output, max_elements_);
    WriteBinaryPOD(output, cur_element_count_);
    WriteBinaryPOD(output, size_fat_node_level0_);
    WriteBinaryPOD(output, global_enterpoint_node_);
    WriteBinaryPOD(output, global_max_level_);
    WriteBinaryPOD(output, mult_);
    WriteBinaryPOD(output, data_offset_);
    WriteBinaryPOD(output, label_offset_);
    WriteBinaryPOD(output, payload_offset_);
    // WriteBinaryPOD(output, size_node_);

    //*slot_enterpoint_nodes_, *slot_maxlevels_, *data_level0_memory_,
    //**link_lists_

    output.write(data_level0_memory_,
                 cur_element_count_ * size_fat_node_level0_);

    for (size_t i = 0; i < cur_element_count_; i++)
    {
      unsigned int linkListSize =
          element_levels_[i] > 0 ? size_node_ * element_levels_[i] : 0;
      WriteBinaryPOD(output, linkListSize);
      if (linkListSize) output.write(link_lists_[i], linkListSize);
    }
    output.write(reinterpret_cast<char *>(slot_enterpoint_nodes_),
                 sizeof(tableint) * num_segments_);
    output.write(reinterpret_cast<char *>(slot_maxlevels_),
                 sizeof(tableint) * num_segments_);

    WriteBinaryPOD(output, (unsigned)skiplist_heads_.size());
    for (tableint id : skiplist_heads_)
    {
      WriteBinaryPOD(output, id);
    }

    size_t bitmap_serial_bytes = 0;
    for (tableint id = 0; id < cur_element_count_; id++)
    {
      for (int level = 0; level <= element_levels_[id]; level++)
      {
        bitmap_serial_bytes +=
            sizeof(unsigned) +
            (global_link_bitmaps_[id][level]->size() + 7) / 8;  // 向上取整
      }
    }
    WriteBinaryPOD(output, bitmap_serial_bytes);

    for (tableint id = 0; id < cur_element_count_; id++)
    {
      for (int level = 0; level <= element_levels_[id]; level++)
      {
        SerializeBitmap(output, *global_link_bitmaps_[id][level]);
      }
    }

    output.close();
  }

  void LoadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                 size_t max_elements_i = 0)
  {
    int nbr_size = 0;
    std::ifstream input(location, std::ios::binary);
    if (!input.is_open()) throw std::runtime_error("Cannot open file");
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    ReadBinaryPOD(input, ef_construction_);
    ReadBinaryPOD(input, max_links_per_slot_);
    ReadBinaryPOD(input, max_links_per_slot_level0_);
    ReadBinaryPOD(input, size_per_slot_level0_);
    ReadBinaryPOD(input, num_segments_);
    ReadBinaryVector(input, slot_ranges_);
    ReadBinaryPOD(input, max_elements_);
    ReadBinaryPOD(input, cur_element_count_);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count_) max_elements = max_elements_;  //?
    max_elements_ = max_elements;
    ReadBinaryPOD(input, size_fat_node_level0_);
    ReadBinaryPOD(input, global_enterpoint_node_);
    ReadBinaryPOD(input, global_max_level_);
    ReadBinaryPOD(input, mult_);
    ReadBinaryPOD(input, data_offset_);
    ReadBinaryPOD(input, label_offset_);
    ReadBinaryPOD(input, payload_offset_);

    data_dim_        = s->get_dim();
    data_size_       = s->get_data_size();
    fstdistfunc_     = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    auto pos = input.tellg();
    /// Optional - check if index is ok:
    input.seekg(cur_element_count_ * size_fat_node_level0_, input.cur);
    for (size_t i = 0; i < cur_element_count_; i++)
    {
      if (input.tellg() < 0 || input.tellg() >= total_filesize)
      {
        throw std::runtime_error("Index seems to be corrupted or unsupported");
      }
      unsigned int linkListSize;
      ReadBinaryPOD(input, linkListSize);
      if (linkListSize != 0)
      {
        input.seekg(linkListSize, input.cur);
      }
    }
    input.seekg(sizeof(tableint) * num_segments_, input.cur);
    input.seekg(sizeof(tableint) * num_segments_, input.cur);

    unsigned int skiplist_heads_size;
    ReadBinaryPOD(input, skiplist_heads_size);
    if (skiplist_heads_size != 0)
    {
      input.seekg(skiplist_heads_size * sizeof(tableint), input.cur);
    }

    size_t bitmap_serial_bytes;
    ReadBinaryPOD(input, bitmap_serial_bytes);
    input.seekg(bitmap_serial_bytes, input.cur);

    if (input.tellg() != total_filesize)
    {
      throw std::runtime_error("Index seems to be corrupted or unsupported");
    }

    input.clear();
    /// Optional check end

    input.seekg(pos, input.beg);
    data_level0_memory_ =
        (char *)malloc(cur_element_count_ * size_fat_node_level0_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: LoadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count_ * size_fat_node_level0_);
    nbr_size += cur_element_count_ * size_fat_node_level0_;

    size_per_slot_ =
        sizeof(tableint) +
        max_links_per_slot_ *
            sizeof(tableint);  // fix: max_links_per_slot -> max_links_per_slot_
    size_node_ = sizeof(tableint) + num_segments_ * size_per_slot_;
    std::vector<std::mutex>(max_elements).swap(link_list_locks_);
    std::vector<std::mutex>(max_update_element_locks)
        .swap(link_list_update_locks_);
    std::vector<std::mutex>(num_segments_).swap(global_slot_locks_);

    visited_list_pool_ = new VisitedListPool(1, max_elements);

    link_lists_ = (char **)malloc(sizeof(void *) * max_elements);
    if (link_lists_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: HSIG failed to allocate linklists");

    element_levels_ = std::vector<int>(max_elements);
    rev_size_       = 1.0 / mult_;
    ef_             = 10;
    for (size_t i = 0; i < cur_element_count_; i++)
    {
      label_lookup_[GetLabelByInternalId(i)] = i;
      unsigned int linkListSize;
      ReadBinaryPOD(input, linkListSize);
      if (linkListSize == 0)
      {
        element_levels_[i] = 0;
        link_lists_[i]     = nullptr;
      }
      else
      {
        element_levels_[i] = linkListSize / size_node_;
        link_lists_[i]     = (char *)malloc(linkListSize);
        if (link_lists_[i] == nullptr)
          throw std::runtime_error(
              "Not enough memory: LoadIndex failed to allocate linklist");
        input.read(link_lists_[i], linkListSize);
        nbr_size += linkListSize;
      }
    }
    slot_enterpoint_nodes_ =
        (tableint *)malloc(sizeof(tableint) * num_segments_);
    input.read(reinterpret_cast<char *>(slot_enterpoint_nodes_),
               sizeof(tableint) * num_segments_);

    slot_maxlevels_ = (int *)malloc(sizeof(int) * num_segments_);
    input.read(reinterpret_cast<char *>(slot_maxlevels_),
               sizeof(tableint) * num_segments_);

    int size;
    ReadBinaryPOD(input, size);
    skiplist_heads_.resize(size);
    for (int i = 0; i < size; i++)
    {
      ReadBinaryPOD(input, skiplist_heads_[i]);
    }

    ReadBinaryPOD(input, bitmap_serial_bytes);
    global_link_bitmaps_ =
        (Bitmap ***)malloc(sizeof(Bitmap **) * max_elements_);
    for (tableint id = 0; id < cur_element_count_; id++)
    {
      auto max_level = element_levels_[id];
      global_link_bitmaps_[id] =
          (Bitmap **)malloc(sizeof(Bitmap *) * (max_level + 1));
      for (int level = 0; level <= max_level; level++)
      {
        auto *ptr                       = new Bitmap();
        global_link_bitmaps_[id][level] = ptr;
        Bitmap bitmap                   = DeserializeBitmap(input);
        ptr->swap(bitmap);
      }
    }

    input.close();

    size_per_slot_level0_ = (max_links_per_slot_level0_ + 1) * sizeof(tableint);
    size_fat_node_level0_ =
        sizeof(tableint) +                     // skiplist link
        num_segments_ * size_per_slot_level0_  // graph links
        + data_size_                           // vector data
        + sizeof(labeltype)                    // label
        + sizeof(Payload);

    al_        = max_links_per_slot_;
    al_level0_ = max_links_per_slot_level0_;



    return;
  }

  void CheckIntegrity()
  {
    int connections_checked = 0;
    std::vector<int> inbound_connections_num(cur_element_count_, 0);
    for (int i = 0; i < cur_element_count_; i++)
    {
      for (int l = 0; l <= element_levels_[i]; l++)
      {
        for (int slot_i = 0; slot_i < num_segments_; slot_i++)
        {
          const tableint *ll_cur = GetLinks(i, l, slot_i);
          int size               = GetLinkCount(ll_cur);
          const tableint *data   = ll_cur + 1;
          std::unordered_set<tableint> s;
          for (int j = 0; j < size; j++)
          {
            assert(data[j] >= 0);
            assert(data[j] < cur_element_count_);
            assert(data[j] != i);
            inbound_connections_num[data[j]]++;
            s.insert(data[j]);
            connections_checked++;
          }
          assert(s.size() == size);
        }
      }
    }
    if (cur_element_count_ > 1)
    {
      int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
      int min_id = 0, max_id = 0;
      for (int i = 0; i < cur_element_count_; i++)
      {
        // assert(inbound_connections_num[i] > 0);
        if (inbound_connections_num[i] == 0)
        {
          PrintFatNode(i);
          std::cout << "Found isolated node: " << i << std::endl;
        }
        min1 = std::min(inbound_connections_num[i], min1);
        max1 = std::max(inbound_connections_num[i], max1);
      }
      std::cout << "Min inbound: " << min1 << ", Max inbound: " << max1 << "\n";
    }
    std::cout << "Integrity ok, checked " << connections_checked
              << " connections\n";
  }

  void PrintFatNode(tableint internal_id)
  {
    auto node = GetFatNodePtrLevel0(internal_id);
    std::cout << "----------"
              << " Fat Node " << internal_id << " ----------\n";
    node.template PrintFatNode<dist_t, Payload>(
        std::cout, num_segments_, max_links_per_slot_level0_, data_dim_,
        data_offset_, label_offset_, payload_offset_);

    std::cout << "Global link bitmap: ";
    const Bitmap &global_link_bitmap = *global_link_bitmaps_[internal_id][0];
    int i                            = 0;
    for (const auto &value : global_link_bitmap)
    {
      if (i % (1 + max_links_per_slot_ * 2) == 0)
      {
        std::cout << "|";
      }
      std::cout << value ? 1 : 0;
      i += 1;
    }
    std::cout << std::endl;
  }

  void PrintNode(tableint internal_id, int level)
  {
    auto node = GetNodePtr(internal_id, level);
    std::cout << "---------- Level " << level << ", "
              << "Node " << internal_id << " ----------\n";
    node.template PrintNode<dist_t>(std::cout, num_segments_,
                                    max_links_per_slot_);

    std::cout << std::endl;
  }

  void LoadOptimizerConf(const std::string &path)
  {
    optimizer_.LoadConf(path);
    // optimizer_.histogram_.PrintHistogram();
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Getters and setters
  ///////////////////////////////////////////////////////////////////////////////
  size_t get_max_elements() const { return max_elements_; }
  size_t get_current_count() const { return cur_element_count_; }
  size_t get_ef() const { return ef_; }
  size_t get_al() const { return al_; }
  size_t get_m() const { return max_links_per_slot_; }
  size_t get_s() const { return num_segments_; }
  size_t get_ef_construction() const { return ef_construction_; }

  void set_ef(size_t ef) { ef_ = ef; }

  void set_al(size_t al)
  {
    al_        = al;
    al_level0_ = al * 2;
  }

  /* Used for range split optimize */
  void set_target_recall(float recall) { target_recall_ = recall; }

  void set_low_range(float low_range) {low_range_ = low_range; }

  void set_high_range(float high_range) {high_range_ = high_range; }

  /* Used for low cardinality optimize */
  void set_search_strategy(int code)
  {
    switch (code)
    {
      case 0:
        search_strategy_ = SearchStrategy::kHybridFiltering;
        break;
      case 1:
        search_strategy_ = SearchStrategy::kPreFiltering;
        break;
      case 2:
        search_strategy_ = SearchStrategy::kPostFiltering;
        break;
      case 3:
        search_strategy_ = SearchStrategy::kCBO;
        break;
      default:
        search_strategy_ = SearchStrategy::kHybridFiltering;
        break;
    }
  }

 private:
  void PruneGlobalLinks(tableint obj, int obj_level)
  {
    {
      // The lock for the new inserted object has already been hold in `Insert`
      // std::unique_lock<std::mutex> el_lock(link_list_locks_[obj]);
      global_link_bitmaps_[obj] =
          (Bitmap **)malloc(sizeof(Bitmap *) * (obj_level + 1));
      for (int level = 0; level <= obj_level; level++)
      {
        global_link_bitmaps_[obj][level] = new Bitmap();
      }
    }

    for (int level = 0; level <= obj_level; level++)
    {
      std::unordered_map<tableint, unsigned> bitmap_pos_map;
      {
        // std::unique_lock<std::mutex> el_lock(link_list_locks_[obj]);
        Bitmap prune_mask = PruneGlobalLinksDetail(obj, level, bitmap_pos_map);
        (*global_link_bitmaps_[obj][level]).swap(prune_mask);
      }

      for (auto [id, _] : bitmap_pos_map)
      {
        std::unique_lock<std::mutex> el_lock(link_list_locks_[id]);
        std::unordered_map<tableint, unsigned> tmp_map;
        Bitmap prune_mask = PruneGlobalLinksDetail(id, level, tmp_map);
        (*global_link_bitmaps_[id][level]).swap(prune_mask);
      }
    }
  }

  Bitmap PruneGlobalLinksDetail(
      tableint obj, int level,
      std::unordered_map<tableint, unsigned> &bitmap_pos_map)
  {
    const void *query = GetDataByInternalId(obj);
    unsigned num_elem_per_slot =
        level == 0 ? (1 + max_links_per_slot_ * 2) : (1 + max_links_per_slot_);
    const size_t n_preserve =
        level == 0 ? (max_links_per_slot_ * 2) : (max_links_per_slot_);
    unsigned bitmap_size = num_segments_ * num_elem_per_slot;
    Bitmap result(bitmap_size);

    std::vector<std::pair<dist_t, tableint>> preserve_list;
    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;

    size_t num_total_neighbors = 0;
    for (unsigned slot_i = 0; slot_i < num_segments_; slot_i++)
    {
      auto *linklist = GetLinks(obj, level, slot_i);
      auto count     = GetLinkCount(linklist);
      for (unsigned j = 0; j < count; j++)
      {
        num_total_neighbors += 1;
        result[slot_i * num_elem_per_slot + 1 + j] = 1;
      }
    }

    if (num_total_neighbors <= n_preserve)
    {
      return result;
    }

    for (unsigned slot_i = 0; slot_i < num_segments_; slot_i++)
    {
      auto *linklist       = GetLinks(obj, level, slot_i);
      const tableint *data = linklist + 1;
      auto count           = GetLinkCount(linklist);
      for (unsigned i = 0; i < count; i++)
      {
        tableint neighbor_id = data[i];
        queue_closest.emplace(
            -fstdistfunc_(query, GetDataByInternalId(neighbor_id),
                          dist_func_param_),
            neighbor_id);
        bitmap_pos_map[neighbor_id] = slot_i * num_elem_per_slot + 1 + i;
      }
    }

    while (queue_closest.size())
    {
      if (preserve_list.size() >= n_preserve) break;
      std::pair<dist_t, tableint> curent_pair =
          queue_closest.top();  // dist最小的
      dist_t dist_to_query = -curent_pair.first;
      queue_closest.pop();
      bool good = true;

      for (std::pair<dist_t, tableint> second_pair : preserve_list)
      {
        dist_t curdist = fstdistfunc_(GetDataByInternalId(second_pair.second),
                                      GetDataByInternalId(curent_pair.second),
                                      dist_func_param_);

        if (curdist < dist_to_query)  //相当于选择候选集中彼此相距较远的点
        {
          good = false;
          break;
        }
      }
      if (good)
      {
        preserve_list.push_back(curent_pair);
      }
    }

    for (auto &pair : preserve_list)
    {
      auto pos    = bitmap_pos_map[pair.second];
      result[pos] = 1;
    }
    return result;
  }

  void GetNeighborsByHeuristic2(
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      const size_t n_preserve)
  {
    if (top_candidates.size() < n_preserve)
    {
      return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0)
    {
      queue_closest.emplace(-top_candidates.top().first,
                            top_candidates.top().second);  //近的排在前面
      top_candidates.pop();
    }

    while (queue_closest.size())
    {
      if (return_list.size() >= n_preserve) break;
      std::pair<dist_t, tableint> curent_pair =
          queue_closest.top();  // dist最小的
      dist_t dist_to_query = -curent_pair.first;
      queue_closest.pop();
      bool good = true;

      for (std::pair<dist_t, tableint> second_pair : return_list)
      {
        dist_t curdist = fstdistfunc_(GetDataByInternalId(second_pair.second),
                                      GetDataByInternalId(curent_pair.second),
                                      dist_func_param_);

        if (curdist < dist_to_query)  //相当于选择候选集中彼此相距较远的点
        {
          good = false;
          break;
        }
      }
      if (good)
      {
        return_list.push_back(curent_pair);
      }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list)
    {
      top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
  }

  tableint MutuallyConnectNewElement(
      unsigned slot_i, const void *data_point, tableint cur_c,
      unsigned cur_c_slot,
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      int level)
  {
    size_t link_num_limit =
        level ? max_links_per_slot_ : max_links_per_slot_level0_;
    GetNeighborsByHeuristic2(
        top_candidates,
        max_links_per_slot_);  //相当于选候选集中彼此相距较远的点

    if (top_candidates.size() > max_links_per_slot_)
      throw std::runtime_error(
          "Should be not be more than max_links_per_slot_ candidates returned "
          "by the heuristic");

    std::vector<tableint> selected_neighbors;
    selected_neighbors.resize(top_candidates.size());
    int i = top_candidates.size() - 1;
    while (i >= 0)
    {
      selected_neighbors[i] =
          top_candidates.top()
              .second;  // selected_neighbors中按距离从小到大排序，selected_neighbors中存放的是id
      top_candidates.pop();
      --i;
    }

    tableint next_closest_entry_point =
        selected_neighbors.back();  //最后一个元素

    // Set links for the new inserted element
    {
      tableint *links = GetMutableLinks(
          cur_c, level,
          slot_i);  // cur_c是当前插入的点，在slot_i中应该不存在连接
      tableint *data = links + 1;

      if (GetLinkCount(links) > 0)
      {
        throw std::runtime_error(
            "The newly inserted element should have blank link list");
      }
      SetLinkCount(
          links,
          selected_neighbors
              .size());  // links对应的内存区的第一个是连接的数量，也就是和cur_c的selected_neighbors的数量

      for (size_t idx = 0; idx < selected_neighbors.size(); idx++)
      {
        if (data[idx]) throw std::runtime_error("Possible memory corruption");
        if (level > element_levels_[selected_neighbors[idx]])
          throw std::runtime_error(
              "Trying to make a link on a non-existent level");

        data[idx] =
            selected_neighbors[idx];  //在links中存放当前插入点的邻居节点的id
      }
    }

    // Set graph links for selected neighbors
    for (
        size_t idx = 0; idx < selected_neighbors.size();
        idx++)  //每个邻居节点和当前插入点之间的连接，在每个邻居节点的link中对应的当前节点的slot中添加当前插入点的id
    {
      tableint neighbor_id = selected_neighbors[idx];

      // PrintLockState(cur_c, cur_c_slot, "waiting", "links", neighbor_id);
      std::unique_lock<std::mutex> lock(link_list_locks_[neighbor_id]);
      // PrintLockState(cur_c, cur_c_slot, "got", "links", neighbor_id);

      tableint *ll_other =
          GetMutableLinks(neighbor_id, level,
                          cur_c_slot);  // neighbor_id在cur_c_slot中的所有连接
      size_t sz_link_list_other = GetLinkCount(ll_other);
      tableint *data            = ll_other + 1;

      if (sz_link_list_other > link_num_limit)
        throw std::runtime_error("Bad value of sz_link_list_other");
      if (selected_neighbors[idx] == cur_c)
        throw std::runtime_error("Trying to connect an element to itself");
      if (level > element_levels_[selected_neighbors[idx]])
        throw std::runtime_error(
            "Trying to make a link on a non-existent level");

      /* Keep the neighbor links ordered */

      dist_t d_max =
          fstdistfunc_(GetDataByInternalId(cur_c),
                       GetDataByInternalId(neighbor_id), dist_func_param_);
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst>
          candidates;
      candidates.emplace(d_max, cur_c);

      for (size_t j = 0; j < sz_link_list_other; j++)  // neighbor_id的所有连接
      {
        candidates.emplace(
            fstdistfunc_(GetDataByInternalId(data[j]),
                         GetDataByInternalId(neighbor_id), dist_func_param_),
            data[j]);
      }

      // An already fulfilled node
      if (sz_link_list_other >= link_num_limit)
      {
        // Heuristic:
        GetNeighborsByHeuristic2(candidates,
                                 link_num_limit);  //去掉里面相距较近的点
      }

      int indx = candidates.size() - 1;
      SetLinkCount(ll_other, candidates.size());
      while (indx >= 0)
      {
        data[indx] = candidates.top().second;
        candidates.pop();
        indx--;
      }

      // PrintLockState(cur_c, cur_c_slot, "release", "links", neighbor_id);
    }

    return next_closest_entry_point;
  }

  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  SearchLayerSlotForInsertion(tableint entrypoint_id, const void *data_point,
                              tableint data_id, int layer, unsigned slot_i)
  {
    VisitedList *vl           = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array    = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidate_set;  //最大的排在前面

    dist_t lower_bound = std::numeric_limits<dist_t>::max();

    dist_t dist = fstdistfunc_(data_point, GetDataByInternalId(entrypoint_id),
                               dist_func_param_);
    unsigned entrypoint_slot = QueryExtension::ComputeSlotIdx(
        GetPayloadByInternalId(entrypoint_id),
        slot_ranges_);  //上一层的最近邻，也就是入口点，所在的slot难道不是slot_i吗

    if (entrypoint_slot == slot_i && entrypoint_id != data_id)
    {
      top_candidates.emplace(dist, entrypoint_id);
      lower_bound = dist;
    }

    candidate_set.emplace(
        -dist, entrypoint_id);  // candidata_set是候选集，top_candidates是结果集

    visited_array[entrypoint_id] = visited_array_tag;

    while (!candidate_set.empty())
    {
      std::pair<dist_t, tableint> curr_el_pair =
          candidate_set.top();  // dist最小的元素
      if ((-curr_el_pair.first) > lower_bound &&
          top_candidates.size() == ef_construction_)
      {
        break;
      }
      candidate_set.pop();  //删除第一个元素

      tableint cur_obj = curr_el_pair.second;

      // Do not include the new inserted point itself as its kNN
      if (cur_obj == data_id) continue;

      std::unique_lock<std::mutex> lock(link_list_locks_[cur_obj]);

      tableint *links = GetMutableLinks(cur_obj, layer, slot_i);  //在slot_i中
      tableint link_count = *links;
      tableint *data      = links + 1;

      for (size_t j = 0; j < link_count; j++)
      {
        tableint candidate_id = data[j];
        //                    if (candidate_id == 0) continue;

        if (visited_array[candidate_id] == visited_array_tag) continue;
        visited_array[candidate_id] = visited_array_tag;
        const void *curr_obj1       = GetDataByInternalId(candidate_id);

        dist_t dist1 = fstdistfunc_(data_point, curr_obj1, dist_func_param_);
        if (top_candidates.size() < ef_construction_ || lower_bound > dist1)
        {
          candidate_set.emplace(-dist1, candidate_id);

          // Do not include the new inserted point itself as its kNN
          if (candidate_id != data_id)
            top_candidates.emplace(dist1, candidate_id);

          if (top_candidates.size() > ef_construction_) top_candidates.pop();

          if (!top_candidates.empty()) lower_bound = top_candidates.top().first;
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  void SearchSlots(std::priority_queue<std::pair<dist_t, tableint>,
                                       std::vector<std::pair<dist_t, tableint>>,
                                       CompareByFirst> &top_ef_results,
                   const std::vector<unsigned int> &activated_slots,
                   const void *query_data, size_t k,
                   PayloadQuery payload_query, int64_t* comp_cnt=nullptr) const
  {
    if (activated_slots.empty())
    {
      return;
    }

    // Number of activated links for per slot
    unsigned al_per_slot = std::floor((double)al_ / activated_slots.size());
    al_per_slot          = std::max(1u, al_per_slot);
    // Find an entry point with the maximum level
    tableint ep_id      = -1;
    int maxlevel        = -1;
    bool is_entry_found = false;

    // std::cout << "Find entry point.\n";
    for (unsigned slot :
         activated_slots)  //找到所有激活的slot中最高层的那个入口点及所在层
    {
      if ((signed)slot_enterpoint_nodes_[slot] != -1 &&
          (signed)slot_maxlevels_[slot] >
              maxlevel)  //该slot中有入口点和最高层次
      {
        ep_id          = slot_enterpoint_nodes_[slot];
        maxlevel       = slot_maxlevels_[slot];
        is_entry_found = true;
      }
    }

    // Cannot find any entry points in all activated slots
    if (!is_entry_found)
    {
      std::cout << "Cannot find any entry points in all activated slots "
                << std::endl;
      return;
    }

    // std::cout << "Start search.\n";

    tableint curr_obj = ep_id;  //当前找到的入口点id

    if(comp_cnt!=nullptr) (*comp_cnt)++;
    dist_t curdist = fstdistfunc_(query_data, GetDataByInternalId(curr_obj),
                                  dist_func_param_);
    for (int level = maxlevel; level > 0; level--)
    {
      bool changed = true;
      while (changed)
      {
        changed = false;
        for (unsigned slot_i : activated_slots)
        {
          const tableint *links = GetLinks(curr_obj, level, slot_i);
          int size              = std::min(GetLinkCount(links), al_per_slot);
          const tableint *data  = links + 1;

          for (int j = 0; j < size; j++)
          {
            tableint cand = data[j];
            if (cand < 0 || cand > max_elements_)
              throw std::runtime_error("cand error");
            if(comp_cnt!=nullptr) (*comp_cnt)++;
            dist_t d = fstdistfunc_(query_data, GetDataByInternalId(cand),
                                    dist_func_param_);

            if (d < curdist)
            {
              curdist  = d;
              curr_obj = cand;
              changed  = true;
            }
          }
        }
      }
    }

    HybridSearchBaseLayer(top_ef_results, curr_obj, query_data,
                          std::max(ef_, k), payload_query, activated_slots,
                          al_per_slot * 2, comp_cnt);

    return;
  }

  void HybridSearchBaseLayer(
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_ef_results,
      tableint ep_id, const void *data_point, size_t ef,
      PayloadQuery payload_query, const std::vector<unsigned> &activated_slots,
      unsigned al_per_slot, int64_t* comp_cnt=nullptr) const
  {
    VisitedList *vl           = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array    = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidate_set;  // candidates

    if(comp_cnt!=nullptr) (*comp_cnt)++;
    dist_t dist =
        fstdistfunc_(data_point, GetDataByInternalId(ep_id), dist_func_param_);

    if (QueryExtension::IsPayloadQualified(GetPayloadByInternalId(ep_id),
                                           payload_query))
    {
      top_ef_results.emplace(dist, ep_id);
    }

    candidate_set.emplace(-dist, ep_id);
    visited_array[ep_id] = visited_array_tag;

    // Add current top-ef results to candidate set
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_ef_copy = top_ef_results;
    while (!top_ef_copy.empty())
    {
      auto &pair = top_ef_copy.top();
      candidate_set.emplace(-pair.first, pair.second);
      visited_array[pair.second] = visited_array_tag;
      top_ef_copy.pop();
    }

    assert(!candidate_set.empty());
    dist_t lower_bound = INFINITY;
    if (candidate_set.empty())
    {
      lower_bound = -candidate_set.top().first;
    }

    while (!candidate_set.empty())
    {
      std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

      if ((-current_node_pair.first) > lower_bound &&
          top_ef_results.size() == ef)
      {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;

      // Visit graph neighbors
      for (unsigned slot_i : activated_slots)
      {
        const tableint *links = GetLinksLevel0(current_node_id, slot_i);
        size_t size           = std::min(GetLinkCount(links), al_per_slot);
        const tableint *data  = links + 1;

        for (size_t j = 0; j < size; j++)
        {
          int candidate_id = data[j];
          if (!(visited_array[candidate_id] == visited_array_tag))
          {
            visited_array[candidate_id] = visited_array_tag;

            const void *curr_obj1 = GetDataByInternalId(candidate_id);
            
            if(comp_cnt!=nullptr) (*comp_cnt)++;
            dist_t dist = fstdistfunc_(data_point, curr_obj1, dist_func_param_);

            if (top_ef_results.size() < ef || lower_bound > dist)
            {
              candidate_set.emplace(-dist, candidate_id);

              if (QueryExtension::IsPayloadQualified(
                      GetPayloadByInternalId(candidate_id), payload_query))
              {
                top_ef_results.emplace(dist, candidate_id);
              }

              if (top_ef_results.size() > ef) top_ef_results.pop();

              if (!top_ef_results.empty())
                lower_bound = top_ef_results.top().first;
            }
          }
        }
      }
    }

    visited_list_pool_->releaseVisitedList(vl);
  }

  void SearchBaseLayer(
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_ef_results,
      tableint ep_id, const void *data_point, size_t ef, int64_t* comp_cnt=nullptr) const
  {
    VisitedList *vl           = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array    = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidate_set;  // candidates
    
    if(comp_cnt!=nullptr) (*comp_cnt)++;
    dist_t dist =
        fstdistfunc_(data_point, GetDataByInternalId(ep_id), dist_func_param_);

    top_ef_results.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);
    visited_array[ep_id] = visited_array_tag;

    assert(!candidate_set.empty());
    dist_t lower_bound = INFINITY;
    if (candidate_set.empty())
    {
      lower_bound = -candidate_set.top().first;
    }

    while (!candidate_set.empty())
    {
      std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

      if ((-current_node_pair.first) > lower_bound &&
          top_ef_results.size() == ef)
      {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;

      // Visit graph neighbors
      const tableint *linklist =
          GetLinks(current_node_id, 0, 0);  // curr_obj在level0的全部链表
      const Bitmap &bitmap = *global_link_bitmaps_[current_node_id][0];

      for (unsigned j = 0; j < bitmap.size(); j++)
      {
        if (!bitmap[j])
        {
          continue;
        }
        int candidate_id = linklist[j];

        if (!(visited_array[candidate_id] == visited_array_tag))
        {
          visited_array[candidate_id] = visited_array_tag;

          const void *curr_obj1 = GetDataByInternalId(candidate_id);
          
          if(comp_cnt!=nullptr) (*comp_cnt)++;
          dist_t dist = fstdistfunc_(data_point, curr_obj1, dist_func_param_);

          if (top_ef_results.size() < ef || lower_bound > dist)
          {
            candidate_set.emplace(-dist, candidate_id);

            top_ef_results.emplace(dist, candidate_id);
            if (top_ef_results.size() > ef) top_ef_results.pop();

            if (!top_ef_results.empty())
              lower_bound = top_ef_results.top().first;
          }
        }
      }
    }

    visited_list_pool_->releaseVisitedList(vl);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Utility subroutines
  ///////////////////////////////////////////////////////////////////////////////
  template <typename T>
  static void WriteBinaryPOD(std::ostream &out, const T &podRef)
  {
    out.write((char *)&podRef, sizeof(T));
  }

  template <typename T>
  static void WriteBinaryVector(std::ostream &out,
                                const std::vector<std::pair<T, T>> &vec)
  {
    // 首先，写入向量的大小
    size_t size = vec.size();
    out.write((char *)&size, sizeof(size));

    // 然后，遍历向量并写入每个pair的元素
    for (const auto &pair : vec)
    {
      out.write((char *)&pair.first, sizeof(T));
      out.write((char *)&pair.second, sizeof(T));
    }
  }

  static void SerializeBitmap(std::ofstream &output,
                              const std::vector<bool> &bitmap)
  {
    unsigned size = bitmap.size();
    // std::cout << "Write bitmap size: " << size << std::endl;
    output.write(reinterpret_cast<const char *>(&size),
                 sizeof(size));  // 写入 bitmap 的大小

    for (size_t i = 0; i < size; ++i)
    {
      unsigned char byte = 0;
      for (int j = 0; j < 8 && i < size; ++j, ++i)
      {
        byte |= (bitmap[i] ? 1 : 0) << j;
      }
      --i;  // 因为 for 循环会再次执行 i++，所以这里需要回退一步
      output.write(reinterpret_cast<const char *>(&byte),
                   sizeof(byte));  // 将字节写入文件
    }
  }

  static std::vector<bool> DeserializeBitmap(std::ifstream &input)
  {
    unsigned size;
    input.read(reinterpret_cast<char *>(&size),
               sizeof(size));  // 读取 bitmap 的大小
    // std::cout << "Read bitmap size: " << size << std::endl;

    std::vector<bool> bitmap(size);
    for (size_t i = 0; i < size;)
    {
      unsigned char byte;
      input.read(reinterpret_cast<char *>(&byte),
                 sizeof(byte));  // 从文件读取字节

      for (int j = 0; j < 8 && i < size; ++j, ++i)
      {
        bitmap[i] = (byte >> j) & 1;
      }
    }

    return bitmap;
  }

  template <typename T>
  static void ReadBinaryPOD(std::istream &in, T &podRef)
  {
    in.read((char *)&podRef, sizeof(T));
  }

  template <typename T>
  static void ReadBinaryVector(std::istream &in,
                               std::vector<std::pair<T, T>> &vec)
  {
    // 首先，读取向量的大小
    size_t size;
    in.read((char *)&size, sizeof(size));
    vec.clear();
    vec.reserve(size);

    // 然后，根据大小读取每个pair的元素
    for (size_t i = 0; i < size; ++i)
    {
      std::pair<T, T> pair;
      in.read((char *)&pair.first, sizeof(T));
      in.read((char *)&pair.second, sizeof(T));
      vec.push_back(pair);
    }
  }

  inline const FatNodePtr GetFatNodePtrLevel0(tableint internal_id) const
  {
    return FatNodePtr(data_level0_memory_ +
                      size_fat_node_level0_ * internal_id);
  };

  inline const NodePtr GetNodePtr(tableint internal_id, int level) const
  {
    return NodePtr(link_lists_[internal_id] + (level - 1) * size_node_);
  };

  inline NodePtr GetMutableNodePtr(tableint internal_id, int level)
  {
    return NodePtr(link_lists_[internal_id] + (level - 1) * size_node_);
  };

  inline FatNodePtr GetMutableFatNodePtrLevel0(tableint internal_id)
  {
    return FatNodePtr(data_level0_memory_ +
                      size_fat_node_level0_ * internal_id);
  };

  inline const void *GetDataByInternalId(tableint internal_id) const
  {
    return GetFatNodePtrLevel0(internal_id).get_data_ptr(data_offset_);
  }

  inline Payload GetPayloadByInternalId(tableint internal_id) const
  {
    return GetFatNodePtrLevel0(internal_id)
        .template get_payload<Payload>(payload_offset_);
  }

  inline labeltype GetLabelByInternalId(tableint internal_id) const
  {
    return GetFatNodePtrLevel0(internal_id).get_label(label_offset_);
  }

  inline const tableint *GetLinks(tableint internal_id, int level,
                                  int slot_i) const
  {
    if (level == 0)
      return GetFatNodePtrLevel0(internal_id)
          .get_links(size_per_slot_level0_, slot_i);
    else
      return GetNodePtr(internal_id, level).get_links(size_per_slot_, slot_i);
  }

  inline const tableint *GetLinksLevel0(tableint internal_id, int slot_i) const
  {
    return GetFatNodePtrLevel0(internal_id)
        .get_links(size_per_slot_level0_, slot_i);
  }

  inline tableint *GetMutableLinks(tableint internal_id, int level, int slot_i)
  {
    if (level == 0)
      return GetMutableFatNodePtrLevel0(internal_id)
          .get_mutable_links(size_per_slot_level0_, slot_i);
    else
      return GetMutableNodePtr(internal_id, level)
          .get_mutable_links(size_per_slot_, slot_i);
  }

  inline void SetLinkCount(tableint *links, tableint count) { *links = count; }

  inline tableint GetLinkCount(const tableint *links) const { return *links; }

  inline tableint GetSkipListNext(tableint obj, int level) const
  {
    if (level != 0)
    {
      return GetNodePtr(obj, level).get_skiplist_next();
    }
    else
    {
      return GetFatNodePtrLevel0(obj).get_skiplist_next();
    }
  }

  inline void SetSkipListNext(tableint obj, int level, tableint next)
  {
    if (level != 0)
    {
      GetMutableNodePtr(obj, level).set_skiplist_next(next);
    }
    else
    {
      GetMutableFatNodePtrLevel0(obj).set_skiplist_next(next);
    }
  }

  int GetRandomLevel(double reverse_size)
  {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

 private:
  ////////////////////////////////////////////////////////////////////////////////
  /// Data attributes
  ///////////////////////////////////////////////////////////////////////////////
  static const tableint max_update_element_locks = 65536;

  /* Core data structures */

  char *data_level0_memory_;  // data and links for level 0
  char **link_lists_;         // links for level 1~n
  // Bitmap for each node in each level
  // Usage: Bitmap* map = global_link_bitmaps_[i][j];
  //   Here map is obj i's bitmap at level j.
  Bitmap ***global_link_bitmaps_;

  std::vector<int> element_levels_;
  std::vector<tableint> skiplist_heads_;  // every layer has a skiplist entry
  std::unordered_map<labeltype, tableint> label_lookup_;
  VisitedListPool *visited_list_pool_;
  SlotRanges slot_ranges_;  // typedef std::vector<std::pair<Scalar, Scalar>>
                            // SlotRanges

  tableint global_enterpoint_node_;
  int global_max_level_;
  tableint *slot_enterpoint_nodes_;
  int *slot_maxlevels_;

  /* Search parameters */
  Optimizer optimizer_;
  float target_recall_            = 0.98;
  float low_range_ = 0.1;
  float high_range_ = 0.5;
  SearchStrategy search_strategy_ = SearchStrategy::kHybridFiltering;
  // Same to ef in HNSW
  size_t ef_;
  // Number of activated links (for level 1~n) during search
  size_t al_;         // defaults to `max_links_per_slot_`
                      // Number of activated links (for level 0) during search
  size_t al_level0_;  // = al_ * 2

  /* Index parameters */
  size_t ef_construction_;
  size_t num_segments_;
  size_t max_links_per_slot_level0_;
  size_t max_links_per_slot_;

  size_t max_elements_;
  size_t cur_element_count_;

  double mult_, rev_size_;

  /* Data sizes and offsets */

  // Sizes and offsets of level 0
  size_t size_per_slot_level0_;
  size_t size_fat_node_level0_;
  size_t data_dim_;
  size_t data_size_;
  size_t data_offset_;
  size_t label_offset_;
  size_t payload_offset_;

  // Sizes and offsets of levels 1~n
  size_t size_per_slot_;
  size_t size_node_;

  /* Locks */

  std::mutex cur_element_count_guard_;

  std::vector<std::mutex> link_list_locks_;

  // Locks to prevent race condition during update/insert of an element at same
  // time. Note: Locks for additions can also be used to prevent this race
  // condition if the querying of KNN is not exposed along with update / inserts
  // i.e multithread insert / update / query in parallel.
  std::vector<std::mutex> link_list_update_locks_;

  DISTFUNC<dist_t> fstdistfunc_;
  void *dist_func_param_;

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  std::vector<std::mutex> global_slot_locks_;
  std::mutex global_;
};

}  // namespace hannlib