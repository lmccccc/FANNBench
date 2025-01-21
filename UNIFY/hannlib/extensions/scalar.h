#pragma once

#include <assert.h>
#include <inttypes.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "hannlib/core/base.h"

namespace hannlib
{
class ScalarRangeExtension
{
 public:
  using Payload      = int64_t;
  using PayloadQuery = std::pair<int64_t, int64_t>;

  inline static Payload Payload2Scalar(Payload payload) { return payload; }

  inline static unsigned int ComputeSlotIdx(Payload payload,
                                            const SlotRanges &ranges)
  {
    for (unsigned i = 0; i < ranges.size(); i++)
    {
      if (payload < ranges[i].second)
      {
        return i;
      }
    }
    return ranges.size() - 1;
  }

  inline static std::vector<unsigned int> GetActivatedSlotIndices(
      PayloadQuery payload_query, const SlotRanges &ranges)
  {
    assert(payload_query.first <= payload_query.second);

    // std::cout << "============================================\n";
    // std::cout << "Query range: " << payload_query.first << ", "
    //           << payload_query.second << "\n";
    // std::cout << "Slot ranges: ";
    // for (auto p : ranges)
    // {
    //   std::cout << "(" << p.first << "," << p.second << "),";
    // }
    // std::cout << "\n";

    std::vector<unsigned int> ret;
    ret.reserve(ranges.size());

    unsigned int i = 0;
    for (auto &range : ranges)
    {
      if (payload_query.first < range.second &&
          payload_query.second >= range.first)
      {
        ret.push_back(i);
      }
      ++i;
    }

    // for (auto slot : ret)
    // {
    //   std::cout << slot << ", ";
    // }
    // std::cout << "\n";
    // std::cout << "============================================\n";

    return ret;
  }

  /**
   * @brief Given some data samples, compute the value interval of each slot.
   *
   * The intervals are of the form [ [l_0, h_0), [l_1, h_1), ..., [l_n, h_n] ].
   * That is, all the intervals (except the last one) are right open.
   *
   * @param scalar_samples
   * @param scalar_min
   * @param scalar_max
   * @param num_slots
   * @param is_samples_sorted
   * @return SlotRanges
   */
  static SlotRanges ComputeSlotRanges(std::vector<Scalar> &scalar_samples,
                                      Scalar scalar_min, Scalar scalar_max,
                                      size_t num_slots,
                                      bool is_samples_sorted = false)
  {
    size_t num_samples = scalar_samples.size();
    assert(num_samples >= num_slots);

    SlotRanges ranges;
    ranges.reserve(num_slots);
    unsigned step = num_samples / num_slots;

    if (!is_samples_sorted)
      std::sort(scalar_samples.begin(), scalar_samples.end());

    for (unsigned i = 0; i < num_slots; i += 1)
    {
      size_t start_idx = i * step;
      size_t end_idx   = i * step + step;
      ranges.emplace_back(
          scalar_samples[start_idx],
          scalar_samples[end_idx < num_samples ? end_idx : num_samples - 1]);
    }

    ranges.front().first = scalar_min;
    ranges.back().second = scalar_max;

    return ranges;
  }

  static inline bool IsPayloadQualified(Payload payload, PayloadQuery query)
  {
    return payload >= query.first && payload <= query.second;
  }

  static void PrintRanges(const SlotRanges &ranges)
  {
    if (!ranges.empty())
    {
      unsigned i = 0;
      while (i < ranges.size() - 1)
      {
        std::cout << i << ":[" << ranges[i].first << "," << ranges[i].second
                  << "),";
        i++;
      }
      std::cout << i << ":[" << ranges.back().first << ","
                << ranges.back().second << "]";
    }
    else
    {
      std::cout << "()";
    }
    std::cout << std::endl;
  }

  template <typename T>
  static void PrintScalars(const std::vector<T> &values)
  {
    if (values.empty())
    {
      std::cout << std::endl;
      return;
    }
    std::cout << "[";
    for (int i = 0; i < values.size() - 1; i++)
    {
      std::cout << values[i] << ",";
    }
    std::cout << values.back() << "]" << std::endl;
  }
};

}  // namespace hannlib