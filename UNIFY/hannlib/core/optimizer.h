#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "hannlib/extensions/scalar.h"
namespace hannlib
{
class StopW
{
  std::chrono::steady_clock::time_point time_begin;

 public:
  StopW() { time_begin = std::chrono::steady_clock::now(); }

  float getElapsedTimeMicro()
  {
    std::chrono::steady_clock::time_point time_end =
        std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::microseconds>(time_end -
                                                                  time_begin)
                .count());
  }

  void reset() { time_begin = std::chrono::steady_clock::now(); }
};

class Histogram
{
 public:
  Histogram() {}

  void LoadFromCsv(const std::string& filename)
  {
    bins_.clear();
    std::ifstream file(filename);
    if (!file.is_open())
    {
      throw std::runtime_error("file not exists: " + filename);
    }

    std::string line;
    std::getline(file, line);  // 跳过表头
    int i = 2;
    while (std::getline(file, line))
    {
      std::istringstream iss(line);
      std::vector<std::string> tokens;
      std::string token;
      while (std::getline(iss, token, ','))
      {
        tokens.push_back(token);
      }

      if (tokens.size() != 3)
      {
        throw std::runtime_error("failed to read histogram bin at line " +
                                 std::to_string(i) + " in file " + filename);
      }

      double bin_start = std::stod(tokens[0]);
      double bin_end   = std::stod(tokens[1]);
      int count        = std::stoull(tokens[2]);

      bins_.emplace_back(bin_start, bin_end, count);
      i += 1;
    }

    return;
  }

  size_t EstimateCardinality(double low, double high) const
  {
    if (bins_.empty())
    {
      throw std::runtime_error("histogram has not been initialized");
    }

    size_t total_count = 0;

    // 使用二分搜索找到范围 [low, high] 的起始和结束 bin 的索引
    auto start_bin_it = bins_.begin();
    if (low > std::get<1>(*start_bin_it))
    {
      start_bin_it = std::lower_bound(bins_.begin(), bins_.end(), low,
                                      [](const auto& bin, double value)
                                      { return std::get<1>(bin) < value; });
    }

    auto end_bin_it = bins_.end();
    if (high < std::get<0>(bins_[bins_.size() - 1]))
    {
      end_bin_it = std::upper_bound(bins_.begin(), bins_.end(), high,
                                    [](double value, const auto& bin)
                                    { return value < std::get<0>(bin); });
    }
    // 遍历范围内的 bin，累加计数
    for (auto it = start_bin_it; it < end_bin_it; ++it)
    {
      double bin_start = std::get<0>(*it);
      double bin_end   = std::get<1>(*it);
      size_t count     = std::get<2>(*it);

      if (bin_start <= high && low <= bin_end)
      {
        total_count += count;
      }
    }

    // if (total_count > 2000000)
    // {
    //   std::cout << "Cardinality estimation error: "
    //             << "low=" << low << ", high=" << high
    //             << ", cardinality=" << total_count << "\n";
    //   std::cout << "Begin bin: " << std::get<0>(*start_bin_it) << ","
    //             << std::get<1>(*start_bin_it) << "\n";
    //   std::cout << "End bin: " << std::get<0>(*end_bin_it) << ","
    //             << std::get<1>(*end_bin_it) << "\n";

    //   size_t c = 0;
    //   for (auto it = start_bin_it; it < end_bin_it; ++it)
    //   {
    //     double bin_start = std::get<0>(*it);
    //     double bin_end   = std::get<1>(*it);
    //     size_t count     = std::get<2>(*it);

    //     if (bin_start <= high && low <= bin_end)
    //     {
    //       c += count;
    //     }

    //     std::cout << "Hit bin: [" << bin_start << "," << bin_end << "], "
    //               << "count=" << count;
    //     std::cout << "; total_count=" << c << "\n";
    //   }

    //   exit(-1);
    // }

    return total_count;
  }

  void PrintHistogram() const
  {
    for (const auto& bin : bins_)
    {
      std::cout << "Bin: [" << std::get<0>(bin) << ", " << std::get<1>(bin)
                << "], Count: " << std::get<2>(bin) << std::endl;
    }
  }

  const auto& get_bins() const { return bins_; };

 private:
  std::vector<std::tuple<double, double, size_t>> bins_;
};

struct CostParams
{
  double a;
  double b;
  // c and d are only effective for graph search
  // and would always be 0 for skiplist search
  double c;
  double d;
};

class Optimizer
{
 public:
  Optimizer() { skiplist_cost_params_ = {-1, -1}; };

  void LoadConf(const std::string& conf_dir)
  {
    graph_al_map_        = LoadAlMapFromCsv(conf_dir + "/al.csv");
    graph_ef_params_map_ = LoadCostParamsFromCsv(conf_dir + "/ef_params.csv");
    graph_latency_params_map_ =
        LoadCostParamsFromCsv(conf_dir + "/latency_params.csv");
    histogram_.LoadFromCsv(conf_dir + "/hist.csv");
    auto params_map =
        LoadCostParamsFromCsv(conf_dir + "/skiplist_latency_params.csv");
    auto it               = params_map.begin();
    skiplist_cost_params_ = std::make_pair(it->second.a, it->second.b);
  }

  size_t EstimateCardinality(
      const ScalarRangeExtension::PayloadQuery& query_range)
  {
    return histogram_.EstimateCardinality(query_range.first,
                                          query_range.second);
  }

  std::tuple<double, int, int> EstimateGraphSearchCost(
      const SlotRanges& ranges, int k,
      const ScalarRangeExtension::PayloadQuery& query_range,
      float target_recall)
  {
    CheckInitialized();

    // Compute i and j
    auto activated_slots =
        ScalarRangeExtension::GetActivatedSlotIndices(query_range, ranges);
    if (activated_slots.size() == 0)
    {
      return std::make_tuple(0.0, 0, 0);
    }

    // The first and last slot
    int i    = activated_slots[0];
    int j    = activated_slots[activated_slots.size() - 1];
    auto key = std::make_tuple(k, i, j);

    auto latency_params_it = graph_latency_params_map_.find(key);
    if (latency_params_it == graph_latency_params_map_.end())
    {
      throw std::runtime_error("error getting skiplist cost params");
    }

    auto al_it = graph_al_map_.find(key);
    if (al_it == graph_al_map_.end())
    {
      throw std::runtime_error("error getting al config");
    }
    int al = al_it->second;

    auto ef_params_it = graph_ef_params_map_.find(key);
    if (ef_params_it == graph_ef_params_map_.end())
    {
      throw std::runtime_error("error getting ef curve params");
    }

    auto& latency_params = latency_params_it->second;
    double cost =
        latency_params.a * target_recall +
        std::exp(latency_params.b * target_recall + latency_params.c) +
        latency_params.d;

    auto& ef_params = ef_params_it->second;
    double ef       = ef_params.a * target_recall +
                std::exp(ef_params.b * target_recall + ef_params.c) +
                ef_params.d;
    return std::make_tuple(cost, int(ef), int(al));
  };

  std::pair<double, size_t> EstimateSkiplistSearchCost(
      const ScalarRangeExtension::PayloadQuery& query_range)
  {
    CheckInitialized();
    size_t cardinality = EstimateCardinality(query_range);
    double cost        = skiplist_cost_params_.first * cardinality +
                  skiplist_cost_params_.second;
    return std::make_pair(cost, cardinality);
  };

  static std::map<std::tuple<int, int, int>, CostParams> LoadCostParamsFromCsv(
      const std::string& filename)
  {
    std::map<std::tuple<int, int, int>, CostParams> curve_params_map;
    std::ifstream file(filename);

    if (!file.is_open())
    {
      std::cerr << "Failed to open CSV file: " << filename << std::endl;
      return curve_params_map;
    }

    std::string line;
    std::getline(file, line);  // 跳过表头

    while (std::getline(file, line))
    {
      std::istringstream iss(line);
      std::vector<std::string> tokens;
      std::string token;
      while (std::getline(iss, token, ','))
      {
        tokens.push_back(token);
      }

      if (tokens.size() != 7)
      {
        std::cerr << "Invalid CSV format: " << line << std::endl;
        continue;
      }

      int k    = std::stoi(tokens[0]);
      int i    = std::stoi(tokens[1]);
      int j    = std::stoi(tokens[2]);
      double a = std::stod(tokens[3]);
      double b = std::stod(tokens[4]);
      double c = std::stod(tokens[5]);
      double d = std::stod(tokens[6]);

      curve_params_map[std::make_tuple(k, i, j)] = {a, b, c, d};
    }

    return curve_params_map;
  }

  static std::map<std::tuple<int, int, int>, int> LoadAlMapFromCsv(
      const std::string& filename)
  {
    std::map<std::tuple<int, int, int>, int> al_map;
    std::ifstream file(filename);

    if (!file.is_open())
    {
      std::cerr << "Failed to open CSV file: " << filename << std::endl;
      return al_map;
    }

    std::string line;
    std::getline(file, line);  // 跳过表头

    while (std::getline(file, line))
    {
      std::istringstream iss(line);
      std::vector<std::string> tokens;
      std::string token;
      while (std::getline(iss, token, ','))
      {
        tokens.push_back(token);
      }

      if (tokens.size() != 4)
      {
        std::cerr << "Invalid CSV format: " << line << std::endl;
        continue;
      }

      int k  = std::stoi(tokens[0]);
      int i  = std::stoi(tokens[1]);
      int j  = std::stoi(tokens[2]);
      int al = std::stod(tokens[3]);

      al_map[std::make_tuple(k, i, j)] = al;
    }

    return al_map;
  }

 public:
  void CheckInitialized()
  {
    if (histogram_.get_bins().empty() || graph_al_map_.empty() ||
        graph_ef_params_map_.empty() || graph_al_map_.empty() ||
        skiplist_cost_params_.first == -1)
    {
      throw std::runtime_error("optimizer has not been initialized");
    }
  }

  Histogram histogram_;
  std::map<std::tuple<int, int, int>, CostParams> graph_latency_params_map_;
  std::map<std::tuple<int, int, int>, CostParams> graph_ef_params_map_;
  std::map<std::tuple<int, int, int>, int> graph_al_map_;
  std::pair<double, double> skiplist_cost_params_;
};

}  // namespace hannlib