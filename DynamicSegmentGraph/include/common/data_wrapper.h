/**
 * @file data_vecs.h
 * @brief 控制原始向量和查询数据结构
 * 
 * 此类用于封装数据集的基本信息以及存储节点和查询的相关数据，
 * 包括维度、权重属性、真实键值、节点及其键值、查询及其范围和键值，
 * 并提供读取数据、生成过滤查询及基准测试的方法。
 * 
 * @date 2023-06-19
 * @copyright Copyright (c) 2023
 */

#pragma once

#include <string>
#include <vector>

// 使用标准库中的pair、string和vector类型
using std::pair;
using std::string;
using std::vector;

class DataWrapper {
public:
    /**
     * 构造函数初始化数据集名称、数据大小、查询数量和查询k值
     */
    DataWrapper(int num, int k_, string dataset_name, int data_size_)
        : dataset(dataset_name),   // 数据集名称
          data_size(data_size_),  // 数据大小
          query_num(num),         // 查询数量
          query_k(k_) {}          // 查询k值

    // 数据集名称（常量）
    const string dataset;
    
    // 版本号
    string version;
    
    // 数据大小（常量）
    const int data_size;
    
    // 查询数量（常量）
    const int query_num;
    
    // 查询k值（常量）
    const int query_k;
    
    // 数据维度
    size_t data_dim;
    
    // 是否为均匀权重
    bool is_even_weight;
    
    // 是否为真实键值
    bool real_keys;
    
    // 节点数据（待优化：从向量改为数组）???
    vector<vector<float>> nodes;
    
    // 节点键值
    vector<int> nodes_keys;
    
    // 原始查询数据
    vector<vector<float>> querys;
    
    // 查询键值
    vector<int> querys_keys;
    
    // 查询范围
    vector<pair<int, int>> query_ranges;
    
    // 地面实况数据
    vector<vector<int>> groundtruth;
    
    // 查询标识符
    vector<int> query_ids;

    /**
     * 读取数据文件
     */
    void readData(string &dataset_path, string &query_path);
    
    /**
     * 生成范围过滤查询和地面实况数据
     */
    void generateRangeFilteringQueriesAndGroundtruth(bool is_save = false, const string path = "");
    
    /**
     * 生成半边界查询和地面实况数据
     */
    void generateHalfBoundedQueriesAndGroundtruth(bool is_save = false, const string path = "");
    
    /**
     * 加载地面实况数据
     */
    void LoadGroundtruth(const string &gt_path);

    /**
     * 生成范围过滤查询和地面实况数据（可扩展性版本）
     */
    void generateRangeFilteringQueriesAndGroundtruthScalability(bool is_save = false, const string path = "");
    
    /**
     * 生成半边界查询和地面实况数据（可扩展性版本）
     */
    void generateHalfBoundedQueriesAndGroundtruthScalability(bool is_save = false, const string path = "");
    
    /**
     * 生成半边界查询和地面实况数据（基准测试版本）
     */
    void generateHalfBoundedQueriesAndGroundtruthBenchmark(bool is_save_to_file, const string save_path = "");
    
    /**
     * 生成范围过滤查询和地面实况数据（基准测试版本）
     */
    void generateRangeFilteringQueriesAndGroundtruthBenchmark(bool is_save_to_file, const string save_path = "");
};
