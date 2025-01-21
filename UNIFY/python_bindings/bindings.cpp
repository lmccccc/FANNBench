#include <assert.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>

#include <atomic>
#include <iostream>
#include <thread>

#include "hannlib/api.h"

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads,
                        Function fn)
{
  if (numThreads <= 0)
  {
    numThreads = std::thread::hardware_concurrency();
  }

  if (numThreads == 1)
  {
    for (size_t id = start; id < end; id++)
    {
      fn(id, 0);
    }
  }
  else
  {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId)
    {
      threads.push_back(std::thread(
          [&, threadId]
          {
            while (true)
            {
              size_t id = current.fetch_add(1);

              if ((id >= end))
              {
                break;
              }

              try
              {
                fn(id, threadId);
              }
              catch (...)
              {
                std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                lastException = std::current_exception();
                /*
                 * This will work even when current is the largest value that
                 * size_t can fit, because fetch_add returns the previous value
                 * before the increment (what will result in overflow
                 * and produce 0 instead of current + 1).
                 */
                current = end;
                break;
              }
            }
          }));
    }
    for (auto &thread : threads)
    {
      thread.join();
    }
    if (lastException)
    {
      std::rethrow_exception(lastException);
    }
  }
}

inline void AssertTrue(bool expr, const std::string &msg)
{
  if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
  return;
}

template <typename dist_t, typename data_t = float>
class HybridIndex
{
 public:
  HybridIndex(const std::string &space_name, const int dim)
      : space_name(space_name), dim(dim)
  {
    normalize = false;
    if (space_name == "l2")
    {
      l2space = new hannlib::L2Space(dim);
    }
    else if (space_name == "ip")
    {
      l2space = new hannlib::InnerProductSpace(dim);
    }
    else if (space_name == "cosine")
    {
      l2space   = new hannlib::InnerProductSpace(dim);
      normalize = true;
    }
    else
    {
      throw new std::runtime_error(
          "Space name must be one of l2, ip, or cosine.");
    }
    appr_alg            = NULL;
    ep_added            = true;
    index_inited        = false;
    num_threads_default = std::thread::hardware_concurrency();
  }

  ~HybridIndex()
  {
    delete l2space;
    if (appr_alg) delete appr_alg;
  }

  /*
    SpaceInterface<dist_t> *s, size_t max_elements,
    SlotRanges slot_ranges, size_t max_links_per_slot = 8,
    size_t ef_construction = 200, size_t random_seed = 100
  */
  void InitNewIndex(py::object slot_ranges_py_object, const size_t max_elements,
                    const size_t M, const size_t ef_construction,
                    const size_t random_seed)
  {
    if (appr_alg)
    {
      throw new std::runtime_error("the index is already initialized");
    }

    py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        slot_ranges_py_array(slot_ranges_py_object);
    auto slot_ranges_buffer =
        slot_ranges_py_array
            .request();  //从Python接收一个NumPy数组，并在C++中获得它的底层数据的访问权，以便进一步处理。

    if (slot_ranges_buffer.ndim != 2)
    {
      throw std::runtime_error("`slot_ranges` must be a 2d array");
    }
    size_t num_slots = slot_ranges_buffer.shape[0];

    hannlib::SlotRanges
        slot_ranges;  // typedef std::vector<std::pair<Scalar, Scalar>>
                      // SlotRanges，每个vector中是一个pair
    int64_t *data = static_cast<int64_t *>(
        slot_ranges_buffer.ptr);  // slot_ranges_buffer 是从 Python NumPy
                                  // 数组获取的缓冲区信息对象。缓冲区的 ptr
                                  // 成员提供了指向 NumPy 数组数据开始处的指针
    for (unsigned i = 0; i < num_slots; i++)
    {
      int64_t start = data[i * 2];
      int64_t end   = data[i * 2 + 1];
      slot_ranges.emplace_back(start, end);
    }

    cur_l    = 0;
    appr_alg = new hannlib::ScalarHSIG<dist_t>(
        l2space, slot_ranges, max_elements, M, ef_construction, random_seed);
    index_inited = true;
    ep_added     = false;
    seed         = random_seed;
  }

  void LoadOptimizerConf(const std::string &path)
  {
    if (appr_alg) appr_alg->LoadOptimizerConf(path);
  }

  void AddItems(py::object data_py_object, py::object scalar_py_object,
                py::object ids_ = py::none(),
                int num_threads = -1)  //调用了hybrid_hnsw.h中的insert
  {
    // std::runtime_error("runtime out num threads: " + std::to_string(num_threads));
    // std::cout << "num_threads: " << num_threads << std::endl;
    // std::cout.flush();
    // Vector data
    py::array_t<dist_t, py::array::c_style | py::array::forcecast>
        data_py_array(data_py_object);
    auto data_buffer = data_py_array.request();
    if (data_buffer.ndim != 2)
      throw std::runtime_error("data must be 2d array");
    size_t rows     = data_buffer.shape[0];
    size_t features = data_buffer.shape[1];

    if (features != dim)
      throw std::runtime_error("wrong dimensionality of the vectors");

    // Scalar data
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        scalar_py_array(scalar_py_object);
    auto scalar_buffer = scalar_py_array.request();

    if (scalar_buffer.ndim != 1)
      throw std::runtime_error("scalar values must be a 1d array");
    if (scalar_buffer.shape[0] != rows)
      throw std::runtime_error(
          "scalar values must be the same as the number of vectors");  //每个向量应该有一个对应的标量值
    const int64_t *scalars = static_cast<const int64_t *>(
        scalar_buffer
            .ptr);  //将缓冲区指针（scalar_buffer.ptr）转换为一个指向int64_t类型的常量指针（scalars）。通过这个指针来访问数组中的实际数据了。

    if (num_threads <= 0) num_threads = num_threads_default;

    // avoid using threads when the number of searches is small:
    if (rows <= (unsigned)num_threads * 4)
    {
      num_threads = 1;
    }
    std::cout << "num_threads: " << num_threads << std::endl;
    std::vector<size_t> ids;  // labels
    if (!ids_.is_none())
    {
      py::array_t<size_t, py::array::c_style | py::array::forcecast> items(
          ids_);
      auto ids_numpy = items.request();
      if (ids_numpy.ndim == 1 && ids_numpy.shape[0] == rows)
      {
        std::vector<size_t> ids1(ids_numpy.shape[0]);
        for (size_t i = 0; i < ids1.size(); i++)
        {
          ids1[i] = items.data()[i];
        }
        ids.swap(ids1);  //将新向量的内容与先前声明的ids向量内容交换。
      }
      else if (ids_numpy.ndim == 0 && rows == 1)
      {
        ids.push_back(*items.data());
      }
      else
        throw std::runtime_error("wrong dimensionality of the labels");
    }

    {
      int start = 0;
      if (!ep_added)  // start=0, 入口点还没有定义
      {
        size_t id            = ids.size() ? ids.at(0) : (cur_l);
        int64_t scalar_value = scalars[0];

        float *vector_data = (float *)data_py_array.data(
            0);  //从一个pybind11包装的NumPy数组中获取数据指针
        std::vector<float> norm_array(dim);  // dim为向量的维度
        if (normalize)
        {
          NormalizeVector(vector_data, norm_array.data());
          vector_data = norm_array.data();
        }
        appr_alg->Insert((void *)vector_data, (size_t)id, scalar_value);
        start    = 1;
        ep_added = true;  //入口点已经添加
      }

      py::gil_scoped_release l;
      if (normalize == false)
      {  //将一系列数据点并行地插入到hybridHNSW中
        ParallelFor(start, rows, num_threads,
                    [&](size_t row, size_t threadId)
                    {
                      size_t id      = ids.size() ? ids.at(row) : (cur_l + row);
                      int64_t scalar = scalars[row];
                      // std::cout << "Thread " << threadId << ", row " << row
                      // << std::endl;
                      appr_alg->Insert((void *)data_py_array.data(row),
                                       (size_t)id, scalar);
                      // std::cout << "Insertion Done.\n";
                    });
      }
      else
      {
        std::vector<float> norm_array(num_threads * dim);
        ParallelFor(start, rows, num_threads,
                    [&](size_t row, size_t threadId)
                    {
                      // normalize vector:
                      size_t start_idx = threadId * dim;
                      NormalizeVector((float *)data_py_array.data(row),
                                      (norm_array.data() + start_idx));

                      size_t id      = ids.size() ? ids.at(row) : (cur_l + row);
                      int64_t scalar = scalars[row];
                      appr_alg->Insert((void *)(norm_array.data() + start_idx),
                                       (size_t)id, scalar);
                    });
      };
      cur_l += rows;
    }
  }

  py::object HybridSearch(py::object query_py_object,
                          py::object ranges_py_object, size_t k = 1)
  {
    // Query vectors
    py::array_t<dist_t, py::array::c_style | py::array::forcecast>
        query_py_array(query_py_object);
    auto query_buffer = query_py_array.request();
    if (query_buffer.ndim != 1)
      throw std::runtime_error("data must be a 1d array");
    size_t features = query_buffer.shape[0];

    if (features != (size_t)dim)
      throw std::runtime_error("wrong dimensionality of the vectors");

    // Query ranges
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        ranges_py_array(ranges_py_object);
    auto ranges_buffer = ranges_py_array.request();

    if (ranges_buffer.ndim != 1)
      throw std::runtime_error("query range must be a 1d array");

    if (ranges_buffer.shape[0] != 2)
    {
      throw std::runtime_error("query ranges must have exact two elements");
    }

    hannlib::labeltype *data_numpy_l;
    dist_t *data_numpy_d;

    data_numpy_l = new hannlib::labeltype[k];
    data_numpy_d = new dist_t[k];

    for (unsigned i = 0; i < k; i++)
    {
      data_numpy_l[i] = 0;
      data_numpy_d[i] = -1;
    }

    std::priority_queue<std::pair<dist_t, hannlib::labeltype>> result;
    int64_t low  = *ranges_py_array.data(0);
    int64_t high = *ranges_py_array.data(1);
    int64_t comp_cnt = 0;
    if (normalize)
    {
      std::vector<float> norm_array(features);
      NormalizeVector((float *)query_py_array.data(), norm_array.data());
      result = appr_alg->OptimizedHybridSearch((void *)norm_array.data(), k,
                                               std::make_pair(low, high), &comp_cnt);
    }
    else
    {
      result = appr_alg->OptimizedHybridSearch((void *)query_py_array.data(), k,
                                               std::make_pair(low, high), &comp_cnt);
    }

    for (int i = result.size() - 1; i >= 0; i--)
    {
      auto &result_tuple = result.top();
      data_numpy_d[i]    = result_tuple.first;
      data_numpy_l[i]    = result_tuple.second;
      result.pop();
    }

    py::capsule free_when_done_l(data_numpy_l, [](void *f) { delete[] f; });
    py::capsule free_when_done_d(data_numpy_d, [](void *f) { delete[] f; });

    return py::make_tuple(
        py::array_t<hannlib::labeltype>(
            {k},                           // shape
            {sizeof(hannlib::labeltype)},  // C-style contiguous strides for
                                           // double
            data_numpy_l,                  // the data pointer
            free_when_done_l),
        py::array_t<dist_t>(
            {k},               // shape
            {sizeof(dist_t)},  // C-style contiguous strides for double
            data_numpy_d,      // the data pointer
            free_when_done_d),
        py::int_(comp_cnt));
  }

  py::object HybridSearchBatch(py::object query_py_object,
                               py::object ranges_py_object, size_t k = 1,
                               int num_threads = -1)
  {
    // Query vectors
    py::array_t<dist_t, py::array::c_style | py::array::forcecast>
        query_py_array(query_py_object);
    auto query_buffer = query_py_array.request();

    if (query_buffer.ndim != 2)
      throw std::runtime_error("query must be a 2d array");
    size_t rows     = query_buffer.shape[0];
    size_t features = query_buffer.shape[1];

    // Query ranges
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        ranges_py_array(ranges_py_object);
    auto ranges_buffer = ranges_py_array.request();

    if (ranges_buffer.ndim != 2)
      throw std::runtime_error("query ranges must be a 2d array");
    if (ranges_buffer.shape[0] != rows || ranges_buffer.shape[1] != 2)
    {
      throw std::runtime_error("query ranges must be of shape (n_query, 2)");
    }
    const int64_t *ranges_ptr = (const int64_t *)ranges_buffer.ptr;

    if (num_threads <= 0) num_threads = num_threads_default;

    hannlib::labeltype *data_numpy_l;
    dist_t *data_numpy_d;
    {
      py::gil_scoped_release l;

      // avoid using threads when the number of searches is small:

      if (rows <= (unsigned)num_threads * 4)
      {
        num_threads = 1;
      }

      data_numpy_l = new hannlib::labeltype[rows * k];
      data_numpy_d = new dist_t[rows * k];

      for (unsigned i = 0; i < rows * k; i++)
      {
        data_numpy_l[i] = 0;
        data_numpy_d[i] = -1;
      }

      if (normalize == false)
      {
        ParallelFor(0, rows, num_threads,
                    [&](size_t row, size_t threadId)
                    {
                      int64_t low  = ranges_ptr[row * 2];
                      int64_t high = ranges_ptr[row * 2 + 1];

                      std::priority_queue<std::pair<dist_t, hannlib::labeltype>>
                          result = appr_alg->HybridFiltering(
                              (void *)query_py_array.data(row), k,
                              std::make_pair(low, high));

                      for (int i = result.size() - 1; i >= 0; i--)
                      {
                        auto &result_tuple        = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                      }
                    });
      }
      else
      {
        std::vector<float> norm_array(num_threads * features);
        ParallelFor(
            0, rows, num_threads,
            [&](size_t row, size_t threadId)
            {
              int64_t low  = ranges_ptr[row * 2];
              int64_t high = ranges_ptr[row * 2 + 1];

              size_t start_idx = threadId * dim;
              NormalizeVector((float *)query_py_array.data(row),
                              (norm_array.data() + start_idx));

              std::priority_queue<std::pair<dist_t, hannlib::labeltype>> result =
                  appr_alg
                      ->HybridFiltering(  // VectorDrivenHybridSearch在hybrid_hnsw.py中
                          (void *)(norm_array.data() + start_idx), k,
                          std::make_pair(low, high));

              for (int i = result.size() - 1; i >= 0; i--)
              {
                auto &result_tuple        = result.top();
                data_numpy_d[row * k + i] = result_tuple.first;
                data_numpy_l[row * k + i] = result_tuple.second;
                result.pop();
              }
            });
      }
    }
    py::capsule free_when_done_l(data_numpy_l, [](void *f) { delete[] f; });
    py::capsule free_when_done_d(data_numpy_d, [](void *f) { delete[] f; });

    return py::make_tuple(
        py::array_t<hannlib::labeltype>(
            {rows, k},  // shape
            {k * sizeof(hannlib::labeltype),
             sizeof(hannlib::labeltype)},  // C-style contiguous strides for
                                           // double
            data_numpy_l,                  // the data pointer
            free_when_done_l),
        py::array_t<dist_t>(
            {rows, k},  // shape
            {k * sizeof(dist_t),
             sizeof(dist_t)},  // C-style contiguous strides for double
            data_numpy_d,      // the data pointer
            free_when_done_d));
  }

  py::object KnnSearch(py::object query_py_object, size_t k = 1)
  {
    // Query vectors
    py::array_t<dist_t, py::array::c_style | py::array::forcecast>
        query_py_array(query_py_object);
    auto query_buffer = query_py_array.request();
    if (query_buffer.ndim != 1)
      throw std::runtime_error("data must be a 1d array");
    size_t features = query_buffer.shape[0];

    if (features != (unsigned)dim)
      throw std::runtime_error("wrong dimensionality of the vectors");

    hannlib::labeltype *data_numpy_l;
    dist_t *data_numpy_d;

    data_numpy_l = new hannlib::labeltype[k];
    data_numpy_d = new dist_t[k];

    for (int i = 0; i < k; i++)
    {
      data_numpy_l[i] = 0;
      data_numpy_d[i] = -1;
    }

    std::priority_queue<std::pair<dist_t, hannlib::labeltype>> result;

    if (normalize)
    {
      std::vector<float> norm_array(features);
      NormalizeVector((float *)query_py_array.data(), norm_array.data());
      result = appr_alg->KnnSearch((void *)norm_array.data(), k);
    }
    else
    {
      result = appr_alg->KnnSearch((void *)query_py_array.data(), k);
    }

    for (int i = result.size() - 1; i >= 0; i--)
    {
      auto &result_tuple = result.top();
      data_numpy_d[i]    = result_tuple.first;
      data_numpy_l[i]    = result_tuple.second;
      result.pop();
    }

    py::capsule free_when_done_l(data_numpy_l, [](void *f) { delete[] f; });
    py::capsule free_when_done_d(data_numpy_d, [](void *f) { delete[] f; });

    return py::make_tuple(
        py::array_t<hannlib::labeltype>(
            {k},                           // shape
            {sizeof(hannlib::labeltype)},  // C-style contiguous strides for
                                           // double
            data_numpy_l,                  // the data pointer
            free_when_done_l),
        py::array_t<dist_t>(
            {k},               // shape
            {sizeof(dist_t)},  // C-style contiguous strides for double
            data_numpy_d,      // the data pointer
            free_when_done_d));
  }

  py::object KnnSearchBatch(py::object query_py_object, size_t k = 1,
                            int num_threads = -1)
  {
    // Query vectors
    py::array_t<dist_t, py::array::c_style | py::array::forcecast>
        query_py_array(query_py_object);
    auto query_buffer = query_py_array.request();

    if (query_buffer.ndim != 2)
      throw std::runtime_error("query must be a 2d array");
    size_t rows     = query_buffer.shape[0];
    size_t features = query_buffer.shape[1];

    if (num_threads <= 0) num_threads = num_threads_default;

    hannlib::labeltype *data_numpy_l;
    dist_t *data_numpy_d;
    {
      py::gil_scoped_release l;

      // avoid using threads when the number of searches is small:

      if (rows <= (unsigned)num_threads * 4)
      {
        num_threads = 1;
      }

      data_numpy_l = new hannlib::labeltype[rows * k];
      data_numpy_d = new dist_t[rows * k];

      for (unsigned i = 0; i < rows * k; i++)
      {
        data_numpy_l[i] = 0;
        data_numpy_d[i] = -1;
      }

      if (normalize == false)
      {
        ParallelFor(0, rows, num_threads,
                    [&](size_t row, size_t threadId)
                    {
                      std::priority_queue<std::pair<dist_t, hannlib::labeltype>>
                          result = appr_alg->KnnSearch(
                              (void *)query_py_array.data(row), k);

                      for (int i = result.size() - 1; i >= 0; i--)
                      {
                        auto &result_tuple        = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                      }
                    });
      }
      else
      {
        std::vector<float> norm_array(num_threads * features);
        ParallelFor(0, rows, num_threads,
                    [&](size_t row, size_t threadId)
                    {
                      size_t start_idx = threadId * dim;
                      NormalizeVector((float *)query_py_array.data(row),
                                      (norm_array.data() + start_idx));

                      std::priority_queue<std::pair<dist_t, hannlib::labeltype>>
                          result = appr_alg->KnnSearch(
                              (void *)(norm_array.data() + start_idx), k);

                      for (int i = result.size() - 1; i >= 0; i--)
                      {
                        auto &result_tuple        = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                      }
                    });
      }
    }
    py::capsule free_when_done_l(data_numpy_l, [](void *f) { delete[] f; });
    py::capsule free_when_done_d(data_numpy_d, [](void *f) { delete[] f; });

    return py::make_tuple(
        py::array_t<hannlib::labeltype>(
            {rows, k},  // shape
            {k * sizeof(hannlib::labeltype),
             sizeof(hannlib::labeltype)},  // C-style contiguous strides for
                                           // double
            data_numpy_l,                  // the data pointer
            free_when_done_l),
        py::array_t<dist_t>(
            {rows, k},  // shape
            {k * sizeof(dist_t),
             sizeof(dist_t)},  // C-style contiguous strides for double
            data_numpy_d,      // the data pointer
            free_when_done_d));
  }
  void SaveIndex(const std::string &path_to_index)
  {
    appr_alg->SaveIndex(path_to_index);
  }

  void LoadIndex(const std::string &path_to_index, size_t max_elements)
  {
    if (appr_alg)
    {
      std::cerr << "Warning: Calling load_index for an already inited index. "
                   "Old index is being deallocated."
                << std::endl;
      delete appr_alg;
    }
    appr_alg = new hannlib::ScalarHSIG<dist_t>(l2space, path_to_index,
                                                       false, max_elements);
    cur_l    = appr_alg->get_current_count();
    index_inited = true;
  }

  size_t get_max_elements() const
  {
    AssertIndexInited();
    return appr_alg->get_max_elements();
  }

  size_t get_current_count() const
  {
    AssertIndexInited();
    return appr_alg->get_current_count();
  }

  size_t get_s() const
  {
    AssertIndexInited();
    return appr_alg->get_s();
  }

  size_t get_m() const
  {
    AssertIndexInited();
    return appr_alg->get_m();
  }

  size_t get_ef_construction() const
  {
    AssertIndexInited();
    return appr_alg->get_ef_construction();
  }

  std::string get_space_name() const { return space_name; }

  int get_dim() const { return dim; }

  size_t get_ef() const
  {
    AssertIndexInited();
    return appr_alg->get_ef();
  }

  void set_ef(size_t ef)
  {
    AssertIndexInited();
    appr_alg->set_ef(ef);
  }

  size_t get_al() const
  {
    AssertIndexInited();
    return appr_alg->get_al();
  }

  void set_al(size_t al)
  {
    AssertIndexInited();
    appr_alg->set_al(al);
  }

  void set_target_recall(float recall)
  {
    AssertIndexInited();
    appr_alg->set_target_recall(recall);
  }

  void set_search_strategy(int value)
  {
    AssertIndexInited();
    appr_alg->set_search_strategy(value);
  }

   void set_low_range(float low_range)
  {
    AssertIndexInited();
    appr_alg->set_low_range(low_range);
  }

  void set_high_range(float high_range)
  {
    AssertIndexInited();
    appr_alg->set_high_range(high_range);
  }

  int get_num_threads() const { return num_threads_default; }

  void set_num_threads(int num_threads)
  {
    // std::cout << "set_num_threads: " << num_threads << std::endl;
    // std::cout.flush();
    this->num_threads_default = num_threads;
  }

 private:
  void NormalizeVector(float *data, float *norm_array)
  {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += data[i] * data[i];
    norm = 1.0f / (sqrtf(norm) + 1e-30f);
    for (int i = 0; i < dim; i++) norm_array[i] = data[i] * norm;
  }

  void AssertIndexInited() const
  {
    if (appr_alg == nullptr)
    {
      throw std::runtime_error("index not initialized");
    }
  }

 private:
  static const int ser_version = 1;  // serialization version

  std::string space_name;
  int dim;
  size_t seed;

  bool index_inited;
  bool ep_added;
  bool normalize;
  int num_threads_default;
  hannlib::labeltype cur_l;
  hannlib::ScalarHSIG<dist_t> *appr_alg;
  hannlib::SpaceInterface<float> *l2space;
};

PYBIND11_PLUGIN(hannlib)
{
  py::module m("hannlib");

  py::class_<HybridIndex<float>>(m, "HybridIndex")
      .def(py::init<const std::string &, const int>(), py::arg("space"),
           py::arg("dim"))
      .def("init_index", &HybridIndex<float>::InitNewIndex,
           py::arg("slot_ranges"), py::arg("max_elements"), py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("random_seed") = 100)
      .def("hybrid_query_batch", &HybridIndex<float>::HybridSearchBatch,
           py::arg("data"), py::arg("ranges"), py::arg("k") = 1,
           py::arg("num_threads") = -1)
      .def("hybrid_query", &HybridIndex<float>::HybridSearch, py::arg("data"),
           py::arg("ranges"), py::arg("k") = 1)
      .def("knn_query_batch", &HybridIndex<float>::KnnSearchBatch,
           py::arg("data"), py::arg("k") = 1, py::arg("num_threads") = -1)
      .def("knn_query", &HybridIndex<float>::KnnSearch, py::arg("data"),
           py::arg("k") = 1)
      .def("add_items", &HybridIndex<float>::AddItems, py::arg("data"),
           py::arg("scalars"), py::arg("ids") = py::none(),
           py::arg("num_threads") = -1)
      .def("set_ef", &HybridIndex<float>::set_ef, py::arg("ef"))
      .def("set_al", &HybridIndex<float>::set_al, py::arg("al"))
      .def("set_target_recall", &HybridIndex<float>::set_target_recall,
           py::arg("recall"))
      .def("set_low_range", &HybridIndex<float>::set_low_range,
           py::arg("low_range"))
      .def("set_high_range", &HybridIndex<float>::set_high_range,
           py::arg("high_range"))
      .def("set_search_strategy", &HybridIndex<float>::set_search_strategy,
           py::arg("stragety_code"))
      .def("load_optimizer_conf", &HybridIndex<float>::LoadOptimizerConf,
           py::arg("path"))
      .def("save_index", &HybridIndex<float>::SaveIndex,
           py::arg("path_to_index"))
      .def("load_index", &HybridIndex<float>::LoadIndex,
           py::arg("path_to_index"), py::arg("max_elements") = 0)
      .def("set_num_threads", &HybridIndex<float>::set_num_threads)
      .def_property(
          "num_threads",
          [](const HybridIndex<float> &index)
          { return index.get_num_threads(); },
          [](HybridIndex<float> &index, const int n)
          { index.set_num_threads(n); })
      .def_property(
          "ef", [](const HybridIndex<float> &index) { return index.get_ef(); },
          [](HybridIndex<float> &index, const size_t ef) { index.set_ef(ef); })
      .def_property(
          "al", [](const HybridIndex<float> &index) { return index.get_al(); },
          [](HybridIndex<float> &index, const size_t al) { index.set_al(al); })
      .def_property_readonly(
          "M", [](const HybridIndex<float> &index) { return index.get_m(); })
      .def_property_readonly(
          "S", [](const HybridIndex<float> &index) { return index.get_s(); })
      .def_property_readonly("space", [](const HybridIndex<float> &index)
                             { return index.get_space_name(); })
      .def_property_readonly("dim", [](const HybridIndex<float> &index)
                             { return index.get_dim(); })
      .def_property_readonly("max_elements", [](const HybridIndex<float> &index)
                             { return index.get_max_elements(); })
      .def_property_readonly("element_count",
                             [](const HybridIndex<float> &index)
                             { return index.get_current_count(); })
      .def_property_readonly("ef_construction",
                             [](const HybridIndex<float> &index)
                             { return index.get_ef_construction(); })
      .def("__repr__",
           [](const HybridIndex<float> &a)
           {
             return "<hannlib.HybridIndex(space='" + a.get_space_name() +
                    "', dim=" + std::to_string(a.get_dim()) + ")>";
           });

  return m.ptr();
}
