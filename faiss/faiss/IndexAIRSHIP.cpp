/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/sorting.h>

namespace faiss {





} // namespace faiss



/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexAIRSHIP.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>

#include <sys/stat.h>
#include <sys/types.h>
#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

namespace faiss {

using MinimaxHeap = AIRSHIP::MinimaxHeap;
using storage_idx_t = AIRSHIP::storage_idx_t;
using NodeDistFarther = AIRSHIP::NodeDistFarther;

AIRSHIPStats airship_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

void airship_add_vertices(
        IndexAIRSHIP& index_airship,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_airship.d;
    AIRSHIP& airship = index_airship.airship;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("airship_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = airship.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = airship.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = airship.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_airship.d * airship.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1;
             pt_level >= !index_airship.init_level0;
             pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_airship.storage));
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    airship.add_with_locks(
                            *dis,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_airship.keep_max_size_level0 && (pt_level == 0));

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        if (index_airship.init_level0) {
            FAISS_ASSERT(i1 == 0);
        } else {
            FAISS_ASSERT((i1 - hist[0]) == 0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexAIRSHIP implementation
 **************************************************************/

IndexAIRSHIP::IndexAIRSHIP(int d, int M, MetricType metric)
        : Index(d, metric), airship(M) {}

IndexAIRSHIP::IndexAIRSHIP(Index* storage, int M)
        : Index(storage->d, storage->metric_type), airship(M), storage(storage) {}

IndexAIRSHIP::~IndexAIRSHIP() {
    if (own_fields) {
        delete storage;
    }
}

void IndexAIRSHIP::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexAIRSHIPFlat (or variants) instead of IndexAIRSHIP directly");
    // airship structure does not require training
    storage->train(n, x);
    is_trained = true;
}

namespace {

template <class BlockResultHandler>
void airship_search(
        const IndexAIRSHIP* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params_in) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexAIRSHIPFlat (or variants) "
            "instead of IndexAIRSHIP directly");
    const SearchParametersAIRSHIP* params = nullptr;
    const AIRSHIP& airship = index->airship;

    int efSearch = airship.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersAIRSHIP*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, ndis = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            airship.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(index->ntotal);
            typename BlockResultHandler::SingleResultHandler res(bres);

            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index->storage));

#pragma omp for reduction(+ : n1, n2, ndis) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                dis->set_query(x + i * index->d);

                AIRSHIPStats stats = airship.search(*dis, res, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                res.end();
            }
        }
        InterruptCallback::check();
    }

    airship_stats.combine({n1, n2, ndis});
}

} // anonymous namespace

void IndexAIRSHIP::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<AIRSHIP::C>;
    RH bres(n, distances, labels, k);

    airship_search(this, n, x, bres, params_in);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexAIRSHIP::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<AIRSHIP::C>;
    RH bres(result, is_similarity_metric(metric_type) ? -radius : radius);

    airship_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexAIRSHIP::add(idx_t n, const float* x) {
}

void IndexAIRSHIP::add(idx_t n, const float* x, const int* _attr) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexAIRSHIPFlat (or variants) instead of IndexAIRSHIP directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;


    airship.attr.resize(n);
    memcpy(airship.attr.data(), _attr, n * sizeof(int));

    prepare_level(n, airship.attr, this->airship);

    airship_add_vertices(*this, n0, n, x, verbose, airship.levels.size() == ntotal);
}

std::vector<idx_t> sort_indices(const std::vector<int>& nums) {
    // Create a vector of indices
    std::vector<idx_t> indices(nums.size());
    for (idx_t i = 0; i < nums.size(); ++i) {
        indices[i] = i;  // Initialize the indices to 0, 1, 2, ...
    }

    // Sort the indices based on the corresponding values in nums
    sort(indices.begin(), indices.end(), [&nums](idx_t i1, idx_t i2) {
        return nums[i1] < nums[i2];  // Compare the values at the indices
    });

    return indices;  // Return the sorted indices
}

std::vector<idx_t> sample_randomly(const std::vector<idx_t>& nums, double percentage) {
    // Calculate the number of items to sample based on the given percentage
    int num_to_sample = std::max(static_cast<idx_t>(nums.size() * percentage), (idx_t)1); // Ensure at least 1 element is selected

    // Randomly generate unique indices
    std::vector<int> indices(nums.size());
    for (int i = 0; i < nums.size(); ++i) {
        indices[i] = i;  // Initialize the indices with 0, 1, 2, ...
    }

    // Shuffle the indices using a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    shuffle(indices.begin(), indices.end(), gen);

    // Select the first 'num_to_sample' indices
    std::vector<idx_t> sampled_values;
    for (int i = 0; i < num_to_sample; ++i) {
        sampled_values.push_back(nums[indices[i]]);
    }

    return sampled_values;
}

int IndexAIRSHIP::prepare_level(size_t n, const std::vector<int>& attr, AIRSHIP& airship) {

    // stat the number of attrs
    // std::map<int, vector<idx_t>> frequency;

    int num_buckets = 1000;
    std::vector<idx_t> indices = sort_indices(attr);
    std::vector<std::vector<idx_t>> buckets;
    int num_elements_per_bucket = ceil(attr.size() * 1.0 / num_buckets);
    int current_idx = 0;
    int cur_bucket = 0;
    for (; cur_bucket < num_buckets; ++cur_bucket) {
        buckets.push_back(std::vector<idx_t>());
        for (int j = 0; j < num_elements_per_bucket; ++j) {
            buckets[cur_bucket].push_back(indices[current_idx++]);
            if (current_idx == attr.size()) break;
        }
        while ((attr[buckets[cur_bucket].back()] == attr[indices[current_idx]] || cur_bucket+1 == num_buckets) && current_idx < attr.size()) {
            buckets[cur_bucket].push_back(indices[current_idx++]);
        }
        if (current_idx == attr.size()) break;
    }
    num_buckets = cur_bucket+1;
        


    std::cout << " Size of buckets: " << num_buckets << std::endl;
    
    double sample_rate = 0.0001; // 0.01% sampled data in each bucket will come about in top layer, but no less than 1.
    
    std::vector<idx_t> sampled_indices;
    for (int i = 0; i < num_buckets; i++) {
        std::vector<idx_t> sampled_indices_in_bucket = sample_randomly(buckets[i], sample_rate);
        sampled_indices.insert(sampled_indices.end(), sampled_indices_in_bucket.begin(), sampled_indices_in_bucket.end());
    }

    size_t n0 = airship.offsets.size() - 1;


    FAISS_ASSERT(n0 == airship.levels.size());
    for (int i = 0; i < n; i++) {
        int pt_level = airship.random_level();
        airship.levels.push_back(pt_level + 1);
    }

    int max_level = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = airship.levels[i + n0] - 1;
        if (pt_level > max_level)
            max_level = pt_level;
    }
    
    for (int i = 0; i < sampled_indices.size(); i++) {
        airship.levels[i+n0] = max_level + 1;
    }


    for (int i = 0; i < n; i++) {
        int pt_level = airship.levels[i + n0] - 1;
        airship.offsets.push_back(airship.offsets.back() + airship.cum_nb_neighbors(pt_level + 1));
    }
    airship.neighbors.resize(airship.offsets.back(), -1);
    airship.sampled_entry_points.clear();
    airship.sampled_entry_points.insert(airship.sampled_entry_points.end(), sampled_indices.begin(), sampled_indices.end());
    airship.upper_beam = airship.sampled_entry_points.size();
    return max_level;
}

void IndexAIRSHIP::reset() {
    airship.reset();
    storage->reset();
    ntotal = 0;
}

void IndexAIRSHIP::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexAIRSHIP::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            airship.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = airship.neighbors[j];
                if (v1 < 0)
                    break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            AIRSHIP::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    airship.neighbors[j] = shrunk_list[j - begin].id;
                else
                    airship.neighbors[j] = -1;
            }
        }
    }
}

void IndexAIRSHIP::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const SearchParametersAIRSHIP* params = nullptr;

    if (params_in) {
        params = dynamic_cast<const SearchParametersAIRSHIP*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
    }

    storage_idx_t ntotal = airship.levels.size();

    using RH = HeapBlockResultHandler<AIRSHIP::C>;
    RH bres(n, distances, labels, k);

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> qdis(
                storage_distance_computer(storage));
        AIRSHIPStats search_stats;
        VisitedTable vt(ntotal);
        RH::SingleResultHandler res(bres);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            qdis->set_query(x + i * d);

            airship.search_level_0(
                    *qdis.get(),
                    res,
                    nprobe,
                    nearest + i * nprobe,
                    nearest_d + i * nprobe,
                    search_type,
                    search_stats,
                    vt,
                    params);
            res.end();
            vt.advance();
        }
#pragma omp critical
        { airship_stats.combine(search_stats); }
    }
    if (is_similarity_metric(this->metric_type)) {
// we need to revert the negated distances
#pragma omp parallel for
        for (int64_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexAIRSHIP::init_level_0_from_knngraph(
        int k,
        const float* D,
        const idx_t* I) {
    int dest_size = airship.nb_neighbors(0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer* qdis = storage_distance_computer(storage);
        std::vector<float> vec(d);
        storage->reconstruct(i, vec.data());
        qdis->set_query(vec.data());

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i)
                continue;
            if (v1 < 0)
                break;
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        AIRSHIP::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        airship.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                airship.neighbors[j] = shrunk_list[j - begin].id;
            else
                airship.neighbors[j] = -1;
        }
    }
}

void IndexAIRSHIP::init_level_0_from_entry_points(
        int n,
        const storage_idx_t* points,
        const storage_idx_t* nearests) {
    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

#pragma omp parallel
    {
        VisitedTable vt(ntotal);

        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));
        std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct(pt_id, vec.data());
            dis->set_query(vec.data());

            airship.add_links_starting_from(
                    *dis, pt_id, nearest, (*dis)(nearest), 0, locks.data(), vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for (int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);
}

void IndexAIRSHIP::reorder_links() {
    int M = airship.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances(M);
        std::vector<size_t> order(M);
        std::vector<storage_idx_t> tmp(M);
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (storage_idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            airship.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = airship.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp[j - begin] = nj;
            }

            fvec_argsort(end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                airship.neighbors[j] = tmp[order[j - begin]];
            }
        }
    }
}

void IndexAIRSHIP::link_singletons() {
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        airship.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = airship.neighbors[j];
            if (ni >= 0)
                seen[ni] = true;
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (airship.levels[i] > 1)
                n_sing_l1++;
        }
    }

    printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
           n_sing,
           ntotal,
           n_sing_l1);

    std::vector<float> recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {
        FAISS_ASSERT(!"not implemented");
    }
}

void IndexAIRSHIP::permute_entries(const idx_t* perm) {
    auto flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_storage, "don't know how to permute this index");
    flat_storage->permute_entries(perm);
    airship.permute_entries(perm);
}

/**************************************************************
 * IndexAIRSHIPFlat implementation
 **************************************************************/

IndexAIRSHIPFlat::IndexAIRSHIPFlat() {
    is_trained = true;
}

IndexAIRSHIPFlat::IndexAIRSHIPFlat(int d, int M, MetricType metric)
        : IndexAIRSHIP(
                  (metric == METRIC_L2) ? new IndexFlatL2(d)
                                        : new IndexFlat(d, metric),
                  M) {
    own_fields = true;
    is_trained = true;
}

} // namespace faiss
