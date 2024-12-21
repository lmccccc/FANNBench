from __future__ import absolute_import

from multiprocessing.pool import ThreadPool

import psutil


class BaseANN(object):
    def done(self):
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X, scalars):
        pass

    def knn_query(self, q, n):
        return []  # array of candidate indices

    def hybrid_query(self, q, interval, n):
        return []  # array of candidate indices

    def batch_knn_query(self, X, n):
        """Provide all queries at once and let algorithm figure out
        how to handle it. Default implementation uses a ThreadPool
        to parallelize query processing."""
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.knn_query(q, n), X)

    def batch_hybrid_query(self, X, ranges, n):
        """Provide all queries at once and let algorithm figure out
        how to handle it. Default implementation uses a ThreadPool
        to parallelize query processing."""
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.hybrid_query(q[0], q[1], n), zip(X, ranges))

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {}

    def saveIndex(self, location):
        raise NotImplemented

    def loadIndex(self, X, location):
        raise NotImplemented

    def __str__(self):
        return self.name
