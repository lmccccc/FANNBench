import itertools
import math
import os
from itertools import chain
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from utils import ParallelMapper


# 保留recall四位小数，将recall浮点数转为整数
def recall_to_int(recall):
    return round(recall * 10000)


def clean_data(xdata, ydata):
    seen = set()
    unique_xdata = []
    unique_ydata = []

    for x, y in zip(xdata, ydata):
        if x not in seen:
            seen.add(x)
            unique_xdata.append(x)
            unique_ydata.append(y)

    return unique_xdata, unique_ydata


def safe_curve_fit(k, i, j, func, x, y):
    # return np.polyfit(x, y, deg=3), 0
    try:
        x, y = clean_data(x, y)
        x, y = np.array(x), np.array(y)
        if len(x) <= 4:
            coeff = np.polyfit(x, y, deg=1)
            return (
                coeff[0],
                0,
                0,
                coeff[1] - 1,
            ), 0
        mask = x > 0.9
        # we have more than 8 points whose recall > 90
        if mask.sum() > 8:
            x = x[mask]
            y = y[mask]

        return curve_fit(
            func,
            x,
            y,
            bounds=([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=100000,
        )

    except RuntimeError as e:
        # Trick: if we cannot fit the exponential curve,
        # use linear curve instead
        coeff = np.polyfit(x, y, deg=1)
        return (
            coeff[0],
            0,
            0,
            coeff[1] - 1,
        ), 0


class RecallCurveFitter:

    @staticmethod
    def output_latency(recall, a, b, c, d):
        # return np.polyval([a, b, c, d], recall)
        return a * recall + np.exp(b * recall + c) + d

    @staticmethod
    def output_ef(recall, a, b, c, d):
        # return np.polyval([a, b, c, d], recall)
        return a * recall + np.exp(b * recall + c) + d

    def fit(self, k, i, j, result_df):
        al_list = set(result_df["al"].values)

        best_al = -1
        best_latency95 = np.inf
        best_latency_curve_params = None

        # 首先寻找最优的al配置：
        # 通过拟合recall-latency曲线，并预测recall=0.95时的latency，
        # 这个latency越小表示al的参数越好
        for al in al_list:
            if al <= 8:
                continue
            sub_df = result_df.loc[
                (result_df["al"] == al).values & (result_df["k"] == k).values
            ]
            time = sub_df["latency(ms)"].values * 1000  # Predict us instead of ms
            recall = sub_df["recall"].values
            ef = sub_df["ef"].values

            latency_curve_params, _ = safe_curve_fit(
                k,
                i,
                j,
                RecallCurveFitter.output_latency,
                recall,
                time,
            )

            latency95 = RecallCurveFitter.output_latency(0.95, *latency_curve_params)
            if latency95 < best_latency95:
                best_al = al
                best_latency95 = latency95
                best_latency_curve_params = latency_curve_params

        # 固定al，拟合recall-ef曲线
        sub_df = result_df.loc[
            (result_df["al"] == best_al).values & (result_df["k"] == k).values
        ]
        recall = sub_df["recall"].values
        ef = sub_df["ef"].values
        ef_curve_params, _ = safe_curve_fit(
            k, i, j, RecallCurveFitter.output_ef, recall, ef
        )

        # 返回最优的al配置，recall-latency曲线的参数，recall-ef曲线的参数
        return best_al, best_latency_curve_params, ef_curve_params


class OpParams:
    def __init__(
        self,
        k_max,
        B,
        al_table,
        graph_latency_curve_params_table,
        graph_ef_curve_params_table,
        histogram: pd.DataFrame,
        skiplist_latency_params: List[float],
    ) -> None:
        self.k_max = k_max  # Optimizer允许的最大k值，超过这个值的kNN查询将得不到优化
        self.B = B  # 分桶数
        self.al_table = al_table  # al的最优配置
        self.latency_curve_params_table = (
            graph_latency_curve_params_table  # recall-latency曲线的参数
        )
        self.ef_curve_params_table = graph_ef_curve_params_table  # recall-ef曲线的参数
        self.histogram = histogram
        self.skiplist_latency_params = skiplist_latency_params

    def save(self, save_dir):
        self._save_al(self.k_max, B, self.al_table, f"{params_save_dir}/al.csv")
        self._save_params(
            self.k_max,
            self.B,
            self.latency_curve_params_table,
            f"{params_save_dir}/latency_params.csv",
        )
        self._save_params(
            self.k_max,
            self.B,
            self.ef_curve_params_table,
            f"{params_save_dir}/ef_params.csv",
        )

        self.histogram.to_csv(f"{params_save_dir}/hist.csv", index=None)
        df = pd.DataFrame(
            [
                {
                    "k": self.k_max,
                    "i": 0,
                    "j": 0,
                    "a": self.skiplist_latency_params[0],
                    "b": self.skiplist_latency_params[1],
                    "c": 0,
                    "d": 0,
                }
            ]
        )
        df.to_csv(
            f"{params_save_dir}/skiplist_latency_params.csv",
            float_format="%.8f",
            index=None,
        )

    @staticmethod
    def load(k_max, B, dir_name) -> "OpParams":
        al_table = OpParams._load_al(k_max, B, f"{dir_name}/al.csv")
        latency_curve_params_table = OpParams._load_params(
            k_max, B, f"{dir_name}/latency_params.csv"
        )
        ef_curve_params_table = OpParams._load_params(
            k_max, B, f"{dir_name}/ef_params.csv"
        )
        histogram = pd.read_csv(f"{params_save_dir}/hist.csv")
        df = pd.read_csv(f"{params_save_dir}/skiplist_latency_params.csv")
        skiplist_latency_params = [df["a"].values[0], df["b"].values[0]]

        return OpParams(
            k_max,
            B,
            al_table,
            latency_curve_params_table,
            ef_curve_params_table,
            histogram,
            skiplist_latency_params,
        )

    @staticmethod
    def _load_params(k_max, B, file_path):
        curve_params_table = np.zeros([k_max + 1, B, B], dtype=object)
        curve_params_table[:] = None
        df = pd.read_csv(file_path)
        print(df)
        for _, row in df.iterrows():
            k, i, j, a, b, c, d = (
                row["k"],
                row["i"],
                row["j"],
                row["a"],
                row["b"],
                row["c"],
                row["d"],
            )
            k, i, j = int(k), int(i), int(j)
            curve_params_table[k, i, j] = a, b, c, d
        return curve_params_table

    @staticmethod
    def _save_params(k_max, B, params_table, save_path):
        data = []
        for k in range(1, k_max + 1):
            for i in range(B):
                for j in range(i, B):
                    a, b, c, d = params_table[k, i, j]
                    data.append(
                        {"k": k, "i": i, "j": j, "a": a, "b": b, "c": c, "d": d}
                    )
        df = pd.DataFrame(data)
        df.to_csv(save_path, float_format="%.8f", index=False)

    @staticmethod
    def _save_al(k_max, B, al_table, save_path):
        data = []
        for k in range(1, k_max + 1):
            for i in range(B):
                for j in range(i, B):
                    al = al_table[k, i, j]
                    data.append({"k": k, "i": i, "j": j, "al": al})
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)

    @staticmethod
    def _load_al(k_max, B, file_path):
        al_table = np.zeros([k_max + 1, B, B], dtype="int64")
        df = pd.read_csv(file_path)
        print(df)
        for _, row in df.iterrows():
            k, i, j, al = (
                row["k"],
                row["i"],
                row["j"],
                row["al"],
            )
            al_table[k, i, j] = al
        return al_table


class Optimizer:
    @staticmethod
    def compute_rso_conf(
        k_factor: int, recall_list: List[float], op_params: OpParams
    ) -> pd.DataFrame:
        """计算range split optimize需要的配置表"""
        data = []
        for recall in recall_list:
            cost_table = Optimizer.compute_cost_table(recall, op_params)
            dp, op_result = Optimizer._dp_optimize(k_factor, cost_table, op_params)
            one_data = Optimizer.encode_op_result(recall, op_params, op_result)
            data.extend(one_data)
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def _dp_optimize(
        k_factor: int, cost_table: np.ndarray, op_params: OpParams
    ) -> Tuple[np.ndarray, np.ndarray]:
        s = cost_table
        k_max = op_params.k_max
        assert s.shape[0] >= k_max
        assert s.shape[1] == s.shape[2]
        B = s.shape[1]
        assert B == op_params.B

        dp = np.zeros([k_max + 1, B, B], dtype="float32")
        result = np.zeros([k_max + 1, B, B], dtype=object)
        result.fill([])

        for k in range(1, k_max + 1):
            for j in range(B):
                for i in range(j, -1, -1):
                    if i == j:
                        dp[k, i, j] = s[k, i, j]
                        result[k, i, j] = [(k, i, j)]
                    else:
                        # print(f"i={i}, j={j}")
                        # print("-"*40)
                        min_cost = s[k, i, j]
                        result[k, i, j] = [(k, i, j)]
                        # print(min_cost)
                        for h in range(j, i, -1):
                            # print(f"h={h}")
                            num_total_slots = j - i + 1
                            k_per_slot = k / num_total_slots

                            # 根据左右两边涉及的桶数，计算相应的k值
                            # 例子：求10NN，左边1个桶，右边9个桶
                            # k_factor=1的情况下，在左边1个桶里搜1NN，右边9个桶里搜9NN
                            # k_factor=2的情况下，在左边1个桶里搜2NN，右边9个桶里搜18NN
                            left_num_slots = (h - 1) - i + 1
                            left_k = min(
                                math.ceil(left_num_slots * k_per_slot) * k_factor, k
                            )
                            right_num_slots = j - h + 1
                            right_k = min(
                                math.ceil(right_num_slots * k_per_slot) * k_factor, k
                            )
                            cost = dp[left_k, i, h - 1] + dp[right_k, h, j]

                            # print(f"[{i},{h-1}] + [{h},{j}] = {cost}")
                            if cost < min_cost:
                                min_cost = cost
                                result[k, i, j] = result[left_k, i, h - 1] + [
                                    (right_k, h, j)
                                ]

                        dp[k, i, j] = min_cost

        return dp, result

    @staticmethod
    def compute_cost_table(target_recall: float, op_params: OpParams):
        """计算指定recall下, 搜索各索引的cost"""
        k_max, B = op_params.k_max, op_params.B
        cost_table = np.zeros([k_max + 1, B, B], dtype="float32")
        for k in range(1, k_max + 1):
            for i in range(B):
                for j in range(i, B):
                    params = op_params.latency_curve_params_table[k, i, j]
                    cost_table[k, i, j] = RecallCurveFitter.output_latency(
                        target_recall, *params
                    )
        return cost_table

    @staticmethod
    def encode_op_result(recall: float, op_params: OpParams, op_result: np.ndarray):
        k_max, B = op_params.k_max, op_params.B
        recall_int = recall_to_int(recall)
        data = []
        for k in range(1, k_max + 1):
            for i in range(B):
                for j in range(i, B):
                    divisions = op_result[k, i, j]
                    division_str = ";".join([f"{l}_{r}" for _, l, r in divisions])
                    ef_list = []
                    al_list = []
                    latency = 0
                    for kk, ii, jj in divisions:
                        ef = RecallCurveFitter.output_ef(
                            recall, *op_params.ef_curve_params_table[kk, ii, jj]
                        )
                        ef = math.ceil(ef)
                        latency += RecallCurveFitter.output_latency(
                            recall, *op_params.latency_curve_params_table[kk, ii, jj]
                        )
                        # Ensure ef > 0
                        ef = ef if ef > 0 else k
                        ef_list.append(f"{ef}")
                        al = op_params.al_table[kk, ii, jj]
                        al_list.append(f"{al}")
                    ef_str = ";".join(ef_list)
                    al_str = ";".join(al_list)
                    item = {
                        "k": k,
                        "i": i,
                        "j": j,
                        "recall": recall_int,
                        "division": division_str,
                        "ef": ef_str,
                        "al": al_str,
                        "latency(ms)": latency,
                    }
                    data.append(item)
        return data


def fit_file(item):
    stat_dir, k_max, i, j = item
    file = f"{stat_dir}/stat_k_max={k_max}_slot_{i}_to_{j}.csv"
    df = pd.read_csv(file)

    print(f"Processing {file}...")
    result_list = []
    for k in range(1, k_max + 1):
        fitter = RecallCurveFitter()
        al, latency_curve_params, ef_curve_params = fitter.fit(k, i, j, df)
        result_list.append(((k, i, j), al, latency_curve_params, ef_curve_params))
    return result_list


# 四个功能：
# 1、获取最优的al配置
# 2、拟合recall-latency曲线
# 3、拟合recall-ef曲线
# 4、拟合跳表的cardinality-latency曲线
def fit_curve(stat_dir: str, k_max: int, B: int, num_threads: int = -1):
    if num_threads == -1:
        num_threads = os.cpu_count() - 1

    # 拟合图索引的相关曲线
    inputs = []
    for i in range(B):
        for j in range(i, B):
            inputs.append((stat_dir, k_max, i, j))

    mapper = ParallelMapper(num_threads)
    fit_results = mapper.map(fit_file, inputs)

    al_table = np.zeros([k_max + 1, B, B], dtype="int64")
    latency_curve_params_table = np.zeros([k_max + 1, B, B], dtype=object)
    latency_curve_params_table[:] = None
    ef_curve_params_table = np.zeros([k_max + 1, B, B], dtype=object)
    ef_curve_params_table[:] = None
    for (k, i, j), al, latency_curve_params, ef_curve_params in chain.from_iterable(
        fit_results
    ):
        al_table[k, i, j] = al
        latency_curve_params_table[k, i, j] = latency_curve_params
        ef_curve_params_table[k, i, j] = ef_curve_params

    # 拟合跳表的cost function
    print(f"Fitting skiplist search cost function...")
    skiplist_search_stats_path = f"{stat_dir}/skiplist_search_stats.csv"
    df = pd.read_csv(skiplist_search_stats_path)
    x, y = df["cardinality"].values, df["latency(ms)"].values * 1000
    # 使用 numpy 的 polyfit 函数进行线性拟合
    a, b = np.polyfit(x, y, 1)  # 1 表示线性拟合

    # 读取Histogram
    hist_df = pd.read_csv(f"{stat_dir}/histogram.csv")
    return OpParams(
        k_max,
        B,
        al_table,
        latency_curve_params_table,
        ef_curve_params_table,
        hist_df,
        [a, b],
    )


def plot_recall_curve(
    stat_dir: str,
    k: int,
    i: int,
    j: int,
    op_params: OpParams,
    fig_save_path: str,
):
    file = f"{stat_dir}/stat_k_max=100_slot_{i}_to_{j}.csv"
    df = pd.read_csv(file)
    lp = op_params.latency_curve_params_table[k, i, j]
    efp = op_params.ef_curve_params_table[k, i, j]
    al = op_params.al_table[k, i, j]
    title = f"Slot {i} to {j}, al={al}, k={k}"
    sub_df = df.loc[(df["al"] == al).values & (df["k"] == k).values]
    ef = sub_df["ef"].values
    recall = sub_df["recall"].values
    latency = sub_df["latency(ms)"].values * 1000  # fitting use us instead of ms

    recall_fit = np.linspace(0.9, 1, 20)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # 在左侧子图中绘制recall-latency曲线
    latency_fit = RecallCurveFitter.output_latency(recall_fit, *lp)
    axs[0].plot(recall, latency, marker="s", label="Original")
    axs[0].plot(recall_fit, latency_fit, marker="^", label="Fit")
    axs[0].legend()
    axs[0].set_xlim([0.9, 1])
    axs[0].set_title(title)
    axs[0].set_xlabel("Recall")
    axs[0].set_ylabel("Latency(us)")

    # 在右侧子图中绘制recall-ef曲线
    ef_fit = RecallCurveFitter.output_ef(recall_fit, *efp)
    axs[1].plot(recall, ef, marker="s", label="Original")
    axs[1].plot(recall_fit, ef_fit, marker="^", label="Fit")
    axs[1].legend()
    axs[1].set_xlim([0.9, 1])
    axs[1].set_title(title)
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Ef")

    plt.savefig(fig_save_path)


params_save_dir = (
    "/data/home/petrizhang/develop/hanncode/benchmark/analysis/sihnsw16x8_glove_config"
)

stat_dir = (
    "/data/home/petrizhang/develop/hanncode/benchmark/analysis/sihnsw16x8_glove_stats"
)

params_save_dir = (
    "/data/home/petrizhang/develop/hanncode/benchmark/analysis/sift_config"
)

stat_dir = "/data/home/petrizhang/develop/hanncode/benchmark/analysis/sift_stats"


if __name__ == "__main__":
    K_MAX = 100
    K_FACTOR = 1
    B = 8

    if not os.path.exists(f"{params_save_dir}/al.csv"):
        print("Fitting recall curve...")
        op_params = fit_curve(stat_dir, K_MAX, B)
        op_params.save(params_save_dir)
    else:
        print("Reading existing recall curve params...")
        op_params = OpParams.load(K_MAX, B, params_save_dir)

    interesting_points = [(10, i, j) for i in range(B) for j in range(i, B)]
    for k, i, j in interesting_points:
        plot_recall_curve(stat_dir, k, i, j, op_params, f"fit_slot{i}-{j}_k{k}.png")

    # print("Computing configuration table for range split optimize...")
    # conf_df = Optimizer.compute_rso_conf(
    #     K_FACTOR,
    #     [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1],
    #     op_params,
    # )
    # conf_df.to_csv(f"{params_save_dir}/rso_conf.csv", float_format="%.6g", index=False)
    # print(
    #     conf_df.loc[
    #         (conf_df["k"] == 10)
    #         # & (conf_df["recall"] == 9900)  # recall
    #         & (conf_df["i"] == 0)
    #     ]
    # )


# print("=" * 50)
# print(s)

# dp, result = DpOptimizer.optimize(K_MAX, 2, s)
# print("=" * 50)
# print(dp)
# print(result)
