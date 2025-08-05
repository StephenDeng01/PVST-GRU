import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


def autocorrelation_delay(x, max_lag=100):
    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-10)
    autocorr = [1.0] + [
        np.corrcoef(x[:-lag], x[lag:])[0, 1] for lag in range(1, max_lag)
    ]
    for i, val in enumerate(autocorr):
        if val < 1 / np.e:
            return i
    return max_lag


def compute_mutual_information(x, lag, bins=64):
    x1 = x[:-lag]
    x2 = x[lag:]
    c_xy = np.histogram2d(x1, x2, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def minimum_mutual_information(x, max_lag=20, bins=64):
    mi_vals = [compute_mutual_information(x, lag, bins) for lag in range(1, max_lag + 1)]
    for i in range(1, len(mi_vals) - 1):
        if mi_vals[i] < mi_vals[i - 1] and mi_vals[i] < mi_vals[i + 1]:
            return i + 1
    return max_lag


def process_all_samples(csv_path):
    df = pd.read_csv(csv_path, header=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())

    num_samples = df.shape[0]
    num_leads = 12
    lead_length = 757

    tau_list = []
    m_list = []

    for sample_idx in range(num_samples):
        beat_id = int(df.iloc[sample_idx, 0])
        for lead_idx in range(num_leads):
            start_col = lead_idx * lead_length + 1  # 第0列是beat_id
            end_col = start_col + lead_length
            signal = df.iloc[sample_idx, start_col:end_col].values.astype(np.float64)

            if len(signal) != 757:
                signal = np.interp(np.linspace(0, len(signal) - 1, 757), np.arange(len(signal)), signal)

            tau = autocorrelation_delay(signal)
            m = minimum_mutual_information(signal)

            tau_list.append(tau)
            m_list.append(m)

            print(f"样本 {sample_idx}, 导联 {lead_idx + 1} (心拍编号: {beat_id}) -> τ = {tau}, m = {m}")

    avg_tau = np.mean(tau_list)
    avg_m = np.mean(m_list)

    print("\n==========================")
    print(f"所有样本平均 τ: {avg_tau:.2f}")
    print(f"所有样本平均 m: {avg_m:.2f}")
    print("==========================")

    return avg_tau, avg_m


if __name__ == "__main__":
    csv_path = "卢显超_denoised.csv"
    process_all_samples(csv_path)
