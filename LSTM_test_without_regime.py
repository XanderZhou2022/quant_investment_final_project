import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# 0. 环境与随机种子
# =========================
np.random.seed(42)
tf.random.set_seed(42)

# =========================
# 1. 读取与预处理数据
# =========================
# 读取最终数据集（包含因子多空收益，不使用 Regime）
df = pd.read_csv("data/final_factor_longshort.csv")

# 确保时间排序正确
if "month" not in df.columns:
    raise ValueError("data/final_factor_longshort.csv 中找不到 'month' 列，请检查列名。")

df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month").reset_index(drop=True)

print("原始数据列：", df.columns.tolist())
print("样本数量：", len(df))

# 需要预测的因子名称（根据你的数据列名调整）
factor_cols = [
    "MOM20",
    "MOM120",
    "RSI",
    "PB",
    "PE",
    "DIV",
    "ROE",
    "PROFIT_GR",
    "VOL",
    "BETA"
]

for col in factor_cols:
    if col not in df.columns:
        raise ValueError(f"数据中缺少因子列: {col}")

# 输入特征列：仅使用因子历史收益，不使用 Regime
input_cols = factor_cols

print("输入特征列：", input_cols)
print("注意：本版本不使用 Regime 特征")

# 去除缺失值（如有）
df_before = len(df)
df = df.dropna(subset=input_cols + factor_cols).reset_index(drop=True)
df_after = len(df)
print(f"删除缺失值后剩余样本数：{df_after}（删除 {df_before - df_after} 行）")

# =========================
# 2. 构造时序滑动窗口数据
# =========================
window = 12  # 使用过去 12 个月预测下一期

X_list = []
y_list = []

data_input = df[input_cols].values.astype(np.float32)    # 输入特征（仅因子）
data_target = df[factor_cols].values.astype(np.float32)  # 目标因子收益（多任务输出）

for i in range(window, len(df)):
    # X: 过去 window 期的输入特征序列
    X_list.append(data_input[i - window:i])
    # y: 当前期的因子多空收益
    y_list.append(data_target[i])

X = np.array(X_list, dtype=np.float32)  # 形状：(样本数, window, 特征数)
y = np.array(y_list, dtype=np.float32)  # 形状：(样本数, 因子数)

print("X shape:", X.shape)
print("y shape:", y.shape)

# 简单检查是否有 NaN / inf
if not np.isfinite(X).all():
    raise ValueError("X 中存在 NaN 或 inf，请检查输入特征。")
if not np.isfinite(y).all():
    raise ValueError("y 中存在 NaN 或 inf，请检查目标因子收益。")

# =========================
# 3. 按时间顺序划分训练 / 验证 / 测试集（7:2:1）
# =========================
n_samples = X.shape[0]
if n_samples < 30:
    print("警告：有效样本数较少，可能影响模型训练效果。")

train_end = int(n_samples * 0.7)
val_end = int(n_samples * 0.9)  # 剩余 10% 为测试集

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

print("Train set:", X_train.shape, y_train.shape)
print("Val set:  ", X_val.shape, y_val.shape)
print("Test set: ", X_test.shape, y_test.shape)

# 再次确保 dtype 正确
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# =========================
# 4. 构建 LSTM 多任务回归模型
# =========================
model = Sequential([
    LSTM(
        units=64,
        return_sequences=False,
        input_shape=(window, X.shape[-1])
    ),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(len(factor_cols))  # 多任务输出：每个因子一个预测值
])

model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

model.summary()

# =========================
# 5. 训练模型
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=16,
    verbose=1
)

# =========================
# 6. 在测试集上进行预测
# =========================
y_pred = model.predict(X_test)

# 将预测结果与对应月份对齐
# X[0] 对应 df[window] 那一行，因此：
# X_test 起始对应的 df 行索引 = window + val_end
start_idx = window + val_end
end_idx = window + n_samples  # 不包含 end_idx
pred_months = df["month"].iloc[start_idx:end_idx].reset_index(drop=True)

if len(pred_months) != y_pred.shape[0]:
    # 如果对不上，给出提示并做一次安全截断
    min_len = min(len(pred_months), y_pred.shape[0])
    print("警告：预测条数与月份条数不一致，已按最小长度对齐。")
    y_pred = y_pred[:min_len]
    pred_months = pred_months[:min_len]

# 构建预测结果 DataFrame
pred_cols = [f"{col}_pred" for col in factor_cols]
df_pred = pd.DataFrame(y_pred, columns=pred_cols)
df_pred["month"] = pred_months

# 按时间排序
df_pred = df_pred.sort_values("month").reset_index(drop=True)

print("预测结果前几行：")
print(df_pred.head())

# =========================
# 7. 导出预测结果到 CSV
# =========================
output_path = "data/factor_prediction_without_regime.csv"
df_pred.to_csv(output_path, index=False)
print(f"预测结果已保存到: {output_path}")
