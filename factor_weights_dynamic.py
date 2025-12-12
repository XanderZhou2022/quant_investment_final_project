import pandas as pd
import numpy as np

# 读取预测结果
df = pd.read_csv("data/factor_prediction.csv")

factor_pred_cols = [
    "MOM20_pred","MOM120_pred","RSI_pred","PB_pred","PE_pred",
    "DIV_pred","ROE_pred","PROFIT_GR_pred","VOL_pred","BETA_pred"
]

# softmax 函数
def softmax(x):
    x = np.array(x)
    x = x - np.max(x)   # 数值稳定性，避免e的溢出
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

# 为每一期计算权重
weights = df[factor_pred_cols].apply(softmax, axis=1, result_type='expand')

# 重命名权重列
weights.columns = [
    "MOM20_w","MOM120_w","RSI_w","PB_w","PE_w",
    "DIV_w","ROE_w","PROFIT_GR_w","VOL_w","BETA_w"
]

# 合并
df_weights = pd.concat([df[["month"]], weights], axis=1)

# 导出
df_weights.to_csv("data/factor_weights_dynamic.csv", index=False)

print("已生成因子权重文件：data/factor_weights_dynamic.csv")
print(df_weights.head())
