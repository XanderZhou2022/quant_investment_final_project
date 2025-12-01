import pandas as pd
import os

# ============================================
# 配置区
# ============================================

data_folder = 'data'  # 数据文件夹路径

# 因子文件映射
factor_files = {
    'MOM20': 'step2_MOM20.xlsx',
    'MOM120': 'step2_MOM120.xlsx',
    'RSI': 'step2_RSI.xlsx',
    'PB': 'step2_市净率PNA.xlsx',
    'PE': 'step2_市盈率TTM.xlsx',
    'DIV': 'step2_股息率.xlsx',
    'ROE': 'step2_净资产收益率ROE_TTM.xlsx',
    'PROFIT_GR': 'step2_营业利润增长率.xlsx',
    'VOL': 'step2_波动率.xlsx',
    'BETA': 'step2_市场敏感度.xlsx'
}

# 因子方向：1=第1组做多，-1=第5组做多
factor_direction = {
    'MOM20': 1,
    'MOM120': 1,
    'RSI': 1,
    'PB': 1,
    'PE': 1,
    'DIV': 1,
    'ROE': 1,
    'PROFIT_GR': 1,
    'VOL': -1,
    'BETA': -1
}

# ============================================
# 主流程
# ============================================

# 1. 先读取MOM20获取日期列
base_path = os.path.join(data_folder, 'step2_MOM20.xlsx')
base_df = pd.read_excel(base_path)
dates = base_df['开始日'].values

# 2. 创建结果DataFrame
result = pd.DataFrame({'日期': dates})

# 3. 逐个处理因子
print("开始处理因子文件...")
for factor_name, filename in factor_files.items():
    filepath = os.path.join(data_folder, filename)
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        direction = factor_direction.get(factor_name, 1)
        
        # 计算多空收益
        if direction == 1:
            long_short = df['第1组'] - df['第5组']
        else:
            long_short = df['第5组'] - df['第1组']
        
        result[factor_name] = long_short.values
        print(f"✓ {factor_name}")
    else:
        print(f"✗ {factor_name}: 文件不存在")

# 4. 保存到data文件夹
output_path = os.path.join(data_folder, 'factor_longshort.csv')
result.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n完成！已保存到 {output_path}")
print(f"数据形状: {result.shape}")
print("\n前5行:")
print(result.head())