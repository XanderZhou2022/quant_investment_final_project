import pandas as pd
import numpy as np


# ======= 配置区 =======
RAW_FILE = "data\step3_上证50收盘情况.xlsx"      # 原始日线数据（建议包含 2014-01 之前一段）
INPUT_SHEET = 0                    # 如果是第一个 sheet，用 0 即可

DAILY_FEATURE_FILE = "data/sh000016_daily_features.csv"
MONTHLY_FEATURE_FILE = "data/sh000016_monthly_features.csv"

START_DATE = "2015-01-01"          # 研究样本起始
END_DATE   = "2024-12-31"          # 研究样本截止（含）
TRADING_DAYS = 252                 # 年化波动率用
# ======================


def load_raw_data(path, sheet=0):
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_daily_features(df):
    df["ret_simple"] = df["close"].pct_change()
    df["ret_log"] = np.log(df["close"]).diff()
    df["vol_20"] = df["ret_log"].rolling(20, min_periods=20).std()
    df["vol_60"] = df["ret_log"].rolling(60, min_periods=60).std()
    df["vol_20_annual"] = df["vol_20"] * np.sqrt(TRADING_DAYS)
    df["vol_60_annual"] = df["vol_60"] * np.sqrt(TRADING_DAYS)
    return df


def filter_daily_for_output(df_daily):
    """日度输出用：限定 2015-01-01~2024-12-31 且有60日波动率"""
    mask = (df_daily["date"] >= START_DATE) & (df_daily["date"] <= END_DATE)
    df = df_daily.loc[mask].copy()
    df = df.dropna(subset=["vol_60"])
    df.reset_index(drop=True, inplace=True)
    return df


def build_monthly_features_from_full(df_daily_full):
    """
    月度特征用“完整日线”计算（保留2014-12等前史），
    然后再在月度层面截取 2015-01~2024-12。
    """
    df = df_daily_full.set_index("date")

    # 月末收盘价 & 月收益
    close_month_end = df["close"].resample("M").last()
    ret_month = close_month_end.pct_change()  # 第一行会是 NaN（前面没有月）

    # 月成交量及变化率
    vol_month_sum = df["vol"].resample("M").sum()
    vol_month_chg = vol_month_sum.pct_change()

    # 月末波动率
    vol20_me = df["vol_20_annual"].resample("M").last()
    vol60_me = df["vol_60_annual"].resample("M").last()

    mdf = pd.DataFrame({
        "month": close_month_end.index,
        "close_month_end": close_month_end.values,
        "ret_month": ret_month.values,
        "vol_month_sum": vol_month_sum.values,
        "vol_month_chg": vol_month_chg.values,
        "vol_20_annual_month_end": vol20_me.values,
        "vol_60_annual_month_end": vol60_me.values,
    })

    # 在“月频”层面截断 2015-01 ~ 2024-12
    mask_m = (mdf["month"] >= pd.to_datetime(START_DATE)) & \
             (mdf["month"] <= pd.to_datetime(END_DATE))
    mdf = mdf.loc[mask_m].reset_index(drop=True)

    # 这时候 2015-01 这一行已经有了上一月(2014-12)的数据，因此
    # - ret_month(2015-01) 可算
    # - vol_month_chg(2015-01) 也可算
    # 唯一仍为 NaN 的是“真正全样本的第一行”（比如200X年第一个月），
    # 但那一行已经被你截掉了。

    mdf["month_str"] = mdf["month"].dt.strftime("%Y-%m")
    return mdf


def main():
    print("1) 加载原始 Excel 日线数据...")
    df_raw = load_raw_data(RAW_FILE, INPUT_SHEET)

    print("2) 在完整样本上计算日度特征（含2014-12）...")
    df_daily_full = add_daily_features(df_raw)

    print("3) 为【日度输出】截取 2015-01-01~2024-12-31 ...")
    df_daily_out = filter_daily_for_output(df_daily_full)
    df_daily_out.to_csv(DAILY_FEATURE_FILE, index=False)
    print(f"日度特征已保存：{DAILY_FEATURE_FILE}")

    print("4) 基于【完整日线】构建月度特征，然后再截取区间 ...")
    df_monthly = build_monthly_features_from_full(df_daily_full)
    df_monthly.to_csv(MONTHLY_FEATURE_FILE, index=False)
    print(f"月度特征已保存：{MONTHLY_FEATURE_FILE}")

    print("完成。")


if __name__ == "__main__":
    main()
