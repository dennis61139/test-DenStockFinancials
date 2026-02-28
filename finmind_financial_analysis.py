"""
AI 台股財報分析系統 (FinMind)
版本：2.0
新增功能：
- 季度/年度數據區間切換
- 起始/結束日期選擇
- AI模型選擇（gpt-5-mini / gpt-4.1-nano）
- 修正市值計算（使用 TaiwanStockShareholding）
- 修正本益比計算
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from openai import OpenAI

# ============================================================
# 頁面基本配置
# ============================================================
st.set_page_config(
    page_title="AI 台股財報分析系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 工具函數模組
# ============================================================

def format_large_number(value):
    """將大數字格式化為易讀的中文單位格式（兆/億/百萬）"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        value = float(value)
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        if abs_value >= 1e12:
            return f"{sign}{abs_value/1e12:.2f}兆"
        elif abs_value >= 1e8:
            return f"{sign}{abs_value/1e8:.2f}億"
        elif abs_value >= 1e6:
            return f"{sign}{abs_value/1e6:.2f}百萬"
        else:
            return f"{sign}{abs_value:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def validate_stock_code(code):
    """驗證台股代碼是否為四位數字格式"""
    if not code:
        return False, "請輸入股票代碼"
    code = code.strip()
    if not code.isdigit():
        return False, f"股票代碼必須為數字（範例：2330、2454、2317、2412）"
    if len(code) != 4:
        return False, f"台股代碼必須為四位數字，您輸入了 {len(code)} 位"
    return True, "格式正確"


def safe_divide(numerator, denominator, default=0.0):
    """安全除法，避免除以零"""
    try:
        if denominator == 0 or denominator is None:
            return default
        result = float(numerator) / float(denominator)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def filter_by_period(df, period_type):
    """
    依據用戶選擇的數據區間過濾財務數據
    - 年度模式：只保留每年 Q4（12月）的資料
    - 季度模式：保留所有季度資料
    """
    if df is None or df.empty:
        return df
    
    if period_type == "年度":
        # 只保留每年最後一季（12月底的年報）
        mask = df.index.month == 12
        return df[mask]
    else:
        # 季度模式：保留所有資料
        return df


# ============================================================
# FinMind API 整合模組
# ============================================================

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

# 損益表欄位對應
INCOME_STATEMENT_MAPPING = {
    "Revenue": "revenues",
    "GrossProfit": "grossprofit",
    "OperatingIncome": "operatingincomeloss",
    "IncomeAfterTaxes": "netincomeloss",
    "PreTaxIncome": "incomelossfromcontinuingoperationsbeforeincometaxes",
    "EPS": "eps_basic",
    "TotalNonoperatingIncomeAndExpense": "total_nonoperating",
}

# 資產負債表欄位對應
BALANCE_SHEET_MAPPING = {
    "TotalAssets": "assets",
    "Liabilities": "liabilities",
    "Equity": "stockholdersequity",
    "CurrentAssets": "assetscurrent",
    "CurrentLiabilities": "liabilitiescurrent",
    "RetainedEarnings": "retainedearningsaccumulateddeficit",
    "NoncurrentLiabilities": "longtermdebtnoncurrent",
}

# 現金流量表欄位對應
CASHFLOW_MAPPING = {
    "CashFlowsFromOperatingActivities": "netcashprovidedbyusedinoperatingactivities",
    "CashProvidedByInvestingActivities": "netcashprovidedbyusedininvestingactivities",
    "CashFlowsProvidedFromFinancingActivities": "netcashprovidedbyusedinfinancingactivities",
    "PropertyAndPlantAndEquipment": "paymentstoacquireproductiveassets",
}


def fetch_finmind_data(dataset, stock_id, start_date, end_date, token):
    """從 FinMind API 獲取指定 dataset 的數據"""
    try:
        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
            "token": token,
        }
        response = requests.get(FINMIND_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != 200:
            msg = data.get("msg", "未知錯誤")
            st.warning(f"FinMind API 警告（{dataset}）：{msg}")
            return None

        records = data.get("data", [])
        if not records:
            return None

        return pd.DataFrame(records)

    except requests.exceptions.ConnectionError:
        st.error("無法連接 FinMind API，請確認網路連線後重試。")
        return None
    except requests.exceptions.Timeout:
        st.error("FinMind API 請求逾時，請稍後重試。")
        return None
    except Exception as e:
        st.error(f"FinMind API 發生錯誤（{dataset}）：{e}")
        return None


def standardize_financial_statement(df, mapping, date_col="date"):
    """將 FinMind type 欄位轉換為內部標準欄位名稱，並以日期為索引"""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        if not all(col in df.columns for col in [date_col, "type", "value"]):
            return pd.DataFrame()

        filtered = df[df["type"].isin(mapping.keys())].copy()
        if filtered.empty:
            return pd.DataFrame()

        filtered["internal_key"] = filtered["type"].map(mapping)
        pivot = filtered.pivot_table(
            index=date_col,
            columns="internal_key",
            values="value",
            aggfunc="first"
        )
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index(ascending=False)

        for col in pivot.columns:
            pivot[col] = pd.to_numeric(pivot[col], errors="coerce")

        return pivot
    except Exception as e:
        st.warning(f"欄位標準化處理發生錯誤：{e}")
        return pd.DataFrame()


def fetch_all_financial_data(stock_id, start_date, end_date, token):
    """從 FinMind API 獲取所有財務報表數據並整合"""
    result = {
        "income_statement": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cash_flow": pd.DataFrame(),
        "stock_price": pd.DataFrame(),
        "shareholding": pd.DataFrame(),
        "company_info": {},
    }

    progress = st.progress(0, text="正在獲取損益表數據...")

    # 1. 損益表
    income_raw = fetch_finmind_data(
        "TaiwanStockFinancialStatements", stock_id, start_date, end_date, token
    )
    result["income_statement"] = standardize_financial_statement(income_raw, INCOME_STATEMENT_MAPPING)
    progress.progress(17, text="正在獲取資產負債表數據...")

    # 2. 資產負債表
    balance_raw = fetch_finmind_data(
        "TaiwanStockBalanceSheet", stock_id, start_date, end_date, token
    )
    result["balance_sheet"] = standardize_financial_statement(balance_raw, BALANCE_SHEET_MAPPING)
    progress.progress(34, text="正在獲取現金流量表數據...")

    # 3. 現金流量表
    cashflow_raw = fetch_finmind_data(
        "TaiwanStockCashFlowsStatement", stock_id, start_date, end_date, token
    )
    result["cash_flow"] = standardize_financial_statement(cashflow_raw, CASHFLOW_MAPPING)
    progress.progress(51, text="正在獲取股價數據...")

    # 4. 股價（使用更長的起始日期以取得最新收盤價）
    price_raw = fetch_finmind_data(
        "TaiwanStockPrice", stock_id, start_date, end_date, token
    )
    if price_raw is not None and not price_raw.empty:
        result["stock_price"] = price_raw
    progress.progress(68, text="正在獲取發行股數數據...")

    # 5. 發行股數（TaiwanStockShareholding）
    shareholding_raw = fetch_finmind_data(
        "TaiwanStockShareholding", stock_id, start_date, end_date, token
    )
    if shareholding_raw is not None and not shareholding_raw.empty:
        result["shareholding"] = shareholding_raw
    progress.progress(84, text="正在獲取公司基本資料...")

    # 6. 公司基本資料
    info_raw = fetch_finmind_data(
        "TaiwanStockInfo", stock_id, "2010-01-01", end_date, token
    )
    if info_raw is not None and not info_raw.empty:
        row = info_raw[info_raw["stock_id"] == stock_id].iloc[0] if "stock_id" in info_raw.columns else info_raw.iloc[0]
        result["company_info"] = row.to_dict()

    progress.progress(100, text="數據獲取完成！")
    progress.empty()

    return result


def compute_derived_fields(income_df, balance_df, cash_flow_df, price_df, shareholding_df):
    """
    計算衍生欄位：
    - 加權平均股數 = 淨利潤 ÷ EPS
    - 利息費用推估
    - 資本支出取絕對值
    - 市值 = 最新收盤價 × number_of_shares_issued
    - 本益比 = 市值 ÷ 淨利潤
    """
    # 加權平均股數與利息費用
    if not income_df.empty:
        if "netincomeloss" in income_df.columns and "eps_basic" in income_df.columns:
            mask = (income_df["eps_basic"] != 0) & (~income_df["eps_basic"].isna())
            income_df["weightedaveragenumberofsharesoutstandingbasic"] = np.nan
            income_df.loc[mask, "weightedaveragenumberofsharesoutstandingbasic"] = (
                income_df.loc[mask, "netincomeloss"] / income_df.loc[mask, "eps_basic"]
            ) * 1000

        if "total_nonoperating" in income_df.columns:
            income_df["interestexpensenonoperating"] = income_df["total_nonoperating"].apply(
                lambda x: abs(x) if (not pd.isna(x) and x < 0) else 0
            )

    # 資本支出取絕對值
    if not cash_flow_df.empty and "paymentstoacquireproductiveassets" in cash_flow_df.columns:
        cash_flow_df["paymentstoacquireproductiveassets"] = (
            cash_flow_df["paymentstoacquireproductiveassets"].abs()
        )

    # 市值計算：使用 TaiwanStockShareholding 的 number_of_shares_issued
    market_cap = None
    latest_price = None
    shares_issued = None

    # 取最新收盤價
    if price_df is not None and not price_df.empty and "close" in price_df.columns:
        price_df["date"] = pd.to_datetime(price_df["date"])
        latest_row = price_df.sort_values("date", ascending=False).iloc[0]
        try:
            latest_price = float(latest_row["close"])
        except (ValueError, TypeError):
            latest_price = None

    # 取最新發行股數
    if shareholding_df is not None and not shareholding_df.empty:
        if "number_of_shares_issued" in shareholding_df.columns:
            shareholding_df["date"] = pd.to_datetime(shareholding_df["date"])
            latest_sh = shareholding_df.sort_values("date", ascending=False).iloc[0]
            try:
                shares_issued = float(str(latest_sh["number_of_shares_issued"]).replace(",", ""))
            except (ValueError, TypeError):
                shares_issued = None

    # 市值 = 收盤價 × 發行股數（單位：股，FinMind 通常以「千股」或「股」回傳，需確認）
    if latest_price and shares_issued:
        market_cap = latest_price * shares_issued

    return income_df, balance_df, cash_flow_df, market_cap, latest_price, shares_issued


def merge_financial_data(income_df, balance_df, cash_flow_df):
    """將三個財務報表依日期合併為統一格式的列表"""
    if income_df.empty and balance_df.empty and cash_flow_df.empty:
        return []

    all_dates = set()
    for df in [income_df, balance_df, cash_flow_df]:
        if not df.empty:
            all_dates.update(df.index.tolist())

    if not all_dates:
        return []

    all_dates = sorted(all_dates, reverse=True)
    merged_data = []

    for d in all_dates:
        record = {"date": d}
        for df in [income_df, balance_df, cash_flow_df]:
            if not df.empty and d in df.index:
                for col in df.columns:
                    record[col] = df.loc[d, col]
        merged_data.append(record)

    return merged_data


# ============================================================
# 數據驗證模組
# ============================================================

def validate_financial_data(financial_data):
    """驗證財務數據完整性，回傳 (is_valid, warnings, errors)"""
    warnings_list = []
    errors_list = []

    if not financial_data:
        errors_list.append("無法獲取任何財務數據，請確認股票代碼和 API Token。")
        return False, warnings_list, errors_list

    if len(financial_data) < 2:
        warnings_list.append("財務數據少於 2 期，部分比較分析將無法進行。")

    required_fields = ["netincomeloss", "assets", "revenues", "stockholdersequity"]
    latest = financial_data[0]
    field_names = {
        "netincomeloss": "淨利潤", "assets": "總資產",
        "revenues": "營收", "stockholdersequity": "股東權益",
    }
    missing = [field_names.get(f, f) for f in required_fields
               if latest.get(f) is None or (isinstance(latest.get(f), float) and np.isnan(latest.get(f)))]
    if missing:
        errors_list.append(f"缺少關鍵財務指標：{', '.join(missing)}")

    return len(errors_list) == 0, warnings_list, errors_list


def generate_data_quality_report(financial_data):
    """生成財務數據品質報告"""
    report = {
        "quality_level": "良好",
        "years_count": len(financial_data),
        "missing_fields": [],
        "computed_fields": [
            "加權平均股數：由「淨利潤 ÷ EPS」計算，可能因 EPS 精度產生誤差",
            "利息費用：由「營業外收入及支出」推估，負值取絕對值",
            "市值：由「最新收盤價 × TaiwanStockShareholding 發行股數」計算，僅供參考",
            "本益比：由「市值 ÷ 淨利潤」計算",
        ],
        "limitations": [],
    }

    if not financial_data:
        report["quality_level"] = "嚴重不足"
        return report

    important_fields = {
        "revenues": "營收", "grossprofit": "毛利",
        "operatingincomeloss": "營業利潤", "netincomeloss": "淨利潤",
        "assets": "總資產", "liabilities": "總負債",
        "stockholdersequity": "股東權益", "assetscurrent": "流動資產",
        "liabilitiescurrent": "流動負債",
        "netcashprovidedbyusedinoperatingactivities": "營運現金流",
        "paymentstoacquireproductiveassets": "資本支出",
    }

    latest = financial_data[0]
    missing_count = sum(
        1 for f in important_fields
        if latest.get(f) is None or (isinstance(latest.get(f), float) and np.isnan(latest.get(f)))
    )
    report["missing_fields"] = [
        f"{name}（{field}）" for field, name in important_fields.items()
        if latest.get(field) is None or (isinstance(latest.get(field), float) and np.isnan(latest.get(field)))
    ]

    if missing_count == 0:
        report["quality_level"] = "良好"
    elif missing_count <= len(important_fields) * 0.3:
        report["quality_level"] = "部分缺失"
    else:
        report["quality_level"] = "嚴重不足"

    if report["years_count"] < 2:
        report["quality_level"] = "部分缺失"
        report["limitations"].append("財務數據期數不足 2 期，無法進行比較分析")

    return report


# ============================================================
# 財務計算模組
# ============================================================

def calculate_piotroski_fscore(annual_data):
    """
    計算 Piotroski F-Score（固定使用年度數據）
    9 項指標，每項 0 或 1 分，總分 0-9
    """
    if len(annual_data) < 2:
        return None

    curr = annual_data[0]
    prev = annual_data[1]

    def gv(record, key, default=0.0):
        v = record.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)

    results = {"total_score": 0, "profitability_scores": [], "leverage_scores": [], "efficiency_scores": []}

    # ---- 獲利能力（4項）----
    curr_net = gv(curr, "netincomeloss")
    curr_assets = gv(curr, "assets", 1)
    curr_roa = safe_divide(curr_net, curr_assets)
    prev_net = gv(prev, "netincomeloss")
    prev_assets = gv(prev, "assets", 1)
    prev_roa = safe_divide(prev_net, prev_assets)
    curr_ocf = gv(curr, "netcashprovidedbyusedinoperatingactivities")

    s1 = 1 if curr_roa > 0 else 0
    results["profitability_scores"].append({
        "description": "ROA 正值（淨利潤 / 總資產 > 0）",
        "current_value": f"{curr_roa*100:.2f}%", "previous_value": "-",
        "score": s1, "passed": s1 == 1,
    })

    s2 = 1 if curr_ocf > 0 else 0
    results["profitability_scores"].append({
        "description": "營運現金流 > 0",
        "current_value": format_large_number(curr_ocf), "previous_value": "-",
        "score": s2, "passed": s2 == 1,
    })

    s3 = 1 if curr_roa > prev_roa else 0
    results["profitability_scores"].append({
        "description": "ROA 年增（最新 > 前期）",
        "current_value": f"{curr_roa*100:.2f}%", "previous_value": f"{prev_roa*100:.2f}%",
        "score": s3, "passed": s3 == 1,
    })

    s4 = 1 if curr_ocf > curr_net else 0
    results["profitability_scores"].append({
        "description": "現金流品質（OCF > 淨利潤）",
        "current_value": f"OCF={format_large_number(curr_ocf)}", "previous_value": f"NI={format_large_number(curr_net)}",
        "score": s4, "passed": s4 == 1,
    })

    # ---- 槓桿與流動性（3項）----
    curr_ltd = gv(curr, "longtermdebtnoncurrent")
    prev_ltd = gv(prev, "longtermdebtnoncurrent")
    curr_ltd_r = safe_divide(curr_ltd, curr_assets)
    prev_ltd_r = safe_divide(prev_ltd, prev_assets)
    s5 = 1 if curr_ltd_r < prev_ltd_r else 0
    results["leverage_scores"].append({
        "description": "長期負債比率改善（最新 < 前期）",
        "current_value": f"{curr_ltd_r*100:.2f}%", "previous_value": f"{prev_ltd_r*100:.2f}%",
        "score": s5, "passed": s5 == 1,
    })

    curr_ca = gv(curr, "assetscurrent", 1)
    curr_cl = gv(curr, "liabilitiescurrent", 1)
    prev_ca = gv(prev, "assetscurrent", 1)
    prev_cl = gv(prev, "liabilitiescurrent", 1)
    curr_cr = safe_divide(curr_ca, curr_cl)
    prev_cr = safe_divide(prev_ca, prev_cl)
    s6 = 1 if curr_cr > prev_cr else 0
    results["leverage_scores"].append({
        "description": "流動比率改善（最新 > 前期）",
        "current_value": f"{curr_cr:.2f}", "previous_value": f"{prev_cr:.2f}",
        "score": s6, "passed": s6 == 1,
    })

    curr_shares = gv(curr, "weightedaveragenumberofsharesoutstandingbasic")
    prev_shares = gv(prev, "weightedaveragenumberofsharesoutstandingbasic")
    s7 = 1 if (curr_shares > 0 and prev_shares > 0 and curr_shares <= prev_shares) else 0
    results["leverage_scores"].append({
        "description": "股份未稀釋（流通股數未增加）",
        "current_value": format_large_number(curr_shares), "previous_value": format_large_number(prev_shares),
        "score": s7, "passed": s7 == 1,
    })

    # ---- 營運效率（2項）----
    curr_gp = gv(curr, "grossprofit")
    curr_rev = gv(curr, "revenues", 1)
    prev_gp = gv(prev, "grossprofit")
    prev_rev = gv(prev, "revenues", 1)
    curr_gpm = safe_divide(curr_gp, curr_rev)
    prev_gpm = safe_divide(prev_gp, prev_rev)
    s8 = 1 if curr_gpm > prev_gpm else 0
    results["efficiency_scores"].append({
        "description": "毛利率改善（最新 > 前期）",
        "current_value": f"{curr_gpm*100:.2f}%", "previous_value": f"{prev_gpm*100:.2f}%",
        "score": s8, "passed": s8 == 1,
    })

    curr_ato = safe_divide(curr_rev, curr_assets)
    prev_ato = safe_divide(prev_rev, prev_assets)
    s9 = 1 if curr_ato > prev_ato else 0
    results["efficiency_scores"].append({
        "description": "資產周轉率改善（最新 > 前期）",
        "current_value": f"{curr_ato:.3f}", "previous_value": f"{prev_ato:.3f}",
        "score": s9, "passed": s9 == 1,
    })

    results["total_score"] = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9
    return results


def calculate_dupont_analysis(annual_data, max_years=3):
    """計算杜邦分析 ROE 三因子分解（固定使用年度數據）"""
    results = []
    for record in annual_data[:max_years]:
        def gv(key, default=0.0):
            v = record.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        net_income = gv("netincomeloss")
        revenues = gv("revenues", 1)
        assets = gv("assets", 1)
        equity = gv("stockholdersequity", 1)

        net_margin = safe_divide(net_income, revenues)
        asset_turnover = safe_divide(revenues, assets)
        equity_multiplier = safe_divide(assets, equity)
        roe_dupont = net_margin * asset_turnover * equity_multiplier
        roe_direct = safe_divide(net_income, equity)

        entry = {
            "date": record["date"].strftime("%Y-%m-%d") if hasattr(record["date"], "strftime") else str(record["date"]),
            "net_margin": net_margin, "asset_turnover": asset_turnover,
            "equity_multiplier": equity_multiplier, "roe_dupont": roe_dupont, "roe_direct": roe_direct,
            "net_margin_change": None, "asset_turnover_change": None,
            "equity_multiplier_change": None, "roe_change": None,
        }
        if results:
            p = results[-1]
            entry["net_margin_change"] = net_margin - p["net_margin"]
            entry["asset_turnover_change"] = asset_turnover - p["asset_turnover"]
            entry["equity_multiplier_change"] = equity_multiplier - p["equity_multiplier"]
            entry["roe_change"] = roe_dupont - p["roe_dupont"]
        results.append(entry)
    return results


def calculate_cashflow_analysis(annual_data, max_years=5):
    """計算現金流分析指標（固定使用年度數據）"""
    results = []
    for record in annual_data[:max_years]:
        def gv(key, default=0.0):
            v = record.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        ocf = gv("netcashprovidedbyusedinoperatingactivities")
        icf = gv("netcashprovidedbyusedininvestingactivities")
        ffcf = gv("netcashprovidedbyusedinfinancingactivities")
        net_income = gv("netincomeloss", 1)
        capex = abs(gv("paymentstoacquireproductiveassets"))  # 確保資本支出為正值
        free_cash_flow = ocf - capex  # 自由現金流 = OCF - 資本支出絕對值
        ocf_quality = safe_divide(ocf, net_income) if net_income != 0 else 0

        if ocf_quality >= 1.2:
            rating = "優秀 🌟"
        elif ocf_quality >= 1.0:
            rating = "良好 ✅"
        elif ocf_quality >= 0.8:
            rating = "尚可 ⚠️"
        else:
            rating = "需關注 🔴"

        results.append({
            "date": record["date"].strftime("%Y-%m-%d") if hasattr(record["date"], "strftime") else str(record["date"]),
            "operating_cash_flow": ocf, "investing_cash_flow": icf,
            "financing_cash_flow": ffcf, "net_income": net_income,
            "capex": capex, "free_cash_flow": free_cash_flow,
            "ocf_quality_ratio": ocf_quality, "quality_rating": rating,
        })
    return results


# ============================================================
# 視覺化模組
# ============================================================

COLORS = {
    "dark_green": "#1B5E20", "dark_red": "#B71C1C",
    "steel_blue": "#1565C0", "gold": "#F57F17",
    "purple": "#4A148C", "teal": "#00695C",
    "light_green": "#4CAF50", "light_red": "#EF5350",
}


def create_bar_chart(x_data, y_data, title, x_label, y_label, color=None, height=400):
    """建立專業柱狀圖"""
    bar_colors = [COLORS["steel_blue"] if (v is not None and not np.isnan(float(v)) and float(v) >= 0) else COLORS["dark_red"] for v in y_data]
    if color:
        bar_colors = [color] * len(y_data)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_data, y=y_data, marker_color=bar_colors,
        text=[format_large_number(v) for v in y_data], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(title=x_label, tickangle=-45),
        yaxis=dict(title=y_label),
        template="plotly_white", height=height,
        margin=dict(t=60, b=80, l=60, r=20),
    )
    return fig


def create_multi_bar_chart(dates, series_data, title, y_label, height=400):
    """建立多系列柱狀圖，series_data: list of (name, values, color)"""
    fig = go.Figure()
    for name, values, color in series_data:
        fig.add_trace(go.Bar(
            name=name, x=dates, y=values, marker_color=color,
            hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        barmode="group",
        xaxis=dict(title="日期", tickangle=-45),
        yaxis=dict(title=y_label),
        template="plotly_white", height=height,
        margin=dict(t=60, b=80, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_line_chart(x_data, y_series, title, y_label, height=400):
    """建立折線圖，y_series: list of (name, values, color)"""
    fig = go.Figure()
    for name, values, color in y_series:
        fig.add_trace(go.Scatter(
            x=x_data, y=values, mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=8),
            hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(title="日期", tickangle=-45),
        yaxis=dict(title=y_label),
        template="plotly_white", height=height,
        margin=dict(t=60, b=80, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_pie_chart(labels, values, title, colors, height=350):
    """建立圓餅圖"""
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels, values=values, marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} 項<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        template="plotly_white", height=height,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def display_fscore_tables(fscore_result):
    """顯示 F-Score 各項指標表格"""
    def make_df(scores):
        return pd.DataFrame([{
            "指標說明": i["description"],
            "當前值": i["current_value"],
            "前期值": i["previous_value"],
            "得分": i["score"],
            "狀態": "✅" if i["passed"] else "❌",
        } for i in scores])

    st.markdown("#### 🏆 獲利能力指標（4項）")
    st.dataframe(make_df(fscore_result["profitability_scores"]), use_container_width=True, hide_index=True)
    st.markdown("#### 🏦 槓桿與流動性指標（3項）")
    st.dataframe(make_df(fscore_result["leverage_scores"]), use_container_width=True, hide_index=True)
    st.markdown("#### ⚙️ 營運效率指標（2項）")
    st.dataframe(make_df(fscore_result["efficiency_scores"]), use_container_width=True, hide_index=True)


# ============================================================
# AI 分析模組
# ============================================================

def prepare_ai_analysis_data(financial_data, fscore_result, dupont_result, cashflow_result,
                              stock_id, company_info, market_cap, latest_price):
    """整合三階段分析結果，準備 AI 提示語所需內容"""
    company_name = company_info.get("stock_name", stock_id)
    industry = company_info.get("industry_category", "未知")

    # F-Score 摘要
    fscore_text = ""
    if fscore_result:
        fscore_text = f"【Piotroski F-Score】總分：{fscore_result['total_score']} / 9\n"
        for group, label in [
            ("profitability_scores", "獲利能力"),
            ("leverage_scores", "槓桿流動性"),
            ("efficiency_scores", "營運效率"),
        ]:
            sub_score = sum(i["score"] for i in fscore_result[group])
            fscore_text += f"\n{label}指標：\n"
            for item in fscore_result[group]:
                fscore_text += f"  {'✅' if item['passed'] else '❌'} {item['description']}：{item['current_value']}（前期：{item['previous_value']}）\n"

    # 杜邦分析摘要
    dupont_text = "\n【杜邦分析（年度）】\n"
    for e in dupont_result:
        dupont_text += (
            f"  {e['date']}：淨利率={e['net_margin']*100:.2f}%，"
            f"資產周轉率={e['asset_turnover']:.3f}，"
            f"權益乘數={e['equity_multiplier']:.2f}，"
            f"ROE={e['roe_dupont']*100:.2f}%\n"
        )

    # 現金流分析摘要
    cashflow_text = "\n【現金流分析（年度）】\n"
    for e in cashflow_result[:3]:
        cashflow_text += (
            f"  {e['date']}：OCF={format_large_number(e['operating_cash_flow'])}，"
            f"FCF={format_large_number(e['free_cash_flow'])}，"
            f"品質比率={e['ocf_quality_ratio']:.2f}（{e['quality_rating']}）\n"
        )

    # 最新財務數據
    latest = financial_data[0] if financial_data else {}
    def fv(key):
        v = latest.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return format_large_number(float(v))

    financial_text = f"""
【最新財務數據（{latest.get('date', 'N/A')}）】
營收：{fv('revenues')} / 毛利：{fv('grossprofit')} / 營業利潤：{fv('operatingincomeloss')}
淨利潤：{fv('netincomeloss')} / 總資產：{fv('assets')} / 股東權益：{fv('stockholdersequity')}
市值（估算）：{format_large_number(market_cap) if market_cap else 'N/A'}
最新股價：{f'{latest_price:.2f} 元' if latest_price else 'N/A'}
產業別：{industry}
"""
    return {
        "company_name": company_name, "stock_id": stock_id, "industry": industry,
        "fscore_text": fscore_text, "dupont_text": dupont_text,
        "cashflow_text": cashflow_text, "financial_text": financial_text,
    }


def run_ai_analysis(openai_api_key, analysis_data, model="gpt-4.1-nano"):
    """
    使用 OpenAI 新版 API 進行 AI 財務分析
    使用 client.chat.completions.create（新版格式，禁止使用舊版 ChatCompletion.create）
    """
    try:
        client = OpenAI(api_key=openai_api_key)

        system_message = """你是一位專精台股財務分析和台灣會計準則（IFRS台版）的資深分析師，
熟悉 FinMind 開源財務資料的特性與限制，以及台股市場的投資環境（法規、產業政策、兩岸關係等）。
請用繁體中文提供客觀、專業且負責任的財務分析報告，避免過度承諾或誤導性內容。"""

        user_prompt = f"""
請根據以下已完成的三階段財務分析結果，對台股 {analysis_data['stock_id']}（{analysis_data['company_name']}）進行深度財務分析。
**請基於已計算完成的數據進行解讀，而非重新計算。**

{analysis_data['fscore_text']}
{analysis_data['dupont_text']}
{analysis_data['cashflow_text']}
{analysis_data['financial_text']}

---
請依以下結構提供完整分析報告：

## 一、三階段評分總結

| 分析階段 | 評分狀態 | 評價 | 主要發現 |
|---------|---------|------|---------|
| Piotroski F-Score | ... | ... | ... |
| 杜邦分析 | ... | ... | ... |
| 現金流分析 | ... | ... | ... |

## 二、Piotroski F-Score 解讀
解讀各項指標的投資意義和公司業務狀況。

## 三、杜邦分析趨勢洞察
分析 ROE 三因子（淨利率、資產周轉率、權益乘數）的趨勢和主要驅動力。

## 四、現金流結構深度分析
分析現金流品質、自由現金流趨勢、資本支出模式和獲利品質一致性。

## 五、台股市場特性分析
分析該公司在台股市場的定位、競爭優勢，以及台灣法規、產業政策、兩岸關係對投資的影響。

## 六、資料來源與限制說明
說明 FinMind 開源資料特性與以下計算欄位的限制：
- 加權平均股數：由「淨利潤 ÷ EPS」計算，可能存在精度誤差
- 利息費用：由「營業外收入及支出」推估
- 市值：由最新收盤價 × TaiwanStockShareholding 發行股數估算

## 七、綜合財務健康診斷

### 主要優勢（3-5點）
### 風險因素
### 後續追蹤重點

### 財報綜合評比

| 評估面向 | 評分 | 說明 |
|---------|------|------|
| 營運績效 | ... | ... |
| 財務結構 | ... | ... |
| 現金流量 | ... | ... |
| 總結 | ... | ... |

---
*本分析僅供教育和研究用途，不構成投資建議。*
"""
        # 使用新版 API：client.chat.completions.create
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4000,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "authentication" in err.lower():
            return f"⚠️ OpenAI API 金鑰錯誤，請確認金鑰是否正確。\n\n錯誤詳情：{err}"
        elif "model" in err.lower():
            return f"⚠️ 模型 {model} 不在您的 API 授權範圍內，請嘗試其他模型。\n\n錯誤詳情：{err}"
        elif "quota" in err.lower() or "rate" in err.lower():
            return f"⚠️ API 使用量超限或頻率過高，請稍後再試。\n\n錯誤詳情：{err}"
        else:
            return f"⚠️ AI 分析發生錯誤，請確認 API 金鑰和網路連線。\n\n錯誤詳情：{err}"


# ============================================================
# 主程式
# ============================================================

def main():
    # ---- 頁面標題 ----
    st.title("📊 AI 台股財報分析系統")
    st.markdown("<hr style='border: 2px solid #1a237e; margin: 0 0 1rem 0;'>", unsafe_allow_html=True)

    # ---- 側邊欄 ----
    with st.sidebar:
        st.markdown("## 📈 AI 財報分析")
        st.markdown("<hr style='border: 2px solid #1a237e;'>", unsafe_allow_html=True)

        stock_id = st.text_input(
            "🏷️ 股票代碼",
            placeholder="例：2330、2454、2317、2412",
            help="請輸入四位數字的台股代碼"
        )

        finmind_token = st.text_input(
            "🔑 FinMind API Token",
            type="password",
            help="請至 FinMind 官網申請免費 Token：https://finmindtrade.com"
        )

        openai_key = st.text_input(
            "🤖 OpenAI API 金鑰",
            type="password",
            help="請至 OpenAI 官網申請：https://platform.openai.com"
        )

        # AI 模型選擇（動態帶入）
        ai_model = st.selectbox(
            "🧠 AI 模型選擇",
            options=["gpt-4.1-nano", "gpt-5-mini"],
            index=0,
            help="選擇用於 AI 分析的 OpenAI 模型版本"
        )

        # 起始/結束日期
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.text_input("📅 起始日期", value="2022-01-01", help="格式：YYYY-MM-DD")
        with col_e:
            end_date = st.text_input("📅 結束日期", value=date.today().strftime("%Y-%m-%d"), help="格式：YYYY-MM-DD")

        # 數據區間選擇
        period_type = st.selectbox(
            "📊 數據區間",
            options=["年度", "季度"],
            index=0,
            help="年度：只保留每年Q4年報數據；季度：保留所有季度數據。三階段財報分析固定使用年度數據。"
        )

        analyze_btn = st.button("🔍 分析股票", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("""
**⚠️ 免責聲明**

本系統僅供**教育和研究用途**，分析結果不構成投資建議。投資有風險，請自行評估。

數據來源：[FinMind 開源平台](https://finmindtrade.com)
        """)
        st.markdown("---")
        st.markdown("""
**使用說明**
1. 輸入台股四位數代碼
2. 填入 FinMind API Token（免費）
3. 填入 OpenAI API 金鑰（AI分析需要）
4. 選擇 AI 模型、日期範圍、數據區間
5. 點擊「分析股票」
        """)

    # ---- 首頁介紹 ----
    if not analyze_btn:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**📊 三大財務報表**\n- 損益表趨勢分析\n- 資產負債表結構\n- 現金流量品質")
        with col2:
            st.info("**🎯 三階段專業分析**\n- Piotroski F-Score\n- 杜邦分析（ROE三因子）\n- 現金流品質評估")
        with col3:
            st.info("**🤖 AI 深度分析**\n- 台股市場特性解讀\n- 財務健康綜合診斷\n- 投資風險評估報告")

        st.markdown("""
### 如何開始？
在左側輸入股票代碼（例如 **2330** 台積電）、FinMind Token 和 OpenAI 金鑰，選擇日期範圍與數據區間，點擊「分析股票」即可開始。
        """)
        return

    # ---- 輸入驗證 ----
    if not finmind_token:
        st.error("❌ 請填入 FinMind API Token。")
        return

    valid, msg = validate_stock_code(stock_id)
    if not valid:
        st.error(f"❌ {msg}")
        return

    # ---- 數據獲取 ----
    st.info(f"⏳ 正在獲取 **{stock_id}** 的財務數據（{start_date} ~ {end_date}），請稍候...")

    raw_data = fetch_all_financial_data(stock_id, start_date, end_date, finmind_token)

    income_df = raw_data["income_statement"]
    balance_df = raw_data["balance_sheet"]
    cashflow_df = raw_data["cash_flow"]
    price_df = raw_data["stock_price"]
    shareholding_df = raw_data["shareholding"]
    company_info = raw_data["company_info"]

    # 計算衍生欄位（市值、本益比等）
    income_df, balance_df, cashflow_df, market_cap, latest_price, shares_issued = compute_derived_fields(
        income_df, balance_df, cashflow_df, price_df, shareholding_df
    )

    # ---- 依數據區間過濾顯示用數據 ----
    display_income = filter_by_period(income_df, period_type)
    display_balance = filter_by_period(balance_df, period_type)
    display_cashflow = filter_by_period(cashflow_df, period_type)

    # ---- 三階段分析固定使用年度數據 ----
    annual_income = filter_by_period(income_df, "年度")
    annual_balance = filter_by_period(balance_df, "年度")
    annual_cashflow = filter_by_period(cashflow_df, "年度")
    annual_financial_data = merge_financial_data(annual_income, annual_balance, annual_cashflow)

    # 顯示用合併數據（依所選區間）
    display_financial_data = merge_financial_data(display_income, display_balance, display_cashflow)

    # ---- 數據驗證 ----
    is_valid, warnings_list, errors_list = validate_financial_data(annual_financial_data or display_financial_data)
    for err in errors_list:
        st.error(f"⚠️ {err}")
    for warn in warnings_list:
        st.warning(f"⚠️ {warn}")

    if not display_financial_data and not annual_financial_data:
        st.error("❌ 無法獲取財務數據，請確認股票代碼和 API Token 是否正確。")
        return

    st.success(f"✅ 成功獲取 **{stock_id}** 的財務數據（{period_type}模式，共 {len(display_financial_data)} 期）")
    st.markdown("---")

    # ---- 公司基本資訊 ----
    company_name = company_info.get("stock_name", stock_id)
    industry = company_info.get("industry_category", "未知")
    sector = company_info.get("type", "")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(f"🏢 {company_name}")
        st.write(f"**產業類別**：{industry}")
        if sector:
            st.write(f"**行業分類**：{sector}")
    with col2:
        if latest_price:
            st.metric(label="💹 最新收盤價", value=f"NT$ {latest_price:,.2f}")
        else:
            st.metric(label="💹 最新收盤價", value="N/A")
    with col3:
        st.write(f"**市值（估算）**：NT$ {format_large_number(market_cap)}")
        # 本益比 = 市值 ÷ 淨利潤
        latest = annual_financial_data[0] if annual_financial_data else (display_financial_data[0] if display_financial_data else {})
        net_income = latest.get("netincomeloss")
        if (market_cap and net_income
                and not np.isnan(float(net_income))
                and float(net_income) > 0):
            pe_ratio = market_cap / float(net_income)
            st.write(f"**本益比（P/E）**：{pe_ratio:.2f}x")
        else:
            st.write("**本益比（P/E）**：N/A")

    st.markdown("---")

    # ---- 財務計算（三階段固定用年度數據）----
    with st.spinner("🧮 正在進行三階段財務分析..."):
        fscore_result = calculate_piotroski_fscore(annual_financial_data)
        dupont_result = calculate_dupont_analysis(annual_financial_data)
        cashflow_result = calculate_cashflow_analysis(annual_financial_data)

    # ---- 頁籤 ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 損益表分析",
        "🏦 資產負債表分析",
        "💰 現金流量表分析",
        "🎯 三階段財報分析",
        "🤖 AI 分析",
    ])

    # ============================================================
    # 頁籤 1：損益表分析
    # ============================================================
    with tab1:
        st.subheader(f"📈 損益表分析（{period_type}數據）")

        if display_income.empty:
            st.warning("無法獲取損益表數據。")
        else:
            dates = [d.strftime("%Y-%m") for d in display_income.index]

            if "revenues" in display_income.columns and "grossprofit" in display_income.columns:
                fig = create_multi_bar_chart(
                    dates,
                    [("營收", display_income["revenues"].tolist(), COLORS["steel_blue"]),
                     ("毛利", display_income["grossprofit"].tolist(), COLORS["dark_green"])],
                    "營收與毛利趨勢", "金額（元）",
                )
                st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if "netincomeloss" in display_income.columns:
                    fig2 = create_bar_chart(dates, display_income["netincomeloss"].tolist(), "淨利潤趨勢", "日期", "金額（元）")
                    st.plotly_chart(fig2, use_container_width=True)
            with col_b:
                if "revenues" in display_income.columns and "grossprofit" in display_income.columns:
                    gpm = [safe_divide(g, r) * 100 for g, r in zip(
                        display_income["grossprofit"].fillna(0),
                        display_income["revenues"].replace(0, np.nan).fillna(1)
                    )]
                    fig3 = create_bar_chart(dates, gpm, "毛利率趨勢（%）", "日期", "毛利率（%）", color=COLORS["gold"])
                    st.plotly_chart(fig3, use_container_width=True)

            st.markdown("#### 完整損益表數據")
            display_cols = {"revenues": "營收", "grossprofit": "毛利", "operatingincomeloss": "營業利潤",
                            "netincomeloss": "淨利潤", "eps_basic": "EPS"}
            show_df = pd.DataFrame(index=[d.strftime("%Y-%m-%d") for d in display_income.index])
            for col, name in display_cols.items():
                if col in display_income.columns:
                    show_df[name] = display_income[col].apply(
                        lambda x: (f"{x:.2f}" if col == "eps_basic" else format_large_number(x)) if not pd.isna(x) else "N/A"
                    )
            st.dataframe(show_df, use_container_width=True)

    # ============================================================
    # 頁籤 2：資產負債表分析
    # ============================================================
    with tab2:
        st.subheader(f"🏦 資產負債表分析（{period_type}數據）")

        if display_balance.empty:
            st.warning("無法獲取資產負債表數據。")
        else:
            dates = [d.strftime("%Y-%m") for d in display_balance.index]
            series = []
            for col, name, color in [("assets", "總資產", COLORS["steel_blue"]),
                                      ("liabilities", "總負債", COLORS["dark_red"]),
                                      ("stockholdersequity", "股東權益", COLORS["dark_green"])]:
                if col in display_balance.columns:
                    series.append((name, display_balance[col].tolist(), color))
            if series:
                fig = create_multi_bar_chart(dates, series, "資產負債結構趨勢", "金額（元）")
                st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if "assetscurrent" in display_balance.columns and "liabilitiescurrent" in display_balance.columns:
                    crs = [safe_divide(ca, cl) for ca, cl in zip(
                        display_balance["assetscurrent"].fillna(0),
                        display_balance["liabilitiescurrent"].replace(0, np.nan).fillna(1)
                    )]
                    fig2 = create_bar_chart(dates, crs, "流動比率趨勢", "日期", "流動比率", color=COLORS["teal"])
                    st.plotly_chart(fig2, use_container_width=True)
            with col_b:
                if "liabilities" in display_balance.columns and "assets" in display_balance.columns:
                    drs = [safe_divide(d, a) * 100 for d, a in zip(
                        display_balance["liabilities"].fillna(0),
                        display_balance["assets"].replace(0, np.nan).fillna(1)
                    )]
                    fig3 = create_bar_chart(dates, drs, "負債比率趨勢（%）", "日期", "負債比率（%）", color=COLORS["purple"])
                    st.plotly_chart(fig3, use_container_width=True)

            st.markdown("#### 財務比率計算")
            ratio_rows = []
            for d, row in display_balance.iterrows():
                a = row.get("assets", np.nan)
                l = row.get("liabilities", np.nan)
                e = row.get("stockholdersequity", np.nan)
                ca = row.get("assetscurrent", np.nan)
                cl = row.get("liabilitiescurrent", np.nan)
                ratio_rows.append({
                    "日期": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                    "負債比率": f"{safe_divide(l, a)*100:.2f}%" if not pd.isna(a) else "N/A",
                    "流動比率": f"{safe_divide(ca, cl):.2f}" if not (pd.isna(ca) or pd.isna(cl)) else "N/A",
                    "股東權益": format_large_number(e),
                    "總資產": format_large_number(a),
                })
            st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True, hide_index=True)

    # ============================================================
    # 頁籤 3：現金流量表分析
    # ============================================================
    with tab3:
        st.subheader(f"💰 現金流量表分析（{period_type}數據）")

        if display_cashflow.empty:
            st.warning("無法獲取現金流量表數據。")
        else:
            dates = [d.strftime("%Y-%m") for d in display_cashflow.index]
            series = []
            for col, name, color in [
                ("netcashprovidedbyusedinoperatingactivities", "營運現金流", COLORS["dark_green"]),
                ("netcashprovidedbyusedininvestingactivities", "投資現金流", COLORS["dark_red"]),
                ("netcashprovidedbyusedinfinancingactivities", "融資現金流", COLORS["steel_blue"]),
            ]:
                if col in display_cashflow.columns:
                    series.append((name, display_cashflow[col].tolist(), color))

            if series:
                fig = create_multi_bar_chart(dates, series, "三大現金流趨勢", "金額（元）")
                st.plotly_chart(fig, use_container_width=True)

            # 自由現金流趨勢（使用年度計算結果）
            if cashflow_result:
                fcf_dates = [r["date"] for r in cashflow_result]
                fcf_values = [r["free_cash_flow"] for r in cashflow_result]
                fig2 = create_bar_chart(fcf_dates, fcf_values, "自由現金流趨勢（年度）", "日期", "金額（元）")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### 詳細現金流數據")
            cf_rows = []
            for d, row in display_cashflow.iterrows():
                ocf = row.get("netcashprovidedbyusedinoperatingactivities", np.nan)
                icf = row.get("netcashprovidedbyusedininvestingactivities", np.nan)
                ffcf = row.get("netcashprovidedbyusedinfinancingactivities", np.nan)
                capex = abs(row.get("paymentstoacquireproductiveassets", 0) or 0)
                total = sum([v for v in [ocf, icf, ffcf] if not pd.isna(v)])
                cf_rows.append({
                    "日期": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                    "營運現金流": format_large_number(ocf),
                    "投資現金流": format_large_number(icf),
                    "融資現金流": format_large_number(ffcf),
                    "資本支出": format_large_number(capex),
                    "現金流總計": format_large_number(total),
                })
            st.dataframe(pd.DataFrame(cf_rows), use_container_width=True, hide_index=True)

    # ============================================================
    # 頁籤 4：三階段財報分析（固定年度數據）
    # ============================================================
    with tab4:
        st.subheader("🎯 三階段財報分析")

        # 若用戶選擇季度模式，顯示提示
        if period_type == "季度":
            st.info("⚠️ 三階段財報分析固定使用**年度數據**，以確保分析準確性。")

        # 數據品質報告
        quality_report = generate_data_quality_report(annual_financial_data)
        with st.expander(f"📋 數據品質報告（{quality_report['quality_level']}）", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**品質等級**：{quality_report['quality_level']}")
                st.write(f"**年度數據期數**：{quality_report['years_count']} 期")
                if quality_report["missing_fields"]:
                    st.write("**缺失欄位**：")
                    for f in quality_report["missing_fields"]:
                        st.write(f"  - {f}")
            with c2:
                st.write("**計算欄位說明**：")
                for f in quality_report["computed_fields"]:
                    st.write(f"  - {f}")
                for l in quality_report["limitations"]:
                    st.warning(l)

        st.markdown("---")

        # ---- 階段一：F-Score ----
        st.markdown("### 📊 階段一：Piotroski F-Score")
        if fscore_result is None:
            st.warning("⚠️ 年度財務數據不足 2 年，無法計算 F-Score。")
        else:
            total = fscore_result["total_score"]
            rating = "強烈看好 🌟" if total >= 7 else ("中性 ⚖️" if total >= 4 else "謹慎看待 ⚠️")

            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                st.metric(label="F-Score 總分", value=f"{total} / 9", delta=rating)
            with col_s2:
                fig_pie = create_pie_chart(
                    ["通過", "未通過"], [total, 9 - total],
                    "F-Score 通過率",
                    [COLORS["dark_green"], COLORS["dark_red"]],
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            display_fscore_tables(fscore_result)

        st.markdown("---")

        # ---- 階段二：杜邦分析 ----
        st.markdown("### 🔬 階段二：杜邦分析")
        if not dupont_result:
            st.warning("⚠️ 無法進行杜邦分析。")
        else:
            st.metric(label="最新年度 ROE", value=f"{dupont_result[0]['roe_dupont']*100:.2f}%")

            st.markdown("#### 年度杜邦分析表格")
            st.dataframe(pd.DataFrame([{
                "日期": e["date"],
                "淨利率": f"{e['net_margin']*100:.2f}%",
                "資產周轉率": f"{e['asset_turnover']:.4f}",
                "權益乘數": f"{e['equity_multiplier']:.2f}",
                "計算ROE": f"{e['roe_dupont']*100:.2f}%",
                "直接ROE": f"{e['roe_direct']*100:.2f}%",
            } for e in dupont_result]), use_container_width=True, hide_index=True)

            dupont_dates = [e["date"] for e in dupont_result]
            fig_dup = create_line_chart(
                dupont_dates,
                [("淨利率(%)", [e["net_margin"]*100 for e in dupont_result], COLORS["dark_green"]),
                 ("ROE(%)", [e["roe_dupont"]*100 for e in dupont_result], COLORS["steel_blue"])],
                "ROE 與淨利率趨勢", "百分比（%）",
            )
            st.plotly_chart(fig_dup, use_container_width=True)

            trend_rows = [e for e in dupont_result if e["net_margin_change"] is not None]
            if trend_rows:
                st.markdown("#### 趨勢變化分析表格")
                st.dataframe(pd.DataFrame([{
                    "日期": e["date"],
                    "淨利率變化": f"{e['net_margin_change']*100:+.2f}%",
                    "資產周轉率變化": f"{e['asset_turnover_change']:+.4f}",
                    "權益乘數變化": f"{e['equity_multiplier_change']:+.2f}",
                    "ROE 變化": f"{e['roe_change']*100:+.2f}%",
                } for e in trend_rows]), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---- 階段三：現金流分析 ----
        st.markdown("### 💧 階段三：現金流分析")
        if not cashflow_result:
            st.warning("⚠️ 無法進行現金流分析。")
        else:
            latest_cf = cashflow_result[0]
            st.metric(
                label=f"現金流品質評估：{latest_cf['quality_rating']}",
                value=f"{latest_cf['ocf_quality_ratio']:.2f}",
                help="OCF品質比率 = 營運現金流 / 淨利潤"
            )

            st.markdown("#### 現金流關鍵指標")
            st.dataframe(pd.DataFrame([
                {"指標": "營運現金流品質比率", "數值": f"{latest_cf['ocf_quality_ratio']:.2f}", "評估": latest_cf["quality_rating"]},
                {"指標": "自由現金流（最新年度）", "數值": format_large_number(latest_cf["free_cash_flow"]),
                 "評估": "正值為佳 ✅" if latest_cf["free_cash_flow"] > 0 else "需關注 🔴"},
            ]), use_container_width=True, hide_index=True)

            st.markdown("#### 現金流結構分析（最新年度）")
            st.dataframe(pd.DataFrame([
                {"類型": "營運現金流", "金額": format_large_number(latest_cf["operating_cash_flow"])},
                {"類型": "投資現金流", "金額": format_large_number(latest_cf["investing_cash_flow"])},
                {"類型": "融資現金流", "金額": format_large_number(latest_cf["financing_cash_flow"])},
            ]), use_container_width=True, hide_index=True)

            st.markdown("#### 詳細現金流數據（多年度）")
            st.dataframe(pd.DataFrame([{
                "日期": r["date"],
                "營運現金流": format_large_number(r["operating_cash_flow"]),
                "投資現金流": format_large_number(r["investing_cash_flow"]),
                "融資現金流": format_large_number(r["financing_cash_flow"]),
                "淨利潤": format_large_number(r["net_income"]),
                "資本支出": format_large_number(r["capex"]),
                "現金流總計": format_large_number(r["operating_cash_flow"] + r["investing_cash_flow"] + r["financing_cash_flow"]),
            } for r in cashflow_result]), use_container_width=True, hide_index=True)

    # ============================================================
    # 頁籤 5：AI 分析
    # ============================================================
    with tab5:
        st.subheader(f"🤖 AI 深度財務分析（模型：{ai_model}）")

        if not openai_key:
            st.warning("⚠️ 請在左側填入 OpenAI API 金鑰以使用 AI 分析功能。")
        else:
            if st.button("🚀 開始 AI 分析", type="primary"):
                with st.spinner(f"🤖 正在使用 {ai_model} 進行三階段財務分析，約需 30-60 秒..."):
                    st.info("📊 正在使用 AI 進行三階段財務分析，包含 F-Score 解讀、杜邦趨勢洞察、現金流深度分析...")

                    ai_data = prepare_ai_analysis_data(
                        annual_financial_data, fscore_result, dupont_result, cashflow_result,
                        stock_id, company_info, market_cap, latest_price
                    )
                    ai_result = run_ai_analysis(openai_key, ai_data, model=ai_model)

                if ai_result.startswith("⚠️"):
                    st.error(ai_result)
                else:
                    st.success("✅ AI 分析完成！")
                    st.markdown(ai_result)
            else:
                st.info(f"""
**AI 分析功能說明**（使用模型：{ai_model}）

點擊「開始 AI 分析」後，系統將進行：
- 🎯 三階段評分總結
- 📊 Piotroski F-Score 指標解讀
- 🔬 杜邦分析趨勢洞察
- 💧 現金流結構深度分析
- 🏛️ 台股市場特性與投資環境分析
- ⚠️ 風險因素與後續追蹤重點

分析約需 30-60 秒，請耐心等待。
                """)


if __name__ == "__main__":
    main()
