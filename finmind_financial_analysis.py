"""
AI 台股財報分析系統 (FinMind) - v3.0
修正：
- session_state 儲存分析結果，AI 分析按鈕不再跳回首頁
- AI 分析結果快取，不因頁面重跑而消失
- 歷史紀錄功能（最多 5 筆）
- 季度/年度數據區間切換
- AI 模型選擇（gpt-4.1-nano / gpt-5-mini）
- 市值使用 TaiwanStockShareholding 正確計算
- 移除免責聲明
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from openai import OpenAI

# ============================================================
# 頁面設定
# ============================================================
st.set_page_config(
    page_title="AI 台股財報分析系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# session_state 初始化
# ============================================================
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}
if "viewing_idx" not in st.session_state:
    st.session_state.viewing_idx = None

MAX_HISTORY = 5

# ============================================================
# 工具函數
# ============================================================

def format_large_number(value):
    """大數字格式化（兆/億/百萬）"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        value = float(value)
        abs_v = abs(value)
        sign = "-" if value < 0 else ""
        if abs_v >= 1e12:
            return f"{sign}{abs_v/1e12:.2f}兆"
        elif abs_v >= 1e8:
            return f"{sign}{abs_v/1e8:.2f}億"
        elif abs_v >= 1e6:
            return f"{sign}{abs_v/1e6:.2f}百萬"
        else:
            return f"{sign}{abs_v:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def validate_stock_code(code):
    """驗證台股四位數代碼"""
    if not code:
        return False, "請輸入股票代碼"
    code = code.strip()
    if not code.isdigit():
        return False, "股票代碼必須為數字（範例：2330、2454、2317、2412）"
    if len(code) != 4:
        return False, f"台股代碼必須為四位數字，您輸入了 {len(code)} 位"
    return True, "OK"


def safe_divide(n, d, default=0.0):
    """安全除法"""
    try:
        if d == 0 or d is None:
            return default
        r = float(n) / float(d)
        return default if (np.isnan(r) or np.isinf(r)) else r
    except Exception:
        return default


def filter_by_period(df, period_type):
    """
    依數據區間過濾：
    - 年度：只保留每年 12 月的資料（年報）
    - 季度：保留全部資料
    """
    if df is None or df.empty:
        return df
    if period_type == "年度":
        return df[df.index.month == 12]
    return df


def save_to_history(result_dict):
    """儲存分析結果到歷史紀錄（最多 MAX_HISTORY 筆，同股票+條件去重）"""
    history = [h for h in st.session_state.analysis_history if not (
        h["stock_id"] == result_dict["stock_id"] and
        h["period_type"] == result_dict["period_type"] and
        h["start_date"] == result_dict["start_date"] and
        h["end_date"] == result_dict["end_date"]
    )]
    history.insert(0, result_dict)
    st.session_state.analysis_history = history[:MAX_HISTORY]
    st.session_state.current_result = result_dict
    st.session_state.viewing_idx = 0


# ============================================================
# FinMind API 整合
# ============================================================

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

INCOME_MAP = {
    "Revenue": "revenues",
    "GrossProfit": "grossprofit",
    "OperatingIncome": "operatingincomeloss",
    "IncomeAfterTaxes": "netincomeloss",
    "PreTaxIncome": "incomelossfromcontinuingoperationsbeforeincometaxes",
    "EPS": "eps_basic",
    "TotalNonoperatingIncomeAndExpense": "total_nonoperating",
}
BALANCE_MAP = {
    "TotalAssets": "assets",
    "Liabilities": "liabilities",
    "Equity": "stockholdersequity",
    "CurrentAssets": "assetscurrent",
    "CurrentLiabilities": "liabilitiescurrent",
    "RetainedEarnings": "retainedearningsaccumulateddeficit",
    "NoncurrentLiabilities": "longtermdebtnoncurrent",
}
CASHFLOW_MAP = {
    "CashFlowsFromOperatingActivities": "netcashprovidedbyusedinoperatingactivities",
    "CashProvidedByInvestingActivities": "netcashprovidedbyusedininvestingactivities",
    "CashFlowsProvidedFromFinancingActivities": "netcashprovidedbyusedinfinancingactivities",
    "PropertyAndPlantAndEquipment": "paymentstoacquireproductiveassets",
}


def fetch_finmind(dataset, stock_id, start_date, end_date, token):
    """呼叫 FinMind API 取得數據"""
    try:
        resp = requests.get(FINMIND_API_URL, params={
            "dataset": dataset, "data_id": stock_id,
            "start_date": start_date, "end_date": end_date, "token": token,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != 200:
            st.warning(f"FinMind（{dataset}）：{data.get('msg', '未知錯誤')}")
            return None
        records = data.get("data", [])
        return pd.DataFrame(records) if records else None
    except requests.exceptions.ConnectionError:
        st.error("無法連接 FinMind API，請確認網路連線。")
        return None
    except requests.exceptions.Timeout:
        st.error("FinMind API 逾時，請稍後重試。")
        return None
    except Exception as e:
        st.error(f"FinMind 錯誤（{dataset}）：{e}")
        return None


def standardize(df, mapping, date_col="date"):
    """將 FinMind type 欄位轉換為內部標準欄位名稱"""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        if not all(c in df.columns for c in [date_col, "type", "value"]):
            return pd.DataFrame()
        f = df[df["type"].isin(mapping.keys())].copy()
        if f.empty:
            return pd.DataFrame()
        f["k"] = f["type"].map(mapping)
        pivot = f.pivot_table(index=date_col, columns="k", values="value", aggfunc="first")
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index(ascending=False)
        for col in pivot.columns:
            pivot[col] = pd.to_numeric(pivot[col], errors="coerce")
        return pivot
    except Exception as e:
        st.warning(f"標準化錯誤：{e}")
        return pd.DataFrame()


def fetch_all(stock_id, start_date, end_date, token):
    """獲取所有財務報表數據"""
    result = {"income": pd.DataFrame(), "balance": pd.DataFrame(),
              "cashflow": pd.DataFrame(), "price": pd.DataFrame(),
              "shareholding": pd.DataFrame(), "company_info": {}}

    progress = st.progress(0, text="正在獲取損益表...")
    result["income"] = standardize(
        fetch_finmind("TaiwanStockFinancialStatements", stock_id, start_date, end_date, token), INCOME_MAP)
    progress.progress(17, text="正在獲取資產負債表...")
    result["balance"] = standardize(
        fetch_finmind("TaiwanStockBalanceSheet", stock_id, start_date, end_date, token), BALANCE_MAP)
    progress.progress(34, text="正在獲取現金流量表...")
    result["cashflow"] = standardize(
        fetch_finmind("TaiwanStockCashFlowsStatement", stock_id, start_date, end_date, token), CASHFLOW_MAP)
    progress.progress(51, text="正在獲取股價...")
    price_raw = fetch_finmind("TaiwanStockPrice", stock_id, start_date, end_date, token)
    if price_raw is not None and not price_raw.empty:
        result["price"] = price_raw
    progress.progress(68, text="正在獲取發行股數...")
    sh_raw = fetch_finmind("TaiwanStockShareholding", stock_id, start_date, end_date, token)
    if sh_raw is not None and not sh_raw.empty:
        result["shareholding"] = sh_raw
    progress.progress(84, text="正在獲取公司基本資料...")
    info_raw = fetch_finmind("TaiwanStockInfo", stock_id, "2010-01-01", end_date, token)
    if info_raw is not None and not info_raw.empty:
        row = info_raw[info_raw["stock_id"] == stock_id].iloc[0] if "stock_id" in info_raw.columns else info_raw.iloc[0]
        result["company_info"] = row.to_dict()
    progress.progress(100, text="完成！")
    progress.empty()
    return result


def compute_derived(income_df, balance_df, cashflow_df, price_df, shareholding_df):
    """計算衍生欄位：加權平均股數、利息費用、資本支出絕對值、市值"""
    # 加權平均股數
    if not income_df.empty:
        if "netincomeloss" in income_df.columns and "eps_basic" in income_df.columns:
            mask = (income_df["eps_basic"] != 0) & (~income_df["eps_basic"].isna())
            income_df["weightedaveragenumberofsharesoutstandingbasic"] = np.nan
            income_df.loc[mask, "weightedaveragenumberofsharesoutstandingbasic"] = (
                income_df.loc[mask, "netincomeloss"] / income_df.loc[mask, "eps_basic"]
            ) * 1000
        # 利息費用推估
        if "total_nonoperating" in income_df.columns:
            income_df["interestexpensenonoperating"] = income_df["total_nonoperating"].apply(
                lambda x: abs(x) if (not pd.isna(x) and x < 0) else 0
            )
    # 資本支出取絕對值
    if not cashflow_df.empty and "paymentstoacquireproductiveassets" in cashflow_df.columns:
        cashflow_df["paymentstoacquireproductiveassets"] = cashflow_df["paymentstoacquireproductiveassets"].abs()

    # 市值：最新收盤價 × TaiwanStockShareholding 發行股數
    market_cap = None
    latest_price = None
    if price_df is not None and not price_df.empty and "close" in price_df.columns:
        price_df["date"] = pd.to_datetime(price_df["date"])
        try:
            latest_price = float(price_df.sort_values("date", ascending=False).iloc[0]["close"])
        except (ValueError, TypeError):
            latest_price = None

    if shareholding_df is not None and not shareholding_df.empty and "number_of_shares_issued" in shareholding_df.columns:
        shareholding_df["date"] = pd.to_datetime(shareholding_df["date"])
        try:
            shares = float(str(shareholding_df.sort_values("date", ascending=False).iloc[0]["number_of_shares_issued"]).replace(",", ""))
            if latest_price and shares:
                market_cap = latest_price * shares
        except (ValueError, TypeError):
            pass

    return income_df, balance_df, cashflow_df, market_cap, latest_price


def merge_data(income_df, balance_df, cashflow_df):
    """將三個報表依日期合併為列表"""
    if income_df.empty and balance_df.empty and cashflow_df.empty:
        return []
    all_dates = set()
    for df in [income_df, balance_df, cashflow_df]:
        if not df.empty:
            all_dates.update(df.index.tolist())
    if not all_dates:
        return []
    merged = []
    for d in sorted(all_dates, reverse=True):
        record = {"date": d}
        for df in [income_df, balance_df, cashflow_df]:
            if not df.empty and d in df.index:
                for col in df.columns:
                    record[col] = df.loc[d, col]
        merged.append(record)
    return merged


# ============================================================
# 財務計算
# ============================================================

def calc_fscore(annual_data):
    """Piotroski F-Score 9 項指標（固定使用年度數據）"""
    if len(annual_data) < 2:
        return None
    curr, prev = annual_data[0], annual_data[1]

    def gv(rec, key, default=0.0):
        v = rec.get(key)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    res = {"total_score": 0, "profitability_scores": [], "leverage_scores": [], "efficiency_scores": []}

    # 獲利能力
    curr_net = gv(curr, "netincomeloss"); curr_assets = gv(curr, "assets", 1)
    prev_net = gv(prev, "netincomeloss"); prev_assets = gv(prev, "assets", 1)
    curr_roa = safe_divide(curr_net, curr_assets); prev_roa = safe_divide(prev_net, prev_assets)
    curr_ocf = gv(curr, "netcashprovidedbyusedinoperatingactivities")

    for score, desc, cur_v, pre_v in [
        (1 if curr_roa > 0 else 0, "ROA 正值（淨利潤 / 總資產 > 0）", f"{curr_roa*100:.2f}%", "-"),
        (1 if curr_ocf > 0 else 0, "營運現金流 > 0", format_large_number(curr_ocf), "-"),
        (1 if curr_roa > prev_roa else 0, "ROA 年增（最新 > 前期）", f"{curr_roa*100:.2f}%", f"{prev_roa*100:.2f}%"),
        (1 if curr_ocf > curr_net else 0, "現金流品質（OCF > 淨利潤）", f"OCF={format_large_number(curr_ocf)}", f"NI={format_large_number(curr_net)}"),
    ]:
        res["profitability_scores"].append({"description": desc, "current_value": cur_v, "previous_value": pre_v, "score": score, "passed": score == 1})

    # 槓桿與流動性
    curr_ltd = gv(curr, "longtermdebtnoncurrent"); prev_ltd = gv(prev, "longtermdebtnoncurrent")
    curr_ltd_r = safe_divide(curr_ltd, curr_assets); prev_ltd_r = safe_divide(prev_ltd, prev_assets)
    curr_ca = gv(curr, "assetscurrent", 1); curr_cl = gv(curr, "liabilitiescurrent", 1)
    prev_ca = gv(prev, "assetscurrent", 1); prev_cl = gv(prev, "liabilitiescurrent", 1)
    curr_cr = safe_divide(curr_ca, curr_cl); prev_cr = safe_divide(prev_ca, prev_cl)
    curr_sh = gv(curr, "weightedaveragenumberofsharesoutstandingbasic")
    prev_sh = gv(prev, "weightedaveragenumberofsharesoutstandingbasic")

    for score, desc, cur_v, pre_v in [
        (1 if curr_ltd_r < prev_ltd_r else 0, "長期負債比率改善（最新 < 前期）", f"{curr_ltd_r*100:.2f}%", f"{prev_ltd_r*100:.2f}%"),
        (1 if curr_cr > prev_cr else 0, "流動比率改善（最新 > 前期）", f"{curr_cr:.2f}", f"{prev_cr:.2f}"),
        (1 if (curr_sh > 0 and prev_sh > 0 and curr_sh <= prev_sh) else 0, "股份未稀釋（流通股數未增加）", format_large_number(curr_sh), format_large_number(prev_sh)),
    ]:
        res["leverage_scores"].append({"description": desc, "current_value": cur_v, "previous_value": pre_v, "score": score, "passed": score == 1})

    # 營運效率
    curr_gp = gv(curr, "grossprofit"); curr_rev = gv(curr, "revenues", 1)
    prev_gp = gv(prev, "grossprofit"); prev_rev = gv(prev, "revenues", 1)
    curr_gpm = safe_divide(curr_gp, curr_rev); prev_gpm = safe_divide(prev_gp, prev_rev)
    curr_ato = safe_divide(curr_rev, curr_assets); prev_ato = safe_divide(prev_rev, prev_assets)

    for score, desc, cur_v, pre_v in [
        (1 if curr_gpm > prev_gpm else 0, "毛利率改善（最新 > 前期）", f"{curr_gpm*100:.2f}%", f"{prev_gpm*100:.2f}%"),
        (1 if curr_ato > prev_ato else 0, "資產周轉率改善（最新 > 前期）", f"{curr_ato:.3f}", f"{prev_ato:.3f}"),
    ]:
        res["efficiency_scores"].append({"description": desc, "current_value": cur_v, "previous_value": pre_v, "score": score, "passed": score == 1})

    res["total_score"] = sum(i["score"] for grp in ["profitability_scores", "leverage_scores", "efficiency_scores"] for i in res[grp])
    return res


def calc_dupont(annual_data, max_years=3):
    """杜邦分析 ROE 三因子分解（固定使用年度數據）"""
    results = []
    for record in annual_data[:max_years]:
        def gv(key, default=0.0):
            v = record.get(key)
            return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        ni = gv("netincomeloss"); rev = gv("revenues", 1); assets = gv("assets", 1); equity = gv("stockholdersequity", 1)
        nm = safe_divide(ni, rev); at = safe_divide(rev, assets); em = safe_divide(assets, equity)
        entry = {
            "date": record["date"].strftime("%Y-%m-%d") if hasattr(record["date"], "strftime") else str(record["date"]),
            "net_margin": nm, "asset_turnover": at, "equity_multiplier": em,
            "roe_dupont": nm * at * em, "roe_direct": safe_divide(ni, equity),
            "net_margin_change": None, "asset_turnover_change": None,
            "equity_multiplier_change": None, "roe_change": None,
        }
        if results:
            p = results[-1]
            entry["net_margin_change"] = nm - p["net_margin"]
            entry["asset_turnover_change"] = at - p["asset_turnover"]
            entry["equity_multiplier_change"] = em - p["equity_multiplier"]
            entry["roe_change"] = entry["roe_dupont"] - p["roe_dupont"]
        results.append(entry)
    return results


def calc_cashflow(annual_data, max_years=5):
    """現金流分析（固定使用年度數據）"""
    results = []
    for record in annual_data[:max_years]:
        def gv(key, default=0.0):
            v = record.get(key)
            return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        ocf = gv("netcashprovidedbyusedinoperatingactivities")
        icf = gv("netcashprovidedbyusedininvestingactivities")
        ffcf = gv("netcashprovidedbyusedinfinancingactivities")
        ni = gv("netincomeloss", 1)
        capex = abs(gv("paymentstoacquireproductiveassets"))
        fcf = ocf - capex  # 自由現金流 = OCF - 資本支出絕對值
        quality = safe_divide(ocf, ni) if ni != 0 else 0
        rating = "優秀 🌟" if quality >= 1.2 else ("良好 ✅" if quality >= 1.0 else ("尚可 ⚠️" if quality >= 0.8 else "需關注 🔴"))
        results.append({
            "date": record["date"].strftime("%Y-%m-%d") if hasattr(record["date"], "strftime") else str(record["date"]),
            "operating_cash_flow": ocf, "investing_cash_flow": icf, "financing_cash_flow": ffcf,
            "net_income": ni, "capex": capex, "free_cash_flow": fcf,
            "ocf_quality_ratio": quality, "quality_rating": rating,
        })
    return results


# ============================================================
# 視覺化
# ============================================================

C = {
    "dark_green": "#1B5E20", "dark_red": "#B71C1C",
    "steel_blue": "#1565C0", "gold": "#F57F17",
    "purple": "#4A148C", "teal": "#00695C",
}


def bar_chart(x, y, title, x_label, y_label, color=None, height=400):
    colors = [C["steel_blue"] if (v is not None and not (isinstance(v, float) and np.isnan(v)) and float(v) >= 0) else C["dark_red"] for v in y]
    if color:
        colors = [color] * len(y)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, marker_color=colors,
                         text=[format_large_number(v) for v in y], textposition="outside",
                         hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
                      xaxis=dict(title=x_label, tickangle=-45), yaxis=dict(title=y_label),
                      template="plotly_white", height=height, margin=dict(t=60, b=80, l=60, r=20))
    return fig


def multi_bar_chart(dates, series, title, y_label, height=400):
    """series: list of (name, values, color)"""
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(go.Bar(name=name, x=dates, y=values, marker_color=color,
                             hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
                      barmode="group", xaxis=dict(title="日期", tickangle=-45), yaxis=dict(title=y_label),
                      template="plotly_white", height=height, margin=dict(t=60, b=80, l=60, r=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def line_chart(x, series, title, y_label, height=400):
    """series: list of (name, values, color)"""
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(go.Scatter(x=x, y=values, mode="lines+markers", name=name,
                                 line=dict(color=color, width=2), marker=dict(size=8),
                                 hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
                      xaxis=dict(title="日期", tickangle=-45), yaxis=dict(title=y_label),
                      template="plotly_white", height=height, margin=dict(t=60, b=80, l=60, r=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def pie_chart(labels, values, title, colors, height=350):
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=labels, values=values, marker=dict(colors=colors),
                         textinfo="label+percent",
                         hovertemplate="<b>%{label}</b><br>%{value} 項<br>%{percent}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
                      template="plotly_white", height=height, margin=dict(t=60, b=20, l=20, r=20))
    return fig


def storable_to_df(stored):
    """將儲存格式還原為 DataFrame"""
    if not stored or not stored.get("data"):
        return pd.DataFrame()
    df = pd.DataFrame(stored["data"], columns=stored["columns"])
    df.index = pd.to_datetime(stored["index"])
    df.index.name = None
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def df_to_storable(df):
    """DataFrame 轉為可序列化格式"""
    if df is None or df.empty:
        return {"data": [], "columns": [], "index": []}
    return {"data": df.values.tolist(), "columns": df.columns.tolist(), "index": [str(i) for i in df.index]}


# ============================================================
# AI 分析
# ============================================================

def prepare_ai_data(r):
    """整合分析結果，準備 AI 提示語"""
    fscore = r.get("fscore_result")
    dupont = r.get("dupont_result", [])
    cashflow = r.get("cashflow_result", [])
    annual = r.get("annual_financial_data", [])
    company_info = r.get("company_info", {})

    fscore_text = ""
    if fscore:
        fscore_text = f"【F-Score】總分：{fscore['total_score']} / 9\n"
        for grp, label in [("profitability_scores", "獲利能力"), ("leverage_scores", "槓桿流動性"), ("efficiency_scores", "營運效率")]:
            fscore_text += f"\n{label}：\n"
            for item in fscore[grp]:
                fscore_text += f"  {'✅' if item['passed'] else '❌'} {item['description']}：{item['current_value']}（前期：{item['previous_value']}）\n"

    dupont_text = "\n【杜邦分析】\n"
    for e in dupont:
        dupont_text += f"  {e['date']}：淨利率={e['net_margin']*100:.2f}%，資產周轉率={e['asset_turnover']:.3f}，權益乘數={e['equity_multiplier']:.2f}，ROE={e['roe_dupont']*100:.2f}%\n"

    cashflow_text = "\n【現金流分析】\n"
    for e in cashflow[:3]:
        cashflow_text += f"  {e['date']}：OCF={format_large_number(e['operating_cash_flow'])}，FCF={format_large_number(e['free_cash_flow'])}，品質比率={e['ocf_quality_ratio']:.2f}（{e['quality_rating']}）\n"

    latest = annual[0] if annual else {}
    def fv(key):
        v = latest.get(key)
        return "N/A" if (v is None or (isinstance(v, float) and np.isnan(v))) else format_large_number(float(v))

    financial_text = f"""
【最新財務數據（{latest.get('date', 'N/A')}）】
營收：{fv('revenues')} / 毛利：{fv('grossprofit')} / 營業利潤：{fv('operatingincomeloss')}
淨利潤：{fv('netincomeloss')} / 總資產：{fv('assets')} / 股東權益：{fv('stockholdersequity')}
市值（估算）：{format_large_number(r.get('market_cap'))}
最新股價：{f"{r.get('latest_price'):.2f} 元" if r.get('latest_price') else 'N/A'}
產業別：{company_info.get('industry_category', '未知')}
"""
    return {
        "company_name": company_info.get("stock_name", r["stock_id"]),
        "stock_id": r["stock_id"],
        "fscore_text": fscore_text,
        "dupont_text": dupont_text,
        "cashflow_text": cashflow_text,
        "financial_text": financial_text,
    }


def run_ai(openai_key, ai_data, model):
    """
    OpenAI 新版 API 呼叫（client.chat.completions.create）
    禁止使用舊版 ChatCompletion.create
    """
    try:
        client = OpenAI(api_key=openai_key)
        system_msg = "你是專精台股財務分析和台灣會計準則（IFRS台版）的資深分析師，熟悉 FinMind 開源資料特性與台股市場投資環境。請用繁體中文提供客觀專業的分析報告。"
        user_msg = f"""
請根據以下已完成的三階段財務分析，對台股 {ai_data['stock_id']}（{ai_data['company_name']}）進行深度財務分析。
請基於已計算完成的數據進行解讀，勿重新計算。

{ai_data['fscore_text']}
{ai_data['dupont_text']}
{ai_data['cashflow_text']}
{ai_data['financial_text']}

請依以下結構提供完整報告：

## 一、三階段評分總結
| 分析階段 | 評分狀態 | 評價 | 主要發現 |
|---------|---------|------|---------|
| Piotroski F-Score | ... | ... | ... |
| 杜邦分析 | ... | ... | ... |
| 現金流分析 | ... | ... | ... |

## 二、Piotroski F-Score 解讀
## 三、杜邦分析趨勢洞察
## 四、現金流結構深度分析
## 五、台股市場特性分析（法規、產業政策、競爭優勢）

## 六、資料來源與限制說明
- 加權平均股數：淨利潤 ÷ EPS 計算，可能有精度誤差
- 利息費用：由營業外收支推估
- 市值：收盤價 × TaiwanStockShareholding 發行股數

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
"""
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4000,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "authentication" in err.lower():
            return f"⚠️ OpenAI API 金鑰錯誤，請確認金鑰是否正確。\n\n{err}"
        elif "model" in err.lower():
            return f"⚠️ 模型 {model} 不在您的 API 授權範圍內。\n\n{err}"
        elif "quota" in err.lower() or "rate" in err.lower():
            return f"⚠️ API 使用量超限或頻率過高，請稍後再試。\n\n{err}"
        else:
            return f"⚠️ AI 分析發生錯誤，請確認 API 金鑰和網路連線。\n\n{err}"


# ============================================================
# 渲染分析結果
# ============================================================

def render_results(r, openai_key, ai_model):
    """渲染完整分析結果（從 session_state 讀取，不重新計算）"""
    company_info = r.get("company_info", {})
    market_cap = r.get("market_cap")
    latest_price = r.get("latest_price")
    stock_id = r["stock_id"]
    period_type = r["period_type"]

    # 還原 DataFrame
    display_income = storable_to_df(r.get("display_income", {}))
    display_balance = storable_to_df(r.get("display_balance", {}))
    display_cashflow = storable_to_df(r.get("display_cashflow", {}))

    fscore_result = r.get("fscore_result")
    dupont_result = r.get("dupont_result", [])
    cashflow_result = r.get("cashflow_result", [])
    annual_data = r.get("annual_financial_data", [])
    display_data = r.get("display_financial_data", [])

    # ── 公司基本資訊 ──
    company_name = company_info.get("stock_name", stock_id)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(f"🏢 {company_name}")
        st.write(f"**產業類別**：{company_info.get('industry_category', '未知')}")
        if company_info.get("type"):
            st.write(f"**行業分類**：{company_info.get('type')}")
    with col2:
        st.metric("💹 最新收盤價", f"NT$ {latest_price:,.2f}" if latest_price else "N/A")
    with col3:
        st.write(f"**市值（估算）**：NT$ {format_large_number(market_cap)}")
        latest = annual_data[0] if annual_data else (display_data[0] if display_data else {})
        net_income = latest.get("netincomeloss")
        if market_cap and net_income and not np.isnan(float(net_income)) and float(net_income) > 0:
            st.write(f"**本益比（P/E）**：{market_cap / float(net_income):.2f}x")
        else:
            st.write("**本益比（P/E）**：N/A")

    st.markdown("---")

    # ── 頁籤 ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 損益表分析", "🏦 資產負債表分析",
        "💰 現金流量表分析", "🎯 三階段財報分析", "🤖 AI 分析"
    ])

    # ── 頁籤 1：損益表 ──
    with tab1:
        st.subheader(f"📈 損益表分析（{period_type}數據）")
        if display_income.empty:
            st.warning("無法獲取損益表數據。")
        else:
            dates = [d.strftime("%Y-%m") for d in display_income.index]
            if "revenues" in display_income.columns and "grossprofit" in display_income.columns:
                st.plotly_chart(multi_bar_chart(dates, [
                    ("營收", display_income["revenues"].tolist(), C["steel_blue"]),
                    ("毛利", display_income["grossprofit"].tolist(), C["dark_green"]),
                ], "營收與毛利趨勢", "金額（元）"), use_container_width=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if "netincomeloss" in display_income.columns:
                    st.plotly_chart(bar_chart(dates, display_income["netincomeloss"].tolist(), "淨利潤趨勢", "日期", "金額（元）"), use_container_width=True)
            with col_b:
                if "revenues" in display_income.columns and "grossprofit" in display_income.columns:
                    gpm = [safe_divide(g, r_) * 100 for g, r_ in zip(
                        display_income["grossprofit"].fillna(0),
                        display_income["revenues"].replace(0, np.nan).fillna(1))]
                    st.plotly_chart(bar_chart(dates, gpm, "毛利率趨勢（%）", "日期", "毛利率（%）", color=C["gold"]), use_container_width=True)
            st.markdown("#### 完整損益表數據")
            show = pd.DataFrame(index=[d.strftime("%Y-%m-%d") for d in display_income.index])
            for col, name in [("revenues","營收"),("grossprofit","毛利"),("operatingincomeloss","營業利潤"),("netincomeloss","淨利潤"),("eps_basic","EPS")]:
                if col in display_income.columns:
                    show[name] = display_income[col].apply(lambda x: (f"{x:.2f}" if col == "eps_basic" else format_large_number(x)) if not pd.isna(x) else "N/A")
            st.dataframe(show, use_container_width=True)

    # ── 頁籤 2：資產負債表 ──
    with tab2:
        st.subheader(f"🏦 資產負債表分析（{period_type}數據）")
        if display_balance.empty:
            st.warning("無法獲取資產負債表數據。")
        else:
            dates = [d.strftime("%Y-%m") for d in display_balance.index]
            series = [(n, display_balance[c].tolist(), col) for c, n, col in [("assets","總資產",C["steel_blue"]),("liabilities","總負債",C["dark_red"]),("stockholdersequity","股東權益",C["dark_green"])] if c in display_balance.columns]
            if series:
                st.plotly_chart(multi_bar_chart(dates, series, "資產負債結構趨勢", "金額（元）"), use_container_width=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if "assetscurrent" in display_balance.columns and "liabilitiescurrent" in display_balance.columns:
                    crs = [safe_divide(ca, cl) for ca, cl in zip(display_balance["assetscurrent"].fillna(0), display_balance["liabilitiescurrent"].replace(0, np.nan).fillna(1))]
                    st.plotly_chart(bar_chart(dates, crs, "流動比率趨勢", "日期", "流動比率", color=C["teal"]), use_container_width=True)
            with col_b:
                if "liabilities" in display_balance.columns and "assets" in display_balance.columns:
                    drs = [safe_divide(d_, a) * 100 for d_, a in zip(display_balance["liabilities"].fillna(0), display_balance["assets"].replace(0, np.nan).fillna(1))]
                    st.plotly_chart(bar_chart(dates, drs, "負債比率趨勢（%）", "日期", "負債比率（%）", color=C["purple"]), use_container_width=True)
            st.markdown("#### 財務比率計算")
            st.dataframe(pd.DataFrame([{
                "日期": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                "負債比率": f"{safe_divide(row.get('liabilities',np.nan), row.get('assets',1))*100:.2f}%" if not pd.isna(row.get('assets', np.nan)) else "N/A",
                "流動比率": f"{safe_divide(row.get('assetscurrent',np.nan), row.get('liabilitiescurrent',1)):.2f}" if not pd.isna(row.get('assetscurrent', np.nan)) else "N/A",
                "股東權益": format_large_number(row.get("stockholdersequity")),
                "總資產": format_large_number(row.get("assets")),
            } for d, row in display_balance.iterrows()]), use_container_width=True, hide_index=True)

    # ── 頁籤 3：現金流量表 ──
    with tab3:
        st.subheader(f"💰 現金流量表分析（{period_type}數據）")
        if display_cashflow.empty:
            st.warning("無法獲取現金流量表數據。")
        else:
            dates = [d.strftime("%Y-%m") for d in display_cashflow.index]
            series = [(n, display_cashflow[c].tolist(), col) for c, n, col in [
                ("netcashprovidedbyusedinoperatingactivities","營運現金流",C["dark_green"]),
                ("netcashprovidedbyusedininvestingactivities","投資現金流",C["dark_red"]),
                ("netcashprovidedbyusedinfinancingactivities","融資現金流",C["steel_blue"]),
            ] if c in display_cashflow.columns]
            if series:
                st.plotly_chart(multi_bar_chart(dates, series, "三大現金流趨勢", "金額（元）"), use_container_width=True)
            if cashflow_result:
                st.plotly_chart(bar_chart([e["date"] for e in cashflow_result], [e["free_cash_flow"] for e in cashflow_result], "自由現金流趨勢（年度）", "日期", "金額（元）"), use_container_width=True)
            st.markdown("#### 詳細現金流數據")
            st.dataframe(pd.DataFrame([{
                "日期": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                "營運現金流": format_large_number(row.get("netcashprovidedbyusedinoperatingactivities")),
                "投資現金流": format_large_number(row.get("netcashprovidedbyusedininvestingactivities")),
                "融資現金流": format_large_number(row.get("netcashprovidedbyusedinfinancingactivities")),
                "資本支出": format_large_number(abs(row.get("paymentstoacquireproductiveassets") or 0)),
            } for d, row in display_cashflow.iterrows()]), use_container_width=True, hide_index=True)

    # ── 頁籤 4：三階段財報分析 ──
    with tab4:
        st.subheader("🎯 三階段財報分析")
        if period_type == "季度":
            st.info("⚠️ 三階段財報分析固定使用**年度數據**，以確保分析準確性。")

        # 數據品質報告
        important_fields = {"revenues":"營收","grossprofit":"毛利","operatingincomeloss":"營業利潤","netincomeloss":"淨利潤","assets":"總資產","liabilities":"總負債","stockholdersequity":"股東權益","assetscurrent":"流動資產","liabilitiescurrent":"流動負債","netcashprovidedbyusedinoperatingactivities":"營運現金流","paymentstoacquireproductiveassets":"資本支出"}
        latest_rec = annual_data[0] if annual_data else {}
        missing = [f"{n}（{f}）" for f, n in important_fields.items() if latest_rec.get(f) is None or (isinstance(latest_rec.get(f), float) and np.isnan(latest_rec.get(f)))]
        years = len(annual_data)
        quality = "良好" if len(missing) == 0 else ("部分缺失" if len(missing) <= 3 else "嚴重不足")
        if years < 2:
            quality = "部分缺失"

        with st.expander(f"📋 數據品質報告（{quality}）", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**品質等級**：{quality}")
                st.write(f"**年度數據期數**：{years} 期")
                if missing:
                    st.write("**缺失欄位**：" + "、".join(missing))
            with c2:
                st.write("**計算欄位說明**：")
                for note in ["加權平均股數：淨利潤 ÷ EPS 計算", "利息費用：由營業外收支推估", "市值：收盤價 × 發行股數（TaiwanStockShareholding）", "本益比：市值 ÷ 淨利潤"]:
                    st.write(f"  - {note}")
        st.markdown("---")

        # 階段一：F-Score
        st.markdown("### 📊 階段一：Piotroski F-Score")
        if fscore_result is None:
            st.warning("⚠️ 年度財務數據不足 2 年，無法計算 F-Score。")
        else:
            total = fscore_result["total_score"]
            rating = "強烈看好 🌟" if total >= 7 else ("中性 ⚖️" if total >= 4 else "謹慎看待 ⚠️")
            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                st.metric("F-Score 總分", f"{total} / 9", delta=rating)
            with col_s2:
                st.plotly_chart(pie_chart(["通過","未通過"],[total, 9-total],"F-Score 通過率",[C["dark_green"],C["dark_red"]]), use_container_width=True)
            def fscore_df(scores):
                return pd.DataFrame([{"指標說明":i["description"],"當前值":i["current_value"],"前期值":i["previous_value"],"得分":i["score"],"狀態":"✅" if i["passed"] else "❌"} for i in scores])
            st.markdown("#### 🏆 獲利能力指標")
            st.dataframe(fscore_df(fscore_result["profitability_scores"]), use_container_width=True, hide_index=True)
            st.markdown("#### 🏦 槓桿與流動性指標")
            st.dataframe(fscore_df(fscore_result["leverage_scores"]), use_container_width=True, hide_index=True)
            st.markdown("#### ⚙️ 營運效率指標")
            st.dataframe(fscore_df(fscore_result["efficiency_scores"]), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 階段二：杜邦分析
        st.markdown("### 🔬 階段二：杜邦分析")
        if not dupont_result:
            st.warning("⚠️ 無法進行杜邦分析。")
        else:
            st.metric("最新年度 ROE", f"{dupont_result[0]['roe_dupont']*100:.2f}%")
            st.dataframe(pd.DataFrame([{"日期":e["date"],"淨利率":f"{e['net_margin']*100:.2f}%","資產周轉率":f"{e['asset_turnover']:.4f}","權益乘數":f"{e['equity_multiplier']:.2f}","計算ROE":f"{e['roe_dupont']*100:.2f}%","直接ROE":f"{e['roe_direct']*100:.2f}%"} for e in dupont_result]), use_container_width=True, hide_index=True)
            st.plotly_chart(line_chart([e["date"] for e in dupont_result], [("淨利率(%)",[e["net_margin"]*100 for e in dupont_result],C["dark_green"]),("ROE(%)",[e["roe_dupont"]*100 for e in dupont_result],C["steel_blue"])], "ROE 與淨利率趨勢", "百分比（%）"), use_container_width=True)
            trend = [e for e in dupont_result if e["net_margin_change"] is not None]
            if trend:
                st.markdown("#### 趨勢變化")
                st.dataframe(pd.DataFrame([{"日期":e["date"],"淨利率變化":f"{e['net_margin_change']*100:+.2f}%","資產周轉率變化":f"{e['asset_turnover_change']:+.4f}","權益乘數變化":f"{e['equity_multiplier_change']:+.2f}","ROE變化":f"{e['roe_change']*100:+.2f}%"} for e in trend]), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 階段三：現金流分析
        st.markdown("### 💧 階段三：現金流分析")
        if not cashflow_result:
            st.warning("⚠️ 無法進行現金流分析。")
        else:
            lcf = cashflow_result[0]
            st.metric(f"現金流品質：{lcf['quality_rating']}", f"{lcf['ocf_quality_ratio']:.2f}")
            st.dataframe(pd.DataFrame([{"指標":"OCF品質比率","數值":f"{lcf['ocf_quality_ratio']:.2f}","評估":lcf['quality_rating']},{"指標":"自由現金流（最新）","數值":format_large_number(lcf['free_cash_flow']),"評估":"正值為佳 ✅" if lcf['free_cash_flow']>0 else "需關注 🔴"}]), use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame([{"類型":"營運現金流","金額":format_large_number(lcf["operating_cash_flow"])},{"類型":"投資現金流","金額":format_large_number(lcf["investing_cash_flow"])},{"類型":"融資現金流","金額":format_large_number(lcf["financing_cash_flow"])}]), use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame([{"日期":e["date"],"營運現金流":format_large_number(e["operating_cash_flow"]),"投資現金流":format_large_number(e["investing_cash_flow"]),"融資現金流":format_large_number(e["financing_cash_flow"]),"淨利潤":format_large_number(e["net_income"]),"資本支出":format_large_number(e["capex"]),"現金流總計":format_large_number(e["operating_cash_flow"]+e["investing_cash_flow"]+e["financing_cash_flow"])} for e in cashflow_result]), use_container_width=True, hide_index=True)

    # ── 頁籤 5：AI 分析 ──
    with tab5:
        st.subheader(f"🤖 AI 深度財務分析（模型：{ai_model}）")
        cache_key = f"{stock_id}_{r['start_date']}_{r['end_date']}_{ai_model}"

        if not openai_key:
            st.warning("⚠️ 請在左側填入 OpenAI API 金鑰。")
        elif cache_key in st.session_state.ai_cache:
            # ✅ 有快取直接顯示，不論怎麼點按鈕都不消失
            st.success("✅ AI 分析報告")
            st.markdown(st.session_state.ai_cache[cache_key])
            if st.button("🔄 重新執行 AI 分析", key=f"rerun_{cache_key}"):
                del st.session_state.ai_cache[cache_key]
                st.rerun()
        else:
            st.info(f"點擊「開始 AI 分析」後，系統將使用 **{ai_model}** 進行深度分析（約 30-60 秒）。")
            if st.button("🚀 開始 AI 分析", type="primary", key=f"start_{cache_key}"):
                with st.spinner(f"🤖 {ai_model} 分析中，請稍候..."):
                    ai_data = prepare_ai_data(r)
                    result_text = run_ai(openai_key, ai_data, ai_model)
                if result_text.startswith("⚠️"):
                    st.error(result_text)
                else:
                    # ✅ 存入快取，重新執行頁面後仍會顯示
                    st.session_state.ai_cache[cache_key] = result_text
                    st.rerun()


# ============================================================
# 主程式
# ============================================================

def main():
    st.title("📊 AI 台股財報分析系統")
    st.markdown("<hr style='border: 2px solid #1a237e; margin: 0 0 1rem 0;'>", unsafe_allow_html=True)

    # ── 側邊欄 ──
    with st.sidebar:
        st.markdown("## 📈 AI 財報分析")
        st.markdown("<hr style='border: 2px solid #1a237e;'>", unsafe_allow_html=True)

        stock_id = st.text_input("🏷️ 股票代碼", placeholder="例：2330、2454、2317、2412")
        finmind_token = st.text_input("🔑 FinMind API Token", type="password")
        openai_key = st.text_input("🤖 OpenAI API 金鑰", type="password")
        ai_model = st.selectbox("🧠 AI 模型", options=["gpt-4.1-nano", "gpt-5-mini"], index=0)

        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.text_input("📅 起始日期", value="2022-01-01")
        with col_e:
            end_date = st.text_input("📅 結束日期", value=date.today().strftime("%Y-%m-%d"))

        period_type = st.selectbox("📊 數據區間", options=["年度", "季度"], index=0,
                                   help="三階段財報分析固定使用年度數據")
        analyze_btn = st.button("🔍 分析股票", type="primary", use_container_width=True)

        # 歷史紀錄
        if st.session_state.analysis_history:
            st.markdown("---")
            st.markdown(f"### 📚 歷史紀錄")
            for i, h in enumerate(st.session_state.analysis_history):
                is_current = (i == st.session_state.viewing_idx)
                label = f"{'🟢 ' if is_current else ''}{h['stock_id']} {h.get('company_name','')}"
                hint = f"{h['period_type']} ｜ {h['start_date']}～{h['end_date']} ｜ {h['timestamp']}"
                if st.button(label, key=f"h_{i}", help=hint, use_container_width=True):
                    st.session_state.viewing_idx = i
                    st.session_state.current_result = st.session_state.analysis_history[i]
                    st.rerun()
            if st.button("🗑️ 清除所有紀錄", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.current_result = None
                st.session_state.viewing_idx = None
                st.rerun()

        st.markdown("---")
        st.markdown("**使用說明**\n1. 輸入台股四位數代碼\n2. 填入 FinMind Token（免費）\n3. 填入 OpenAI 金鑰（AI分析需要）\n4. 選擇模型、日期、區間\n5. 點擊「分析股票」")

    # ── 執行分析 ──
    if analyze_btn:
        if not finmind_token:
            st.error("❌ 請填入 FinMind API Token。")
            return
        valid, msg = validate_stock_code(stock_id)
        if not valid:
            st.error(f"❌ {msg}")
            return

        st.info(f"⏳ 獲取 **{stock_id}** 的財務數據（{start_date} ～ {end_date}）中...")

        raw = fetch_all(stock_id, start_date, end_date, finmind_token)
        income_df, balance_df, cashflow_df, price_df, shareholding_df = (
            raw["income"], raw["balance"], raw["cashflow"], raw["price"], raw["shareholding"]
        )
        company_info = raw["company_info"]
        income_df, balance_df, cashflow_df, market_cap, latest_price = compute_derived(
            income_df, balance_df, cashflow_df, price_df, shareholding_df
        )

        display_income = filter_by_period(income_df, period_type)
        display_balance = filter_by_period(balance_df, period_type)
        display_cashflow = filter_by_period(cashflow_df, period_type)
        annual_income = filter_by_period(income_df, "年度")
        annual_balance = filter_by_period(balance_df, "年度")
        annual_cashflow = filter_by_period(cashflow_df, "年度")
        annual = merge_data(annual_income, annual_balance, annual_cashflow)
        display_data = merge_data(display_income, display_balance, display_cashflow)

        if not display_data and not annual:
            st.error("❌ 無法獲取財務數據，請確認股票代碼和 API Token。")
            return

        with st.spinner("🧮 計算三階段財務分析..."):
            fscore = calc_fscore(annual)
            dupont = calc_dupont(annual)
            cashflow = calc_cashflow(annual)

        # 儲存結果至 session_state（DataFrame 轉為可序列化格式）
        result_dict = {
            "stock_id": stock_id,
            "company_name": company_info.get("stock_name", stock_id),
            "start_date": start_date,
            "end_date": end_date,
            "period_type": period_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "company_info": company_info,
            "market_cap": market_cap,
            "latest_price": latest_price,
            "display_income": df_to_storable(display_income),
            "display_balance": df_to_storable(display_balance),
            "display_cashflow": df_to_storable(display_cashflow),
            "display_financial_data": display_data,
            "annual_financial_data": annual,
            "fscore_result": fscore,
            "dupont_result": dupont,
            "cashflow_result": cashflow,
        }

        save_to_history(result_dict)
        st.success(f"✅ {stock_id} 分析完成！（{period_type}，共 {len(display_data)} 期）")
        st.rerun()  # 重跑以進入顯示結果的流程

    # ── 顯示結果（從 session_state 讀取）──
    if st.session_state.current_result is not None:
        r = st.session_state.current_result
        if st.session_state.viewing_idx is not None and len(st.session_state.analysis_history) > 1:
            st.caption(f"📌 {r['stock_id']} {r.get('company_name','')} ｜ {r['period_type']} ｜ {r['start_date']} ～ {r['end_date']} ｜ 分析時間：{r['timestamp']}")
        render_results(r, openai_key, ai_model)

    elif not analyze_btn:
        # 首頁介紹
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**📊 三大財務報表**\n- 損益表趨勢分析\n- 資產負債表結構\n- 現金流量品質")
        with c2:
            st.info("**🎯 三階段專業分析**\n- Piotroski F-Score\n- 杜邦分析（ROE三因子）\n- 現金流品質評估")
        with c3:
            st.info("**🤖 AI 深度分析**\n- 台股市場特性解讀\n- 財務健康綜合診斷\n- 投資風險評估報告")
        st.markdown("### 如何開始？\n在左側輸入股票代碼（例如 **2330** 台積電）、FinMind Token 和 OpenAI 金鑰，選擇日期與區間，點擊「分析股票」。")


if __name__ == "__main__":
    main()
