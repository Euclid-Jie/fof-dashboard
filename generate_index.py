import os
import argparse
import glob
import json
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import sqlalchemy
from jinja2 import Environment, FileSystemLoader
from pyecharts.charts import Line, Pie, Grid, HeatMap
from pyecharts import options as opts
from config import SQL_HOST, SQL_PASSWORDS
from utils import generate_trading_date
from Nav_Show.performance_report import PerformanceReportGenerator
from Nav_Show.nav_interval_metric.nav_metric import NavMetric

engine_data = sqlalchemy.create_engine(
    f"mysql+pymysql://dev:{SQL_PASSWORDS}@{SQL_HOST}:3306/UpdatedData?charset=utf8mb4"
)


def read_csvs(data_dir, output_dir) -> tuple:
    # Check for fund.json
    json_path = os.path.join(data_dir, "fund.json")
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    funds_data = config.get("funds", [])
    if not funds_data:
        raise ValueError("No funds found in fund.json")

    funds_list = []
    navs_list = []
    txns_list = []

    for fund in funds_data:
        fund_id = fund["id"]
        fund_name = fund["name"]
        nav_path_hint = fund.get("nav_data_path", "")

        # Try to find the file
        # 1. Direct path
        fpath = os.path.join(data_dir, os.path.basename(nav_path_hint))
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Warning: Could not find NAV file for {fund_name} (hint: {nav_path_hint})"
            )
        # Read NAVs
        df = pd.read_excel(fpath)
        assert "日期" in df.columns, f"Error: '日期' column not found in {fpath}"
        assert (
            "复权净值" in df.columns
        ), f"Error: '复权净值' column not found in {fpath}"

        sub_nav = df[["日期", "复权净值"]].copy()
        sub_nav.columns = ["date", "nav"]
        sub_nav["fund_code"] = fund_id
        sub_nav["date"] = pd.to_datetime(sub_nav["date"])
        begin_date = sub_nav["date"].min()
        end_date = sub_nav["date"].max()
        sub_nav.set_index("date", inplace=True)

        _, weekly_trade_date = generate_trading_date(
            begin_date - np.timedelta64(10, "D"),
            end_date + np.timedelta64(5, "D"),
        )
        nav_series = sub_nav["nav"].reindex(weekly_trade_date)
        nav_series = nav_series[nav_series.index >= begin_date]
        nav_series = nav_series[nav_series.notna()]
        benchmark = fund.get("benchmark", None)
        if benchmark is not None:
            bench_df = pd.read_sql_query(
                f"SELECT date,CLOSE FROM bench_basic_data WHERE code = '{benchmark}'",
                engine_data,
            )
            bench_df["date"] = pd.to_datetime(bench_df["date"])
            bench_df.set_index("date", inplace=True)
            bench_df = bench_df.reindex(nav_series.index)
        # Generate report
        nav = nav_series.values
        date = nav_series.index.values
        report = PerformanceReportGenerator(
            fund_name,
            date,
            nav,
            benchmark=bench_df["CLOSE"].values if benchmark is not None else None,
        )
        report.render(os.path.join(output_dir, fund_id + ".html"))

        funds_list.append(
            {
                "fund_code": fund_id,
                "name": fund_name,
                "link_template": os.path.join(output_dir, fund_id + ".html"),
            }
        )
        # NAVs
        navs_sub = sub_nav.reset_index()[["date", "fund_code", "nav"]]
        navs_list.append(navs_sub)
        # Transaction
        txns_list.append(
            {
                "date": pd.to_datetime(fund["purchaseDate"]),
                "fund_code": fund_id,
                "shares": float(fund["shares"]),
                "price": float(fund["purchaseNav"]),
                "type": "buy",
            }
        )

    funds_df = pd.DataFrame(funds_list)
    navs_df = pd.concat(navs_list, ignore_index=True)
    txns_df = pd.DataFrame(txns_list)

    return funds_df, navs_df, txns_df


def prepare_navs(navs_df, fund_codes):
    # pivot navs to wide with date index and columns fund_codes
    pivot = navs_df.pivot(index="date", columns="fund_code", values="nav").sort_index()
    # forward fill to carry last known nav
    pivot = pivot.ffill()
    # ensure columns include fund_codes
    for c in fund_codes:
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot = pivot[fund_codes]
    return pivot


def compute_holdings(txns_df, dates, fund_codes):
    tx = txns_df.copy()
    tx["signed_shares"] = tx.apply(
        lambda r: (
            -abs(float(r["shares"]))
            if str(r.get("type", "")).lower() == "sell"
            else float(r.get("shares", 0.0))
        ),
        axis=1,
    )
    # aggregate by date and fund
    agg = tx.groupby(["date", "fund_code"], as_index=False).agg(
        {"signed_shares": "sum"}
    )
    # build full grid
    idx = pd.MultiIndex.from_product([dates, fund_codes], names=["date", "fund_code"])
    grid = pd.DataFrame(index=idx).reset_index()
    merged = grid.merge(agg, on=["date", "fund_code"], how="left").fillna(0)
    deltas = (
        merged.pivot(index="date", columns="fund_code", values="signed_shares")
        .fillna(0)
        .sort_index()
    )
    holdings = deltas.cumsum()
    # ensure fund_codes order
    holdings = holdings.reindex(columns=fund_codes).fillna(0)
    return holdings


def build_portfolio_grid_chart(dates, series, name="组合净值"):
    # Calculate drawdown
    pv = pd.Series(series, index=dates)
    cummax = pv.cummax()
    drawdown = (pv - cummax) / cummax

    # Handle dates: if numpy array of datetime64, convert to string list
    date_list = dates.astype("M8[D]").astype(str).tolist()

    # Top chart: NAV
    line = (
        Line()
        .add_xaxis(date_list)
        .add_yaxis(
            name,
            [float(round(v, 4)) for v in series],
            is_smooth=False,
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#d9534f"),
        )
    )
    line.set_global_opts(
        title_opts=opts.TitleOpts(
            pos_left="center",
            title_textstyle_opts=opts.TextStyleOpts(
                font_size=20, font_weight="bold", color="#333"
            ),
        ),
        legend_opts=opts.LegendOpts(pos_top="8%", pos_left="68%", is_show=False),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True, link=[{"xAxisIndex": "all"}]
        ),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        yaxis_opts=opts.AxisOpts(
            name="收益率 (%)", axislabel_opts=opts.LabelOpts(formatter="{value} %")
        ),
        datazoom_opts=[
            opts.DataZoomOpts(
                type_="slider",
                xaxis_index=[0, 1],
                range_start=0,
                range_end=100,
            )
        ],
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            pos_left="right",
            feature=opts.ToolBoxFeatureOpts(
                # 启用保存为图片功能
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    title="保存为图片",
                    pixel_ratio=4,  # 提高分辨率
                    background_color="white",  # 设置背景色
                    name="performance_report_chart",  # 设置文件名
                ),
                # 启用还原按钮（重置视图）
                restore=True,
                # 禁用所有其他默认功能
                magic_type=False,  # 关闭动态类型切换（如折线/柱状切换）
                brush=False,  # 关闭区域选择
                data_view=False,  # 关闭数据视图
            ),
        ),
    )

    # Bottom chart: Drawdown
    dd_chart = (
        Line()
        .add_xaxis(date_list)
        .add_yaxis(
            "回撤",
            [float(round(v * 100, 2)) for v in drawdown.values],
            is_smooth=False,
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=1, color="#d9534f"),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5, color="#d9534f"),
        )
    )
    dd_chart.set_global_opts(
        yaxis_opts=opts.AxisOpts(
            name="回撤 (%)", axislabel_opts=opts.LabelOpts(formatter="{value} %")
        ),
        legend_opts=opts.LegendOpts(is_show=False, pos_left="73%", pos_top="70%"),
        xaxis_opts=opts.AxisOpts(is_show=False),
    )
    grid = Grid(init_opts=opts.InitOpts(width="100%", height="700px"))
    grid.add(line, grid_opts=opts.GridOpts(pos_top="12%", pos_bottom="33%"))
    grid.add(dd_chart, grid_opts=opts.GridOpts(pos_top="80%", pos_bottom="10%"))

    return grid


def build_area_stack_chart(dates, df_market_values):
    xaxis = [d.strftime("%Y-%m-%d") for d in dates]
    chart = Line(init_opts=opts.InitOpts(width="100%", height="420px"))
    chart.add_xaxis(xaxis)
    for col in df_market_values.columns:
        vals = df_market_values[col].fillna(0).astype(float).tolist()
        chart.add_yaxis(
            col,
            vals,
            is_smooth=False,
            is_symbol_show=False,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.6),
            stack="总量",
            label_opts=opts.LabelOpts(is_show=False),
        )
    chart.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[
            opts.DataZoomOpts(
                type_="slider", xaxis_index=[0, 1], range_start=0, range_end=100
            ),
        ],
        yaxis_opts=opts.AxisOpts(name="市值"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            pos_left="right",
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    title="保存为图片",
                    pixel_ratio=4,
                    background_color="white",
                    name="area_stack_chart",
                ),
                restore=opts.ToolBoxFeatureRestoreOpts(),
                magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False),
            ),
        ),
    )
    return chart


def build_pie_chart(series):
    data = [list(z) for z in zip(series.index.tolist(), series.values.tolist())]
    pie = Pie(init_opts=opts.InitOpts(width="100%", height="360px"))
    pie.add(
        "",
        data,
        radius=["30%", "60%"],
        center=["50%", "50%"],
        rosetype="radius",
    )
    pie.set_series_opts(
        label_opts=opts.LabelOpts(
            formatter="{b}: {d}%",
            position="outside",
            font_size=12,
        )
    )
    pie.set_global_opts(
        legend_opts=opts.LegendOpts(
            orient="vertical",
            pos_top="15%",
            pos_left="5%",
            is_show=False,
        ),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            pos_left="right",
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    title="保存为图片",
                    pixel_ratio=4,
                    background_color="white",
                    name="area_stack_chart",
                ),
                restore=opts.ToolBoxFeatureRestoreOpts(),
                magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False),
            ),
        ),
    )
    return pie


def get_chart_options(chart):
    """
    从 pyecharts Chart 获取 options 的 JSON 字典
    """
    try:
        return json.loads(chart.dump_options_with_quotes())
    except Exception:
        return json.loads(chart.dump_options())


def generate_table_html(df_holdings, fund_url_template):
    df = df_holdings.copy()

    def link(name, code):
        url = fund_url_template.format(code=code)
        return f'<a href="{url}" target="_blank">{name}</a>'

    df["产品名称"] = df.apply(lambda r: link(r["name"], r["fund_code"]), axis=1)
    df_display = df[
        [
            "产品名称",
            "fund_code",
            "purchase_date",
            "total_shares",
            "avg_cost",
            "total_cost",
            "nav",
            "current_value",
            "pnl",
        ]
    ].rename(
        columns={
            "purchase_date": "购买日期",
            "fund_code": "代码",
            "total_shares": "份额",
            "avg_cost": "成本价",
            "total_cost": "总成本",
            "nav": "当前净值",
            "current_value": "当前市值",
            "pnl": "持有收益",
        }
    )
    # format
    for c in ["份额", "成本价", "总成本", "当前净值", "当前市值", "持有收益"]:
        df_display[c] = df_display[c].apply(
            lambda x: (
                f"{x:,.2f}"
                if (pd.notnull(x) and not isinstance(x, str))
                else (x if pd.notnull(x) else "")
            )
        )
    html = df_display.to_html(index=False, escape=False)
    return html


def build_heatmap_chart(corr_df):
    # 构造热力图数据
    x_labels = corr_df.columns.tolist()
    # 为了让对角线从左上到右下，这里将 y 轴顺序反转
    y_labels = corr_df.index.tolist()[::-1]
    min_value = corr_df.min().min()
    max_value = corr_df.max().max()
    data = []
    for i, yi in enumerate(y_labels):
        for j, xj in enumerate(x_labels):
            val = (
                float(corr_df.loc[yi, xj]) if not pd.isna(corr_df.loc[yi, xj]) else None
            )
            data.append([j, i, round(val, 4) if val is not None else None])

    # 使用 Grid 控制整体布局，压缩图的主体区域，给轴标签留空间
    heat = HeatMap(init_opts=opts.InitOpts(width="100%", height="420px"))
    heat.add_xaxis(x_labels)
    heat.add_yaxis(
        "",
        y_labels,
        data,
        label_opts=opts.LabelOpts(is_show=False),
    )
    heat.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        legend_opts=opts.LegendOpts(is_show=False),
        visualmap_opts=opts.VisualMapOpts(
            min_=min_value,
            max_=max_value,
            is_show=False,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026",
            ],
        ),
        # 减小热力图主体，给产品名(轴标签)更多空间
        xaxis_opts=opts.AxisOpts(
            type_="category",
            axislabel_opts=opts.LabelOpts(rotate=45, font_size=10, margin=10),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="category",
            axislabel_opts=opts.LabelOpts(rotate=0, font_size=10, margin=10),
        ),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            pos_left="right",
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    title="保存为图片",
                    pixel_ratio=4,
                    background_color="white",
                    name="correlation_heatmap",
                ),
                restore=opts.ToolBoxFeatureRestoreOpts(),
                magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False),
            ),
        ),
    )

    # 用 Grid 调整 plot 区域，预留上、下、左空间给标签
    grid = Grid(init_opts=opts.InitOpts(width="100%", height="420px"))
    grid.add(
        heat,
        grid_opts=opts.GridOpts(
            # 这些百分比可以根据需要微调
            pos_left="30%",  # 给左侧 y 轴产品名留更大空间
            pos_right="10%",
            pos_top="10%",  # 给上方 x 轴旋转后的标签留空间
            pos_bottom="40%",  # 给下侧（如果有标签/工具条）留空间
        ),
        is_control_axis_index=True,
    )
    return grid


def build_interval_returns_table_html(
    portfolio_nav: NDArray[np.float64],
    portfolio_date: NDArray[np.datetime64],
    navs_pivot: pd.DataFrame,
    funds_df: pd.DataFrame,
    ref_date: pd.Timestamp,
    fund_url_template: str,
):
    _base_interval = NavMetric.generate_intervals(
        last_day=np.datetime64("2025-12-05"), last_week_day=np.datetime64("2025-11-28")
    )
    # 组合行
    porf_metric = NavMetric(
        "portfolio",
        portfolio_nav,
        portfolio_date,
        freq="W",
    )
    porf_metric = porf_metric.calculate_interval_return(_base_interval)
    interval_return = {}
    for interval in porf_metric:
        interval_return[interval.name] = interval.interval_return
    rows = []
    rows.append(
        {
            "名称": "组合",
            "代码": "",
            "近一周收益": interval_return.get("recent_week", np.nan),
            "近一个月收益": interval_return.get("recent_month", np.nan),
            "年初至今回报": interval_return.get("ytd", np.nan),
            "近一年收益": interval_return.get("recent_year", np.nan),
        }
    )
    # 子基金行
    for _, r in funds_df.iterrows():
        code = r["fund_code"]
        name = r["name"]
        s = navs_pivot[code]
        metric = NavMetric(
            name,
            s.values,
            s.index.values,
            freq="W",
        )
        metric = metric.calculate_interval_return(_base_interval)
        ret = {}
        for interval in metric:
            ret[interval.name] = interval.interval_return

        url = fund_url_template.format(code=code)
        rows.append(
            {
                "名称": f'<a href="{url}" target="_blank">{name}</a>',
                "代码": code,
                "近一周收益": ret.get("recent_week", np.nan),
                "近一个月收益": ret.get("recent_month", np.nan),
                "年初至今回报": ret.get("ytd", np.nan),
                "近一年收益": ret.get("recent_year", np.nan),
            }
        )

    df = pd.DataFrame(rows)
    # 百分比格式化
    for c in ["近一周收益", "近一个月收益", "年初至今回报", "近一年收益"]:
        df[c] = df[c].apply(lambda x: (f"{x*100:.2f}%" if pd.notnull(x) else ""))
    html = df[
        ["名称", "近一周收益", "近一个月收益", "年初至今回报", "近一年收益"]
    ].to_html(index=False, escape=False)
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="docs")
    parser.add_argument("--base-value", type=float, default=1.0)
    parser.add_argument("--fund-url-template", default="./{code}.html")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    funds_df, navs_df, txns_df = read_csvs(args.data_dir, args.out_dir)
    fund_codes = funds_df["fund_code"].tolist()

    navs_pivot = prepare_navs(navs_df, fund_codes)
    dates = navs_pivot.index

    # New Portfolio Calculation Logic
    # 1. Total Cost
    # Assuming txns_df contains all buy transactions
    # We need to handle multiple buys if any, but based on current logic it's one buy per fund.
    # Let's aggregate cost per fund just in case.

    # We need a way to map fund_code to its purchase price and shares.
    # Since we might have multiple transactions, let's process txns_df.
    # However, the user requirement is specific: "sum(purchaseNav*shares)" is total cost.
    # And "Daily Profit = (NAV - PurchaseNAV) * Shares".

    # Let's build a DataFrame of holdings parameters indexed by fund_code.
    # We assume one buy transaction per fund for simplicity as per the json structure.
    # If there are multiple, we might need weighted average price, but let's stick to the user's simple formula.

    # Filter for 'buy' transactions
    buys = txns_df[txns_df["type"] == "buy"].copy()
    total_cost = (buys["shares"] * buys["price"]).sum()

    # 2. Calculate Daily Profit
    # Profit_t = Sum_i [ (NAV_i_t - PurchaseNAV_i) * Shares_i ]
    # If NAV_i_t is NaN (before inception/purchase), profit is 0 (or we treat it as not started).
    # The user said: "if the fund is not yet purchased, profit is 0".
    # So we need to ensure we only calculate profit for dates >= purchaseDate.

    # Let's create a profit matrix
    profit_matrix = pd.DataFrame(0.0, index=dates, columns=fund_codes)

    for _, row in buys.iterrows():
        code = row["fund_code"]
        shares = row["shares"]
        purchase_price = row["price"]
        purchase_date = row["date"]

        if code not in navs_pivot.columns:
            continue

        # Get NAV series for this fund
        nav_series = navs_pivot[code]

        # Calculate profit: (NAV - PurchasePrice) * Shares
        # Only for dates >= purchase_date
        # And also ensure NAV is not NaN (though prepare_navs ffills, it might have NaNs at start)

        # Create a mask for valid dates
        mask = (nav_series.index >= purchase_date) & (nav_series.notna())

        # Calculate profit for valid days
        # Note: nav_series[mask] - purchase_price gives per share profit
        fund_profit = (nav_series[mask] - purchase_price) * shares

        # Assign to matrix
        profit_matrix.loc[mask, code] += fund_profit

    # Total daily profit
    total_daily_profit = profit_matrix.sum(axis=1)

    # 3. Portfolio NAV
    # NAV = 1 + (Total Profit / Total Cost)
    portfolio_nav = 1.0 + (total_daily_profit / total_cost)

    # For compatibility with existing code that uses 'market' (market value matrix) for area chart
    # We should also calculate market value: MarketValue = NAV * Shares
    # But wait, the area chart expects "market values".
    # If we use the new logic, "Market Value" of a fund is technically (PurchaseCost + Profit).
    # Or simply NAV * Shares.
    # Let's stick to NAV * Shares for the area chart as it represents the actual value of holdings.
    # We need to compute holdings (shares) over time.
    holdings = compute_holdings(txns_df, dates, fund_codes)
    market = holdings * navs_pivot

    # Note: portfolio_nav calculated above is "Net Asset Value" of the FOF (normalized to 1).
    # The 'portfolio_value' variable in original code was sum of market values.
    # Let's see if they align.
    # Old: portfolio_value = market.sum(axis=1).
    # New: portfolio_value_implied = Total_Cost + Total_Profit.
    # portfolio_nav = portfolio_value_implied / Total_Cost.
    # This aligns perfectly if we assume cash is held at cost.

    # So we can use our new portfolio_nav directly.
    # And we don't need to normalize it by base_value again if we want it to start at 1.0.
    # But if args.base_value is not 1.0, we should scale it.
    portfolio_nav = portfolio_nav * args.base_value

    # Define portfolio_value for template rendering (Total Assets)
    # Portfolio Value = Total Cost + Total Daily Profit
    portfolio_value = total_cost + total_daily_profit

    # Use NavMetric for metrics and chart data consistency
    pm = NavMetric(
        "Portfolio",
        portfolio_nav.values,
        portfolio_nav.index.values,
        freq="D",
    )
    stats = pm.base_metric_dict

    total_return = stats["区间收益率"]
    ann_return = stats["年化收益率"]
    ann_vol = stats["年化波动率"]
    max_dd = stats["最大回撤"]

    # Create mapping from code to name
    code_to_name = dict(zip(funds_df["fund_code"], funds_df["name"]))

    # charts
    # Use pm.date and pm.nav which might be truncated to 2020-01-01
    portfolio_chart = build_portfolio_grid_chart(pm.date, pm.nav, name="组合净值")

    # Rename columns for area chart
    market_named = market.rename(columns=code_to_name)
    area_chart = build_area_stack_chart(dates, market_named)

    last_date = dates[-1]
    current_holdings = holdings.loc[last_date]
    current_navs = navs_pivot.loc[last_date]
    current_vals = (current_holdings * current_navs).fillna(0.0)

    # Rename index for pie chart
    current_vals_named = current_vals.rename(index=code_to_name)
    pie_chart = build_pie_chart(current_vals_named[current_vals_named > 0])

    # 相关性热力图：使用子基金的日收益率相关系数
    daily_returns = navs_pivot.pct_change().dropna(how="all")
    corr = daily_returns.corr()
    # 将列和索引替换成基金名称
    name_map = dict(zip(funds_df["fund_code"], funds_df["name"]))
    corr_named = corr.rename(index=name_map, columns=name_map)
    heatmap_chart = build_heatmap_chart(corr_named)

    # 区间收益表 HTML
    interval_table_html = build_interval_returns_table_html(
        portfolio_nav.values,
        dates,
        navs_pivot,
        funds_df,
        last_date,
        args.fund_url_template,
    )

    # Build chart options
    charts_options = {
        "portfolio": get_chart_options(portfolio_chart),
        "area": get_chart_options(area_chart),
        "pie": get_chart_options(pie_chart),
        "heatmap": get_chart_options(heatmap_chart),
    }

    # table summarization
    tx = txns_df.copy()
    tx["signed_shares"] = tx.apply(
        lambda r: (
            -abs(float(r["shares"]))
            if str(r.get("type", "")).lower() == "sell"
            else float(r.get("shares", 0.0))
        ),
        axis=1,
    )
    holdings_summary = []
    for _, row in funds_df.iterrows():
        code = row["fund_code"]
        name = row["name"]
        tx_f = tx[tx["fund_code"] == code]
        if tx_f.empty:
            total_shares = 0.0
            avg_cost = 0.0
            total_cost = 0.0
        else:
            buys = tx_f[tx_f["signed_shares"] > 0]
            total_shares = tx_f["signed_shares"].sum()
            if not buys.empty:
                total_buys = buys["signed_shares"].sum()
                total_cost = (buys["signed_shares"] * buys["price"]).sum()
                avg_cost = total_cost / total_buys if total_buys != 0 else 0.0
            else:
                avg_cost = 0.0
                total_cost = 0.0
        nav_last = (
            float(current_navs.get(code, np.nan))
            if pd.notnull(current_navs.get(code, np.nan))
            else 0.0
        )
        current_value = float(current_vals.get(code, 0.0))
        pnl = current_value - (avg_cost * total_shares)
        holdings_summary.append(
            {
                "fund_code": code,
                "name": name,
                "purchase_date": (
                    tx_f[tx_f["signed_shares"] > 0]["date"].min().strftime("%Y-%m-%d")
                    if not tx_f[tx_f["signed_shares"] > 0].empty
                    else ""
                ),
                "total_shares": float(total_shares),
                "avg_cost": float(avg_cost),
                "total_cost": float(avg_cost * total_shares),
                "nav": nav_last,
                "current_value": current_value,
                "pnl": pnl,
            }
        )
    df_hold = pd.DataFrame(holdings_summary)

    table_html = generate_table_html(df_hold, args.fund_url_template)

    # render template
    env = Environment(loader=FileSystemLoader(searchpath="templates"))
    tpl = env.get_template("index_template.html")
    html = tpl.render(
        latest_date=last_date.strftime("%Y-%m-%d"),
        total_start=float(portfolio_value.iloc[0]),
        total_end=float(portfolio_value.iloc[-1]),
        total_return_pct=total_return * 100,
        annualized_return_pct=ann_return * 100,
        annual_vol_pct=ann_vol * 100,
        max_drawdown_pct=max_dd * 100,
        portfolio_chart_id=portfolio_chart.chart_id,
        area_chart_id=area_chart.chart_id,
        pie_chart_id=pie_chart.chart_id,
        heatmap_chart_id=heatmap_chart.chart_id,
        charts_options=json.dumps(charts_options),
        table_html=table_html,
        interval_table_html=interval_table_html,
        fund_url_template=args.fund_url_template,
    )

    out_path = os.path.join(args.out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("生成完成：", out_path)


if __name__ == "__main__":
    main()
