import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm


def scatter_with_trend(used_data, x: str, y: str) -> go.Figure:
    xv = used_data[x].to_numpy(dtype=float)
    yv = used_data[y].to_numpy(dtype=float)
    # 在中心化/缩放后的 X 上拟合带截距 OLS，降低大数值变量的病态风险。
    center, scale = xv.mean(), xv.std()
    standardized = (xv - center) / scale
    model = sm.OLS(yv, sm.add_constant(standardized, has_constant="add")).fit()
    grid = np.linspace(xv.min(), xv.max(), 100)
    predicted = model.predict(sm.add_constant((grid - center) / scale, has_constant="add"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers", name="有效样本",
                            marker=dict(color="#157A78", size=7, opacity=0.58),
                            hovertemplate="X = %{x}<br>Y = %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=predicted, mode="lines", name="OLS 线性趋势",
                            line=dict(color="#D4943A", width=3)))
    fig.update_layout(title=f"{x} 与 {y} 的样本关系", xaxis_title=x, yaxis_title=y,
                      template="plotly_white", height=430, margin=dict(l=30, r=25, t=65, b=35),
                      font=dict(family="Microsoft YaHei, sans-serif", color="#192B40"),
                      legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def bootstrap_distribution_chart(bootstrap: dict) -> go.Figure:
    values = [value for value in bootstrap["bootstrap_r"] if value is not None]
    fig = go.Figure(go.Histogram(x=values, nbinsx=40, name="有效 Bootstrap r",
                                marker_color="#157A78", opacity=.78))
    markers = [(0, "0", "#8895A7", "top left"),
               (bootstrap["original_r"], "原始 r", "#C48326", "bottom right")]
    if bootstrap["available"]:
        markers += [(bootstrap["ci_lower"], "95% CI 下界", "#A34E57", "top left"),
                    (bootstrap["ci_upper"], "95% CI 上界", "#A34E57", "top right")]
    for value, label, color, position in markers:
        fig.add_vline(x=value, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position=position)
    fig.update_layout(title="Bootstrap Pearson r 分布（有效重抽样）", xaxis_title="Pearson r",
                      yaxis_title="重抽样次数", template="plotly_white", height=420,
                      margin=dict(t=70, b=45, l=40, r=30), bargap=.05,
                      font=dict(family="Microsoft YaHei, sans-serif", color="#192B40"))
    return fig
