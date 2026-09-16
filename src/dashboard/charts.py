import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CYAN = "#54CBE3"
AMBER = "#F2BA63"
RED = "#F0828C"
MUTED = "#9BACBF"


def style(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=16, t=30, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#CFDBE8", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
        uirevision="ouranos",
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, title_text="Simulation elapsed · hours"
    )
    fig.update_yaxes(gridcolor="#24303F", zeroline=False)
    return fig


def risk_chart(result, cursor, events):
    s = result.scores.iloc[:cursor]
    x = np.arange(len(s)) / 12
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=s.risk,
            name="Ouranos risk",
            line=dict(color=CYAN, width=2.3),
            fill="tozeroy",
            fillcolor="rgba(84,203,227,.07)",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=result.model.config.warning_threshold,
        line=dict(color=MUTED, dash="dot", width=1),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=s.anomaly, name="Anomaly", line=dict(color="#96A8FC", width=1.4)
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=s.degradation, name="Degradation", line=dict(color=AMBER, width=1.5)
        ),
        row=2,
        col=1,
    )
    for label, hour, color in events:
        if hour is not None and hour <= x[-1]:
            fig.add_vline(
                x=hour, line_width=1, line_dash="dash", line_color=color, row=1, col=1
            )
            fig.add_annotation(
                x=hour,
                y=100,
                text=label,
                showarrow=False,
                textangle=-90,
                yanchor="top",
                xanchor="right",
                font=dict(size=10, color=color),
                row=1,
                col=1,
            )
    fig.update_yaxes(range=[0, 105], title_text="Risk / 100", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], title_text="Score", row=2, col=1)
    fig = style(fig, 410)
    fig.update_xaxes(title_text=None, row=1, col=1)
    return fig


def telemetry_chart(result, cursor, channel, history_hours=48):
    start = max(0, cursor - int(history_hours * 12))
    sl = slice(start, cursor)
    x = np.arange(start, cursor) / 12
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=result.telemetry[channel].iloc[sl],
            name="Observed",
            line=dict(color=CYAN, width=1.5),
        )
    )
    if channel in result.expected:
        expected = result.expected[channel].iloc[sl]
        sigma = result.model.scale[channel]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=expected + 3 * sigma,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=expected - 3 * sigma,
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(150,168,252,.13)",
                name="Healthy ±3 residual σ",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=expected,
                name="Learned expected",
                line=dict(color="#96A8FC", width=1, dash="dot"),
            )
        )
    limit = result.model.config.thresholds.get(channel)
    if limit:
        fig.add_hline(
            y=limit[1],
            line_color=RED,
            line_dash="dash",
            annotation_text="Conventional limit",
        )
    return style(fig, 290)
