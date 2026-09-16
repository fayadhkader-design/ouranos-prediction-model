CSS = """
<style>
:root { --ink:oklch(0.95 0.008 240); --muted:oklch(0.74 0.022 240); --accent:oklch(0.79 0.11 220); --line:oklch(0.30 0.025 245); }
.stApp {background:oklch(0.18 0.023 250);}
.block-container {padding-top:2rem; padding-bottom:3rem; max-width:1540px;}
h1,h2,h3 {font-family:system-ui,sans-serif; letter-spacing:-.02em; text-wrap:balance;}
h1 {font-size:2rem!important;font-weight:650!important;}
h2 {font-size:1.22rem!important;}
h3 {font-size:1.05rem!important;}
p,li {text-wrap:pretty;}
[data-testid="stSidebar"] {background:oklch(0.21 0.025 250);}
[data-testid="stSidebar"] .block-container {padding-top:1.6rem;}
[data-testid="stMetricValue"] {font-family:ui-monospace,SFMono-Regular,monospace;font-size:1.8rem;}
.stButton button {min-height:44px;border-radius:6px;transition:background-color 180ms ease;}
.stButton button[kind="primary"] {color:oklch(.18 .023 250);background:var(--accent);border-color:transparent;}
.stButton button[kind="primary"] p {color:oklch(.18 .023 250)!important;}
[role="tab"] p {white-space:nowrap;text-wrap:nowrap;}
[data-testid="stAlert"] p {color:var(--ink);}
.stButton button:focus-visible {outline:2px solid var(--accent);outline-offset:3px;}
.masthead {display:flex;align-items:baseline;justify-content:space-between;gap:1rem;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:1rem;margin-bottom:1.3rem;}
.wordmark {font-size:1.55rem;letter-spacing:.16em;font-weight:650;}
.provenance {color:var(--accent);font-size:.8rem;border:1px solid var(--line);padding:.35rem .65rem;border-radius:4px;}
.asset {color:var(--muted);font-size:.95rem;margin-bottom:.3rem;}
.score-row {display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin:.6rem 0 1.3rem;}
.score {font:500 4.7rem/1 ui-monospace,SFMono-Regular,monospace;letter-spacing:-.04em;}
.score small {font:400 1.2rem system-ui;color:var(--muted);letter-spacing:0;}
.status {font-size:.88rem;letter-spacing:.06em;}
.quiet {color:var(--muted);font-size:.9rem;max-width:68ch;}
.subsystem-row {display:grid;grid-template-columns:minmax(0,1fr) 2.5rem;gap:.3rem 1rem;padding:.60rem 0;border-bottom:1px solid var(--line);}
.subsystem-row strong {font-family:ui-monospace,monospace;text-align:right;}
.subsystem-row .track {grid-column:1/3;height:3px;background:oklch(.27 .025 250);margin-bottom:.2rem;}
.subsystem-row .track span {display:block;height:3px;background:var(--accent);}
.stTabs [data-baseweb="tab-list"] {gap:1.5rem;}
footer {visibility:hidden;}
[data-testid="stAppDeployButton"] {display:none;}
@media(max-width:1100px) {[data-testid="stHorizontalBlock"] {flex-wrap:wrap;} [data-testid="stColumn"] {min-width:min(100%,260px)!important;flex:1 1 260px!important;} }
@media(prefers-reduced-motion:reduce) {*{transition:none!important;animation:none!important;}}
@media(max-width:700px) {.block-container{padding:1rem;} .score{font-size:3.7rem;} .score-row{gap:1rem;} .masthead{margin-top:1.5rem;} }
</style>
"""
