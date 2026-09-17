CSS = """
<style>
:root {--ink:#e3e7ec;--muted:#a5adb8;--accent:#86b8c8;--line:#30353d;}
.stApp {background:#101216;color:var(--ink);}
.block-container {padding-top:5rem!important;padding-bottom:3rem;max-width:1800px;padding-left:1.5rem;padding-right:1.5rem;}
h1,h2,h3 {font-family:system-ui,sans-serif;letter-spacing:-.015em;text-wrap:balance;}
h1 {font-size:1.65rem!important;} h2 {font-size:1.2rem!important;} h3 {font-size:1rem!important;font-weight:600!important;}
p,li {text-wrap:pretty;} [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:var(--muted)!important;}
[data-testid="stSidebar"] {background:#191b20;border-right:1px solid var(--line);}
[data-testid="stSidebar"] h3 {font-size:.9rem!important;}
[data-testid="stSidebar"] button p {font-size:.86rem;}
[data-testid="stMetricValue"] {font-size:1.6rem;font-variant-numeric:tabular-nums;}
.stButton button {min-height:40px;border-radius:4px;transition:background-color 150ms ease;border-color:#3c434d;}
.stButton button[kind="primary"] {color:#14181c;background:var(--accent);border-color:var(--accent);}
.stButton button[kind="primary"] p {color:#14181c!important;}
.stButton button:focus-visible {outline:2px solid var(--accent);outline-offset:3px;}
[data-testid="stSlider"] p,[role="tab"] p {white-space:nowrap;text-wrap:nowrap;}
.masthead {display:flex;align-items:center;justify-content:space-between;gap:1rem;border-bottom:1px solid var(--line);padding-bottom:1rem;margin-bottom:.2rem;}
.wordmark {font-size:1.2rem;letter-spacing:-.02em;font-weight:650;}
.wordmark span {font-weight:400;color:var(--muted);font-size:.85rem;margin-left:1rem;letter-spacing:0;}
.provenance {color:var(--muted);font-size:.8rem;}
.asset {font-size:1.35rem;font-weight:600;margin-top:1rem;}
.score-row {display:flex;gap:1.5rem;align-items:center;margin:.8rem 0 1rem;}
.score {font:500 2.35rem/1.2 system-ui,sans-serif;font-variant-numeric:tabular-nums;letter-spacing:-.025em;}
.score small {font:400 1rem system-ui;color:var(--muted);letter-spacing:0;}
.status {font-size:.85rem;font-weight:600;}
.quiet {color:var(--muted);font-size:.85rem;max-width:68ch;}
.subsystem-row {display:grid;grid-template-columns:minmax(0,1fr) 2.5rem;gap:.4rem 1rem;padding:.6rem 0;border-bottom:1px solid var(--line);font-size:.9rem;}
.subsystem-row strong {font-weight:500;font-variant-numeric:tabular-nums;text-align:right;}
.subsystem-row .track {grid-column:1/3;height:2px;background:#30353d;}
.subsystem-row .track span {display:block;height:2px;background:#8194a4;}
.stTabs [data-baseweb="tab-list"] {gap:1.5rem;border-bottom:1px solid var(--line);}
[data-testid="stAlert"] {border-radius:4px;}
footer {visibility:hidden;} [data-testid="stAppDeployButton"] {display:none;}
@media(max-width:1100px) {[data-testid="stHorizontalBlock"] {flex-wrap:wrap;} [data-testid="stColumn"] {min-width:min(100%,260px)!important;flex:1 1 260px!important;}}
@media(max-width:700px) {.block-container {padding:1rem;} .masthead {margin-top:1.5rem;} .wordmark span {display:none;} .score-row {gap:1rem;}}
@media(prefers-reduced-motion:reduce) {* {transition:none!important;animation:none!important;}}

.side-brand {font-size:1.2rem;font-weight:600;padding:0 0 1.4rem;border-bottom:1px solid var(--line);margin-bottom:1rem;}
.side-brand span {display:block;font-size:.8rem;font-weight:400;color:var(--muted);margin-top:.3rem;}
.masthead {padding-bottom:.8rem;margin-bottom:.4rem;}
.wordmark {font-size:.88rem;font-weight:500;letter-spacing:0;}
.wordmark span {margin:0 .65rem;color:#9ba3af;}
.summary-strip {display:flex;flex-wrap:wrap;gap:1.5rem;justify-content:space-between;border-bottom:1px solid var(--line);padding:1rem 0 1.25rem;margin-bottom:.5rem;}
.summary-strip>div {min-width:110px;}
.summary-strip span {display:block;font-size:.78rem;color:#a5adb8;margin-bottom:.45rem;}
.summary-strip strong {font-size:1.15rem;font-weight:500;font-variant-numeric:tabular-nums;}
.summary-strip small {font-size:.8rem;color:#a5adb8;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:4px!important;}
[data-testid="stVerticalBlockBorderWrapper"]:has(>div>div>[data-testid="stVerticalBlock"]) {background:#191b20;}
[data-testid="stSidebar"] hr {margin:1rem 0;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.65rem;}
[data-testid="stSidebar"] {min-width:250px;max-width:270px;}
.stButton button {min-height:34px;}
@media(max-width:700px) {.summary-strip {gap:1rem;} .summary-strip>div {min-width:95px;} .wordmark span {display:inline;} .block-container {padding-left:1rem;padding-right:1rem;}}
</style>
"""
