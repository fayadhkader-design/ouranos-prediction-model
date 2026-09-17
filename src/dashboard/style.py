CSS = """
<style>
:root {--ink:#263442;--muted:#5b6774;--accent:#3979b9;--line:#dce0e5;}
.stApp {background:#f1f2f4;color:var(--ink);}
.block-container {padding:4rem 1.2rem 2rem!important;max-width:1800px;}
h1,h2,h3 {font-family:"Segoe UI",system-ui,sans-serif;letter-spacing:0;}
h3 {font-size:.95rem!important;font-weight:500!important;}
p,li {text-wrap:pretty;}
[data-testid="stCaptionContainer"] p {color:var(--muted)!important;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid var(--line);min-width:250px;max-width:270px;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.65rem;}
[data-testid="stSidebar"] hr {margin:.8rem 0;}
.side-brand {background:#243442;color:#fff;padding:1rem;margin-bottom:1rem;font-size:1.2rem;font-weight:500;}
.side-brand span {display:block;font-size:.75rem;margin-top:.3rem;color:#dce5ee;}
.masthead {display:flex;justify-content:space-between;align-items:center;gap:1rem;background:#3979b9;color:#fff;padding:.9rem 1rem;margin-bottom:.5rem;}
.wordmark {font-size:.9rem;}.wordmark span {margin:0 .65rem;color:#dae7f5;}
.provenance {font-size:.8rem;color:#fff;}
.summary-strip {display:flex;flex-wrap:wrap;justify-content:space-between;gap:1rem;background:#fff;border:1px solid var(--line);padding:1rem;margin-bottom:.4rem;}
.summary-strip>div {min-width:110px;}.summary-strip span {display:block;color:var(--muted);font-size:.78rem;margin-bottom:.5rem;}
.summary-strip strong {font-size:1.2rem;font-weight:500;font-variant-numeric:tabular-nums;}.summary-strip small {font-size:.8rem;color:var(--muted);}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff;border-radius:2px;}
[data-testid="stVerticalBlockBorderWrapper"] h3 {border-bottom:1px solid #e4e7eb;padding-bottom:.65rem!important;}
.stButton button {min-height:34px;border-radius:2px;border-color:#cbd2da;}
.stButton button[kind="primary"] {background:#3979b9;color:#fff;border-color:#3979b9;}
.stButton button:focus-visible {outline:2px solid #22598e;outline-offset:2px;}
.stTabs [data-baseweb="tab-list"] {gap:1.5rem;border-bottom:1px solid #ccd4dc;}
[role="tab"] p {white-space:nowrap;}
.subsystem-row {display:flex;justify-content:space-between;flex-wrap:wrap;padding:.5rem 0;border-bottom:1px solid var(--line);font-size:.85rem;}
.subsystem-row strong {font-weight:500;font-variant-numeric:tabular-nums;}
.track {width:100%;height:3px;background:#edf0f3;margin-top:.4rem;}.track span {display:block;height:3px;background:#5692cd;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
footer,[data-testid="stAppDeployButton"] {display:none;}
@media(max-width:900px) {[data-testid="stHorizontalBlock"] {flex-wrap:wrap;} [data-testid="stColumn"] {min-width:min(100%,280px)!important;flex:1 1 280px!important;}}
@media(max-width:600px) {.block-container {padding:4rem .7rem 1rem!important;} .summary-strip>div {min-width:95px;} .masthead {font-size:.8rem;} }
@media(prefers-reduced-motion:reduce) {* {transition:none!important;animation:none!important;}}

.st-key-subsystem_panel,.st-key-composition_panel,.st-key-history_panel,.st-key-status_panel,.st-key-telemetry_panel {background:#fff!important;border-radius:2px!important;}
.st-key-subsystem_panel h3,.st-key-composition_panel h3,.st-key-history_panel h3,.st-key-status_panel h3,.st-key-telemetry_panel h3 {border-bottom:1px solid #e4e7eb;padding-bottom:.65rem!important;}
</style>
"""
