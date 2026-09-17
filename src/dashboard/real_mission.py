"""Real ESA replay and actual CSV neural inference, separate from synthetic risk."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from src.neural.predict_contextual import ContextualDetector
from src.neural.lifecycle import apply_recovery
from src.neural.supervised import CHANNELS
from src.dashboard.charts import CYAN, AMBER, MUTED

ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource
def real_resources():
    data=ROOT/'data/processed/esa_supervised'
    return {'raw':np.load(data/'features.npy',mmap_mode='r'),
            'index':pd.to_datetime(np.load(data/'timestamps.npy',mmap_mode='r'),utc=True),
            'detector':ContextualDetector(ROOT/'models/esa_contextual'),
            'policy':json.loads((ROOT/'models/esa_contextual/lifecycle.json').read_text())}



@st.cache_data(max_entries=8)
def real_window(onset, end, hours):
    resources=real_resources(); index=resources['index']
    onset=pd.Timestamp(onset)
    left=index.searchsorted(onset-pd.Timedelta(hours=hours/4))
    right=min(len(index),index.searchsorted(onset+pd.Timedelta(hours=hours*3/4)))
    context=max(0,left-1440)
    scored=resources['detector'].score_bins(index[context:right],resources['raw'][context:right])
    scored=apply_recovery(scored,resources['policy']).iloc[left-context:].reset_index(drop=True)
    telemetry=pd.DataFrame(np.asarray(resources['raw'][left:right,::4]),columns=CHANNELS)
    telemetry.insert(0,'timestamp',index[left:right])
    return telemetry,scored


def plot_replay(telemetry, scored, channel, annotation_start, annotation_end, threshold):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.12,row_heights=[.52,.48])
    # Keep full-resolution alerts and extrema around the selected event; at most 5760 minutes.
    fig.add_trace(go.Scatter(x=telemetry.timestamp,y=telemetry[channel],name=channel,line=dict(color=CYAN,width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=scored.timestamp,y=scored.model_score,name='Neural score',line=dict(color=CYAN,width=1.3),connectgaps=False),row=2,col=1)
    active=scored.active_alert
    fig.add_trace(go.Scatter(x=scored.loc[active,'timestamp'],y=scored.loc[active,'model_score'],name='Active alert',mode='markers',marker=dict(color=AMBER,size=5)),row=2,col=1)
    fig.add_hline(y=threshold,line_dash='dash',line_color=AMBER,row=2,col=1)
    # Ground truth is a retrospective overlay and never enters the detector.
    start=pd.Timestamp(annotation_start)
    if len(scored) and start <= scored.timestamp.iloc[-1]:
        for row in [1,2]:
            fig.add_vline(x=start.timestamp()*1000,line_color=MUTED,line_dash='dot',row=row,col=1)
    fig.update_yaxes(title_text='Recorded value',row=1,col=1)
    fig.update_yaxes(title_text='Model score',row=2,col=1)
    fig.update_layout(height=530,margin=dict(l=20,r=20,t=15,b=15),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#dce7ef',family='system-ui'),legend=dict(orientation='h',y=1.08),hovermode='x unified')
    if len(scored)==1:
        fig.update_xaxes(range=[scored.timestamp.iloc[0]-pd.Timedelta(minutes=1),scored.timestamp.iloc[0]+pd.Timedelta(minutes=1)])
    fig.update_xaxes(gridcolor='#263541');fig.update_yaxes(gridcolor='#263541')
    return fig


def render():
    st.markdown('<div class="masthead"><span class="wordmark">OURANOS</span><span class="provenance">REAL MISSION DATA</span></div>',unsafe_allow_html=True)
    st.title('ESA Mission 1 · Anomaly workbench')
    st.caption('Six anonymous telemetry channels · Neural detection with recent context · Experimental')
    needed=[ROOT/'models/esa_contextual/context_nn.joblib',ROOT/'models/esa_contextual/lifecycle.json',ROOT/'data/processed/esa_supervised/features.npy']
    if any(not p.exists() for p in needed):
        st.info('Real-model artifacts are not installed. Follow docs/ESA_CONTEXTUAL.md to prepare data and train the model. The synthetic demo remains available in the sidebar.')
        return
    resources=real_resources(); policy=resources['policy']
    metrics=json.loads((ROOT/'reports/esa_contextual/lifecycle_metrics.json').read_text())
    replay,validation,upload=st.tabs(['Mission replay','Measured performance','Score a CSV'])
    with st.sidebar:
        st.divider();st.caption('REAL DATA · RECORDED REPLAY')
        st.caption('Scores identify unusual telemetry. They are not failure probabilities or the synthetic Ouranos Risk Score.')
    with replay:
        events=pd.read_csv(ROOT/'reports/esa_contextual/lifecycle_events.csv')
        options=events.event_id.tolist()
        selected=st.selectbox('Annotated event',options,index=options.index('id_184'),format_func=lambda k:f"{k} · {'detected' if bool(events.loc[events.event_id.eq(k),'detected'].iloc[0]) else 'missed'}")
        event=events.loc[events.event_id.eq(selected)].iloc[0]
        c1,c2=st.columns([2,1])
        channel=c1.selectbox('Recorded channel',CHANNELS,index=0)
        hours=c2.selectbox('Replay window',[6,24,72,96],index=0,format_func=lambda n:f'{n} hours')
        telemetry,scored=real_window(event.onset,event.end,hours)
        key=f'{selected}_{hours}'
        if st.session_state.get('real_replay_key')!=key:
            st.session_state.real_replay_key=key;st.session_state.real_cursor=len(scored);st.session_state.real_play=False
        a,b,c=st.columns([1,1,2])
        if a.button('Pause replay' if st.session_state.real_play else 'Start replay',type='primary',width='stretch'):
            if st.session_state.real_cursor>=len(scored):st.session_state.real_cursor=1
            st.session_state.real_play=not st.session_state.real_play
        if b.button('Reset replay',width='stretch'):
            st.session_state.real_cursor=1;st.session_state.real_play=False
        speed=c.selectbox('Replay speed',[1,5,15,60],index=2,format_func=lambda n:f'{n} recorded minutes / second')
        if not st.session_state.real_play:
            st.session_state.real_cursor=st.slider('Recorded minute',1,len(scored),int(st.session_state.real_cursor),key=f'position_{key}_{st.session_state.real_cursor}')

        @st.fragment(run_every=1.0 if st.session_state.real_play else None)
        def replay_body():
            if st.session_state.real_play:
                st.session_state.real_cursor=min(len(scored),st.session_state.real_cursor+speed)
                if st.session_state.real_cursor==len(scored):
                    st.session_state.real_play=False;st.rerun()
            cursor=st.session_state.real_cursor
            visible=scored.iloc[:cursor];row=visible.iloc[-1]
            a,b,c=st.columns(3)
            a.metric('Current assessment',row.status)
            b.metric('Neural score',f'{row.model_score:.2f}' if pd.notna(row.model_score) else 'Unavailable')
            c.metric('Alert entry threshold',f'{policy["entry_threshold"]:.2f}')
            st.caption(f'Recorded UTC: {row.timestamp.strftime("%Y-%m-%d %H:%M UTC")} · Raw model score; not a probability. Dashed line: alert entry. Dotted vertical line: retrospective annotation onset.')
            st.plotly_chart(plot_replay(telemetry.iloc[:cursor],visible,channel,event.onset,event.end,policy['entry_threshold']),width='stretch',key='real_replay_chart')
            if row.status=='ALERT ACTIVE':
                st.warning('Unusual telemetry detected. Check the channel evidence; no physical failure diagnosis is available.')
            elif row.status=='UNSCORED':
                st.info('Insufficient recent observations. Unscored does not mean healthy.')
            with st.expander('Why is this alert active?'):
                st.write(f"Entry requires a score above {policy['entry_threshold']:.2f}. A warning closes when the score falls to {policy['exit_threshold']:.2f} or below for {policy['quiet_minutes']} minute(s), a gap occurs, or {policy['max_minutes']} minutes pass without another high-score trigger.")
                st.write(f"Largest recent deviation: **{row.largest_recent_deviation_channel or 'Unavailable'}**. This is descriptive evidence, not causal attribution or a subsystem diagnosis.")
                st.caption('The network compares minute extrema, variability and changes against past-hour and past-day behavior. Event annotations are for review only; they are not model inputs.')
                alerts=visible.loc[visible.active_alert,['timestamp','model_score','largest_recent_deviation_channel']]
                st.dataframe(alerts.tail(25),hide_index=True,width='stretch')
        replay_body()
        st.download_button('Download this replay · CSV',scored.to_csv(index=False),f'ouranos_real_{selected}.csv','text/csv')
    with validation:
        st.subheader('Real-data results, with their limits')
        a,b,c=st.columns(3)
        a.metric('Events detected',f"{metrics['detected_events']} / {metrics['total_events']}")
        b.metric('False episodes / 1,000 nominal h',f"{metrics['false_alerts_per_1000_nominal_hours']:.2f}")
        c.metric('Anomalous minutes flagged',f"{metrics['point']['recall']:.1%}")
        st.warning('Mission 1 results are retrospective development evidence. Most anomalous minutes remain unflagged. These numbers do not demonstrate advance failure prediction.')
        st.write('Warnings now remain active while lower-level evidence persists. Fewer episodes can also mean longer warnings; both time in alarm and episode counts matter.')
        st.write(f"Minute precision: **{metrics['point']['precision']:.1%}** · Nominal time in alarm: **{metrics['point']['false_positive_rate']:.3%}**")
        st.subheader('Separate mission evaluation')
        transfer=ROOT/'reports/esa_transfer/metrics.json'
        if transfer.exists():
            report=json.loads(transfer.read_text())
            st.caption(report['scope'])
            table=pd.DataFrame([{'Variant':r['variant'].replace('_',' '), 'Events':f"{r['events_detected']} / {r['events_total']}", 'False episodes / 1,000 h':round(r['false_episodes_per_1000_nominal_h'],2), 'Anomalous minutes flagged':f"{r['minute_recall']:.2%}", 'Nominal minutes flagged':f"{r['nominal_alarm_fraction']:.2%}"} for r in report['summary']])
            st.dataframe(table,hide_index=True,width='stretch')
            st.info(report['conclusion'])
            st.download_button('Download separate-mission results',transfer.read_text(),'ouranos_mission2_evaluation.json','application/json')
        else:
            st.info('The separate ESA Mission 2 evaluation has not produced a result yet. No independent-validation claim is made.')
        st.download_button('Download Mission 1 evaluation',(ROOT/'reports/esa_contextual/lifecycle_metrics.json').read_text(),'ouranos_real_metrics.json','application/json')
    with upload:
        st.subheader('Run the trained network on telemetry')
        st.write('CSV columns: timestamp and channel_41 through channel_46, using the Mission 1 channel meanings and original numeric representation. Include at least one day of history and preserve subminute readings.')
        st.caption('Arbitrary spacecraft channels cannot be substituted into this model. Uploaded data provenance remains unverified.')
        file=st.file_uploader('Mission 1 telemetry CSV',type=['csv'],key='real_model_csv',max_upload_size=50)
        fingerprint=hashlib.sha256(file.getvalue()).hexdigest() if file is not None else None
        if file is not None:
            if file.size>50*1024**2:
                st.error('Upload a CSV below 50 MB; use the command-line scorer for larger files.')
            elif st.button('Run neural inference',type='primary'):
                try:
                    frame=pd.read_csv(file)
                    result=apply_recovery(resources['detector'].score_frame(frame),policy)
                    st.session_state.real_upload_result=result
                    st.session_state.real_upload_fingerprint=fingerprint
                except (ValueError,KeyError,TypeError) as exc:
                    st.session_state.pop('real_upload_result',None)
                    st.error(f'Could not score this CSV: {exc}')
        result=st.session_state.get('real_upload_result')
        if result is not None and file is not None and st.session_state.get('real_upload_fingerprint')==fingerprint:
            st.success(f'Scored {result.model_score.notna().sum():,} of {len(result):,} minute bins. Source: user supplied, unverified.')
            st.dataframe(result.tail(100),hide_index=True,width='stretch')
            st.download_button('Download predictions',result.to_csv(index=False),'ouranos_predictions.csv','text/csv')
    st.caption('Experimental anomaly monitoring · No flight qualification, calibrated failure probability, physical subsystem diagnosis or RUL estimate.')
