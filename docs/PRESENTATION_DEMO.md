# Presenting the Ouranos scenario demo

Open the app and choose **Synthetic simulation** (the default workspace).

1. Choose reaction wheel, battery, or solar degradation under **Presentation demo**.
2. Click **RUN GUIDED DEMO**. It starts at 36 simulated hours, with degradation beginning at hour 48. At the default speed, the main story unfolds in about one minute.
3. Explain the changing risk score, leading subsystem, telemetry departures and score audit.
4. When the conventional threshold is reached, the app displays the measured difference between its first warning and that threshold.

For a short meeting, expand **Presentation checkpoints** and use **SHOW HEALTHY OPERATIONS**, **SHOW FIRST OURANOS WARNING**, and **SHOW CONVENTIONAL ALERT**. These jump to actual calculated results for the selected scenario. Scores and lead time are not fabricated slide numbers. The telemetry and fault evolution are synthetic.

Suggested introduction: “This is a synthetic scenario demonstrating the intended Ouranos workflow. The backend is running real detection and scoring code on simulated telemetry. These results are not evidence of accuracy or advance failure prediction on real spacecraft.”

The demo illustrates healthy operations → controlled degradation → abnormal relationships and trends → subsystem risk → explanation → conventional alarm. It does not simulate what a future perfected model is guaranteed to achieve. Real-model experiments and their limitations remain available under **Real ESA telemetry**.

Use **RESET** for manual operation and injection controls. The guided demo uses seed 42 for repeatability. Existing simulation evaluation and raw score-audit downloads remain available. The current suite has 57 passing tests, including guided playback and all three presentation checkpoints.
