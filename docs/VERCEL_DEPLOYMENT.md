# Vercel deployment

Live dashboard: https://ouranos-prediction-model.vercel.app/

Vercel builds `Dockerfile.vercel` as a container. The application runs the actual Streamlit Python server and uses WebSockets for interactive sessions. `requirements-deploy.txt` pins the runtime to the versions used by the saved demo model.

## Included

The synthetic scenario demo, healthy model, risk engine, interactive charts, explanations, and evaluation reports. The controls generate telemetry and run Python analysis on the server. Demo performance does not establish real-spacecraft predictive accuracy.

## Not included

Large ESA processed datasets and optional PyTorch dependencies are excluded. The ESA workspace explains this when those files are absent; it does not substitute demo telemetry. To enable it, separately provision the mission data using the repository's ESA instructions and add the neural dependencies to the image.

## Deploy

With the Vercel CLI authenticated, run `vercel deploy --prod` from the repository root. The linked project is `ouranos-prediction-model` in team `fayadh1`. GitHub integration is connected to this repository.

`.vercelignore` and `.dockerignore` exclude credentials, local environment files, raw data, tests, and local caches. Never commit `.env.local` or `.vercel`.

## Runtime limits

Vercel containers and WebSockets are beta features. Instances can scale down when idle and connections are subject to platform duration limits. Streamlit session state is in memory: reconnects or instance replacement may reset playback. This is a demo deployment, not durable mission operations infrastructure. First access after idle may have a cold start.

References: https://vercel.com/docs/functions/container-images and https://vercel.com/docs/functions/websockets
