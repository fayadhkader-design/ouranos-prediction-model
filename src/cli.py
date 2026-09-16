import argparse, json
from pathlib import Path
import joblib
from src.pipeline import train_default, analyze
from src.simulation.generator import generate
from src.risk.explain import explain
from src.evaluation import run_metrics


def main():
    p = argparse.ArgumentParser(description="Ouranos synthetic telemetry pipeline")
    p.add_argument("command", choices=["train", "simulate"])
    p.add_argument(
        "--scenario",
        choices=["normal", "reaction_wheel", "battery", "solar"],
        default="reaction_wheel",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    path = Path("models/healthy.joblib")
    if args.command == "train":
        print(json.dumps(train_default(path).training_metadata, indent=2))
        return
    model = joblib.load(path) if path.exists() else train_default(path)
    sim = generate(args.scenario, seed=args.seed)
    r = analyze(sim.telemetry, model)
    Path("data/simulated").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    sim.telemetry.to_csv("data/simulated/demo.csv", index=False)
    sim.truth.to_csv("data/simulated/truth.csv", index=False)
    r.scores.to_csv("data/processed/scores.csv", index=False)
    r.subsystems.to_csv("data/processed/subsystems.csv", index=False)
    audit = r.global_components.copy()
    audit.insert(0, "timestamp", r.telemetry.timestamp)
    for c in r.global_components:
        audit[c + "_change"] = r.global_components[c].diff().fillna(0)
    audit.to_csv("data/processed/score_audit.csv", index=False)
    Path("data/processed/explanation.json").write_text(json.dumps(explain(r), indent=2))
    Path("data/simulated/metadata.json").write_text(json.dumps(sim.metadata, indent=2))
    print(json.dumps(run_metrics(sim, r), indent=2))


if __name__ == "__main__":
    main()
