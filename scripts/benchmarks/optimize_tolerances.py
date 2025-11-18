#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import optuna

from loudspeaker.data import build_msd_dataset
from loudspeaker.msd_sim import MSDConfig
from loudspeaker.neuralode import LinearMSDModel, build_loss_fn, solve_with_model
from loudspeaker.testsignals import ControlSignal, build_control_signal


def _deterministic_control(num_samples: int, dt: float) -> ControlSignal:
    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples, dtype=jnp.float32)
    values = jnp.sin(2 * jnp.pi * 5.0 * ts).astype(jnp.float32)
    return build_control_signal(ts, values)


def _time_callable(fn: Callable[[], jax.Array], repeats: int) -> float:
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _ = jax.block_until_ready(result)
        total += time.perf_counter() - start
    return total / repeats


def build_environment():
    loss_config = MSDConfig(num_samples=128, sample_rate=400.0)
    dataset = build_msd_dataset(
        loss_config,
        dataset_size=4,
        key=jax.random.PRNGKey(1),
        forcing_fn=lambda num_samples, dt, key, **kw: _deterministic_control(
            num_samples, dt
        ),
    )
    model_loss = LinearMSDModel(loss_config)
    batch = (dataset.forcing[:2], dataset.reference[:2])

    solve_config = MSDConfig(num_samples=256, sample_rate=500.0)
    solve_ts = jnp.linspace(
        0.0, solve_config.duration, solve_config.num_samples, dtype=jnp.float32
    )
    control = _deterministic_control(solve_config.num_samples, solve_config.dt)
    model_solve = LinearMSDModel(solve_config)

    return {
        "loss_config": loss_config,
        "dataset_ts": dataset.ts,
        "model_loss": model_loss,
        "batch": batch,
        "solve_config": solve_config,
        "control": control,
        "solve_ts": solve_ts,
        "model_solve": model_solve,
    }


def main(args: argparse.Namespace) -> None:
    env = build_environment()
    repeats = args.repeats

    def objective(trial: optuna.Trial) -> float:
        rtol = trial.suggest_float("rtol", 1e-6, 1e-3, log=True)
        atol = trial.suggest_float("atol", 1e-6, 1e-3, log=True)

        loss_time = float("inf")
        solve_time = float("inf")
        try:
            loss_fn = build_loss_fn(
                ts=env["dataset_ts"],
                initial_state=env["loss_config"].initial_state,
                dt=env["loss_config"].dt,
                rtol=rtol,
                atol=atol,
            )

            def run_loss():
                return loss_fn(env["model_loss"], env["batch"])

            loss_time = _time_callable(run_loss, repeats)

            def run_solve():
                return solve_with_model(
                    env["model_solve"],
                    env["solve_ts"],
                    env["control"],
                    env["solve_config"].initial_state,
                    env["solve_config"].dt,
                    rtol=rtol,
                    atol=atol,
                )

            solve_time = _time_callable(run_solve, repeats)
            total = loss_time + solve_time
        except Exception:
            total = float("inf")

        trial.set_user_attr("loss_time", loss_time if total != float("inf") else None)
        trial.set_user_attr("solve_time", solve_time if total != float("inf") else None)
        return total

    if args.storage:
        study = optuna.create_study(
            direction="minimize",
            storage=args.storage,
            study_name=args.study_name,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    Path("logs").mkdir(exist_ok=True)
    output = Path("logs") / "benchmark_optuna_best.json"
    output.write_text(
        json.dumps(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            indent=2,
        )
    )
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize benchmark tolerances with Optuna."
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timing repeats per evaluation."
    )
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI.")
    parser.add_argument("--study-name", type=str, default="benchmark_tolerances")
    args = parser.parse_args()
    main(args)
