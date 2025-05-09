import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
from hydra import initialize_config_dir, compose
from main import main
from src.utils import run_2d_plot


config_dir = os.path.abspath("configs")
run_id = lambda: str(uuid.uuid4())[:8]  # or use timestamp/seed/etc

config_names = ["config_example_kfrflow.yaml"]

run_dirs = []

# Run experiments
for cfg_name in config_names:
    with initialize_config_dir(config_dir=config_dir, job_name="multi_run", version_base=None):
        cfg = compose(config_name=cfg_name)

        # Create unique run folder
        run_name = f"{cfg_name.replace('.yaml', '')}_{uuid.uuid4().hex[:6]}"
        run_dir = os.path.join("outputs", run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Store folder path in config
        cfg.run_dir = run_dir

        print(f"\nRunning: {cfg_name} | Saving to: {run_dir}")
        main(cfg)
        run_dirs.append((cfg_name, run_dir))  # save for later


        try:
            run_2d_plot(
                run_dir, sampling_method="Halton"
            )
        except Exception as e:
            print(f"[WARNING] Plotting failed for {run_dir}: {e}")

