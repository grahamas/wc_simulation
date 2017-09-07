import runner
import traceback

dt = 1e-4
effective_rate = 0.01
subsample = int(effective_rate / dt)
if subsample is 0:
    subsample = 1

runner.run_simulation(json_file_name="replicate_neuman.json",
        # Simulation parameters here
        stimulus = [0.15, 1.2, 3.5],
        noiseless= True,
        space = {"length": 100.5, "step": 0.5},
        time = {"length": 2.81, "step": dt},
        solver = {
            "name": "euler"
            },
        # Post-simulation parameters here
        analysis=False,
        show_figs=False,
        movie_params={
            "subsample": subsample
        }
    )

