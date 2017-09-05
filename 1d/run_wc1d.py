import diffeq
import traceback

dt = 1e-4
effective_rate = 0.01
subsample = int(effective_rate / dt)
if subsample is 0:
    subsample = 1

# try:
diffeq.run_simulation("replicate_neuman.json",
    {
        "stimulus": [0.15, 1.2, 3.5],
        "noiseless": True,
        "space": [100.5, 0.5],
        "time": [1, dt],
        "solver": {
            "name": "euler"
            }
    }, analysis=False, show_figs=False,
    movie_params={
            "subsample": subsample
        })