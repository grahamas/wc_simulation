import wc1d
import traceback

dt = 1e-6
effective_rate = 0.01
subsample = int(effective_rate / dt)
if subsample is 0:
    subsample = 1

# try:
wc1d.run_simulation("replicate_neuman.json",
    {
        "stimulus": [0.15, 1.2, 3.5],
        "noiseless": True,
        "space": [100.5, 0.5],
        "time": [1, dt],
        "solver": {
            "name": "ode45"
            }
    }, analysis=False, show_figs=False,
    movie_params={
            "subsample": subsample
        })
# except Exception as err:
#     print(err)
#     traceback.print_tb(err.__traceback__)