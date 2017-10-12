import runner
import traceback
import numpy as np

dt = 1e-4
effective_rate = 0.01
subsample = int(effective_rate / dt)
if subsample is 0:
    subsample = 1

runner.run_simulation(json_file_name="replicate_neuman.json",
        json_dir = 'params',
        run_name = "traveling_wave_tests",
        # Simulation parameters here
        model_modifications = {
            'stimulus': {
                'name': 'square_pulse',
                'args': {
                    'duration': 1,
                    'strength': 2,
                    'width': 1
                    }
                },
            "w": [[16.0, -18.2],
                [27.0, -4.0]],
            "nonlinearity": {
                "name": "sigmoid_norm_rectify",
                "args": {
                    "a": [1.2, 1.0],
                    "theta": [2.6, 8.0]
                    }
            },
            'noiseless': True,
            'lattice': {
                'space_extent': 400.5, 'space_step': 0.5,
                'time_extent': 4, 'time_step': dt,
                'n_populations': 2, 'population_names': ['E', 'I']
                },
            'solver': {
                "name": "ode45"
                }
            },
        # Post-simulation parameters here
        results_params = {
            'analyses_dct': {
                    "e_i": {}
                },
            'figure_params': {
                "show": False,
                'movie_params': {
                    "show": False,
                    "subsample": subsample
                    }
                }
            }
    )

