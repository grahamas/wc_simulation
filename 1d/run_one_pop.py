import runner
import traceback
import numpy as np

from args import ParameterSet as P
from itertools import chain

dt = 1e-4
effective_rate = 0.01
subsample = int(effective_rate / dt)
if subsample is 0:
    subsample = 1


runner.run_simulation(json_file_name="one_pop.json",
        json_dir = 'params',
        run_name = 'one_pop_test',
        # Simulation parameters here
        model_modifications = {
            "w": [[P(np.linspace(0.2, 2, 5), name='w')]],
            "s": [[P(np.linspace(0.25, 3, 5), name='s')]],
            "beta": [P(np.linspace(0.8, 1.2, 4), name='beta')],
            "alpha": [P(np.linspace(0.5, 2, 4), name='alpha')],
            'stimulus': {
                'name': 'square_pulse',
                'args': {
                    'duration': P(np.concatenate([np.linspace(0.01, 0.2, 3), np.linspace(0.25, 2, 3)]), name='duration'),
                    'strength': P(np.concatenate([np.linspace(0.01, 0.2, 3), np.linspace(0.25, 2, 3), np.linspace(2, 10, 4)]), name='strength'),
                    'width': P(np.linspace(1, 20, 5), name='width')
                    }
                },
            'lattice': {
                'space_extent': 50.5, 'space_step': 0.5,
                'time_extent': 2.81, 'time_step': dt,
                'n_populations': 1, 'population_names': ['E']
                },
            },
        # Post-simulation parameters here
        results_params = {
            'analyses_dct': {
                "one_pop": {}
            },
            'figure_params': {
                "show": False,
                'movie_params': {
                }
            }
        }
    )

