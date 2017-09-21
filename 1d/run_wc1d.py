import runner
import traceback

dt = 1e-4
effective_rate = 0.01
subsample = int(effective_rate / dt)
if subsample is 0:
    subsample = 1

runner.run_simulation(json_file_name="replicate_neuman.json",
        json_dir = 'params',
        run_name = "restructure_test",
        # Simulation parameters here
        model_modifications = {
            'stimulus': {
                'name': 'square_pulse',
                'args': {
                    'duration': 0.15,
                    'strength': 1.2,
                    'width': 3.5
                    }
                },
            'noiseless': True,
            'lattice': {
                'space_extent': 100.5, 'space_step': 0.5,
                'time_extent': 2.81, 'time_step': dt,
                'n_populations': 2, 'population_names': ['E', 'I']
                },
            'solver': {
                "name": "euler"
                }
            },
        # Post-simulation parameters here
        results_params = {
            'analyses_dct': {
                    "e_i": {}
                },
            'figure_params': {
                "show": False
                },
            'movie_params': {
               "show": False,
                "subsample": subsample
                }
            }
    )

