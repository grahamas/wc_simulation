import diffeq
import traceback
import numpy as np

dt = 1e-4
effective_rate = 0.01
subsample = int(effective_rate / dt)
if subsample is 0:
    subsample = 1

# parameters = [
#     {
#         'my_run_name': 'neuman',
#         'w':[[16.0, -18.2],
#             [27.0, -4.0]]
#     },
#     {
#         'my_run_name': 'neuman_balanced_interaction',
#         'w':[[16.0,-18.2],
#             [18.2,-4.0]]
#     },
#     {
#         'my_run_name': 'neuman_flipped_interaction',
#         'w':[[16.0,-27.0],
#             [18.2,-4.0]]
#     },
#     {
#         'my_run_name': 'neuman_flipped_selfeffect',
#         'w':[[4.0, -18.2],
#             [27.0, -16.0]]
#     }
# ]

# neuman_w = np.array({
#         'my_run_name': 'neuman',
#         'w':[[16.0, -18.2],
#             [27.0, -4.0]]
#     }['w'])
# parameters = [
#     {
#         'my_run_name': 'neuman_scaled_{:.2f}'.format(scale),
#         'w': (neuman_w * scale).tolist()
#     } for scale in np.linspace(0.1,2.0,num=5)
# ]

durations = [4]
strengths = [0.1, 0.5, 1.2, 2]
widths = [1, 2, 3.5, 10]

import itertools
parameters = [
    {
        'stimulus': tup,
        'my_run_name': 'neuman_longstim_dur{:.2f}_str{:.2f}_wid{:.2f}'.format(*tup)
    } for tup in itertools.product(durations, strengths, widths)
]

print("Starting loop.")
for d_params in parameters:
    print("Running: {}".format(d_params['my_run_name']))
    diffeq.run_simulation("replicate_neuman.json",
        {
            #"stimulus": [0.15, 1.2, 3.5],
            "noiseless": True,
            "space": [600.5, 0.5],
            "time": [7, dt],
            "solver": {
                "name": "euler"
                },
            **d_params
        }, analysis=False, show_figs=False,
        movie_params={
                "subsample": subsample
            })
print('done.')