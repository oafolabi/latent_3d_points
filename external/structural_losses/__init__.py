try:    
    from .tf_nndistance import nn_distance
    from .tf_approxmatch import approx_match, match_cost
except Exception as e:
    print('External Losses (Chamfer-EMD) were not loaded.')
    print(e)
