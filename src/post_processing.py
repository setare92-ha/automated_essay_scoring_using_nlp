def denormalize_scores(y, min_range, max_range):
    a = 10/(max_range - min_range) # slope
    b = -10*min_range/(max_range - min_range) # intercept

    return (y-b)/a
