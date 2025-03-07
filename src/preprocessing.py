def normalize_scores(row, col_name,min_score, max_score):
    
    a = 10/(max_score - min_score) # slope
    b = -10*min_score/(max_score - min_score) # intercept

    return a*row[col_name] + b