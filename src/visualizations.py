import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats import inter_rater as irr

def compare_raters(df):
    fig, ax = plt.subplots()
    ax.hist([df['rater1_score_norm'],df['rater2_score_norm']], bins=10, density=True, label=('Rater 1', 'Rater 2'), color=['#2ca25f', '#99d8c9'])
    ax.set_xlabel('Normalized score Range')
    ax.set_ylabel('Normalized count of essays')
    ax.set_title("All Essays")
    ax.legend()

    fig, ax = plt.subplots(2,4, figsize=(12,6))
    cohen_kappa=[]
    for i in range(8):
        df_to_plot = df[df['essay_set']==i+1].copy()
        ax[i//4,i%4].hist([df_to_plot['rater1_score'],df_to_plot['rater2_score']], rwidth=0.8, density=True, label=('Rater 1', 'Rater 2'), color=['#2ca25f', '#99d8c9'])
        
        
        ax[i//4,i%4].set_xlabel('Score Range') if i//4==1 else ax[i//4,i%4].set_xlabel('')
        ax[i//4,i%4].set_ylabel('Count of essays') if i%4==0 else ax[i//4,i%4].set_ylabel('')
        ax[i//4, i%4].set_title("Essay set # "+str(i+1), fontsize=10)
        ax[i//4, i%4].legend(loc='upper left', fontsize=10) if i//4==0 and i%4==0 else ax[i//4, i%4].legend('')
        ax[i//4,i%4].tick_params(labelsize=10)
        
        cohen_kappa.append(cohen_kappa_score(df_to_plot['rater1_score'], df_to_plot['rater2_score'], weights='quadratic'))

        ax[i//4,i%4].text(0.05, 0.65, rf"$\kappa = {cohen_kappa[-1]:.2f}$", 
                       transform=ax[i//4, i%4].transAxes, 
                       fontsize=10, color='black')
        
        ax[i//4,i%4].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        
    print(f"Average of Cohen's Kappa is: {np.mean(cohen_kappa):.2f}")
    print(f"Standard deviation of Cohen's Kappa is: {np.std(cohen_kappa, ddof=1):.2f}")



        

            