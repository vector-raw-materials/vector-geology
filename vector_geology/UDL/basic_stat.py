def parameters(data, data_name):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats
    
    print('---------------------------------------\n')
    print(f"Main statistical parameters of {data_name}\n")
    
    print('NaN:', data.isna().sum())
    print('NaN(%):', '{:.3f}'.format(100*data.isna().sum()/len(data)), '\n')
    print('Lenght:', len(data))
    print('Min:', data.min())
    print('Max:', data.max())
    print('Range:', max(data) - min(data), '\n')
    
    data = data.dropna()
    
    print('Mean:', '{:.3f}'.format(np.mean(data)))
    print('Median:', np.median(data))
    
    if max(np.unique(data, return_counts=True)[1]) == 1:
        print('Mode has not sence because all values count 1')
    else: print('Mode:', stats.mode(data), '\n')
    
    if np.mean(data) < np.median(data):
        print('Mean < median: left tail')
    else: print('Mean < median: right tail')
        
    print('Skew:', '{:.3f}'.format(stats.skew(data)))
    print("Fisher's kurtosis:", '{:.3f}'.format(stats.kurtosis(data)))
    print("Pearson's kurtosis:", '{:.3f}'.format(stats.kurtosis(data, fisher=False), '\n'))
    print('Variance:', '{:.3f}'.format(np.var(data, ddof=1)))
    print('Standart deviation:', '{:.3f}'.format(np.std(data, ddof=1)))
    print('Mean - 3std:', '{:.3f}'.format(np.mean(data) - 3*(np.std(data, ddof=1))))
    print('Mean + 3std:', '{:.3f}'.format(np.mean(data) + 3*(np.std(data, ddof=1)), '\n'))
    
def plots(data, data_name):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    data = data.dropna()

    nbins = round(np.sqrt(len(data)))
    xmin = np.mean(data) - 10*np.std(data)
    xmax = np.mean(data) + 10*np.std(data)
    
    plt.figure(figsize=(13,8))
    plt.subplot(311)
    y1 = np.mean(data) + 3*np.std(data)
    y2 = np.mean(data)
    y3 = np.median(data)
    y4 = np.mean(data) - 3*np.std(data)
    plt.hist(data, bins=nbins, density=1, color='skyblue', edgecolor='k')
    plt.axvline(x=y2, ls="-", color='red', label='mean')
    plt.axvline(x=y3, ls="-", color='blue', label='median')
    plt.axvline(x=y1, ls="--", color='black', label='mean $\pm$ 3std')
    plt.axvline(x=y4, ls="--", color='black')
    plt.xlim(xmin, xmax)
    plt.ylabel('Probability')
    plt.legend()
    plt.title(data_name+' Statistical Plots')
    plt.grid()
    
    plt.subplot(312)
    plt.hist(data, bins=nbins, density=1, cumulative=True, color='skyblue', edgecolor='k')
    plt.axvline(x=y3, ls="-", color='black', label='median')
    plt.xlim(xmin, xmax)
    plt.ylabel('Probability')
    plt.grid()
    
    plt.subplot(313)
    plt.boxplot(data, vert=False)
    plt.xlim(xmin, xmax)
    plt.xlabel(data_name)
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def metrics(true_values, predicted_values):

    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    residuals = true_values - predicted_values
    sum_of_residuals = np.sum(residuals)
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    
    return {
        'Sum of Residuals': sum_of_residuals,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }