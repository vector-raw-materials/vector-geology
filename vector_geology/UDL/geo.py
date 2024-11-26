def plot_tops(df):
    
    '''Take a dataframe and make a list of dictionaries with the name, top, base, and color of the markers.
    Then the dictionary can be use to plot the markers matplotlib or plotly'''

    tops = []

    for i in range(0, len(df) - 1):
        tops.append({
            'name': df['name'][i],
            'top': df['depth'][i],
            'base': df['depth'][i + 1],
            'color': df['color'][i]})

    return tops