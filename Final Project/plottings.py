import matplotlib.pyplot as plt
import seaborn as sns
colors = ['#001c57', '#50248f', '#00ff00', '#38d1ff','#cc3181','#FFBA08']

def plot_countplot(df, col_name, title=None, figsize=(15,5)):
    '''
    Function is called to plot count (i.e. bars)
    :return: bars
    '''
    plt.figure(figsize=figsize, dpi=100)
    sns.countplot(df[col_name], order=df[col_name].value_counts().index)
    plt.title(f'{title} distribution\n', fontsize=15)
    plt.xlabel(f'{title}')
    plt.ylabel('Quantity (frequency)')
    plt.show()



def get_boxplot(data,
                X_axis,
                Y_axis,
                hue=None,
                figsize=(7, 5),
                take_less_box=False):
    '''Function is called to plot boxplots
    ------
    data - pandas dataframe
    X_axis - column need to be reflected in X axis
    Y_axis - column need to be reflected in X axis
    take_less_box - Default is False. If need to plot only 7 boxes
    '''
    if take_less_box:
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x=X_axis,
                    y=Y_axis,
                    hue=hue,
                    data=data.loc[data.loc[:, X_axis].isin(
                        data.loc[:, X_axis].value_counts().index[:7])],
                    palette=colors)
        plt.xticks(rotation=45)
        ax.set_title(f'Boxplot for {X_axis} and {Y_axis}', fontsize=14)
        plt.show()

    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x=X_axis, y=Y_axis, hue=hue, data=data, palette=colors)
        plt.xticks(rotation=45)
        ax.set_title(f'Boxplot for {X_axis} and {Y_axis}', fontsize=14)
        plt.show()




def get_scatter_plot(data, X_axis, Y_axis, title=None, width=600, height=500):

    title = title if title is not None else f"Scatterplot for {X_axis} VS {Y_axis}"
    fig = px.scatter(data, x=X_axis, y=Y_axis, size=Y_axis)
    # Edit the layout
    fig.update_layout(title=title)
    fig.update_xaxes(title=(f'{X_axis}'))
    fig.update_yaxes(title=(f'{Y_axis}'))
    fig.update_layout(width=width)
    fig.update_layout(height=height)
    fig.show()



def get_variable_distribution(col, title=None):
    '''Function is called to plot feture distribution'''

    title = title if title is not None else f"Distribution for '{col}"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5),)
    fig = sm.qqplot(col, fit=True, line='45', ax=ax1)
    fig.suptitle(title, fontsize=20)

    sns.distplot(col.values, bins=20, color=colors[1], ax=ax2)
#     sns.violinplot(col.values, color=colors[3], bw=.3, cut=1, linewidth=4)
    sns.boxplot(col.values,color=colors[3])

    ax1.set_title('QQ-plot')
    ax2.set_title('Distribution')
    ax3.set_title('Boxplot')

    plt.show()

