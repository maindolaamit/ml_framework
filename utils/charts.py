import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def reset_rc():
    """
    Reset the matplotlib rc parameters
    """
    mpl.rcParams.update(mpl.rcParamsDefault)


def set_rc(titlesize=10):
    """
    Set the RC parameters for matplotlib and set Seaborn settings
    """
    pd.set_option('display.max_colwidth', 100)
    # sns.set(style="ticks", color_codes=True)
    # plt.style.use('seaborn-whitegrid')
    sns.set_theme(style='whitegrid', color_codes=True, palette='Set2')

    # Set Matplotlib defaults
    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelweight='bold', labelsize='large',
           titleweight='bold', titlesize=titlesize, titlepad=10)

    plt.rcParams['text.color'] = '#191c1b'
    purple_color = "#b753e6"
    _ = plt.tight_layout()
    # plt.rcParams['axes.labelcolor']= '#ffaa80'
    # plt.rcParams['xtick.color'] = '#e27caa'
    # plt.rcParams['ytick.color'] = '#799fec'
    # plt.rcParams['font.size']=12


def donut(df_col, title=None, figsize=None, titlesize=17):
    """
    Create a Donut chart for the given Pandas Series
    :param figsize: Figure size
    :param title: Title of the figure
    :param df_col: Pandas Series
    """
    df_count = df_col.apply(str).str.lower().value_counts()

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#66b3ff', '#99ff99', '#ffcc99']
    patches, text, autotext = ax.pie(df_count, labels=df_count.index, autopct='%1.1f%%',
                                     shadow=True, startangle=90)
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    if title is None:
        title = f'Distribution of {df_col.name.capitalize()}'
    plt.title(title, fontsize=titlesize)
    plt.tight_layout()
    plt.show()


def bi_donut(df_cols, sup_title):
    """
    Creates two donut charts side by side for the passed DataFrame of two columns
    :param df_cols: DataFrame
    :param sup_title: Sup Title of the cart
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 6), constrained_layout=False)
    columns = df_cols.columns
    if len(columns) < 2:
        raise Exception("Number of columns less than 2.")

    for i in range(2):
        ax = ax1 if i == 0 else ax2
        col = columns[i]
        df_col = df_cols[col]
        df_count = df_col.str.upper().value_counts()
        title = f'{df_col.name.upper()}'
        ax.set_title(label=title, ha='center', va='bottom',
                     fontdict={'fontsize': 15}, color='r')
        patches, text, autotext = ax.pie(df_count, labels=df_count.index, autopct='%1.1f%%',
                                         startangle=90)
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        ax.add_artist(centre_circle)
        ax.legend(loc='upper left')

    plt.tight_layout()
    fig.suptitle(sup_title, ha='center', va='bottom', size=18,
                 fontweight='bold', fontdict={'color': 'g'})
    _ = plt.show()


def stacked_donut(grp_data):
    """
    Creates a stacked Donut chart by grouping second column inside the first column
    :param grp_data: Pandas Dataframe of two columns
    """
    columns = grp_data.columns
    if len(columns) < 2:
        raise Exception("Number of columns less than 2.")

    # Create groups and subgroups
    from matplotlib import cm
    group_names = grp_data.index.levels[0].to_list()
    group_size = [grp_data[name].sum() for name in group_names]
    subgroup_names = [f"{x[1]}" for x in grp_data.index.values]
    subgroup_size = grp_data.values

    # Create colors
    color_pallete = [cm.Blues, cm.Reds, cm.Greens,
                     cm.Oranges, cm.Purples, cm.Greys]

    grp_cmap = {group_names[i]: color_pallete[i]
                for i in range(len(group_names))}

    # Create color palette for group and subgroups
    grp_colors = [grp_cmap[color](0.8) for color in grp_cmap.keys()]
    subgroup_colors = []
    for grp in grp_cmap.keys():
        variance = np.linspace(.1, .9, grp_data[grp].count())  # Divide based on Sub-group count for a group
        for i in variance:
            subgroup_colors.append(grp_cmap[grp](i))

    # First Ring (outside)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis('equal')
    mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=grp_colors)
    plt.setp(mypie, width=0.3, edgecolor='white')
    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=1.3 - 0.3, labels=subgroup_names, labeldistance=0.7,
                       colors=subgroup_colors)
    plt.setp(mypie2, width=.9, edgecolor='white')
    plt.margins(0, 0)

    # show it
    plt.show()


def strip_violinplot(data, x, y=None, hue=None, title="Data Distribution", xlabel=None, ylabel=None, figsize=(12, 8)):
    """
    Creates a Violinplot with stripplot showing the distribution density of data as well
    :param data: Dataframe
    :param x: X data
    :param y: y data
    :param hue: hue column to be used
    :param title: Title of the plot
    :param xlabel: X label name
    :param ylabel: Y label name
    :param figsize: figsize
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.stripplot(x=x, y=y, data=data
                       , hue=hue, jitter=0.2, size=2.5)
    ax = sns.violinplot(x=x, y=y, data=data)
    if xlabel is None:
        xlabel = data[x].name
    if y and ylabel is None:
        ylabel = data[y].name
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    _ = plt.show()


def plot_stripviolens(df, cols, figsize=(20, 20)):
    """Plot the subplots of violinplot and stripplot in 3 columns"""
    import math
    ncols = 3
    nrows = math.ceil(len(cols) / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    fig.tight_layout()

    purple_color = "#b753e6"
    for i, colname in enumerate(cols):
        row = int(i / ncols)
        col = i % ncols
        sns.stripplot(data=df, x=colname, ax=axs[row][col], color=purple_color, alpha=0.5)
        sns.violinplot(data=df, x=colname, ax=axs[row][col])
        axs[row][col].set(xlabel=colname)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)


def plot_boxes(df, cols, figsize=(20, 16)):
    """Plot the subplots of boxplots in 3 columns"""
    import math
    ncols = 3
    nrows = math.ceil(len(cols) / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    fig.tight_layout()

    for i, colname in enumerate(cols):
        row = int(i / ncols)
        col = i % ncols
        sns.boxplot(x=df[colname], ax=axs[row][col]).set(xlabel=colname)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)


def qq_plot(df, features):
    """
    Plot a Quantile to Quantile plot for the features
    :param df:
    :param features:
    """
    import scipy.stats as stats
    import pylab
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df[features].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[features], dist='norm', plot=pylab)
    plt.show()


def grouped_bar(df, title, figsize=(20, 8)):
    """
    Creates a beautiful Bar chart with groups
    :param df: Dataframe
    :param title: Title of the chart
    :param figsize: Figure size
    """
    colors_list = ['#5cb85c', '#5bc0de', '#d9534f']
    # Create graph with figure size 20,8 | width 0.8
    ax = df.plot.bar(figsize=figsize, width=0.8, color=colors_list)
    print('')
    # Set Title with font size 16
    plt.title(title, size=16)
    # Shift Legend to upper right with font 14
    plt.legend(loc='upper right', fontsize=14)
    plt.xticks(fontsize=14, rotation=0)
    # Hide Y axis values
    ax.yaxis.set_major_locator(plt.NullLocator())

    # Remove the borders
    ax.spines['left'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)

    # Show Percentage on Yticks
    value_format = "{:.1%}"
    # Loop through the Rectangle objects of the plot to get the bars
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2 + .05,  # X position of text, start position + half width + margin
                bar.get_height() + 1,  # Y position of text
                f"{bar.get_height()}%",  # Text Value
                fontsize=14,  # Fontsize of text
                ha="center"  # Alignment of text
                )

    # Show plot
    plt.show()


def value_count_bar(data, title=None, normalize=True, figsize=(12, 7),
                    titlesize=17, labelsize=14, ticksize=12):
    """
    Plots a beautiful bar plot for the count of columns
    :param data: Pandas Series
    :param title: Title of the plot
    :param normalize: If true distribution will be shown in normalized fashion
    :param labelsize: Label size
    :param figsize: Figure size
    :param titlesize: Title size
    """
    grp_data = data.value_counts(normalize=normalize).reset_index()
    column_name = grp_data.columns[1]
    grp_data.columns = [column_name, 'count']
    _, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(x=column_name, y="count", data=grp_data, palette=sns.color_palette("tab10"))
    if title is None:
        title = f'Distribution of {column_name.capitalize()}'
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(column_name.capitalize(), fontsize=labelsize)
    ax.set_ylabel(f'{column_name.capitalize()} count', fontsize=labelsize)
    # ax.set(title=title, xlabel=column_name.capitalize(), ylabel=f'{column_name.capitalize()} count')

    plt.xticks(fontsize=ticksize, rotation=10)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.spines['left'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)

    # Print values above the bar
    for bar in ax.patches:
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = bar.get_height() + bar.get_height() * .01
        value = f"{bar.get_height() * 100:.2f} %" if normalize else round(bar.get_height())
        ax.text(x_pos,  # X Position of the text
                y_pos,  # Y Position of the text
                value,  # Value to print above bar
                fontsize=labelsize,  # Fontsize
                ha="center"  # Alignment of text
                )
    plt.show()


def plot_images(images_path_list, row=1, col=1, figsize=(8, 8)):
    num_plots = row * col
    assert images_path_list is not None and len(images_path_list) >= num_plots, 'Number of items in the list are lesser than subplots count.'
    fig, axs = plt.subplots(row, col, figsize=figsize)
    for i in range(row):
        for j in range(col):
            image_path = images_path_list[i+j]
            axs[i, j].imshow(plt.imread(image_path))
    plt.show()


def plot_confusion_matrix(cnf_matrix_data, target_names,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # from sklearn.metrics import confusion_matrix
    df_cm = pd.DataFrame(cnf_matrix_data, columns=target_names, index=target_names).astype('float')

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 5))
    plt.title(title, color='green', fontsize=25)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, color='indigo')
    plt.yticks(tick_marks, target_names, color='indigo')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap="YlGnBu", annot=True, annot_kws={"size": 16}, fmt=".0f")
    # plt.imshow(df_cm, interpolation='nearest', cmap="YlGnBu")
    plt.tight_layout()
    plt.ylabel('Actual', color='crimson', fontsize=20)
    plt.xlabel('Predicted', color='crimson', fontsize=20)


def plot_loss_acc(history, title_remarks=None, figsize=(8,6)):
    """
    Method to plot the Loss and Accuracy with the given History dataframe as input
    """
    history_df = pd.DataFrame(history.history)
    # plot loss during training
    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    title = f'Loss and Accurracy Graph\n{"" if title_remarks is None else title_remarks}'
    fig.suptitle(title, size=15)

    ax[0].set_title('Loss')
    ax[0].plot(history_df['loss'], label='train')
    if 'val_loss' in history_df.columns:
        ax[0].plot(history_df['val_loss'], label='test')
    ax[0].legend()
    # plot mse during training
    ax[1].set_title('Accurracy')
    ax[1].plot(history_df['accuracy'], label='train')
    if 'val_accuracy' in history_df.columns:
        ax[1].plot(history_df['val_accuracy'], label='test')
    ax[1].legend()
    plt.show()
