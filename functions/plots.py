import os

def plot_mi_scores(scores):
    """"Mutual information (MI) between two random variables is a non-negative value, which measures the dependency
        between the variables. It is equal to zero if and only if two random variables are independent, and higher
        values mean higher dependency."""
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

    filename = str(f'figures/MI_Scores.png')
    plt.savefig(filename)
    print(f'Plot MI scores saved: {filename}')


def plot_hist(df, cols):
    """Plot Histogram for numerical features."""
    for col in cols:
        plt.clf()
        sns.displot(x=col, data=df, hue="Survived", element="step", palette={0: "darkred", 1: "lightgreen"})
        plt.title(f'Histogram numerical feature: {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        dir = 'figures/Num_Histograms'
        os.makedirs(dir, exist_ok=True)
        filename = str(f'Hist_ord_{col}.png')
        filepath = os.path.join(dir, filename)

        plt.savefig(filepath)
        print(f'Histogram saved: {filepath}')


def plot_count(df, cols):
    """Plot Countplot for categorical features with."""
    for col in cols:
        if df[col].nunique() < 32:  # only plot categorical features with less than 32 unique values
            plt.clf()
            sns.countplot(x=col, hue='Survived', data=df, palette={0: "darkred", 1: "lightgreen"})
            plt.title(f'Countplot categorical feature: {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

            dir = 'figures/Cat_Countplots'
            os.makedirs(dir, exist_ok=True)
            filename = str(f'Count_cat_{col}.png')
            filepath = os.path.join(dir, filename)

            plt.savefig(filepath)
            print(f'Countplot saved: {filepath}')
        else:
            print(f'No plot created as #{df[col].nunique()} unique values were retrieved for Categorical Feature: {col}')


def plot_swarm(df, cols, y):
    """Plot Swarmplot for categorical features."""
    for col in cols:
        plt.clf()
        sns.swarmplot(x=col, y=y, data=df)
        plt.title(f'Swarmplot categorical feature: {col}')
        plt.xlabel(col)
        plt.ylabel('Target')

        filename = str(f'figures/Cat_Swarmplots/Swarm_cat_{col}.png')
        plt.savefig(filename)
        print(f'Swarmplot saved: {filename}')


def plot_cdf(df, cols):
    """Plot CDF for categorical features."""
    for col in cols:
        plt.clf()
        pmf = df[col].value_counts().sort_index()
        cdf = pmf.cumsum()
        cdf_norm = cdf / pmf.sum()  # normalized CDF
        cdf_norm.plot.bar()

        filename = str(f'figures/Cat_CDF/CDF_cat_{col}.png')
        plt.savefig(filename)
        print(f'CDF saved: {filename}')

def plot_violin(df, cols):
    """Plot Violin for numerical features."""
    plt.clf()
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    sns.violinplot(ax=ax, data=df[cols])
    plt.title(f'Violin-plot of numerical features:')

    filename = str(f'figures/Violin_num.png')
    plt.savefig(filename)
    print(f'Violin saved: {filename}')

def plot_corr(df, cols):
    """Plot Heatmap for numerical features."""
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    sns.heatmap(ax=ax, data=df[cols].corr().round(2), annot=True, cmap='RdYlGn')
    plt.title(f'Heatmap of numerical features:')

    filename = str(f'figures/Heatmap_corr_num.png')
    plt.savefig(filename)
    print(f'Heatmap saved: {filename}')