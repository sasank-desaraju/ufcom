# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "matplotlib==3.10.5",
#     "numpy==2.3.2",
#     "pandas==2.3.2",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    #import seaborn as sns
    import altair as alt
    import math
    return alt, math, mo, np, pd, plt


@app.cell
def _(mo):
    sick_number = mo.ui.number(value=60, label='Sick Count')
    healthy_number = mo.ui.number(value=60, label='Healthy Count')

    sick_mean = mo.ui.slider(start=0, stop=50, value=30, label='Sick Mean', show_value=True)
    healthy_mean = mo.ui.slider(start=0, stop=50, value=20, label='Healthy Mean', show_value=True)

    sick_std = mo.ui.slider(start=0.5, stop=10, value=5, label='Sick Std Dev', show_value=True)
    healthy_std = mo.ui.slider(start=0.5, stop=10, value=5, label='Healthy Std Dev', show_value=True)

    mo.vstack([mo.hstack([sick_number, sick_mean, sick_std]),
              mo.hstack([healthy_number, healthy_mean, healthy_std])])
    return (
        healthy_mean,
        healthy_number,
        healthy_std,
        sick_mean,
        sick_number,
        sick_std,
    )


@app.cell
def _(
    healthy_mean,
    healthy_number,
    healthy_std,
    np,
    pd,
    sick_mean,
    sick_number,
    sick_std,
):
    sick_data = np.random.normal(sick_mean.value, sick_std.value, sick_number.value)
    healthy_data = np.random.normal(healthy_mean.value, healthy_std.value, healthy_number.value)

    # Create a DataFrame for Altair
    df = pd.DataFrame({
        'Value': np.concatenate([sick_data, healthy_data]),
        'Condition': ['A'] * len(sick_data) + ['B'] * len(healthy_data)
    })

    return df, healthy_data, sick_data


@app.cell
def _(alt, df):
    density_points_plot = alt.Chart(df).transform_density(
        'Value',
        as_=['Value', 'Density'],
        groupby=['Condition']
    ).mark_point().encode(
        x='Value:Q',
        y='Density:Q',
        color='Condition:N'
    ) + alt.Chart(df).mark_point(filled=True, opacity=0.5).encode(
        x='Value:Q',
        y='Density:Q',
        color='Condition:N'
    ).properties(
        title='Density and Points Plot of Values',
        width=600,
        height=400
    )

    # density_points_plot
    return


@app.cell
def _(alt, cutoff_line, df):
    dp = alt.Chart(df).mark_point().encode(
        # x='Value:Q',
        x=alt.X('Value:Q'),
        yOffset="jitter:Q",
        color='Condition:N'
    ).properties(
        title='Scatter Plot of Values',
        # width=600,
        height=50
    ).transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    )
    # gaussian_jitter = alt.Chart(source, title='Normally distributed jitter').mark_circle(size=8).encode(
    #     y="Major_Genre:N",
    #     x="IMDB_Rating:Q",
    #     yOffset="jitter:Q",
    #     color=alt.Color('Major_Genre:N').legend(None)
    # ).transform_calculate(
    #     # Generate Gaussian jitter with a Box-Muller transform
    #     jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    # )
    #
    # uniform_jitter = gaussian_jitter.transform_calculate(
    #     # Generate uniform jitter
    #     jitter='random()'
    # ).encode(
    #     alt.Y('Major_Genre:N').axis(None)
    # ).properties(
    #     title='Uniformly distributed jitter'
    # )
    dp = dp + cutoff_line
    dp
    return


@app.cell
def _(df, math, mo):
    cutoff_slider = mo.ui.slider(
        start=math.floor(min(df['Value'])),
        stop=math.ceil(max(df['Value'])),
        value=20,
        label='Cutoff Line Position',
        show_value=True
    )
    mo.hstack([cutoff_slider])
    return (cutoff_slider,)


@app.cell
def _(alt, cutoff_slider, pd):
    cutoff = cutoff_slider.value
    cutoff_line = alt.Chart(pd.DataFrame({'cutoff': [cutoff]})).mark_rule(color='red').encode(
    # cutoff_line = alt.Chart().mark_rule(color='red').encode(
        x='cutoff:Q',
        size=alt.value(2),
    )
    return cutoff, cutoff_line


@app.cell
def _(cutoff, healthy_data, np, sick_data):
    def eval_cutoff(sick, healthy, cutoff):
        # Sensitivity and specificity calculation
        tp = np.sum(sick_data > cutoff)
        fn = np.sum(sick_data <= cutoff)
        tn = np.sum(healthy_data <= cutoff)
        fp = np.sum(healthy_data > cutoff)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sensitivity, specificity, tp, fn, tn, fp
    sensitivity, specificity, tp, fn, tn, fp = eval_cutoff(sick_data, healthy_data, cutoff)


    return (eval_cutoff,)


@app.cell
def _():
    # Confusion Matrix
    return


@app.cell
def _(df, eval_cutoff, healthy_data, math, pd, sick_data):
    # ROC Dataframe
    roc_data = []
    # for cutoff_val in range(math.floor(min(df['Value'])), math.ceil(max(df['Value'])), 20):
    for cutoff_val in range(0, math.ceil(max(df['Value'])), 1):
        sens, spec, _tp, _fn, _tn, _fp = eval_cutoff(sick_data, healthy_data, cutoff_val)
        roc_data.append({'Cutoff': cutoff_val, 'True Positive Rate': sens, 'False Positive Rate': 1 - spec})
    roc_data = pd.DataFrame(roc_data)    # roc_data
    return (roc_data,)


@app.cell
def _(alt, cutoff, eval_cutoff, healthy_data, pd, roc_data, sick_data):
    # ROC Curve
    roc = alt.Chart(roc_data).mark_line(point=True).encode(
        y=alt.Y('True Positive Rate:Q'),
        x=alt.X('False Positive Rate:Q'),
        tooltip=['Cutoff', 'True Positive Rate', 'False Positive Rate']
    )
    _sens, _spec, _tp, _fn, _tn, _fp = eval_cutoff(sick_data, healthy_data, cutoff)
    # current_stats = pd.DataFrame({'Cutoff': cutoff, 'True Positive Rate': _sens, 'False Positive Rate': _spec})
    current_stats = pd.DataFrame([[cutoff, _sens, 1 - _spec]], columns=['Cutoff', 'True Positive Rate', 'False Positive Rate'])
    current_dot = alt.Chart(current_stats).mark_point(color='red', size=100).encode(
        y=alt.Y('True Positive Rate:Q'),
        x=alt.X('False Positive Rate:Q'),
        tooltip=['Cutoff', 'True Positive Rate', 'False Positive Rate']
    )
    roc = roc + current_dot
    roc
    return


@app.cell(disabled=True)
def _(cutoff, display, healthy_data, np, pd, plt, sick_data):
    import seaborn as sns
    import ipywidgets as widgets
    from sklearn.metrics import roc_curve

    # Initialize parameters
    num_sick = 100
    mean_sick = 0
    std_sick = 1
    num_healthy = 100
    mean_healthy = 3
    std_healthy = 1
    poo_cutoff = 1.5

    # Generate data
    _sick_data = np.random.normal(loc=mean_sick, scale=std_sick, size=num_sick)
    _healthy_data = np.random.normal(loc=mean_healthy, scale=std_healthy, size=num_healthy)

    def update_plot(num_sick, mean_sick, std_sick, num_healthy, mean_healthy, std_healthy, cutoff):
        plt.figure(figsize=(12, 6))

        # Density plot
        sns.kdeplot(sick_data, color='red', label='Sick', fill=True)
        sns.kdeplot(healthy_data, color='blue', label='Healthy', fill=True)

        # Show points
        plt.scatter(sick_data, np.zeros_like(sick_data), color='red', alpha=0.5)
        plt.scatter(healthy_data, np.zeros_like(healthy_data), color='blue', alpha=0.5)

        # Cutoff line
        plt.axvline(x=cutoff, color='black', linestyle='--', label='Cutoff')

        plt.title('Density Plot of Sick and Healthy Populations')
        plt.xlabel('Biomarker Level')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.gca()

    # Sensitivity and specificity calculation
    def calculate_metrics(cutoff):
        tp = np.sum(sick_data > cutoff)
        fn = np.sum(sick_data <= cutoff)
        tn = np.sum(healthy_data <= cutoff)
        fp = np.sum(healthy_data > cutoff)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return sensitivity, specificity

    # Create sliders
    num_sick_slider = widgets.IntSlider(value=num_sick, min=1, max=500, description='Sick Count')
    mean_sick_slider = widgets.FloatSlider(value=mean_sick, min=-5, max=5, description='Sick Mean')
    std_sick_slider = widgets.FloatSlider(value=std_sick, min=0.1, max=5, description='Sick Std Dev')

    num_healthy_slider = widgets.IntSlider(value=num_healthy, min=1, max=500, description='Healthy Count')
    mean_healthy_slider = widgets.FloatSlider(value=mean_healthy, min=-5, max=5, description='Healthy Mean')
    std_healthy_slider = widgets.FloatSlider(value=std_healthy, min=0.1, max=5, description='Healthy Std Dev')

    poo_cutoff_slider = widgets.FloatSlider(value=cutoff, min=-5, max=5, step=0.1, description='Cutoff')

    def update_table(cutoff):
        sensitivity, specificity = calculate_metrics(cutoff)
        metrics_table = pd.DataFrame({
            'Metric': ['Sensitivity', 'Specificity'],
            'Value': [sensitivity, specificity]
        })
        return metrics_table

    def update_roc(cutoff):
        y_true = [1] * num_sick + [0] * num_healthy
        y_scores = np.concatenate([sick_data, healthy_data])
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC Curve')
        plt.scatter(fpr[np.abs(thresholds - cutoff).argmin()], tpr[np.abs(thresholds - cutoff).argmin()], color='red', label='Current Point')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.axhline(0.5, color='black', linestyle='--')
        plt.axvline(0.5, color='black', linestyle='--')
        plt.legend()
        plt.grid()
        plt.gca()

    # Output display
    def update_all(*args):
        update_plot(num_sick_slider.value, mean_sick_slider.value, std_sick_slider.value,
                     num_healthy_slider.value, mean_healthy_slider.value, std_healthy_slider.value,
                     poo_cutoff_slider.value)

        metrics = update_table(poo_cutoff_slider.value)
        display(metrics)

        update_roc(poo_cutoff_slider.value)

    # Attach update function to sliders
    num_sick_slider.observe(update_all, 'value')
    mean_sick_slider.observe(update_all, 'value')
    std_sick_slider.observe(update_all, 'value')
    num_healthy_slider.observe(update_all, 'value')
    mean_healthy_slider.observe(update_all, 'value')
    std_healthy_slider.observe(update_all, 'value')
    poo_cutoff_slider.observe(update_all, 'value')

    # Initial plot and table
    update_all()
    return


if __name__ == "__main__":
    app.run()
