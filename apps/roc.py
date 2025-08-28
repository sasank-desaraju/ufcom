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
    return alt, mo, np, plt


@app.cell
def _(mo):
    sick_number = mo.ui.number(value=100, label='Sick Count')
    sick_mean = mo.ui.number(value=20, label='Sick Mean')
    sick_std = mo.ui.number(value=5, label='Sick Std Dev')
    healthy_number = mo.ui.number(value=1000, label='Healthy Count')
    healthy_mean = mo.ui.number(value=15, label='Healthy Mean')
    healthy_std = mo.ui.number(value=5, label='Healthy Std Dev')

    mo.vstack([mo.hstack([sick_number, sick_mean, sick_std]),
              mo.hstack([healthy_number, healthy_mean, healthy_std])])
    return healthy_number, sick_mean, sick_number, sick_std


@app.cell
def _(np, sick_mean, sick_number, sick_std):
    sick_data = np.random.normal(sick_mean.value, sick_std.value, sick_number.value)
    return (sick_data,)


@app.cell
def _(np, plt, sick_data):

    # Parameters for the Gaussian distribution
    n_points = 1000
    mean = 0
    std_dev = 1

    # Generate Gaussian distribution
    data = sick_data

    # Plotting the Gaussian distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Gaussian Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid()
    plt.gca()
    return


@app.cell
def _(alt, sick_data):
    # Create a point density plot using Altair
    alt_point_density_chart = alt.Chart(sick_data).mark_circle(opacity=0.5).encode(
        x='Values:Q',
        y='Condition:N',
        color='Condition:N',
        tooltip=['Condition:N', 'Values:Q']
    ).properties(
        title='Point Density Plot of Sick and Healthy Classes',
        width=600,
        height=300
    ).interactive()

    alt_point_density_chart
    return


@app.cell(hide_code=True)
def _(healthy_number, sick_number):
    import altair as alt
    import pandas as pd

    # Assuming sick_number and healthy_number are defined as pd.DataFrame
    data = pd.DataFrame({
        'Condition': ['Sick'] * len(sick_number.value) + ['Healthy'] * len(healthy_number.value),
        'Values': list(sick_number.value) + list(healthy_number.value)
    })

    point_density_chart = alt.Chart(data).mark_circle(opacity=0.5).encode(
        x='Values:Q',
        y='Condition:N',
        color='Condition:N',
        tooltip=['Condition:N', 'Values:Q']
    ).properties(
        title='Point Density Plot of Sick and Healthy Classes',
        width=600,
        height=300
    ).interactive()

    point_density_chart
    return (alt,)


@app.cell(disabled=True)
def _(display, healthy_data, sick_data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
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
    cutoff = 1.5

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

    cutoff_slider = widgets.FloatSlider(value=cutoff, min=-5, max=5, step=0.1, description='Cutoff')

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
                     cutoff_slider.value)
    
        metrics = update_table(cutoff_slider.value)
        display(metrics)

        update_roc(cutoff_slider.value)

    # Attach update function to sliders
    num_sick_slider.observe(update_all, 'value')
    mean_sick_slider.observe(update_all, 'value')
    std_sick_slider.observe(update_all, 'value')
    num_healthy_slider.observe(update_all, 'value')
    mean_healthy_slider.observe(update_all, 'value')
    std_healthy_slider.observe(update_all, 'value')
    cutoff_slider.observe(update_all, 'value')

    # Initial plot and table
    update_all()
    return np, plt


if __name__ == "__main__":
    app.run()
