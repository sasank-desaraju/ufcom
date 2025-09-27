# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "ipywidgets==8.1.7",
#     "matplotlib==3.10.6",
#     "numpy==2.3.3",
#     "pandas==2.3.2",
#     "scikit-learn==1.7.2",
#     "scipy==1.16.2",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    from scipy import stats
    import altair as alt
    import math
    return alt, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Stats Examples

    This app helps visualize some examples and connect them to stats concepts.

    Use the tabs below to explore the different the different examples.
    """
    )
    return


@app.cell
def _(mo):
    tabs = mo.ui.tabs({
        # For the larynhoscopy odds example, I want to show odds ratio ofc. I also want to show how this would look with risk (a probability) instead of odds. I think I can also include the concepts of absolute and relative risk reduction and number needed to treat.
        "Laryngoscopy (Odds)": mo.md(
                      "Test results of a new intubation technique: 100 patients died using direct laryngoscopy, while 10 died using a blindfolded technique; 900 patients survived using direct laryngoscopy, while 200 survived using the blindfolded technique. What is the odds ratio for dying using direct laryngoscopy vs blindfolded?"
        ),
        # For the whipple tests example, I want to show a t-test and ANOVA. I also want to show the concept of paired vs unpaired tests. I think I can also include the concepts of type 1 and type 2 errors here.
        "Whipple (Tests)": mo.md(
            "Patients scheduled to undergo a whipple are randomized into 2 groups.  Group 1 receives GETA + regional anesthesia and Group 2 receives GETA + a ketamine infusion.  PACU opoid usage is measured for 4 hours.  Which of the following statistical tests would be MOST appropriate to determine whether opioid usage changes with respect to the type of anesthesia?"
        ),
    })
    tabs
    return (tabs,)


@app.cell
def _(mo, tabs):
    mo.stop(tabs.value != "Laryngoscopy (Odds)")
    mo.md(
        r"""
        ## Odds Ratios vs Risk Analysis

        **Clinical Scenario:** Comparing two intubation techniques - direct laryngoscopy vs a blindfolded technique.

        Understanding the difference between **odds** and **risk** is crucial in medical decision-making:
        - **Risk (probability)** = Deaths / Total patients in each group
        - **Odds** = Deaths / Survivors in each group  
        - **Odds Ratio** = (Odds in exposed group) / (Odds in control group)
        - **Relative Risk** = (Risk in exposed group) / (Risk in control group)

        Use the sliders below to explore how different patient outcomes affect these key statistics.
        """
    )
    return


@app.cell
def _(mo, tabs):
    mo.stop(tabs.value != "Laryngoscopy (Odds)")

    # Interactive controls for the 2x2 table
    direct_deaths = mo.ui.slider(1, 200, value=100, label="Direct Laryngoscopy Deaths", show_value=True)
    direct_survivors = mo.ui.slider(100, 1000, value=900, label="Direct Laryngoscopy Survivors", show_value=True)
    blind_deaths = mo.ui.slider(1, 200, value=10, label="Blindfolded Deaths", show_value=True)
    blind_survivors = mo.ui.slider(50, 500, value=200, label="Blindfolded Survivors", show_value=True)

    mo.vstack([
        mo.hstack([direct_deaths, direct_survivors]),
        mo.hstack([blind_deaths, blind_survivors])
    ])
    return blind_deaths, blind_survivors, direct_deaths, direct_survivors


@app.cell
def _(
    alt,
    blind_deaths,
    blind_survivors,
    direct_deaths,
    direct_survivors,
    mo,
    pd,
    tabs,
):
    mo.stop(tabs.value != "Laryngoscopy (Odds)")

    # Calculate key statistics
    direct_total = direct_deaths.value + direct_survivors.value
    blind_total = blind_deaths.value + blind_survivors.value

    # Risk calculations
    direct_risk = direct_deaths.value / direct_total
    blind_risk = blind_deaths.value / blind_total
    relative_risk = direct_risk / blind_risk
    absolute_risk_reduction = direct_risk - blind_risk
    number_needed_to_treat = 1 / abs(absolute_risk_reduction) if absolute_risk_reduction != 0 else float('inf')

    # Odds calculations  
    direct_odds = direct_deaths.value / direct_survivors.value
    blind_odds = blind_deaths.value / blind_survivors.value
    odds_ratio = direct_odds / blind_odds

    # Create 2x2 contingency table data
    table_data = pd.DataFrame({
        'Technique': ['Direct Laryngoscopy', 'Direct Laryngoscopy', 'Blindfolded', 'Blindfolded'],
        'Outcome': ['Death', 'Survival', 'Death', 'Survival'], 
        'Count': [direct_deaths.value, direct_survivors.value, blind_deaths.value, blind_survivors.value],
        'Row': ['Direct', 'Direct', 'Blindfolded', 'Blindfolded'],
        'Col': ['Death', 'Survival', 'Death', 'Survival']
    })

    # Create heatmap
    heatmap = alt.Chart(table_data).mark_rect().encode(
        x=alt.X('Col:O', title='Outcome'),
        y=alt.Y('Row:O', title='Technique'),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title="Count")),
        tooltip=['Technique', 'Outcome', 'Count']
    ).properties(
        title='2x2 Contingency Table',
        width=300,
        height=200
    )

    # Add text labels
    text = alt.Chart(table_data).mark_text(
        fontSize=16,
        fontWeight='bold'
    ).encode(
        x='Col:O',
        y='Row:O', 
        text='Count:Q',
        color=alt.value('white')
    )

    contingency_viz = heatmap + text

    # Statistics summary
    stats_text = mo.md(f"""
    **Key Statistics:**
    - **Direct Laryngoscopy Risk:** {direct_risk:.1%} ({direct_deaths.value}/{direct_total})
    - **Blindfolded Risk:** {blind_risk:.1%} ({blind_deaths.value}/{blind_total})
    - **Relative Risk:** {relative_risk:.2f}
    - **Absolute Risk Reduction:** {abs(absolute_risk_reduction):.1%}
    - **Number Needed to Treat:** {number_needed_to_treat:.1f} patients

    **Odds Calculations:**
    - **Direct Laryngoscopy Odds:** {direct_odds:.3f} ({direct_deaths.value}/{direct_survivors.value})
    - **Blindfolded Odds:** {blind_odds:.3f} ({blind_deaths.value}/{blind_survivors.value})
    - **Odds Ratio:** {odds_ratio:.2f}
    """)

    mo.hstack([contingency_viz, stats_text])
    return blind_risk, direct_risk, odds_ratio, relative_risk


@app.cell
def _(alt, blind_risk, direct_risk, mo, odds_ratio, pd, relative_risk, tabs):
    mo.stop(tabs.value != "Laryngoscopy (Odds)")

    # Risk comparison visualization
    risk_data = pd.DataFrame({
        'Technique': ['Direct Laryngoscopy', 'Blindfolded'],
        'Risk': [direct_risk, blind_risk],
        'Risk_Percent': [direct_risk * 100, blind_risk * 100]
    })

    risk_bars = alt.Chart(risk_data).mark_bar().encode(
        x=alt.X('Technique:O', title='Intubation Technique'),
        y=alt.Y('Risk_Percent:Q', title='Death Risk (%)', scale=alt.Scale(domain=[0, max(direct_risk, blind_risk) * 110])),
        color=alt.Color('Technique:N', scale=alt.Scale(domain=['Direct Laryngoscopy', 'Blindfolded'], 
                                                     range=['#FF6B6B', '#4ECDC4'])),
        tooltip=['Technique', 'Risk_Percent:Q']
    ).properties(
        title='Death Risk by Technique',
        width=300,
        height=250
    )

    # Add text labels on bars
    risk_text = alt.Chart(risk_data).mark_text(
        align='center',
        baseline='bottom',
        fontSize=12,
        fontWeight='bold'
    ).encode(
        x='Technique:O',
        y='Risk_Percent:Q',
        text=alt.Text('Risk_Percent:Q', format='.1f'),
        color=alt.value('black')
    )

    risk_viz = risk_bars + risk_text

    # Odds ratio and relative risk comparison
    comparison_data = pd.DataFrame({
        'Metric': ['Odds Ratio', 'Relative Risk'],
        'Value': [odds_ratio, relative_risk],
        'Reference_Line': [1.0, 1.0]  # Reference line at 1
    })

    comparison_bars = alt.Chart(comparison_data).mark_bar(width=60).encode(
        x=alt.X('Metric:O', title='Statistical Measure'),
        y=alt.Y('Value:Q', title='Ratio Value', scale=alt.Scale(domain=[0, max(odds_ratio, relative_risk) * 1.1])),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=['Odds Ratio', 'Relative Risk'], 
                                                   range=['#FF9999', '#99CCFF'])),
        tooltip=['Metric', 'Value:Q']
    ).properties(
        title='Odds Ratio vs Relative Risk',
        width=250,
        height=250
    )

    # Reference line at 1.0
    ref_line = alt.Chart(comparison_data).mark_rule(strokeDash=[5, 5], color='black').encode(
        y=alt.datum(1.0)
    )

    # Add value labels
    comparison_text = alt.Chart(comparison_data).mark_text(
        align='center',
        baseline='bottom',
        fontSize=12,
        fontWeight='bold'
    ).encode(
        x='Metric:O',
        y='Value:Q',
        text=alt.Text('Value:Q', format='.2f'),
        color=alt.value('black')
    )

    comparison_viz = comparison_bars + ref_line + comparison_text

    mo.hstack([risk_viz, comparison_viz])
    return


@app.cell
def _(
    alt,
    blind_risk,
    direct_risk,
    mo,
    np,
    odds_ratio,
    pd,
    relative_risk,
    tabs,
):
    mo.stop(tabs.value != "Laryngoscopy (Odds)")

    # Educational visualization showing the relationship between OR and RR
    # Create a range of baseline risks to show how OR and RR relate
    baseline_risks = np.linspace(0.01, 0.5, 50)

    # For a fixed odds ratio, calculate what the relative risk would be at different baseline risks
    or_fixed = odds_ratio  # Current odds ratio from our data
    rr_values = []

    for baseline_risk in baseline_risks:
        # Convert baseline risk to odds
        baseline_odds = baseline_risk / (1 - baseline_risk)

        # Apply our odds ratio
        new_odds = baseline_odds * or_fixed

        # Convert back to risk
        new_risk = new_odds / (1 + new_odds)

        # Calculate relative risk
        rr = new_risk / baseline_risk
        rr_values.append(rr)

    relationship_data = pd.DataFrame({
        'Baseline_Risk': baseline_risks * 100,  # Convert to percentage
        'Relative_Risk': rr_values,
        'Odds_Ratio': [or_fixed] * len(baseline_risks)
    })

    # Plot showing relationship between OR and RR
    rr_line = alt.Chart(relationship_data).mark_line(color='blue', strokeWidth=3).encode(
        x=alt.X('Baseline_Risk:Q', title='Baseline Risk (%)'),
        y=alt.Y('Relative_Risk:Q', title='Relative Risk'),
        tooltip=['Baseline_Risk:Q', 'Relative_Risk:Q']
    )

    or_line = alt.Chart(relationship_data).mark_line(color='red', strokeDash=[5, 5], strokeWidth=2).encode(
        x='Baseline_Risk:Q',
        y='Odds_Ratio:Q',
        tooltip=['Baseline_Risk:Q', 'Odds_Ratio:Q']
    )

    # Current study point
    current_baseline = min(direct_risk, blind_risk) * 100
    current_point = alt.Chart(pd.DataFrame({
        'x': [current_baseline],
        'y': [relative_risk],
        'label': ['Our Study']
    })).mark_circle(size=100, color='orange').encode(
        x='x:Q',
        y='y:Q',
        tooltip='label:N'
    )

    relationship_viz = (rr_line + or_line + current_point).properties(
        title=f'OR vs RR Relationship (OR = {or_fixed:.2f})',
        width=500,
        height=300
    ).resolve_scale(y='independent')

    explanation = mo.md(f"""
    **Key Insights:**

    1. **When is OR ≈ RR?** When the baseline risk is low (rare disease assumption)
    2. **Our study:** Baseline risk = {min(direct_risk, blind_risk):.1%}, so OR ({odds_ratio:.2f}) {'≈' if abs(odds_ratio - relative_risk) < 0.5 else '≠'} RR ({relative_risk:.2f})
    3. **Clinical interpretation:** The {'blindfolded' if odds_ratio < 1 else 'direct laryngoscopy'} technique has {'lower' if odds_ratio < 1 else 'higher'} odds of death

    **The blue line shows how RR changes with baseline risk for a fixed OR of {or_fixed:.2f}**
    """)

    mo.vstack([relationship_viz, explanation])
    return


if __name__ == "__main__":
    app.run()
