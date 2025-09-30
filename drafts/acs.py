# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic==0.68.1",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ACS Case and Stats Concepts

    Let's discuss the Acute Coronary Syndrome (ACS) case and how it relates to the statistics concepts on your board exams.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Statistics Concepts

    Here are the statistical concepts on your boards along with the ones that we'll discuss in this case.

    - [ ] 1. Odds ratio
    - [ ] 2. Relative risk
    - [ ] 3. Number needed to treat
    - [ ] 4. Sensitivity, specificity
    - [ ] 5. ROC
    - [ ] 6. Sample size estimate
    - [ ] 7. Regression analysis
    - [ ] 8. Statistical tests: Non-parametric (e.g.; Wilcoxon, Mann Whitney U test, Chi squared, Kaplan Meier curve), Parametric (e.g.; t-test, ANOVA)
    - [ ] 9. Type 1 and type 2 errors
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    > "Odds" versus "risk":
    >
    > The terms "odds" and "risk" sound similar, but the difference is what is used as the denominator.
    > For a risk, the denominator is all things that could happen. It is a probability. For odds, the denominator is only the other event that could have happened. One consequence of this is that odds can be greater than 1 whereas risk (a probability) can never exceed 1.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
