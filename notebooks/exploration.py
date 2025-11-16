import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import duckdb

    DATABASE_URL = "notebooks/public/mimic3-demo.duckdb"
    engine = duckdb.connect(DATABASE_URL, read_only=True)
    return


if __name__ == "__main__":
    app.run()
