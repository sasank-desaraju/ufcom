import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import datetime
    from datetime import datetime
    import altair as alt
    return alt, datetime, duckdb, mo, pl


@app.cell
def _(duckdb):
    DATABASE_URL = "notebooks/public/mimic3-demo.duckdb"
    engine = duckdb.connect(DATABASE_URL, read_only=True)
    return (engine,)


@app.cell
def _(mo):
    mo.md(f"""
    # MIMIC-III Demo: Exploring EHR Data with Marimo

    Welcome to an interactive exploration of the MIMIC-III demo dataset using Marimo.

    - Instantly browse and filter real EHR data
    - Build & explore patient cohorts
    - Visualize results with interactive charts and tables

    ---

    Let's get started by viewing the raw data!
    """)
    return


@app.cell
def _(mo):
    # Step 1: Interactive exploration of tables

    table_options = [
        "PATIENTS", "ADMISSIONS", "ICUSTAYS", "DIAGNOSES_ICD", "PROCEDURES_ICD", "DRGCODES", "LABEVENTS", "PRESCRIPTIONS"
    ]
    table_dropdown = mo.ui.dropdown(table_options, value="PATIENTS", label="Choose a table to explore")
    table_dropdown
    return (table_dropdown,)


@app.cell
def _(engine, mo, table_dropdown):
    # Display the selected table interactively
    selected_table = table_dropdown.value
    explore_df = mo.sql(f"""SELECT * FROM {selected_table} LIMIT 100""", engine=engine)
    mo.ui.data_explorer(explore_df)  # Interactive browser
    return


@app.cell
def _(engine, mo, patients):
    # Step 2: Simple Cohort Builder UI
    # We'll let users filter the PATIENTS table by gender and age

    # Fetch a sample from PATIENTS
    patients_sample = mo.sql(f'''SELECT * FROM PATIENTS''', engine=engine)
    # Compute min and max birth year for slider
    birth_years = patients_sample['DOB'].dt.year()
    min_year = int(birth_years.min())
    max_year = int(birth_years.max())

    # Gender dropdown
    gender_dropdown = mo.ui.dropdown(['M', 'F'], value=None, label='Gender')
    # Birth year range slider
    year_slider = mo.ui.range_slider(min_year, max_year, value=(min_year, max_year), label='Year of Birth Range')
    mo.hstack([gender_dropdown, year_slider])
    return gender_dropdown, patients_sample, year_slider


@app.cell
def _(gender_dropdown, mo, patients_sample, pl, year_slider):
    # Filter the PATIENTS table using UI
    # Wait for user selection and filter accordingly, showing a filtered cohort

    selected_gender = gender_dropdown.value
    selected_years = year_slider.value

    cohort_df = patients_sample
    if selected_gender is not None:
        cohort_df = cohort_df.filter(pl.col('GENDER') == selected_gender)
    cohort_df = cohort_df.filter(
        (pl.col('DOB').dt.year() >= selected_years[0]) &
        (pl.col('DOB').dt.year() <= selected_years[1])
    )

    mo.ui.table(cohort_df)
    return (cohort_df,)


@app.cell
def _(cohort_df, datetime, mo):
    # Step 3: Summary Stats & Visualizations

    # Compute age from DOB
    now = datetime.now()
    age_years = (now.year - cohort_df['DOB'].dt.year()).alias('AGE')
    cohort_df_with_age = cohort_df.with_columns(age_years)

    # Summary statistics
    num_patients = cohort_df_with_age.shape[0]
    gender_counts = cohort_df_with_age.group_by('GENDER').len().to_dict(as_series=False)
    avg_age = float(cohort_df_with_age['AGE'].mean()) if num_patients > 0 else None

    mo.md(f"""
    ### Summary Statistics (Cohort)
    - **N patients**: {num_patients}
    - **Average Age**: {avg_age:.1f} years
    - **Gender split:** M: {gender_counts.get('count', [0,0])[0]}, F: {gender_counts.get('count', [0,0])[1]}
    """)
    return (cohort_df_with_age,)


@app.cell
def _(alt, cohort_df_with_age, mo):
    # Age distribution histogram
    age_hist = alt.Chart(cohort_df_with_age).mark_bar(color='steelblue').encode(
        alt.X('AGE:Q', bin=alt.Bin(maxbins=30), title='Age'),
        alt.Y('count()', title='Number of patients'),
        tooltip=['AGE', 'count()']
    ).properties(
        width=400, height=300, title='Cohort Age Distribution'
    )

    mo.ui.altair_chart(age_hist)
    return


@app.cell
def _(mo, patients_sample):
    # Step 4: Patient Drill-down UI
    # Search by SUBJECT_ID and show this patient's admissions, diagnoses, and procedures
    subject_ids = patients_sample['SUBJECT_ID'].unique().sort()
    patient_id_dropdown = mo.ui.dropdown(subject_ids.to_list(), value=None, label="Search by SUBJECT_ID")
    patient_id_dropdown
    return (patient_id_dropdown,)


@app.cell
def _(diagnoses_icd, engine, mo, patient_id_dropdown, procedures_icd):
    # Show all of this patient's admissions, diagnoses, procedures when an ID is selected
    selected_patient_id = patient_id_dropdown.value

    if selected_patient_id is not None:
        admissions = mo.sql(f"SELECT * FROM ADMISSIONS WHERE SUBJECT_ID = {selected_patient_id}", engine=engine)
        diagnoses = mo.sql(f"SELECT * FROM DIAGNOSES_ICD WHERE SUBJECT_ID = {selected_patient_id}", engine=engine)
        procedures = mo.sql(f"SELECT * FROM PROCEDURES_ICD WHERE SUBJECT_ID = {selected_patient_id}", engine=engine)
    
        mo.md(f"""### Patient {selected_patient_id} Drilldown
        **Admissions:**
        """)
        mo.ui.table(admissions)
        mo.md('**Diagnoses:**')
        mo.ui.table(diagnoses)
        mo.md('**Procedures:**')
        mo.ui.table(procedures)
    else:
        mo.md('Select a patient to see details.')
    return (admissions,)


@app.cell
def _(mo):
    # Step 5: Outcome Analysis
    # Group outcomes by gender or admission type, show LOS distribution and mortality

    # Options for grouping
    group_options = ["None", "GENDER", "ADMISSION_TYPE"]
    group_dropdown = mo.ui.dropdown(group_options, value="None", label="Group outcomes by:")
    group_dropdown
    return (group_dropdown,)


@app.cell
def _(admissions, alt, cohort_df, engine, group_dropdown, mo, patients, pl):
    # Use UI to select outcome variable and draw length-of-stay & mortality
    selected_group = group_dropdown.value

    # Join admissions with cohort for this analysis
    admissions_joined = mo.sql(f'''
        SELECT 
            p.SUBJECT_ID, 
            a.ADMISSION_TYPE, 
            a.HOSPITAL_EXPIRE_FLAG, 
            date_diff('day', a.ADMITTIME, a.DISCHTIME) AS LOS, 
            p.GENDER
        FROM ADMISSIONS a
        JOIN PATIENTS p ON a.SUBJECT_ID = p.SUBJECT_ID
    ''', engine=engine)

    # Filter only patients in the current cohort
    cohort_ids = cohort_df['SUBJECT_ID'].to_list()
    adm_cohort = admissions_joined.filter(pl.col('SUBJECT_ID').is_in(cohort_ids))

    # Group as needed and compute summary per group
    group_col = selected_group if selected_group != 'None' else None
    if group_col:
        los_box = alt.Chart(adm_cohort).mark_boxplot(extent='min-max').encode(
            x=alt.X(f'{group_col}:N', title=group_col.capitalize()),
            y=alt.Y('LOS:Q', title='Length of Stay (days)'),
            color=alt.Color(f'{group_col}:N'),
            tooltip=[group_col, 'LOS']
        ).properties(width=300, height=300, title='LOS by ' + group_col.capitalize())
    
        death_bar = alt.Chart(adm_cohort).mark_bar().encode(
            x=alt.X(f'{group_col}:N', title=group_col.capitalize()),
            y=alt.Y('mean(HOSPITAL_EXPIRE_FLAG):Q', title='In-hospital Mortality Rate'),
            color=alt.Color(f'{group_col}:N'),
            tooltip=[group_col, alt.Tooltip('mean(HOSPITAL_EXPIRE_FLAG):Q', title='Mortality Rate', format='.2%')]
        ).properties(width=300, height=300, title='Mortality Rate by ' + group_col.capitalize())
        mo.tabs({
            'LOS': mo.ui.altair_chart(los_box),
            'Mortality': mo.ui.altair_chart(death_bar)
        })
    else:
        los_box = alt.Chart(adm_cohort).mark_boxplot(extent='min-max').encode(
            y=alt.Y('LOS:Q', title='Length of Stay (days)'),
            tooltip=['LOS']
        ).properties(width=200, height=300, title='LOS (All Cohort)')
        death_rate = float(adm_cohort['HOSPITAL_EXPIRE_FLAG'].mean()) if adm_cohort.shape[0] > 0 else None
        death_text = mo.md(f"**Cohort mortality rate:** {death_rate:.2%}" if death_rate is not None else "No data")
        mo.vstack([mo.ui.altair_chart(los_box), death_text])
    return


@app.cell
def _(mo):
    # Step 6: Live SQL Query Interface
    # Let the user write and run raw SQL and show result interactively!

    user_query = mo.ui.text_area(label="Enter a SQL query to run on the MIMIC-III database", value="SELECT * FROM PATIENTS LIMIT 10")
    run_query_btn = mo.ui.run_button(label="Run Query")

    mo.vstack([user_query, run_query_btn])
    return


@app.cell
def _(admissions, diagnoses_icd, engine, mo):
    # Step 7: What-if Analysis: Advanced Cohort Filters
    # Include additional filters: diagnosis code & admission type

    diag_codes = mo.sql(f'SELECT DISTINCT ICD9_CODE FROM DIAGNOSES_ICD ORDER BY ICD9_CODE LIMIT 50', engine=engine)['ICD9_CODE'].to_list()
    diag_dropdown = mo.ui.dropdown(diag_codes, value=None, label='Filter: ICD9 Diagnosis Code (any)')
    adm_types = mo.sql(f'SELECT DISTINCT ADMISSION_TYPE FROM ADMISSIONS ORDER BY ADMISSION_TYPE', engine=engine)['ADMISSION_TYPE'].to_list()
    adm_type_dropdown = mo.ui.dropdown(['Any'] + adm_types, value='Any', label='Admission Type')
    los_slider = mo.ui.range_slider(0, 60, value=(0, 60), label='Length of Stay (days)')

    mo.hstack([diag_dropdown, adm_type_dropdown, los_slider])
    return adm_type_dropdown, diag_dropdown, los_slider


@app.cell
def _(
    adm_type_dropdown,
    admissions,
    cohort_df,
    diag_dropdown,
    diagnoses_icd,
    engine,
    los_slider,
    mo,
    pl,
):
    # Apply the advanced filters to the current cohort
    # Build cohort using gender/year + diag code + admission type + LOS
    selected_diag = diag_dropdown.value
    selected_adm_type = adm_type_dropdown.value
    selected_los = los_slider.value

    # Start with previously filtered cohort
    whatif_cohort = cohort_df

    if selected_diag is not None:
        # Keep people with any matching diagnosis. Join with DIAGNOSES_ICD.
        diag_ids = mo.sql(f"SELECT DISTINCT SUBJECT_ID FROM DIAGNOSES_ICD WHERE ICD9_CODE = '{selected_diag}'", engine=engine)['SUBJECT_ID'].to_list()
        whatif_cohort = whatif_cohort.filter(pl.col('SUBJECT_ID').is_in(diag_ids))
    if selected_adm_type != 'Any':
        adm_ids = mo.sql(f"SELECT DISTINCT SUBJECT_ID FROM ADMISSIONS WHERE ADMISSION_TYPE = '{selected_adm_type}'", engine=engine)['SUBJECT_ID'].to_list()
        whatif_cohort = whatif_cohort.filter(pl.col('SUBJECT_ID').is_in(adm_ids))
    if selected_los:
        # Only keep if any of their admissions falls in the LOS window (compute LOS from ADMITTIME/DISCHTIME)
        los_ids = mo.sql(
            f"""
            SELECT DISTINCT SUBJECT_ID
            FROM ADMISSIONS
            WHERE date_diff('day', ADMITTIME, DISCHTIME) >= {int(selected_los[0])}
              AND date_diff('day', ADMITTIME, DISCHTIME) <= {int(selected_los[1])}
            """,
            engine=engine
        )['SUBJECT_ID'].to_list()
        whatif_cohort = whatif_cohort.filter(pl.col('SUBJECT_ID').is_in(los_ids))

    mo.ui.table(whatif_cohort)
    return


@app.cell
def _(mo):
    mo.md(f"""
    # Step 8: Save & Share / Export

    You can export any cohort or analysis result as a CSV using Polars' or Marimo's built-in functionality. For example, click the 3-dot menu on any table in the UI to download.

    - **Export Table as CSV:** Use the table UI menu or Polars `.write_csv()`
    - **Share Notebook:** Marimo notebooks can be saved and versioned for reproducibility—share with classmates or instructors.
    - **Re-run Analysis Anytime:** All UI selections and results are reproducible!

    _This illustrates end-to-end reproducibility and shareability for clinical data analyses in Marimo._
    """)
    return


if __name__ == "__main__":
    app.run()
