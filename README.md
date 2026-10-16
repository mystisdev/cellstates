# cellstates

A class project predicting whether CD14-positive monocyte expression resembles cells from
COVID-19 patients or healthy donors in the Newcastle subset of Stephenson et al. (2021).

## Run the presentation app

Use Python 3.12. From this project directory, create an environment and install the app dependencies:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Start Streamlit:

```bash
.venv/bin/python -m streamlit run app.py
```

If you already have the project environment, skip `uv venv` and just install the requirements.

Open **http://localhost:8501** if the browser doesn't open automatically. Keep the terminal
running during the presentation; press **Ctrl+C** there to stop the app.

Choose a donor, click **Predict**, explore the cell predictions, and optionally reveal the donor's
recorded condition. **Download predictions as CSV** exports full cell IDs, donor ID, predicted
condition, and COVID-19 probability on a 0–1 scale. The on-screen probabilities are percentages.

After predicting, **Why this prediction?** lets you choose an individual cell. It shows the ten
largest signed gene contributions for that cell: positive contributions push the model toward
COVID-19 and negative ones toward healthy. These are XGBoost's native SHAP contributions in
log-odds, not probability percentage points or causal effects. Contributions from all genes plus
the reference score reconstruct the prediction; the chart displays only the largest ten.
Explanations are cached for each cell and don't require retraining or an extra SHAP package.

The app runs the saved model on expression data when you click Predict. It doesn't run training,
display stored test scores, or accept uploads. Predictions refer to donor-condition profiles in
this study and aren't a diagnosis or a test of individual cell infection.

## Files used by the app

- `app.py` contains the interface, model loading, and prediction code.
- `.streamlit/config.toml` sets the colors for the interface.
- `examples/heldout_cells.h5ad` contains all 1,594 cells from the 11 actual test donors.
- `models/xgboost.pkl` contains the fitted pipeline, gene order, and label meanings.
- `requirements.txt` records the app dependencies, including the model's training-library versions.

The example cells retain their downloaded expression values and use the model's saved input-gene
order. Their IDs come from the saved `test_predictions.csv`; no new split is made. Model inputs
contain only expression, while donor information is used to select examples and reveal the known condition.

The saved model and example cells are included in the repository. The full source dataset and
notebooks are not needed to run the app. Keep the pinned library versions when loading the
saved model.

## Training and exporting examples

Install the additional notebook dependencies into the same environment:

```bash
uv pip install --python .venv/bin/python -r requirements-notebook.txt
```

Download `stephenson_2021_subsampled.h5ad` using the link in the notebook's **Load the Data**
section and put it in `data/`. Run the notebook from the project root using the `.venv` kernel.
The source dataset is already in that folder locally; it is not included in GitHub.

`pipeline.ipynb` prepares the data, tunes XGBoost using five donor-grouped folds, evaluates it on
held-out donors, and saves the model. The **Export Examples for the App** cell exports those same
test cells to `examples/heldout_cells.h5ad`. Run the notebook through the model-saving cell before
exporting examples. The example file is already prepared, so you don't need to rerun the notebook
to use the app. If you replace the saved model or examples, restart Streamlit to reload them.

The notebook saves the fitted model to `models/xgboost.pkl` and evaluation tables to
`outputs/covid_newcastle_xgboost/`. Earlier notebooks, the old reference notebook, and melanoma
files are preserved locally in `archive/`. The `data/`, `archive/`, `outputs/`, and `.venv/`
folders are ignored by Git.

## Publish the app

Commit the app, notebook, dependency files, theme, saved model, and example cells to GitHub.
No training step is needed during deployment. From the project root:

```bash
git status --short
git add -A
git commit -m "Prepare Streamlit demo for deployment"
git push origin main
```

The two old tracked notebooks will appear as deletions in Git. Their local copies remain in
`archive/`, and previous commits retain their history.

In [Streamlit Community Cloud](https://share.streamlit.io/), connect your GitHub account and
choose **Create app**. Use these settings for this repository:

- Repository: `mystisdev/cellstates`
- Branch: `main`
- Main file path: `app.py`
- Advanced settings → Python version: `3.12`

Leave secrets empty; this app does not use API keys or external services. Choose an optional
app URL and click **Deploy**. Community Cloud installs `requirements.txt`; the separate
`requirements-notebook.txt` is only for running the training notebook locally. Later pushes
to the deployed branch update the app.

See the [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
If deployment fails, check the build logs first. A successful local check does not verify the
Linux environment used by Community Cloud.

Dataset: [Stephenson et al., Single-cell multi-omics analysis of the immune response in COVID-19](https://doi.org/10.1038/s41591-021-01329-2).
