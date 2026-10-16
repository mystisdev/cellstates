from pathlib import Path
import pickle

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import DMatrix


# Use paths relative to this script, so the app can find its files from any working directory
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models/xgboost.pkl"
EXAMPLE_PATH = PROJECT_DIR / "examples/heldout_cells.h5ad"

st.set_page_config(page_title="CellStates | Monocyte Explorer")


@st.cache_resource
def load_demo():
    """Load the trained pipeline and held-out cells once when the app starts."""
    with MODEL_PATH.open("rb") as f:
        model_bundle = pickle.load(f)
    examples = ad.read_h5ad(EXAMPLE_PATH)
    return model_bundle, examples


model_bundle, examples = load_demo()
model = model_bundle["pipeline"]
donor_counts = examples.obs.groupby("donor_id", observed=True).size()


@st.cache_data
def explain_cell(cell_id):
    """Get the gene contributions for one cell using the fitted XGBoost model."""
    # Select the cell and apply the same fitted gene filter used for prediction
    cell = examples[[cell_id], model_bundle["input_genes"]]
    cell_expression = model.named_steps["constant_genes"].transform(cell.X)

    # XGBoost returns one SHAP contribution per retained gene, followed by a reference score.
    # Contributions plus the reference score add up to the prediction in log-odds.
    contributions = model.named_steps["xgb"].get_booster().predict(
        DMatrix(cell_expression), pred_contribs=True
    )[0]
    gene_scores = pd.Series(
        contributions[:-1], index=model_bundle["selected_genes"], name="Contribution"
    )
    return gene_scores, float(contributions[-1])


st.caption("CELLSTATES · SINGLE-CELL DEMO")
st.title("Explore monocyte profiles")
st.write(
    "Choose a held-out donor and use the model to classify each cell's expression "
    "as resembling a healthy or COVID-19 donor profile."
)
st.caption("Newcastle · CD14-positive monocytes · Built-in examples")

# Select a donor without showing the recorded condition before prediction
with st.container(border=True):
    st.subheader("Try the model")
    selection, action = st.columns([3, 1], vertical_alignment="bottom")
    donor = selection.selectbox(
        "Choose a donor",
        options=sorted(donor_counts.index),
        format_func=lambda name: f"{name} · {donor_counts[name]} cells",
        key="donor",
    )
    predict_clicked = action.button(
        "Predict", type="primary", width="stretch", key="predict"
    )
    st.caption(
        "These donors were held out during training. Each prediction uses the saved model; "
        "the app doesn't retrain it."
    )

if predict_clicked:
    # Select this donor's cells and arrange the genes in the order used during training
    donor_cells = examples[
        examples.obs["donor_id"] == donor, model_bundle["input_genes"]
    ]

    # Predict from expression only. The recorded donor condition isn't a model feature.
    with st.spinner("Predicting cell profiles..."):
        labels = model.predict(donor_cells.X)
        covid_column = np.flatnonzero(model.classes_ == 1)[0]
        probabilities = model.predict_proba(donor_cells.X)[:, covid_column]

    predictions = pd.DataFrame({
        "cell_id": donor_cells.obs_names,
        "donor_id": donor,
        "predicted_condition": pd.Series(labels).map(model_bundle["labels"]).to_numpy(),
        "covid_probability": probabilities,
    })

    # Keep the results when Streamlit reruns after a checkbox or download interaction
    st.session_state["predictions"] = predictions
    st.session_state["predicted_donor"] = donor

# Only show results for the donor currently selected, so switching donors doesn't show old results
if st.session_state.get("predicted_donor") == donor:
    predictions = st.session_state["predictions"]
    n_covid = int((predictions["predicted_condition"] == "COVID-19").sum())
    n_healthy = len(predictions) - n_covid

    st.subheader("Cell predictions")
    st.write(
        "These predictions come from our trained XGBoost model, using each cell's normalized "
        "gene expression. The COVID-19 probability is the model's estimate that the cell came "
        "from a COVID-19 donor in this study."
    )
    total, healthy, covid = st.columns(3)
    total.metric("Cells analyzed", len(predictions), border=True)
    healthy.metric("Healthy-like", n_healthy, border=True)
    covid.metric("COVID-19-like", n_covid, border=True)

    # Plot one probability per cell, colored by the predicted class
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor("#F7F9F8")
    ax.set_facecolor("#F7F9F8")
    bins = np.linspace(0, 1, 21)
    for condition, color in [("Healthy", "#7198B8"), ("COVID-19", "#0F766E")]:
        values = predictions.loc[
            predictions["predicted_condition"] == condition, "covid_probability"
        ]
        ax.hist(values, bins=bins, color=color, label=f"{condition}-like", alpha=0.9)
    ax.axvline(0.5, color="#5B6663", linestyle="--", linewidth=1)
    ax.set(xlabel="COVID-19 probability", ylabel="Number of cells", xlim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Each cell contributes once. A probability above 0.5 (50%) gives a COVID-19-like label; "
        "otherwise the label is healthy-like. The dashed line marks this threshold. "
        "These probabilities are model estimates and have not been calibrated."
    )

    # Use short row numbers in the display; the downloaded CSV keeps the full cell IDs
    table = pd.DataFrame({
        "Cell": np.arange(1, len(predictions) + 1),
        "Prediction": predictions["predicted_condition"],
        "COVID-19 probability": predictions["covid_probability"] * 100,
    })
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "COVID-19 probability": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%",
                help="XGBoost's estimated probability of COVID-19 donor condition, shown as a percentage.",
            ),
        },
    )
    st.download_button(
        "Download predictions as CSV",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{donor}_predictions.csv",
        mime="text/csv",
        on_click="ignore",
        key="download",
    )

    with st.container(border=True):
        st.subheader("Why this prediction?")
        st.write(
            "Choose a cell to see which genes moved its prediction toward either profile."
        )
        st.caption(
            "These are local SHAP values, calculated directly by XGBoost for the selected cell. "
            "Larger absolute values mean a larger contribution to this particular prediction."
        )
        cell_row = st.selectbox(
            "Choose a cell",
            options=range(len(predictions)),
            format_func=lambda row: (
                f"Cell {row + 1} · {predictions.iloc[row]['predicted_condition']}-like "
                f"· {predictions.iloc[row]['covid_probability']:.1%} COVID-19 probability"
            ),
            key=f"cell_{donor}",
        )
        selected_cell = predictions.iloc[cell_row]

        # Explain the selected cell. We cache this result so returning to the same cell is quick.
        gene_scores, reference_score = explain_cell(selected_cell["cell_id"])

        # Select the ten largest contributions by magnitude, keeping their positive/negative signs
        top_genes = gene_scores.loc[gene_scores.abs().nlargest(10).index].sort_values()
        colors = np.where(top_genes > 0, "#0F766E", "#7198B8")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("#F7F9F8")
        ax.set_facecolor("#F7F9F8")
        ax.barh(top_genes.index, top_genes.to_numpy(), color=colors)
        ax.axvline(0, color="#5B6663", linewidth=1)
        ax.set(xlabel="Contribution to prediction (log-odds)", ylabel="Gene")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            handles=[
                Patch(color="#7198B8", label="Toward healthy"),
                Patch(color="#0F766E", label="Toward COVID-19"),
            ],
            loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False,
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Translate the reference score to a probability to put the explanation in context
        reference_probability = 1 / (1 + np.exp(-reference_score))
        st.caption(
            f"Reference model probability: {reference_probability:.1%}. "
            f"All gene contributions together produce this cell's "
            f"{selected_cell['covid_probability']:.1%} COVID-19 probability. "
            "Only the 10 largest contributions are shown."
        )
        st.caption(
            "Bars explain the model's score for this cell. They are in log-odds, not "
            "percentage-point changes in probability, and do not show that a gene causes disease."
        )
        with st.expander("How do these values give the probability?"):
            st.write(
                "The reference score is the starting point of the explanation. XGBoost adds "
                "each gene's SHAP value to that score, then converts the total from log-odds "
                "to a probability. Positive contributions move the score toward COVID-19; "
                "negative contributions move it toward healthy."
            )
            st.latex(r"z = b + \sum_j \phi_j, \qquad p = \frac{1}{1 + e^{-z}}")
            st.caption(
                "Here b is the reference score, each φ is one gene's SHAP contribution, "
                "and p is the COVID-19 probability. "
                "A total score z of 0 gives 50%; positive totals give more than 50%, and "
                "negative totals give less than 50%. All genes enter the total, even though "
                "the chart shows only ten."
            )
            st.markdown(
                "We use [XGBoost's built-in SHAP calculation]"
                "(https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict)."
            )

    # Let the audience compare the predictions with the recorded condition afterward
    if st.checkbox("Show the donor's recorded condition", key="show_condition"):
        condition = examples.obs.loc[examples.obs["donor_id"] == donor, "condition"].iloc[0]
        st.info(f"Recorded donor condition: {condition}")
else:
    st.info("Choose a donor and click Predict to explore their cell profiles.")

st.divider()
st.caption(
    "A class-project demonstration using "
    "[Stephenson et al. (2021)](https://doi.org/10.1038/s41591-021-01329-2) data. "
    "Predictions describe donor-condition profiles in this study, not individual cell infection "
    "or a clinical diagnosis."
)
