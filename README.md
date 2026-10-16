# CellStates

## Project question

Can we use a monocyte's gene expression to predict whether it comes from a COVID-19 patient
or a healthy donor, when the donor was not seen during training?

This class project uses single-cell RNA sequencing data to explore that question. We build an
XGBoost classifier, evaluate it on held-out donors, and use gene contributions to examine
individual predictions.

## Data

We use the processed, subsampled PBMC dataset from
[Stephenson et al. (2021)](https://doi.org/10.1038/s41591-021-01329-2).
We select CD14-positive monocytes from Newcastle donors with either a COVID-19 or healthy label.
Keeping one cell type and one collection site makes the comparison more focused and reduces
differences due to cell composition and collection site.

The selected data contain **5,934 cells from 54 donor groups**, with 16,299 genes before feature
filtering. There are 5,134 cells from COVID-19 donors and 800 from healthy donors. We retain the
processed file's normalized, log-transformed expression values rather than normalizing them again.
The target is the donor's recorded clinical condition; the model uses gene expression as input.

## What we did

The analysis is in [pipeline.ipynb](pipeline.ipynb).

1. Inspect the processed data and select the cells and donors for our question.
2. Split donors into training and test sets, keeping every cell from a donor in the same set.
   This gives 43 training donors (4,340 cells) and 11 test donors (1,594 cells).
3. Remove constant genes within the fitted pipeline and train an XGBoost classifier. Use balanced
   sample weights to account for the unequal numbers of cells in the two classes.
4. Tune the number of trees, learning rate, and tree depth over eight parameter combinations
   using five-fold cross-validation within the training donors. Each fold also keeps donors
   separate, and gene filtering and class weights are fitted using only that fold's training data.
5. Select the parameters with the highest mean validation balanced accuracy, refit on the
   training set, and evaluate on the held-out test donors. Compare with a majority-class baseline
   and inspect performance for individual donors.

The selected model uses 200 trees, a learning rate of 0.1, and a maximum tree depth of 2.

## Results

The selected model's mean cross-validation balanced accuracy was **92.7%**.
On the held-out test donors, the cell-level results were:

| Metric | Majority-class baseline | XGBoost |
| --- | ---: | ---: |
| Balanced accuracy | 50.0% | 85.2% |
| ROC AUC | 0.500 | 0.959 |
| Matthews correlation coefficient | 0.000 | 0.579 |

We emphasize balanced accuracy because the classes are unequal. These results come from one
held-out donor split, with only two healthy test donors. Cells from the same donor are related,
so the number of cells should not be treated as the number of independent samples. Performance
also varies between donors, and the results do not establish generalization to other sites or studies.

## Interactive demonstration

The Streamlit app in [app.py](app.py) lets users select a held-out donor and predict the expression
profile of each cell using the saved model. Users can inspect predicted probabilities, reveal the
recorded donor condition, and explore the genes contributing to an individual prediction.

The explanations use local SHAP values calculated directly by XGBoost. Positive contributions
push the prediction toward COVID-19 and negative contributions toward healthy, on a log-odds
scale. These describe the model's predictions and do not establish that a gene causes disease.

The saved model is in `models/`, and the example cells are in `examples/`. All example donors
come from the actual test set and were excluded from training and parameter tuning.
The predictions describe donor-condition expression profiles; they do not identify whether an
individual cell is infected.
