# Book Genres Classification

Multilabel book-genre classification using each book's description. The project extracts the top-level genres from hierarchical topic labels and compares convolutional and recurrent neural network models implemented with PyTorch.

## Project contents

```text
Book Genres Classification/
├── bgc_train.csv
├── bgc_dev.csv
├── bgc_test.csv
├── train.ipynb
└── README.md
```

## Dataset

The data was obtained from the [Blurb Genre Collection](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html) provided by the University of Hamburg.

The dataset contains book metadata with the following fields:

- `Number`: book identifier
- `Title`: book title
- `Body`: book description
- `Topics` / `Labels`: hierarchical genre labels separated by `>`

The notebook uses the following predefined splits:

| Split | Examples |
|---|---:|
| Training | 58,715 |
| Validation (`bgc_dev.csv`) | 14,785 |
| Test | 18,394 |
| **Total** | **91,894** |

The test file calls the label column `Labels`, so it is renamed to `Topics` before the datasets are combined.

## Objective

The original labels are hierarchical, for example:

```text
Fiction>Historical Fiction
```

The notebook keeps only the first level of each hierarchy, producing these seven target labels:

- Children's Books
- Classics
- Fiction
- Humor
- Nonfiction
- Poetry
- Teen & Young Adult

The task is multilabel classification because a book can belong to more than one top-level genre. The rare `Crafts` label was removed because it had very few examples and was not present in the training data.

## Workflow

The notebook implements the following pipeline:

1. Mount Google Drive in Google Colab.
2. Load the training, validation, and test CSV files with pandas.
3. Standardize the test label-column name.
4. Combine the splits temporarily for exploratory analysis.
5. Check missing values and duplicate rows.
6. Extract top-level genres by splitting hierarchical labels at `>`.
7. Remove the rare `Crafts` label.
8. Convert multilabel genres into binary vectors using `MultiLabelBinarizer`.
9. Tokenize each book description using whitespace splitting.
10. Inspect description lengths. Descriptions contain 1 to 2,675 tokens, with a mean of about 157 and a median of 142.
11. Build a vocabulary from the training descriptions only.
12. Keep the 50,000 most frequent training tokens and add:
    - `<PAD>` → `0`
    - `<UNK>` → `1`
13. Encode tokens as integer indices, mapping unseen words to `<UNK>`.
14. Pad batches dynamically and truncate sequences to a maximum of 500 tokens.
15. Create a custom `BGCDataset` and PyTorch `DataLoader` objects.
16. Train four neural network architectures.
17. Save the model state with the lowest validation loss.
18. Evaluate the best checkpoint on the held-out test set.

## Models

### CNN

The CNN uses:

- A learned embedding layer with dimension `100`
- Three parallel one-dimensional convolutions with kernel sizes `2`, `3`, and `4`
- ReLU activations
- Global max pooling for each convolution
- Concatenation of the pooled representations
- Dropout with probability `0.3`
- A fully connected multilabel output layer

### Vanilla RNN

The recurrent baseline uses:

- A learned embedding layer
- A one-layer `nn.RNN`
- Hidden size `128`
- Dropout with probability `0.3`
- A fully connected output layer

### LSTM

The LSTM has the same general configuration as the vanilla RNN, but uses gated cell and hidden states to preserve information across longer descriptions.

### GRU

The GRU also uses a hidden size of `128` and replaces the vanilla recurrent layer with `nn.GRU`.

For all recurrent models, `pack_padded_sequence` is used with the real sequence lengths so padded tokens are not processed as meaningful input.

## Training configuration

- Embedding dimension: `100`
- Maximum sequence length: `500`
- Batch size: `128`
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `1e-4`
- Dropout: `0.3`
- Loss: `BCEWithLogitsLoss`
- Device: CUDA (`torch.device("cuda")`)
- CNN epochs: `30`
- RNN, LSTM, and GRU epochs: `20`

## Handling Class imbalance

Genre frequencies are imbalanced, so weighted loss computation was used to give greater importance to underrepresented genres during training. Per-label positive weights are calculated from the training split using:

```python
pos_weight = sqrt(negative_examples / positive_examples)
```

These weights are passed to `BCEWithLogitsLoss` for every model. The best validation-loss checkpoint is kept with `copy.deepcopy(model.state_dict())` and restored before test evaluation.

## Results

The following results are the metrics recorded in the notebook after evaluating the best checkpoint on the test set. Accuracy and F1 scores are multilabel metrics from `torchmetrics`.

| Model | Test loss | Accuracy | Macro F1 | Micro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| CNN | 0.2063 | **0.9553** | **0.6658** | **0.8492** | 0.8548 |
| Vanilla RNN | 0.4352 | 0.8948 | 0.4199 | 0.6547 | 0.6440 |
| LSTM | 0.2078 | 0.9526 | 0.6456 | 0.8420 | **0.8578** |
| GRU | 0.2141 | 0.9502 | 0.6275 | 0.8366 | 0.8541 |

The CNN achieved the highest multilabel accuracy, macro F1, and micro F1 in this experiment. The LSTM achieved the highest weighted F1. The vanilla RNN was the weakest model, while the CNN and gated recurrent models handled the book descriptions substantially better.

The notebook also computes a separate confusion matrix for each of the seven genres.

## Running the notebook

### Google Colab

1. Upload `train.ipynb` and the three CSV files to Google Drive.
2. Update the path in the data-loading cell if necessary:

   ```python
   train_df = pd.read_csv("/content/drive/My Drive/Book_Genres/bgc_train.csv")
   test_df = pd.read_csv("/content/drive/My Drive/Book_Genres/bgc_test.csv")
   val_df = pd.read_csv("/content/drive/My Drive/Book_Genres/bgc_dev.csv")
   ```

3. Run the notebook from top to bottom.
4. Select a CUDA runtime because the notebook explicitly selects a CUDA device.

### Local Jupyter environment

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchmetrics jupyter
```

Replace the Google Drive paths with local paths, then launch the notebook:

```bash
jupyter notebook train.ipynb
```

## Important implementation details

- The vocabulary is built from training descriptions only, preventing vocabulary leakage from validation and test data.
- Unknown validation and test words are mapped to `<UNK>`.
- `<PAD>` is used by the embedding layer as its padding index.
- CNN batches are padded to the longest sequence in each batch after truncation to 500 tokens.
- Recurrent batches retain sequence lengths and use packed sequences.
- `BCEWithLogitsLoss` is used because this is a multilabel, rather than single-label, classification problem.
- Validation loss, not training loss, determines which model checkpoint is evaluated.
- The test set is evaluated only after restoring the best validation checkpoint.
