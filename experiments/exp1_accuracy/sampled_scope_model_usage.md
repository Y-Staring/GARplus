# Sampled Scope Data Usage for TransE, RotatE, and NBFNet

This document explains how to use the sampled-scope datasets exported from GAR rule mining.

The key point is that GAR rules were mined on sampled subgraphs, so TransE, RotatE, and NBFNet should train/evaluate on the same sampled scope instead of the whole graph. Otherwise, many GAR rules will not match the model prediction pairs.

## Exported Dataset Paths

### DDA

```bash
/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/DDA_depth6_20260709_093322
```

Summary:

```text
sampled e0 pairs: 9273
positive: 5544
negative: 3729
generated no_edge: 5544
final triples: 14817
entities: 3525
```

### TI

```bash
/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/TI_processed_20260709_093517
```

Summary:

```text
sampled e0 pairs: 2740
positive: 1380
negative: 1360
generated no_edge: 1380
final triples: 4120
entities: 1990
```

## TransE and RotatE

TransE and RotatE should use the `openke` subdirectory.

DDA:

```bash
/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/DDA_depth6_20260709_093322/openke
```

TI:

```bash
/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/TI_processed_20260709_093517/openke
```

Each `openke` directory contains:

```text
entity2id.txt
relation2id.txt
train2id.txt
valid2id.txt
test2id.txt
```

Use this directory as the OpenKE input path. For example:

```bash
--in_path "/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/DDA_depth6_20260709_093322/openke"
```

For TI, replace the path with:

```bash
--in_path "/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/TI_processed_20260709_093517/openke"
```

## NBFNet

NBFNet should use the `nbfnet` subdirectory.

DDA:

```bash
/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/DDA_depth6_20260709_093322/nbfnet
```

TI:

```bash
/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/TI_processed_20260709_093517/nbfnet
```

Each `nbfnet` directory contains:

```text
train.txt
valid.txt
test.txt
node.csv
edge_update.csv
```

The `train.txt`, `valid.txt`, and `test.txt` files are tab-separated triples without a header:

```text
head_id    relation    tail_id
```

The relation labels are:

```text
0 = no_edge
1 = positive
2 = negative
```

## Sampled-Scope Prediction Candidates

Each export directory also contains a candidate file:

```bash
candidates/sampled_prediction_candidates.csv
```

Use this file for prediction and rule verification. The model should not predict over the full graph node-pair space for this experiment. It should predict only over the pairs in `sampled_prediction_candidates.csv`, so that the prediction scope matches the GAR rule mining scope.

Example DDA candidate file:

```bash
/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/DDA_depth6_20260709_093322/candidates/sampled_prediction_candidates.csv
```

Example TI candidate file:

```bash
/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/sampled_scope_exports/TI_processed_20260709_093517/candidates/sampled_prediction_candidates.csv
```

## Required Outputs to Return

After training and prediction, please return one result folder per dataset and per model.

Recommended folder structure:

```text
results/
  DDA/
    TransE/
    RotatE/
    NBFNet/
  TI/
    TransE/
    RotatE/
    NBFNet/
```

Each model result folder should contain the following files.

### 1. Prediction File

Required file name:

```text
predictions_with_original_ids.csv
```

This file should contain predictions only for the pairs in:

```text
candidates/sampled_prediction_candidates.csv
```

Required columns:

```text
source_id,target_id,true_label,pred_label,score
```

Column meaning:

```text
source_id = original sampled source-side node id, for example drug id or gene id
target_id = original sampled disease-side node id
true_label = ground-truth label: no_edge / positive / negative
pred_label = model predicted label: no_edge / positive / negative
score = model confidence or ranking score for the predicted label
```

If the model naturally outputs one score per relation, also include:

```text
score_no_edge,score_positive,score_negative
```

Preferred full format:

```text
source_id,target_id,true_label,pred_label,score,score_no_edge,score_positive,score_negative
```

Relation label mapping:

```text
0 = no_edge
1 = positive
2 = negative
```

String labels are preferred in `true_label` and `pred_label`, but numeric labels are acceptable if the mapping is clearly kept.

### 2. Metrics File

Required file name:

```text
metrics.json
```

Please include at least:

```json
{
  "dataset": "DDA",
  "model": "TransE",
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "positive_f1": 0.0,
  "negative_f1": 0.0,
  "no_edge_f1": 0.0,
  "num_test_pairs": 0
}
```

If available, also include precision and recall per class.

### 3. Training Log

Required file name:

```text
train.log
```

This should contain the training command, important hyperparameters, epoch logs, validation performance, and final test performance.

### 4. Config File

Required file name:

```text
config.json
```

This should record the actual model settings used for the run, including:

```text
dataset path
model name
embedding dimension
learning rate
batch size
number of epochs
negative sampling settings
random seed
checkpoint path, if any
```

### 5. Optional Checkpoint

If the model checkpoint is not too large, also return it:

```text
checkpoint.pt
```

If the checkpoint is large, returning only the path to the checkpoint is enough.

## Important Evaluation Requirement

For this experiment, the most important returned file is:

```text
predictions_with_original_ids.csv
```

It must be aligned with `sampled_prediction_candidates.csv`. In other words, every predicted pair should come from the sampled-scope candidate file, and the returned rows should keep the original `source_id` and `target_id`. This is required for the later step where GAR negative-edge rules are checked one by one against model predictions.

## Important Notes

- Entity IDs are namespaced before being mapped to integer IDs. For example, `drug:13` and `disease:13` are treated as different entities.
- This avoids incorrectly merging source-side and target-side nodes that happen to share the same numeric index.
- The generated `no_edge` triples are sampled only inside the sampled source-node and target-node scope.
- Downstream rule verification should use the sampled-scope candidate CSV, not whole-graph prediction outputs.
