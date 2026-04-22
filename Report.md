# Self-Pruning Neural Network — Report

## Why L1 Penalty Encourages Sparsity

The sparsity loss sums all gate values across the network.
The optimizer minimizes this by pushing gates toward exactly 0.
L1 is used because it applies constant pressure toward zero
unlike L2 which tapers off and rarely reaches exactly 0.
A gate of 0 means that weight is completely removed from the network.

## Results Table

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|------------------|-------------------|
| 1e-5   |        83.04%    |     47.3%         |
| 5e-4   |        84.99%    |     99.2%         |
| 2e-3   |                  |                   |

## Observations

- Higher lambda = more sparsity but lower accuracy
- Lower lambda = better accuracy but fewer weights pruned
- Best model: lambda =
