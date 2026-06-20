# Project Summary for Grading

Start from this file if reviewing the repository.

## Claim

The active implementation is a Learning-based Insertion Heuristic for CVRP.
It inherits the learning-policy component from Nazari et al. (NeurIPS 2018),
"Reinforcement Learning for Solving the Vehicle Routing Problem", then uses the
learned sequence as a priority order for Best Insertion.

## Active Implementation

The relevant code is in `nazari_vrp/`.

- `data.py`: generates the same synthetic CVRP distribution used in the paper.
- `model.py`: implements static embeddings, dynamic embeddings, an RNN decoder,
  and attention over feasible destinations.
- `solver.py`: provides the Nazari-style autoregressive rollout and beam search.
- `insertion.py`: converts the learned action sequence into a customer priority
  order, then applies Best Insertion with Delta(a,j,b).
- `train.py`: trains the actor with policy-gradient loss and a critic baseline.
  The default training objective is the negative route length after insertion.

## Important Difference From the Old Version

The previous project used RL4CO with PPO + Attention Model. That made the
paper connection unclear.

The current active version keeps the paper-inspired RNN + attention learning
component, but the final solver is explicitly insertion-based to match the
thesis title.

## How to Check

Run:

```bash
py -3.12 run.py train --num-customers 10 --steps 5 --batch-size 8 --output smoke_checkpoint.pt
py -3.12 run.py eval --checkpoint smoke_checkpoint.pt --num-customers 10 --method insertion --decode greedy --eval-instances 3
py -3.12 -m pytest tests/test_nazari_vrp.py
```
