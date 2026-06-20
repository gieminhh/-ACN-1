# Learning-based Insertion Heuristic for CVRP

This project implements a learning-based insertion heuristic for the
Capacitated Vehicle Routing Problem (CVRP). It inherits the learning policy
idea from:

Nazari et al., "Reinforcement Learning for Solving the Vehicle Routing Problem",
NeurIPS 2018.

## Method

The active method is:

```text
Nazari-style learned priority policy
-> customer priority order
-> best insertion heuristic
-> CVRP routes
```

For each customer `j` in the learned priority order, the insertion heuristic
tests every feasible edge `a -> b` in the current routes and inserts `j` where
the extra distance is smallest:

```text
Delta(a, j, b) = c(a, j) + c(j, b) - c(a, b)
```

If no current route can accept `j` without exceeding capacity, a new route
`0 -> j -> 0` is opened.

## What is inherited from Nazari et al.

- CVRP instances are sampled in the unit square.
- Customer demands are integer values in `[1, 9]`.
- Capacities follow the paper setup: VRP10=20, VRP20=30, VRP50=40, VRP100=50.
- The learning policy uses a recurrent decoder with attention.
- Dynamic state includes remaining demand and current vehicle load.
- Feasibility masks block served customers and customers that exceed remaining load.
- Greedy and beam-search decoding are used to create the learned priority order.

The difference is intentional: this project uses the learned policy to guide
Best Insertion, so it matches the thesis topic "Learning-based Insertion
Heuristic" rather than being a line-by-line reproduction of the paper.

## Main files

- `nazari_vrp/`: learning policy and insertion implementation.
- `app.py`: Streamlit visual demo.
- `run.py`: command-line train/eval entrypoint.
- `train_cvrp_final.py`: training entrypoint kept for the original project naming.
- `PROJECT_SUMMARY_FOR_CLAUDE.md`: short grading guide.

The old RL4CO source tree is kept only as legacy/reference material. The active
implementation for grading is `nazari_vrp`.

## Install

```bash
py -3.12 -m pip install torch streamlit plotly pandas pytest
```

## Train

```bash
py -3.12 run.py train --num-customers 20 --steps 1000 --batch-size 128 --output nazari_vrp_checkpoint.pt
```

For a quick smoke run:

```bash
py -3.12 run.py train --num-customers 10 --steps 5 --batch-size 8 --output smoke_checkpoint.pt
```

## Evaluate

```bash
py -3.12 run.py eval --checkpoint nazari_vrp_checkpoint.pt --num-customers 20 --method insertion --decode greedy
py -3.12 run.py eval --checkpoint nazari_vrp_checkpoint.pt --num-customers 20 --method insertion --decode beam --beam-width 5
```

## Run the app

```bash
py -3.12 -m streamlit run app.py
```

If `nazari_vrp_checkpoint.pt` is missing, the app still runs with an untrained
model so the route construction pipeline can be inspected.
