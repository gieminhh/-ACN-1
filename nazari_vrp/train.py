from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nazari_vrp.data import VRPConfig, generate_batch
from nazari_vrp.insertion import learning_based_insertion
from nazari_vrp.model import NazariVRPModel
from nazari_vrp.solver import beam_search_decode, rollout


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> NazariVRPModel:
    checkpoint = torch.load(path, map_location=device)
    model = NazariVRPModel(
        embed_dim=int(checkpoint.get("embed_dim", 128)),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def save_checkpoint(path: str | Path, model: NazariVRPModel, config: VRPConfig) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "embed_dim": model.embed_dim,
            "hidden_dim": model.hidden_dim,
            "num_customers": config.num_customers,
            "vehicle_capacity": config.capacity,
            "objective": "learning_based_insertion",
            "paper": "Nazari et al., Reinforcement Learning for Solving the Vehicle Routing Problem, NeurIPS 2018",
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    config = VRPConfig(
        num_customers=args.num_customers,
        vehicle_capacity=args.capacity,
    )
    model = NazariVRPModel(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        batch = generate_batch(args.batch_size, config, device=device)
        if args.objective == "insertion":
            result = learning_based_insertion(model, batch, decode_type="sampling")
        else:
            result = rollout(model, batch, decode_type="sampling")

        advantage = result.reward - result.value.detach()
        actor_loss = -(advantage * result.log_likelihood).mean()
        critic_loss = F.mse_loss(result.value, result.reward.detach())
        entropy_bonus = result.entropy.mean()
        loss = actor_loss + args.critic_coef * critic_loss - args.entropy_coef * entropy_bonus

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:05d} "
                f"objective={args.objective} "
                f"cost={result.cost.mean().item():.4f} "
                f"actor={actor_loss.item():.4f} "
                f"critic={critic_loss.item():.4f}"
            )

    save_checkpoint(args.output, model, config)
    print(f"saved checkpoint: {args.output}")


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    config = VRPConfig(
        num_customers=args.num_customers,
        vehicle_capacity=args.capacity,
    )
    model = load_checkpoint(args.checkpoint, device=device)

    costs = []
    with torch.no_grad():
        for idx in range(args.eval_instances):
            batch = generate_batch(1, config, device=device, seed=args.seed + idx)
            if args.method == "insertion":
                result = learning_based_insertion(
                    model,
                    batch,
                    decode_type="beam" if args.decode == "beam" else "greedy",
                    beam_width=args.beam_width,
                )
            elif args.decode == "beam":
                result = beam_search_decode(model, batch, beam_width=args.beam_width)
            else:
                result = rollout(model, batch, decode_type="greedy")
            costs.append(float(result.cost.item()))

    mean_cost = sum(costs) / len(costs)
    print(
        f"method={args.method} decode={args.decode} "
        f"instances={len(costs)} mean_cost={mean_cost:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nazari-style RL solver for CVRP")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--num-customers", type=int, default=20)
    train_parser.add_argument("--capacity", type=float, default=None)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--embed-dim", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--critic-coef", type=float, default=0.5)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--objective", choices=["insertion", "direct"], default="insertion")
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--output", default="nazari_vrp_checkpoint.pt")
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", default="nazari_vrp_checkpoint.pt")
    eval_parser.add_argument("--num-customers", type=int, default=20)
    eval_parser.add_argument("--capacity", type=float, default=None)
    eval_parser.add_argument("--eval-instances", type=int, default=100)
    eval_parser.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    eval_parser.add_argument("--method", choices=["insertion", "direct"], default="insertion")
    eval_parser.add_argument("--beam-width", type=int, default=5)
    eval_parser.add_argument("--seed", type=int, default=1234)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.set_defaults(func=evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        args = parser.parse_args(["train"])
    args.func(args)


if __name__ == "__main__":
    main()
