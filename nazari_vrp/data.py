from __future__ import annotations

from dataclasses import dataclass

import torch

CAPACITIES_BY_SIZE = {
    10: 20.0,
    20: 30.0,
    50: 40.0,
    100: 50.0,
}


def capacity_for_size(num_customers: int) -> float:
    """Return the capacity schedule used in the Nazari et al. VRP experiments."""
    if num_customers in CAPACITIES_BY_SIZE:
        return CAPACITIES_BY_SIZE[num_customers]

    closest = min(CAPACITIES_BY_SIZE, key=lambda n: abs(n - num_customers))
    return CAPACITIES_BY_SIZE[closest]


@dataclass(frozen=True)
class VRPConfig:
    """Synthetic CVRP distribution from the paper."""

    num_customers: int = 20
    vehicle_capacity: float | None = None
    demand_low: int = 1
    demand_high: int = 9

    @property
    def capacity(self) -> float:
        # Nếu người dùng không truyền capacity riêng, dùng lịch capacity chuẩn
        # theo số khách hàng trong bài Nazari et al.
        if self.vehicle_capacity is not None:
            return float(self.vehicle_capacity)
        return capacity_for_size(self.num_customers)


@dataclass
class VRPBatch:
    """One batch of CVRP instances."""

    depot: torch.Tensor
    locs: torch.Tensor
    demand: torch.Tensor
    capacity: torch.Tensor

    def to(self, device: torch.device | str) -> VRPBatch:
        return VRPBatch(
            depot=self.depot.to(device),
            locs=self.locs.to(device),
            demand=self.demand.to(device),
            capacity=self.capacity.to(device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.depot.size(0))

    @property
    def num_customers(self) -> int:
        return int(self.locs.size(1))

    def coords_with_depot(self) -> torch.Tensor:
        return torch.cat((self.depot[:, None, :], self.locs), dim=1)

    def demand_with_depot(self) -> torch.Tensor:
        depot_demand = torch.zeros(
            self.batch_size,
            1,
            dtype=self.demand.dtype,
            device=self.demand.device,
        )
        return torch.cat((depot_demand, self.demand), dim=1)


def generate_batch(
    batch_size: int,
    config: VRPConfig,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> VRPBatch:
    """Generate random CVRP instances in the unit square.

    This follows the experiment setup in Nazari et al.: depot/customer
    coordinates are sampled uniformly from [0, 1]^2 and customer demands are
    integers in [1, 9].
    """
    # Hàm này tạo dữ liệu đầu vào cho model:
    # depot là kho, locs là tọa độ khách hàng, demand là nhu cầu từng khách,
    # capacity là tải trọng tối đa của xe.
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    depot = torch.rand(batch_size, 2, generator=generator)
    locs = torch.rand(batch_size, config.num_customers, 2, generator=generator)
    demand = torch.randint(
        low=config.demand_low,
        high=config.demand_high + 1,
        size=(batch_size, config.num_customers),
        generator=generator,
    ).float()
    capacity = torch.full((batch_size,), float(config.capacity))

    return VRPBatch(
        depot=depot.to(device),
        locs=locs.to(device),
        demand=demand.to(device),
        capacity=capacity.to(device),
    )
