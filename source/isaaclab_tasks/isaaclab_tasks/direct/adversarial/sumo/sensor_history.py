"""CPU-testable observation utilities; no simulator or CUDA initialization."""

import torch
import torch.nn.functional as F


def rgb_features(image):
    """No segmentation labels/detections: just pooled RGB, same operation on real frames."""
    return F.adaptive_avg_pool2d(image[..., :3].float().permute(0, 3, 1, 2) / 255.0, (9, 16)).flatten(1)


def lidar_features(points, maximum=6.0):
    """Pool local-frame returns from the +Z hemisphere into 3 elevation x 32 azimuth bins.

    Unitree's tilted radar frame comes from its URDF. This is a common point-cloud
    adapter, not a reproduction of the proprietary L1/L2 scan pattern.
    """
    radius = torch.linalg.vector_norm(points, dim=-1)
    valid = torch.isfinite(points).all(-1) & (radius >= 0.05) & (radius < maximum) & (points[..., 2] >= 0)
    safe = torch.nan_to_num(points)
    azimuth = torch.atan2(safe[..., 1], safe[..., 0])
    elevation = torch.atan2(safe[..., 2], torch.linalg.vector_norm(safe[..., :2], dim=-1))
    column = ((azimuth + torch.pi) / (2 * torch.pi) * 32).long().clamp(0, 31)
    row = (elevation / (torch.pi / 2) * 3).long().clamp(0, 2)
    output = torch.zeros((points.shape[0], 96), device=points.device)
    proximity = torch.where(valid, 1 - radius / maximum, 0.0)
    output.scatter_reduce_(1, row * 32 + column, proximity, reduce="amax", include_self=True)
    return output


def gravity_estimate(previous, gyro, acceleration, dt):
    """Simple complementary IMU filter, not simulator attitude or a certified estimator."""
    predicted = F.normalize(previous - torch.cross(gyro, previous, dim=-1) * dt, dim=-1)
    norm = torch.linalg.vector_norm(acceleration, dim=-1, keepdim=True)
    stationary = (norm - 9.81).abs() < 1.0
    blend = stationary.float() * 0.02
    measured = -acceleration / norm.clamp_min(1e-6)
    return F.normalize((1 - blend) * predicted + blend * measured, dim=-1)


def depth_proximity(depth, maximum=6.0, shape=(3, 32)):
    """Pool metric ranges; missing returns represent no detected surface."""
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth = torch.nan_to_num(depth, nan=maximum, posinf=maximum, neginf=maximum)
    return F.adaptive_max_pool2d((1 - depth.clamp(0, maximum) / maximum)[:, None], shape).flatten(1)


class DepthHistory:
    """Newest-first history. Repeated reads do not advance time or resample noise."""

    def __init__(self, count, features=96, frames=4, device="cpu"):
        self.values = torch.zeros(count, frames, features, device=device)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)
        self.last_tick = torch.full((count,), -1, dtype=torch.long, device=device)

    def reset(self, ids):
        self.values[ids] = 0
        self.initialized[ids] = False
        self.last_tick[ids] = -1

    def update(self, current, tick, eligible):
        fresh = eligible & (self.last_tick != tick)
        first = fresh & ~self.initialized
        shifted = torch.cat((current[:, None], self.values[:, :-1]), dim=1)
        self.values[fresh] = shifted[fresh]
        self.values[first] = current[first, None].expand(-1, self.values.shape[1], -1)
        self.initialized |= fresh
        self.last_tick[fresh] = tick
        return self.values.flatten(1)


def joint_targets(actions, nominal, limits, scale):
    """Bound commanded offsets and enforce the asset's soft joint limits."""
    return (nominal + actions.clamp(-3, 3) * scale).maximum(limits[..., 0]).minimum(limits[..., 1])


def stall_update(counter, engaged, relative_speed, dt, grace_seconds=2.0):
    """Only stationary engagements accumulate stall cost; ordinary bracing is allowed."""
    counter = torch.where(engaged & (relative_speed < 0.08), counter + 1, torch.zeros_like(counter))
    return counter, counter.to(torch.float32) * dt >= grace_seconds
