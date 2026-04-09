from __future__ import annotations

from typing import Iterable, Sequence


MAD_CAMERA_ALIASES = {
    "first": "gripperPOV",
    "third1": "corner2",
    "front": "behindGripper",
    "third2": "corner3",
}


VIT_JOINT_CAMERA_ALIASES = {
    "first": "gripperPOV",
    "third1": "corner",
    "front": "behindGripper",
    "third2": "corner2",
}


def normalize_task_name(task: str) -> str:
    normalized = str(task)
    for suffix in ("-v2-goal-observable", "-v2", "-v3"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def resolve_camera_alias_profile(agent_name: str | None, requested_profile: str = "auto") -> str:
    profile = str(requested_profile)
    if profile == "auto":
        agent = str(agent_name or "")
        if agent.startswith("vit_latent_"):
            return "vit_joint"
        return "mad"
    if profile not in ("mad", "vit_joint"):
        raise ValueError(f"Unknown camera alias profile: {requested_profile}")
    return profile


def normalize_camera_name(camera_name: str, profile: str) -> str:
    mapping = MAD_CAMERA_ALIASES if profile == "mad" else VIT_JOINT_CAMERA_ALIASES
    return mapping.get(str(camera_name), str(camera_name))


def normalize_camera_names(camera_names: Sequence[str], profile: str) -> list[str]:
    return [normalize_camera_name(name, profile) for name in camera_names]


def describe_camera_mapping(camera_names: Iterable[str], profile: str) -> list[tuple[str, str]]:
    return [(str(name), normalize_camera_name(str(name), profile)) for name in camera_names]
