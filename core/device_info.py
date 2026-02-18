from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeviceInfo:
    device_type: str
    name: Optional[str] = None
    total_memory_gb: Optional[float] = None

    def format_lines(self):
        lines = [f"- type: {self.device_type}"]
        if self.name:
            lines.append(f"- name: {self.name}")
        if self.total_memory_gb is not None:
            lines.append(f"- memory: {self.total_memory_gb:.1f} GB")
        return lines


def get_device_info(device: str) -> DeviceInfo:
    if device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / (1024 ** 3)
        return DeviceInfo(device_type="cuda", name=props.name, total_memory_gb=mem_gb)
    return DeviceInfo(device_type=device)
