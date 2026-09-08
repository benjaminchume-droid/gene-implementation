from __future__ import annotations

import json
import platform

import torch


def hardware_report() -> dict:

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available":
            torch.cuda.is_available(),
    }

    if torch.cuda.is_available():

        report.update(
            {
                "cuda_version":
                    torch.version.cuda,

                "gpu_name":
                    torch.cuda.get_device_name(0),

                "gpu_count":
                    torch.cuda.device_count(),

                "gpu_memory_bytes":
                    torch.cuda.get_device_properties(0)
                    .total_memory,
            }
        )

    return report


if __name__ == "__main__":

    print(
        json.dumps(
            hardware_report(),
            indent=2,
        )
    )
