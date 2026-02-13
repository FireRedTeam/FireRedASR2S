# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)
"""CTC module for timestamp generation."""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class CTC(torch.nn.Module):
    def __init__(self, odim: int, encoder_output_size: int, subsampling: int = 4):
        super().__init__()
        self.ctc_lo = torch.nn.Linear(encoder_output_size, odim)
        self.subsampling = subsampling

    @classmethod
    def from_pretrained(cls, ctc_path: str, device_id: int = 0):
        """Load CTC module from saved weights."""
        ckpt = torch.load(ctc_path, map_location="cpu", weights_only=True)

        odim = ckpt["odim"]
        encoder_output_size = ckpt["encoder_output_size"]
        subsampling = ckpt.get("subsampling", 4)

        model = cls(odim, encoder_output_size, subsampling)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model = model.cuda(device_id)

        return model

    def forward(self, encoder_output_pad: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output_pad: (N, T, H) encoder output
        Returns:
            log_probs: (N, T, odim) log softmax output
        """
        return F.log_softmax(self.ctc_lo(encoder_output_pad), dim=2)

    @staticmethod
    def ctc_alignment_to_timestamp(ys_with_blank: List[int], subsampling: int, blank_id: int = 0):
        """Convert CTC alignment to timestamps.

        Args:
            ys_with_blank: CTC alignment result with blanks
            subsampling: encoder subsampling factor
            blank_id: blank token id

        Returns:
            start_times: list of start times in seconds
            end_times: list of end times in seconds
        """
        start_times: List[float] = []
        end_times: List[float] = []
        frame_shift = 10  # ms, hard coded
        T = len(ys_with_blank)
        t = 0
        ctc_durs = []

        while t < T:
            token = ys_with_blank[t]
            t += 1
            if token != blank_id:
                start_t = t
                timestamp = frame_shift * subsampling * t / 1000.0  # s
                start_times.append(timestamp)
                if len(start_times) == len(end_times) + 2:
                    end_times.append(start_times[-1])
                # skip repeat token
                while t < T and token == ys_with_blank[t]:
                    t += 1
                assert t - start_t >= 0
                ctc_durs.append((t - start_t + 1) * frame_shift * subsampling / 1000.0)

        end_times.append((frame_shift * subsampling * T + 25) / 1000.0)
        if len(start_times) == 0:
            start_times.append(0.0)

        # Refine end_times
        assert len(ctc_durs) == len(end_times) and len(start_times) == len(end_times)
        avg_dur = sum(e - s for s, e in zip(start_times, end_times)) / len(end_times)
        new_end_times = []
        for s, e, ctc_dur in zip(start_times, end_times, ctc_durs):
            if e - s > 2 * avg_dur:
                e = s + max(1.5 * avg_dur, ctc_dur)
            new_end_times.append(round(e, 3))
        end_times = new_end_times

        return start_times, end_times
