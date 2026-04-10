# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import torch
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio


class AudioCodecLhotseDataset(torch.utils.data.Dataset):
    """
    A Lhotse-based dataset for audio codec model training.

    Receives a mini-batch of Lhotse Cut objects (pre-assembled by the Lhotse
    sampler) and produces a batch dict with ``audio`` and ``audio_lens`` tensors,
    compatible with ``AudioCodecModel._process_batch()``.

    Duration filtering
    """

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        if True:
            # Resample the audio to the target sample rate (not done automatically by
            # Lhotse for custom fields)
            for cut in cuts:
                cut.target_audio = cut.target_audio.resample(self.sample_rate)
            # Load and collate the audio, applying any transformations that were
            # configured in Lhotse in the process Note: fault_tolerant=False for now to
            # be aware of errors
            batch_audio, batch_audio_len = collate_audio(cuts, recording_field="target_audio", fault_tolerant=False)

        return {
            "audio": batch_audio,
            "audio_lens": batch_audio_len,
        }
