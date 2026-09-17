# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
import torch

from nemo.collections.tts.modules.magpietts_modules import FeatureMasking


def _set_seed():
    torch.manual_seed(42)


class TestFeatureMasking:

    @pytest.mark.unit
    @pytest.mark.parametrize('batch_size', [2, 5])
    @pytest.mark.parametrize('time', [5, 10])
    @pytest.mark.parametrize('hidden_size', [128, 256])
    def test_feature_masking_infer(self, batch_size, time, hidden_size):
        _set_seed()
        feature_masking = FeatureMasking(hidden_size=hidden_size)
        masked_emb = feature_masking.masked_emb[0, 0]
        inputs = torch.randn(size=(batch_size, time, hidden_size), dtype=torch.float32)
        mask = torch.randint(low=0, high=2, size=(batch_size, time), dtype=torch.bool)
        masked_tensor = feature_masking.infer(inputs=inputs, mask=mask)
        for i in range(batch_size):
            for j in range(time):
                output_val = masked_tensor[i, j]
                if mask[i, j]:
                    torch.testing.assert_close(actual=output_val, expected=masked_emb)
                else:
                    torch.testing.assert_close(actual=output_val, expected=inputs[i, j])

    @pytest.mark.unit
    def test_feature_masking_hides_only_maskable_timesteps(self):
        _set_seed()
        feature_masking = FeatureMasking(hidden_size=4, mask_min=1.0, mask_max=1.0)
        with torch.no_grad():
            feature_masking.masked_emb.fill_(7.0)
        masked_emb = feature_masking.masked_emb[0, 0]
        inputs = torch.randn(size=(2, 5, 4), dtype=torch.float32)
        input_len = torch.tensor([5, 4])
        maskable = torch.tensor(
            [
                [True, False, True, True, True],
                [True, True, False, True, True],
            ]
        )

        feature_masking.train()
        masked_tensor = feature_masking(inputs=inputs, input_len=input_len, maskable=maskable)

        for i in range(2):
            for j in range(5):
                hidden = maskable[i, j] and j < input_len[i]
                expected = masked_emb if hidden else inputs[i, j]
                torch.testing.assert_close(actual=masked_tensor[i, j], expected=expected)

        feature_masking.eval()
        torch.testing.assert_close(
            feature_masking(inputs=inputs, input_len=input_len, maskable=maskable),
            inputs,
        )

    @pytest.mark.unit
    def test_feature_masking_rejects_shares_outside_the_unit_interval(self):
        with pytest.raises(AssertionError, match="mask_min <= mask_max"):
            FeatureMasking(hidden_size=4, mask_min=0.5, mask_max=0.25)
        with pytest.raises(AssertionError, match="cannot exceed 1"):
            FeatureMasking(hidden_size=4, mask_max=1.5)
