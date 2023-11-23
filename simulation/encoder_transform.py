from tensordict import TensorDictBase
from torchrl.envs.transforms import Transform, RewardSum, ObservationTransform

from typing import Sequence
from tensordict.utils import NestedKey
import torch
from torchrl.envs.transforms.utils import (
    _set_missing_tolerance,
)
from autoencoder.encoder import VariationalEncoder
from torchrl.data.tensor_specs import TensorSpec, CompositeSpec
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from torchrl.envs import Compose
from typing import List, Optional, Union

from torchrl.envs.transforms.transforms import (
    CatTensors,
    Compose,
    FlattenObservation,
    ObservationNorm,
    Resize,
    ToTensorImage,
    Transform,
    UnsqueezeTransform,
)

class EncodeImage(Transform):

    inplace = False

    def __init__(self, in_keys, out_keys, encoder, del_keys: bool = True):
        convnet = encoder
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.convnet = convnet
        self.del_keys = del_keys
        self.outdim = 101



    def _call(self, tensordict):
        image_obs = tensordict.get(self.in_keys[0])
        nav_obs = tensordict.get(self.in_keys[1])
        encoder_obs = self._apply_transform(image_obs)
        out = torch.cat((encoder_obs, nav_obs), 1)
        tensordict.set(
            self.out_keys[0],
            out,
        )

        return tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # TODO: Check this makes sense
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @torch.no_grad()
    def _apply_transform(self, obs: torch.Tensor) -> None:
        image_obs = obs.permute(0, 3, 2, 1).to(torch.float)
        encoder_obs = self.convnet(image_obs).to(torch.float64)
        return encoder_obs

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        keys = [key for key in observation_spec.keys(True, True) if key in self.in_keys]
        device = observation_spec.device
        dim = observation_spec[keys[0]].shape[0]
        observation_spec = observation_spec.clone()
        if self.del_keys:
            for in_key in keys:
                del observation_spec[in_key]

        for out_key in self.out_keys:
            observation_spec[out_key] = UnboundedContinuousTensorSpec(
                shape=torch.Size([dim, self.outdim]), device=device, dtype=torch.float64
            )

        return observation_spec
