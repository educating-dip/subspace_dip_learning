from typing import Union, Tuple, Optional, Callable
from torch import Tensor
from subspace_dip.data.trafo.base_ray_trafo import BaseRayTrafo


class IdentityTrafo(BaseRayTrafo):
    """
    Identity transform (useful for denoising tasks, need to refactor base class in the future to be more general).
    """

    def __init__(self,
            im_shape: Union[Tuple[int, int], Tuple[int, int, int]],
            pinv_fun: Optional[Callable[[Tensor], Tensor]] = None,
            ):
        super().__init__(im_shape=im_shape, obs_shape=im_shape)

        self.pinv_fun = pinv_fun

    def trafo_flat(self, x: Tensor) -> Tensor:
        return x

    def trafo_adjoint_flat(self, observation: Tensor) -> Tensor:
        return observation

    def fbp(self, observation: Tensor) -> Tensor:
        return self.pinv_fun(observation) if self.pinv_fun is not None else observation

    trafo = BaseRayTrafo._trafo_via_trafo_flat
    trafo_adjoint = BaseRayTrafo._trafo_adjoint_via_trafo_adjoint_flat
