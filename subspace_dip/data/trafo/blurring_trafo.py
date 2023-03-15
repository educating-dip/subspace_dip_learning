from typing import Union, Tuple, Optional, Callable
import torch
import numpy as np
from torch import Tensor
from subspace_dip.data.trafo.base_ray_trafo import BaseRayTrafo


def fft_np(x, s):
    # s = (Ny, Nx)
    H,W = x.shape
    x_ = np.roll(x, ((H//2), (W//2)), axis=(0,1))
    x_pad = np.pad(x_, ((0, s[0] - H), (0, s[1] - W)))
    x_pad_ = np.roll(x_pad, (- (H//2), -(W//2)), axis=(0,1))
    return np.fft.fft2(x_pad_)


def zero_SV(H, eps):
    abs_H = np.abs(H)
    H[abs_H / np.max(abs_H) <= eps] = 0
    return H


def fft_Filter_(x, A):
    X_fft = torch.fft.fftn(x, dim=(-2, -1))
    HX = A * X_fft
    return torch.fft.ifftn(HX, dim=(-2, -1))


def build_flt(f, size):
    is_even_x = not size[1] % 2
    is_even_y = not size[0] % 2

    grid_x = np.linspace(-(size[1] // 2 - is_even_x * 0.5), (size[1] // 2 - is_even_x * 0.5), size[1])
    grid_y = np.linspace(-(size[0] // 2 - is_even_y * 0.5), (size[0] // 2 - is_even_y * 0.5), size[0])

    x, y = np.meshgrid(grid_x, grid_y)

    h = f(x, y)
    h = np.roll(h, (- (h.shape[0] // 2), -(h.shape[1] // 2)), (0, 1))

    return torch.tensor(h).float().unsqueeze(0).unsqueeze(0)


def get_gauss_flt(flt_size, std):
    f = lambda x, y: np.exp(-(x ** 2 + y ** 2) / 2 / std ** 2)
    h = build_flt(f, (flt_size, flt_size))
    return h


def dagger(H, method='Naive', eps=1e-3):
    abs_H = torch.abs(H)
    if method == 'Naive':
        H_pinv = torch.zeros_like(H)
        H_pinv[(abs_H / abs_H.max()) > 0] = 1 / H[(abs_H / abs_H.max()) > 0]
    else:
        H_pinv = H.conjugate() / (abs_H ** 2 + eps ** 2)
    return H_pinv


class BlurringTrafo(BaseRayTrafo):
    """
    Blurring transform (useful for deblurring tasks, need to refactor base class in the future to be more general).

    Implementation is based on https://github.com/shadyabh/PGSURE
    """

    def __init__(self,
            im_shape: Union[Tuple[int, int], Tuple[int, int, int]],
            flt_size: int = 15,
            std: float = 1.6,
            P_eps: float = 5e-2,
            pinv_fun: Optional[Callable[[Tensor], Tensor]] = None,
            ):
        super().__init__(im_shape=im_shape, obs_shape=im_shape)

        h = get_gauss_flt(flt_size, std)
        def flip_np(x):
            x_ = np.flip(np.roll(x, ((x.shape[0] // 2), (x.shape[1] // 2)), (0, 1)))
            return np.roll(x_, (- (x_.shape[0] // 2), -(x_.shape[1] // 2)), (0, 1))
        h_np = np.array(h.clone().cpu())[0, 0, :, :]
        H = fft_np(h_np, s=im_shape)
        H = zero_SV(H, P_eps)
        H_ = torch.tensor(H).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('H_', H_, persistent=False)

        self.pinv_fun = pinv_fun
        if self.pinv_fun is None:
            Ht = H  # blurring is self-adjoint
            HtH_dag = dagger(torch.from_numpy(Ht * H)).numpy()
            HtH_dag_Ht_np = HtH_dag * Ht
            HtH_dag_Ht = torch.tensor(HtH_dag_Ht_np).float().cuda()
            self.register_buffer('HtH_dag_Ht', HtH_dag_Ht, persistent=False)

    def trafo(self, x: Tensor) -> Tensor:
        return fft_Filter_(x, self.H_).real

    def trafo_adjoint(self, observation: Tensor) -> Tensor:
        return self.trafo(observation)  # blurring is self-adjoint

    def fbp(self, observation: Tensor) -> Tensor:
        if self.pinv_fun is not None:
            x = self.pinv_fun(observation)
        else:
            x = fft_Filter_(observation, self.HtH_dag_Ht).real
        return x

    trafo_flat = BaseRayTrafo._trafo_flat_via_trafo
    trafo_adjoint_flat = BaseRayTrafo._trafo_adjoint_flat_via_trafo_adjoint
