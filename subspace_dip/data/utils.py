from .trafo import (
        get_parallel_beam_2d_matmul_ray_trafo)

def get_ray_trafo(name, kwargs):
    if name == 'ellipses':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
    elif name == 'rectangles':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
    else:
        raise ValueError

    return ray_trafo
