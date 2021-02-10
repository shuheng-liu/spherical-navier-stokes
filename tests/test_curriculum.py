import torch
from spherical_navier_stokes.curriculum import RadialCurriculumLearner
from neurodiffeq.generators import GeneratorSpherical, StaticGenerator
from neurodiffeq.solvers import SolverSpherical
from neurodiffeq.conditions import NoCondition
from neurodiffeq.networks import FCNN, Swish
from neurodiffeq.function_basis import ZonalSphericalHarmonics


def test_radial_curriculum():
    def train_gen_getter(size, r0, r1):
        return GeneratorSpherical(size, r0, r1, method='equally-spaced-noisy')

    def valid_gen_getter(size, r0, r1):
        return StaticGenerator(GeneratorSpherical(size, r0, r1, method='equally-spaced-noisy'))

    BASE_SIZE = 1024

    cr = RadialCurriculumLearner(
        n_curricula=20,
        train_gen_getter=train_gen_getter,
        valid_gen_getter=valid_gen_getter,
        base_size=BASE_SIZE,
        r_in_from=1.0, r_in_to=0.1,
        r_out_from=10.0, r_out_to=10.0,
        advance_method='log',
        size_method='log',
        min_size=1,
        max_size=1024,
    )

    pde = lambda R, r, theta, phi: [torch.sum(R ** 2, dim=1, keepdim=True)]
    degrees = [0, 2, 4, 6, 8]
    basis = ZonalSphericalHarmonics(degrees=degrees)
    conditions = [NoCondition()]
    nets = [FCNN(1, len(degrees), actv=Swish)]

    solver = SolverSpherical(pde_system=pde, conditions=conditions, r_min=0.1, r_max=10.0, nets=nets)

    epochs_per_curriculum = 10

    def check_callback(solver):
        curr_i = (solver.global_epoch - 1) // epochs_per_curriculum
        local_epoch = (solver.global_epoch - 1) % epochs_per_curriculum
        if local_epoch == 0:
            print(f'{curr_i}: ({solver.generator["train"].size}, {solver.generator["valid"].size})')

    cr.fit(solver, epochs_per_curriculum=epochs_per_curriculum, callbacks=[check_callback])
    assert cr.base_size['train'] == BASE_SIZE
    assert cr.base_size['valid'] == BASE_SIZE
