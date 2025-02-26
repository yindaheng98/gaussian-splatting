
import torch

from gaussian_splatting import GaussianModel, Camera
from .trainer import AbstractTrainer, TrainerWrapper, BaseTrainer


class IncrementalSHTrainer(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            sh_degree_up_interval=1000,
            initial_sh_degree=0,
    ):
        super().__init__(base_trainer)
        self.sh_degree_up_interval = sh_degree_up_interval
        self.model.active_sh_degree = initial_sh_degree

    def oneupSHdegree(self):
        if self.model.active_sh_degree < self.model.max_sh_degree:
            self.model.active_sh_degree += 1

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if self.curr_step % self.sh_degree_up_interval == 0:
            self.oneupSHdegree()
        return super().loss(out, camera)

