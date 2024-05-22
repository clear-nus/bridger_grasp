from .denoising_loss import ProjectedSE3DenoisingLoss, SE3DenoisingLoss
from .sdf_loss import SDFLoss
from .si_loss import StochasticInterpolantsLoss
from .ddpm_loss import DDPMLoss
from .residual_loss import ResidualLoss
from .vae_loss import VAELoss

def get_losses(args):
    losses = args['Losses']

    loss_fns = {}
    if 'sdf_loss' in losses:
        loss_fns['sdf'] = SDFLoss().loss_fn
    if 'projected_denoising_loss' in losses:
        loss_fns['denoise'] = ProjectedSE3DenoisingLoss().loss_fn
    if 'denoising_loss' in losses:
        loss_fns['denoise'] = SE3DenoisingLoss().loss_fn
    if 'si_loss' in losses:
        loss_fns['si'] = StochasticInterpolantsLoss().loss_fn
    if 'ddpm_loss' in losses:
        loss_fns['ddpm'] = DDPMLoss().loss_fn
    if 'residual_loss' in losses:
        loss_fns['residual'] = ResidualLoss().loss_fn
    if 'vae_loss' in losses:
        loss_fns['vae'] = VAELoss().loss_fn

    loss_dict = LossDictionary(loss_dict=loss_fns)
    return loss_dict


class LossDictionary():
    def __init__(self, loss_dict):
        self.fields = loss_dict.keys()
        self.loss_dict = loss_dict

    def loss_fn(self, model, model_input, ground_truth, val=False):

        losses = {}
        infos = {}
        for field in self.fields:
            loss_fn_k = self.loss_dict[field]
            loss, info = loss_fn_k(model, model_input, ground_truth, val)
            losses = {**losses, **loss}
            infos = {**infos, **info}

        return losses, infos