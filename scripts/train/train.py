import os
import configargparse
from se3dif.utils import get_root_src

import torch
from torch.utils.data import DataLoader


from se3dif import datasets, losses, summaries, trainer
from se3dif.models import loader

from se3dif.utils import load_experiment_specifications

from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules

base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))


def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--specs_file_dir', type=str, default=os.path.join(base_dir, 'params')
                   , help='root for saving logging')

    p.add_argument('--spec_file', type=str, default='pcl_si'
                   , help='root for saving logging')

    p.add_argument('--summary', type=bool, default=False
                   , help='activate or deactivate summary')

    p.add_argument('--model_name', type=str, default='si'
                   , help='model to use')

    p.add_argument('--load_pretrain', type=bool, default=False
                   , help='activate or deactivate summary')

    p.add_argument('--pretrain_path', type=str, default='/pcl_si/checkpoints/model_current.pth'
                   , help='root for saving logging')

    p.add_argument('--partial', type=bool, default=False
                   , help='If partial observe point cloud')

    p.add_argument('--use_ot', type=bool, default=False
                   , help='If use optimal transport')

    p.add_argument('--saving_root', type=str, default=os.path.join(get_root_src(), 'logs')
                   , help='root for saving logging')

    p.add_argument('--models_root', type=str, default=root_dir
                   , help='root for saving logging')

    p.add_argument('--prior_type', type=str, default=root_dir
                   , help='{cvae, heuristic, gaussian}')

    p.add_argument('--dataset_name', type=str, default='vanilla'
                   , help='dataset to use')

    p.add_argument('--device', type=str, default='cuda:3', )
    p.add_argument('--class_type', type=str, default='Mug')

    opt = p.parse_args()
    return opt


def main(opt):
    ## Load training args ##
    spec_file = os.path.join(opt.specs_file_dir, opt.spec_file)
    args = load_experiment_specifications(spec_file)

    # Saving directories
    root_dir = opt.saving_root
    exp_dir = os.path.join(root_dir, args['exp_log_dir'])
    args['saving_folder'] = exp_dir

    device = opt.device

    ## Dataset
    train_dataset = datasets.PointcloudAcronymAndSDFDataset(augmented_rotation=True, partial=opt.partial,
                                                            one_object=args['single_object'], )
    train_dataloader = DataLoader(train_dataset, batch_size=args['TrainSpecs']['batch_size'], shuffle=True,
                                  drop_last=True)
    test_dataset = datasets.PointcloudAcronymAndSDFDataset(augmented_rotation=True, partial=opt.partial,
                                                           one_object=args['single_object'])  # copy.deepcopy(train_dataset)
    test_dataset.set_test_data()
    test_dataloader = DataLoader(test_dataset, batch_size=args['TrainSpecs']['batch_size'], shuffle=True,
                                 drop_last=True)

    ## Model
    args['device'] = device
    model = loader.load_model(args)

    if opt.load_pretrain:
        model_path = './logs' + opt.pretrain_path
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(device)

    # Losses
    loss = losses.get_losses(args)
    loss_fn = val_loss_fn = loss.loss_fn

    ## Summaries
    summary = summaries.get_summary(args, opt.summary)

    ## Optimizer
    if opt.model_name == 'si':
        lr_schedules = get_learning_rate_schedules(args)
        optimizer = torch.optim.Adam([
            {
                "params": model.vision_encoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": model.feature_encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": model.v_decoder.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
            {
                "params": model.s_decoder.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
            {
                "params": model.b_decoder.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ])
    else:
        raise NotImplementedError

    # Train
    trainer.train(model=model.float(), train_dataloader=train_dataloader, epochs=args['TrainSpecs']['num_epochs'], model_dir=exp_dir,
                  summary_fn=summary, device=device, lr=1e-4, optimizers=[optimizer],
                  steps_til_summary=args['TrainSpecs']['steps_til_summary'],
                  epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
                  loss_fn=loss_fn, iters_til_checkpoint=args['TrainSpecs']['iters_til_checkpoint'],
                  clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
                  val_dataloader=test_dataloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
