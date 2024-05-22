import os
import torch
import torch.nn as nn
import numpy as np

from se3dif import models

from se3dif.utils import get_pretrained_models_src, load_experiment_specifications

pretrained_models_dir = get_pretrained_models_src()


# pretrained_models_dir = '../../../logs/'

def gt_load_model(args):
    if 'pretrained_model' in args:
        model_args = load_experiment_specifications(os.path.join(pretrained_models_dir,
                                                                 args['pretrained_model']))
        args["NetworkArch"] = model_args["NetworkArch"]
        args["NetworkSpecs"] = model_args["NetworkSpecs"]

    if args['NetworkArch'] == 'GraspDiffusion':
        model = tg_load_grasp_diffusion(args)
    elif args['NetworkArch'] == 'PointcloudGraspDiffusion':
        model = load_pointcloud_grasp_diffusion(args)
    elif args['NetworkArch'] == 'PointcloudSI':
        model = load_pointcloud_si(args)
    elif args['NetworkArch'] == 'PointcloudDDPM':
        model = load_pointcloud_ddpm(args)
    elif args['NetworkArch'] == 'PointcloudVAE':
        model = load_pointcloud_vae(args)

    if 'pretrained_model' in args:
        # model_path = os.path.join(pretrained_models_dir, args['pretrained_model'], 'checkpoints', 'model.pth')
        model_path = os.path.join(pretrained_models_dir, args['pretrained_model'], 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        if args['device'] != 'cpu':
            model = model.to(args['device'], dtype=torch.float32)

    elif 'saving_folder' in args:
        load_model_dir = os.path.join(args['saving_folder'], 'checkpoints', 'model_current.pth')
        try:
            if args['device'] == 'cpu':
                model.load_state_dict(torch.load(load_model_dir, map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(load_model_dir))
        except:
            pass

    return model


def tg_load_grasp_diffusion(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    lat_params = params['latent_codes']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.LatentCodes(num_scenes=lat_params['num_scenes'], latent_size=lat_params['latent_size'])
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
        enc_dim=feat_enc_params['enc_dim'],
        latent_size=lat_params['latent_size'],
        dims=feat_enc_params['dims'],
        out_dim=feat_enc_params['out_dim'],
        dropout=feat_enc_params['dropout'],
        dropout_prob=feat_enc_params['dropout_prob'],
        norm_layers=feat_enc_params['norm_layers'],
        latent_in=feat_enc_params["latent_in"],
        xyz_in_all=feat_enc_params["xyz_in_all"],
        use_tanh=feat_enc_params["use_tanh"],
        latent_dropout=feat_enc_params["latent_dropout"],
        weight_norm=feat_enc_params["weight_norm"]
    )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points=points_params['n_points'],
                                          loc=np.array(points_params['loc']),
                                          scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']
    task_dim = points_params['task_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
        nn.Linear(in_dim + hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

    task_net = nn.Sequential(
        nn.Linear(task_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )

    model = models.TGGraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                          decoder=energy_net, points=points, task_encoder=task_net).to(device)
    return model


def load_pointcloud_grasp_diffusion(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
        enc_dim=feat_enc_params['enc_dim'],
        latent_size=v_enc_params['latent_size'],
        dims=feat_enc_params['dims'],
        out_dim=feat_enc_params['out_dim'],
        dropout=feat_enc_params['dropout'],
        dropout_prob=feat_enc_params['dropout_prob'],
        norm_layers=feat_enc_params['norm_layers'],
        latent_in=feat_enc_params["latent_in"],
        xyz_in_all=feat_enc_params["xyz_in_all"],
        use_tanh=feat_enc_params["use_tanh"],
        latent_dropout=feat_enc_params["latent_dropout"],
        weight_norm=feat_enc_params["weight_norm"]
    )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points=points_params['n_points'],
                                          loc=np.array(points_params['loc']),
                                          scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']
    task_dim = points_params['task_dim']
    hidden_dim = 512
    task_hidden_dim = 256
    energy_net = nn.Sequential(
        nn.Linear(in_dim + task_hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

    task_net = nn.Sequential(
        nn.Linear(task_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, task_hidden_dim),
    )

    model = models.TGGraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                          decoder=energy_net, points=points, task_encoder=task_net).to(device)
    return model


def load_pointcloud_ddpm(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
        enc_dim=feat_enc_params['enc_dim'],
        latent_size=v_enc_params['latent_size'],
        dims=feat_enc_params['dims'],
        out_dim=feat_enc_params['out_dim'],
        dropout=feat_enc_params['dropout'],
        dropout_prob=feat_enc_params['dropout_prob'],
        norm_layers=feat_enc_params['norm_layers'],
        latent_in=feat_enc_params["latent_in"],
        xyz_in_all=feat_enc_params["xyz_in_all"],
        use_tanh=feat_enc_params["use_tanh"],
        latent_dropout=feat_enc_params["latent_dropout"],
        weight_norm=feat_enc_params["weight_norm"]
    )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points=points_params['n_points'],
                                          loc=np.array(points_params['loc']),
                                          scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']
    task_dim = points_params['task_dim']
    hidden_dim = 512
    task_hidden_dim = 256
    energy_net = nn.Sequential(
        nn.Linear(in_dim + task_hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

    task_net = nn.Sequential(
        nn.Linear(task_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, task_hidden_dim),
    )

    model = models.TGGraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                        decoder=energy_net, points=points, task_encoder=task_net).to(device)
    return model


def load_pointcloud_si(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
        enc_dim=feat_enc_params['enc_dim'],
        latent_size=v_enc_params['latent_size'],
        dims=feat_enc_params['dims'],
        out_dim=feat_enc_params['out_dim'],
        dropout=feat_enc_params['dropout'],
        dropout_prob=feat_enc_params['dropout_prob'],
        norm_layers=feat_enc_params['norm_layers'],
        latent_in=feat_enc_params["latent_in"],
        xyz_in_all=feat_enc_params["xyz_in_all"],
        use_tanh=feat_enc_params["use_tanh"],
        latent_dropout=feat_enc_params["latent_dropout"],
        weight_norm=feat_enc_params["weight_norm"]
    )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points=points_params['n_points'],
                                          loc=np.array(points_params['loc']),
                                          scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']
    task_dim = points_params['task_dim']

    hidden_dim = 512
    task_hidden_dim = 256
    v_decoder = nn.Sequential(
        nn.Linear(in_dim + task_hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 6),
    )

    s_decoder = nn.Sequential(
        nn.Linear(in_dim + task_hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 6),
    )

    b_decoder = nn.Sequential(
        nn.Linear(in_dim + task_hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 6),
    )

    task_net = nn.Sequential(
        nn.Linear(task_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, task_hidden_dim),
    )
    model = models.TGSIFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                            v_decoder=v_decoder, s_decoder=s_decoder, b_decoder=b_decoder, points=points, task_encoder=task_net).to(device)
    return model


def load_pointcloud_vae(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
        enc_dim=feat_enc_params['enc_dim'],
        latent_size=v_enc_params['latent_size'],
        dims=feat_enc_params['dims'],
        out_dim=feat_enc_params['out_dim'],
        dropout=feat_enc_params['dropout'],
        dropout_prob=feat_enc_params['dropout_prob'],
        norm_layers=feat_enc_params['norm_layers'],
        latent_in=feat_enc_params["latent_in"],
        xyz_in_all=feat_enc_params["xyz_in_all"],
        use_tanh=feat_enc_params["use_tanh"],
        latent_dropout=feat_enc_params["latent_dropout"],
        weight_norm=feat_enc_params["weight_norm"]
    )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points=points_params['n_points'],
                                          loc=np.array(points_params['loc']),
                                          scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']
    latent_dim = 128
    hidden_dim = 512
    decoder = nn.Sequential(
        nn.Linear(latent_dim + in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 6),
    )

    encoder = models.nets.NormalMLP(input_dim=6 + in_dim, hidden_dim=hidden_dim,
                                    output_dim=latent_dim, layers=2)

    model = models.VAEFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                             decoder=decoder, encoder=encoder, points=points, latent_size=latent_dim).to(device)
    return model
