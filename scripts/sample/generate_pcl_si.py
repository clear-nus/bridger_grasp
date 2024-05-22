# Object Classes :['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
# 'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
# 'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
# 'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
# 'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
# 'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']

from se3dif.samplers.grasp_vae_samplers import Grasp_VAE

batch = 2000
def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=int, default=0)
    p.add_argument('--n_grasps', type=str, default='200')
    p.add_argument('--obj_class', type=str, default='Bottle')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--eval_sim', type=bool, default=False)
    p.add_argument('--model', type=str, default='pcl_si')
    p.add_argument('--prior_type', type=str, default='heuristic', help='{heuristic, gaussian, cvae}')

    opt = p.parse_args()
    return opt


def get_cvae(p, num_of_grasps=10, model_name='pcl_vae_prior', device='cpu'):
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_name
    }
    model = load_model(model_args)
    model.eval()

    context = to_torch(p[None, ...], device)
    model.set_latent(context, batch=num_of_grasps)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_VAE(model, batch=batch, device=device)

    return generator, model


def get_approximated_grasp_diffusion_field(p, args, device='cpu'):
    model_params = args.model
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    context = to_torch(p[None, ...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_SI(model, T=5, device=device)

    return generator, model


def sample_pointcloud(obj_id=0, obj_class='Mug'):
    scale = 8.

    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class)
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()

    P = mesh.sample(400)

    sampled_rot = scipy.spatial.transform.Rotation.random()
    # rot = sampled_rot.as_matrix()
    rot_quat = sampled_rot.as_quat()

    # P = np.einsum('mn,bn->bm', rot, P)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    H = np.eye(4)
    # H[:3, :3] = rot
    mesh.apply_transform(H)
    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3, -1] = -P_mean
    mesh.apply_transform(H)
    translational_shift = copy.deepcopy(H)

    # sample good grasps
    grasp_obj = acronym_grasps.avail_obj[obj_id]
    rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=10)
    H_grasps = grasp_obj.good_grasps[rix, ...]

    H = np.eye(4)
    # H[:3, :3] = rot
    H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
    H_grasps[..., :3, -1] = H_grasps[..., :3, -1] * scale
    H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - P_mean

    return P, mesh, translational_shift, rot_quat, H_grasps


if __name__ == '__main__':
    import copy
    import configargparse

    args = parse_args()

    EVAL_SIMULATION = args.eval_sim
    # isaac gym has to be imported here as it is supposed to be imported before torch
    if (EVAL_SIMULATION):
        # Alternatively: Evaluate Grasps in Simulation:
        from isaac_evaluation.grasp_quality_evaluation import GraspSuccessEvaluator

    from theseus import SO3
    from se3dif.utils import SO3_R3
    import theseus as th
    from se3dif.samplers.grasp_si_samplers import Grasp_SI
    import scipy.spatial.transform
    import numpy as np
    from se3dif.datasets import AcronymGraspsDirectory
    from se3dif.models.loader import load_model
    from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
    from se3dif.utils import to_numpy, to_torch

    import torch

    print('##########################################################')
    print('Object Class: {}'.format(args.obj_class))
    print(args.obj_id)
    print('##########################################################')

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    obj_class = args.obj_class
    n_envs = 30
    device = args.device

    ## Set Model and Sample Generator ##
    P, mesh, trans, rot_quad, H_grasps = sample_pointcloud(obj_id, obj_class)
    generator, model = get_approximated_grasp_diffusion_field(P, args, device)

    prior_type = args.prior_type

    num_samples = batch

    import time
    # Record the start time
    start_time = time.time()
    if prior_type == 'heuristic':
        scale = 8
        sampled_rot = scipy.spatial.transform.Rotation.random(num_samples)
        rot = sampled_rot.as_matrix()
        H_initial = np.eye(4)[np.newaxis,].repeat(num_samples, 0)
        H_initial[:, :3, :3] = rot
        H_initial[..., :3, -1] = H_initial[..., :3, -1] * scale

        p_radius = np.sqrt(np.power(P, 2).sum(-1)).max()

        base = np.zeros(3)[np.newaxis, :, np.newaxis].repeat(num_samples, 0)
        base[..., 2, :] = 1.0 * p_radius + 0.5
        H_initial[..., :3, -1:] -= np.matmul(rot, base)
    elif prior_type == 'cvae':
        generator_vae, model = get_cvae(P, num_of_grasps=num_samples, device=device)
        with torch.no_grad():
            H_initial, _ = generator_vae.sample(num_sample=num_samples)
            H_initial = H_initial.detach().cpu().numpy()
    elif prior_type == 'gaussian':
        xw = torch.randn((batch, 6)).cpu()
        H_initial = SO3_R3().exp_map(xw).to_matrix().cpu().float()
    else:
        raise NotImplementedError

    H_initial = torch.as_tensor(H_initial).float().to(device)
    # Call the function you want to time

    H = generator.sample(H_initial, save_path=False)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f'Time taken: {elapsed_time} seconds')

    H_grasp = H.clone()
    H_init_grasp = H_initial.clone()
    H[..., :3, -1] *= 1 / 8.

    ## Visualize results ##
    from se3dif.visualization import grasp_visualization

    vis_H = H_grasp.squeeze()
    P *= 1 / 8
    mesh = mesh.apply_scale(1 / 8)
    grasp_visualization.visualize_grasps(to_numpy(H_grasp), p_cloud=P, mesh=mesh)
