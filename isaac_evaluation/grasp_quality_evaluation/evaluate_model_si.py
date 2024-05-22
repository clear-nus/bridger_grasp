from isaac_evaluation.grasp_quality_evaluation.grasps_sucess import GraspSuccessEvaluator

from scipy.spatial.transform import Rotation as R

from se3dif.utils import to_numpy, to_torch
from se3dif.datasets.acronym_dataset import AcronymGraspsDirectory
import numpy as np
import torch
import scipy
from se3dif.utils import SO3_R3
class EvaluatePCLSI():
    def __init__(self, generator, prior_generator, prior_type, n_grasps=500, batch=100, obj_id=0, obj_class='Mug', n_envs=10, net_scale=8.,
                 viewer=True, args=None, center_P=True, visualize_grasp=False):

        ## Set args
        self.center_P = center_P
        self.n_grasps = n_grasps
        self.n_envs = n_envs
        self.net_scale = net_scale
        self.obj_id = obj_id
        self.obj_class = obj_class
        self.batch = batch
        self.viewer = viewer
        self.args = self._set_args(args)

        self.prior_type = prior_type

        ## Load generator model
        self.generator = generator
        self.prior_generator = prior_generator
        self.visualize_grasp = visualize_grasp

    def _set_args(self, args):
        if args is None:
            args = {
                'generation_batch': self.batch,
                'empirical_dist_episodes': 1000,
            }
        return args

    def generate_and_evaluate(self, success_eval=True, earth_moving_distance=True):
        H = self.generate_grasps()
        if success_eval:
            success_cases, success_distance = self.evaluate_grasps_success(H)
            success_rate = success_cases / H.shape[0]
            print('Success rate : {}'.format(success_rate))
        else:
            success_rate = 0.
            success_distance = []
        if earth_moving_distance:
            print('EMD Distance: Dataset-Dataset')
            edd_data_mean, edd_data_std, divergence = self.measure_empirircal_dist_distance()
            print('EMD Distance: Samples-Dataset')
            edd_mean, edd_std, data_divergence = self.measure_empirircal_dist_distance(H)
        else:
            edd_mean = 0.
            edd_std = 0.
            divergence = []
            edd_data_mean = 0.
            edd_data_std = 0.
            data_divergence = []
        return success_rate, success_distance, edd_mean, edd_std, divergence, edd_data_mean, edd_data_std, data_divergence

    def pointcloud_conditioning(self):
        acronym_grasps = AcronymGraspsDirectory(data_type=self.obj_class)
        mesh = acronym_grasps.avail_obj[self.obj_id].load_mesh()
        P = mesh.sample(400)
        P = to_torch(P, self.generator.device)
        rot = to_torch(R.from_quat(self.q).as_matrix(), self.generator.device)
        P = torch.einsum('mn,bn->bm', rot, P)
        self.P = P.clone()

        P *= self.net_scale
        if self.center_P:
            self.P_mean = torch.mean(P, 0)
            P += -self.P_mean

        self.generator.model.set_latent(P[None, ...], batch=self.generator.batch)

        if self.prior_type == 'vae':
            self.prior_generator.model.set_latent(P[None, ...], batch=self.prior_generator.batch)
        return P

    def generate_grasps(self, n_grasps=None):
        if n_grasps == None:
            n_grasps = self.n_grasps

        ## Set a Random Rotations ##
        q = np.random.randn(4)
        # q = np.array([0., 0., 0., 1.])
        self.q = q / np.linalg.norm(q)

        ## Set SE3 Langevin Dynamics for generating Grasps
        P = self.pointcloud_conditioning()

        ## Generate Grasps in batch
        H = torch.zeros(0, 4, 4).to(self.generator.device)
        batch = self.generator.batch
        for i in range(0, self.n_grasps, batch):
            print('Generating of {} to {} samples'.format(i, i + batch))

            if self.prior_type == 'heuristic':
                sampled_rot = scipy.spatial.transform.Rotation.random(batch)
                rot = sampled_rot.as_matrix()
                H_initial = np.eye(4)[np.newaxis,].repeat(batch, 0)
                H_initial[:, :3, :3] = rot
                H_initial[..., :3, -1] = H_initial[..., :3, -1] * self.net_scale

                p_radius = np.sqrt(np.power(to_numpy(P.cpu()), 2).sum(-1)).max()

                base = np.zeros(3)[np.newaxis, :, np.newaxis].repeat(batch, 0)
                base[..., 2, :] = 1.0 * p_radius + 0.5
                H_initial[..., :3, -1:] -= np.matmul(rot, base)
            elif self.prior_type == 'vae':
                H_initial, _ = self.prior_generator.sample(num_sample=batch)
                H_initial = H_initial.detach().cpu().numpy()
            elif self.prior_type == 'gaussian':
                xw = torch.randn((batch, 6)).to(self.generator.device)
                H_initial = SO3_R3().exp_map(xw).to_matrix().to(self.generator.device).float()
            else:
                raise NotImplementedError

            H_initial = torch.as_tensor(H_initial).float().to(self.generator.device)
            H_episode = H_initial
            # H_episode = self.generator.sample(H_initial)

            ## Shift to CoM of the object
            if self.center_P:
                H_episode[:, :3, -1] = H_episode[:, :3, -1] + self.P_mean
            ## Rescale
            H_episode[:, :3, -1] = H_episode[:, :3, -1] / self.net_scale

            H = torch.cat((H, H_episode), 0)

        if self.visualize_grasp:
            num_visual_grasp = 10
            H_vis = H[:num_visual_grasp].clone()
            # H_vis = H_initial[:num_visual_grasp].clone()
            from se3dif.visualization import grasp_visualization

            H_vis = H_vis.squeeze()
            H_vis[:, :3, -1] = (H_vis[:, :3, -1] * self.net_scale - self.P_mean) / self.net_scale
            grasp_visualization.visualize_grasps(to_numpy(H_vis.cpu()), p_cloud=to_numpy(P.cpu()) / self.net_scale, mesh=None)

        return H

    def evaluate_grasps_success(self, H):
        ## Load grasp evaluator
        grasp_evaluator = GraspSuccessEvaluator(n_envs=self.n_envs, idxs=[self.obj_id] * self.n_envs, obj_class=self.obj_class,
                                                rotations=[self.q] * self.n_envs, viewer=self.viewer, enable_rel_trafo=False)
        result = grasp_evaluator.eval_set_of_grasps(H)
        grasp_evaluator.grasping_env.kill()
        return result

    def measure_empirircal_dist_distance(self, H_sample=None):

        from scipy.optimize import linear_sum_assignment
        from pytorch3d.transforms.so3 import so3_rotation_angle

        ## Load Acronym Dataset and Sample n_grasps ##
        grasps_directory = AcronymGraspsDirectory(data_type=self.obj_class)
        grasps = grasps_directory.avail_obj[self.obj_id]
        Hgrasps = grasps.good_grasps

        rot = R.from_quat(self.q).as_matrix()
        H_map = np.eye(4)
        H_map[:3, :3] = rot
        Hgrasps = np.einsum('mn,bnd->bmd', H_map, Hgrasps)

        # Set Samples
        if H_sample is None:
            idx = np.random.randint(0, Hgrasps.shape[0], self.n_grasps)
            H_sample = torch.Tensor(Hgrasps[idx, ...])
        p_sample = H_sample[:, :3, -1]
        R_sample = H_sample[:, :3, :3]

        divergence = np.zeros(0)
        for k in range(self.args['empirical_dist_episodes']):
            ## Sample Candidates ##
            idx = np.random.randint(0, Hgrasps.shape[0], self.n_grasps)
            H_eval = torch.Tensor(Hgrasps[idx, ...]).to(H_sample)

            p_eval = H_eval[:, :3, -1]
            R_eval = H_eval[:, :3, :3]

            xyz_dist = (p_eval[None, ...] - p_sample[:, None, ...]).pow(2).sum(-1).pow(.5)
            R12 = torch.einsum('bmn,knd->bkmd', R_eval.transpose(-1, -2), R_sample)

            R12_ = R12.reshape(-1, 3, 3)
            R_dist_ = (1. - so3_rotation_angle(R12_, cos_angle=True))
            R_dist = R_dist_.reshape(R12.shape[0], R12.shape[1])

            distance = xyz_dist + R_dist

            distance = to_numpy(distance)
            row_ind, col_ind = linear_sum_assignment(distance)
            min_distance = distance[row_ind, col_ind].mean()
            divergence = np.concatenate((divergence, np.array([min_distance])), 0)

        mean = np.mean(divergence)
        std = np.std(divergence)
        print('Wasserstein Distance Mean: {}, Variance: {}'.format(mean, std))
        return mean, std, divergence
