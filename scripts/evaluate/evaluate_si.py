# Object Classes :['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
# 'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
# 'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
# 'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
# 'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
# 'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']


import copy
import isaacgym
import configargparse
from se3dif.models.loader import load_model
from se3dif.samplers import Grasp_SI
from se3dif.samplers import Grasp_VAE


def get_vae(model_name='pcl_vae_prior', batch=10, device='cpu'):
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_name
    }
    model = load_model(model_args)
    model.eval()

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_VAE(model, batch=batch, device=device)

    return generator, model


def get_model(model_name, batch, diffuse_step=120, device='cuda:0'):
    model_params = model_name
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)
    model.eval()

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_SI(model, T=diffuse_step, batch=batch, device=device)

    return generator, model


def evaluate_si(model_name, prior_type, diffuse_step, obj_class, obj_id, n_grasps, n_envs, batch, viewer, visualize_grasp, device):
    print('##########################################################')
    print('Object Class: {}'.format(obj_class))
    print(obj_id)
    print('##########################################################')

    ## Get Model and Sample Generator ##
    generator, model = get_model(model_name, batch, diffuse_step, device)

    if prior_type == 'vae':
        generator_vae, model_vae = get_vae(batch=batch, device=device)
    else:
        generator_vae, model_vae = None, None

    #### Build Model Generator ####
    from isaac_evaluation.grasp_quality_evaluation.evaluate_model_si import EvaluatePCLSI
    evaluator = EvaluatePCLSI(generator, prior_generator=generator_vae, prior_type=prior_type,
                              n_grasps=n_grasps, obj_id=obj_id, obj_class=obj_class, n_envs=n_envs,
                              viewer=viewer, visualize_grasp=visualize_grasp)

    success_rate, success_distance, edd_mean, edd_std, divergence, edd_data_mean, edd_data_std, data_divergence = evaluator.generate_and_evaluate(success_eval=True, earth_moving_distance=True)
    print(edd_mean, edd_std)
    return success_rate, success_distance, edd_mean, edd_std, divergence, edd_data_mean, edd_data_std, data_divergence


if __name__ == '__main__':
    success_rate, success_distance, edd_mean, edd_std, divergence, edd_data_mean, edd_data_std, data_divergence = evaluate_si(model_name='pcl_gaussian', prior_type='vae',
                                                                                                                              diffuse_step=20, obj_class='Bottle', obj_id=0, n_grasps=20, n_envs=20, batch=20,
                                                                                                                              viewer=True, visualize_grasp=True, device='cpu')
    print(success_rate, edd_mean, edd_std, edd_data_mean, edd_data_std)
