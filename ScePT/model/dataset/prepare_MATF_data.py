import numpy as np
import torch
import pdb
import dill as pickle
import sys
sys.path.append('../../')




# config should be the architecture intended to run,
# which should be in ['baseline', 'single_agent_scene', 'multi_agent', 'multi_agent_scene', 'GAN_D', 'GAN_G', 'jointGAN_D', 'jointGAN_G']
# input_list should be a list, whose each element comes from a scene:
# each element is a list of [scene_id, agent_id, scene_image, number_agents, past_list, future_list, 
# weights_list, coordinates_list, lanes, absolute_coordinate_list]

# def preprocess_nusc(scene):
#         scene_id


def main():
	with open('../../../experiments/processed/nuScenes_mini_train.pkl','rb') as f:
		env = pickle.load(f)
		for scene in env.scenes:
			scene_id = env.scenes.index(scene)
			
	pdb.set_trace()


if __name__=='__main__':
	main()