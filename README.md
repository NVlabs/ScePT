### ScePT ###
### Environment Setup ###
First, we'll create a conda environment to hold the dependencies.
```
conda create --name ScePT python=3.8 -y
source activate ScePT
pip install -r requirements.txt
```

### Data Setup ###
#### Pedestrian Datasets ####
We've already included preprocessed data splits for the ETH and UCY Pedestrian datasets in this repository, you can see them in `experiments/pedestrians/raw`. In order to process them into a data format that our model can work with, execute the follwing.
```
cd experiments/pedestrians
python process_data.py # 
```

#### nuScenes Dataset ####
Download the nuScenes dataset (this requires signing up on [their website](https://www.nuscenes.org/)). You can start with the mini dataset as it is smaller. Extract the downloaded zip file's contents and place them in the `experiments/nuScenes` directory. Then, download the map expansion pack (v1.1) and copy the contents of the extracted `maps` folder into the `experiments/nuScenes/v1.0-mini/maps` folder. Finally, process them into a data format that our model can work with.
```
cd experiments/nuScenes

# For the mini nuScenes dataset, use the following
python process_data.py --data=./v1.0-mini --version="v1.0-mini" --output_path=../processed --num_worker=X

# For the full nuScenes dataset, use the following
python process_data.py --data=./v1.0 --version="v1.0-trainval" --output_path=../processed --num_worker=X
```
In case you also want a validation set generated (by default this will just produce the training and test sets), replace line 406 in `process_data.py` with:
```
    val_scene_names = val_scenes
```

## Model Training ##
### Pedestrian Dataset ###
To train a model on the ETH and UCY Pedestrian datasets, you can execute a version of the following command from within the `ScePT/` directory.
```
python -m torch.distributed.launch --nproc_per_node=X train_clique.py --train_data_dict <dataset>_train.pkl --eval_data_dict <dataset>_val.pkl --offline_scene_graph yes --preprocess_workers X --log_dir ../experiments/pedestrians/models  --train_epochs X --augment --conf ../config/clique_ped_config.json --indexing_workers=X --batch_size=X --vis_every=X --eval_every=X
```
### nuScenes Dataset ###
To train a model on the nuScenes dataset, you can execute a version of the following command from within the `ScePT/` directory.
```
python -m torch.distributed.launch --nproc_per_node=1 train_clique.py --train_data_dict nuScenes_train.pkl --eval_data_dict nuScenes_val.pkl --offline_scene_graph yes --preprocess_workers X --log_dir ../experiments/nuScenes/models  --train_epochs X --augment --conf ../config/clique_nusc_config.json --indexing_workers=X --batch_size=X --vis_every=X --map_encoding --incl_robot_node --eval_every=X
```




