# Train Semantic Segmentation with torchseg

## Install environment
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Training the model
```bash
python train.py --config config/COCO_train_config.yaml
```
The arg `--config` specifies the configuration file to be used. The configuration files are located in the `config` folder. You can create your own configuration file or modify the existing ones.
Log files and checkpoints are saved in the `train_output` folder.

Example of a configuration file:
```yaml
dataset_id : "COCO"
file_list_path: "datasets/COCO/lists/train_data_list.txt" # list containing the path to the images and the annotations
base_path: "datasets/COCO" # path to the dataset folder, concatenates with the paths in the list if not empty
output_folder: "/mnt/train_output" # path to save the output files
checkpoints: # absolute path to checkpoint
num_classes_checkpoint: 201 #classes used for training the checkpoint

num_classes: 81 #with background

batch_size: 32
lr : 0.001
weight_decay: 0.005
epochs: 200
workers: 4
num_checkpoints: 3 # number of checkpoints to save during training

scheduler:
  name: "cosine"
  warmup_epochs: 5
  base_lr: 1.25e-3 # 4096 batch-size
  warmup_lr: 1.25e-6
  min_lr: 1.25e-5

model:
  id: "FPNSwin2Small-224"
  freeze_decoder: False
  freeze_encoder: False
```
In particular, `the model id` specifies the model to be used. You can find available models and settings in the `models.yaml` file. Models are taken from `torchseg` library, see [here](https://github.com/isaaccorley/torchseg/tree/main).
The `freeze_decoder` and `freeze_encoder` parameters specify whether to freeze the encoder and decoder parts of the model, respectively.

**TO DO:** Add support to [MM Segmentation](https://github.com/open-mmlab/mmsegmentation). 
**TO DO:** Add support to [HF Transformers](https://huggingface.co/docs/transformers/index). 

## Evaluation
```bash
python test.py --config config/COCO_test.yaml
```
The script tests all the pre-trained models specified in the configuration file. The results are saved in the `test_output` folder.