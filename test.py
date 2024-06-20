import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
from tqdm import tqdm
from utils.generic import load_config, print_config
from utils.SemSegDataset import SemSegDataset
from utils.metrics import SegmentationMetrics
import torchseg
import time

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3_ResNet101_Weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--config', type=str, default='config/COCO_test.yaml',
                        help='Path to test configuration file')
    parser.add_argument('--model_desc', type=str, default='config/models.yaml',
                        help='Path to model description file')
    args = parser.parse_args()

    # Load test configuration
    config = load_config(args.config)
    print_config(config)
    config['model_desc'] = args.model_desc

    # Create test dataset
    # Read file list from txt file
    with open(config["test_data_list"], 'r') as f:
        file_list = f.read().splitlines()
    test_dataset = SemSegDataset(file_list, base_path=config["base_path"], transform=None)

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize dictionary to store results
    results = {}

    # Output file
    os.makedirs(config["output_folder"], exist_ok=True)
    output_file = os.path.join(config["output_folder"], f"{config['dataset_id']}_results.yaml")

    # load models descriptions
    model_desc = load_config(config['model_desc'])
    class_ids = list(range(config['num_classes']))

    # Check if the file exists
    for model_path in config["model_list"]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The file {model_path} does not exist.")

    # Load model
    # Iterate over models in the list
    for model_path in config["model_list"]:

        # # Load model
        # model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        # model.classifier = DeepLabHead(2048, config['num_classes'])  # Assuming num_classes is defined in the config
        # model.load_state_dict(torch.load(model_path))

        # Load model
        model_id = os.path.basename(model_path).split('_')[1]
        model_params = model_desc.get(model_id)

        if model_params:
            if model_params['head'] == 'Unet':
                model = getattr(torchseg, model_params['head'])(
                    model_params['backbone'],
                    encoder_weights=None,
                    classes=config['num_classes'],
                    encoder_depth=model_params['encoder_depth'],
                    encoder_indices=model_params.get('encoder_indices', None),
                    head_upsampling=model_params.get('head_upsampling', 1),
                    decoder_channels=model_params['decoder_channels'],
                    encoder_params=model_params['encoder_params']
                )
            elif model_params['head'] == 'FPN':
                model = getattr(torchseg, model_params['head'])(
                    model_params['backbone'],
                    encoder_weights=None,
                    classes=config['num_classes'],
                    encoder_depth=model_params['encoder_depth'],
                    encoder_indices=model_params.get('encoder_indices', None),
                    decoder_pyramid_channels=model_params['decoder_pyramid_channels'],
                    decoder_segmentation_channels=model_params['decoder_segmentation_channels'],
                    upsampling=model_params.get('upsampling', 4),
                    encoder_params=model_params['encoder_params']
                )
            else:
                print(f"Model head {model_params['head']} not supported")
                continue
        else:
            print(f"Model parameters not found for model {model_id}")
            continue

        model.load_state_dict(torch.load(model_path))
        print(f"Loaded  {model_id} from {model_path}")
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        meter = SegmentationMetrics(config['num_classes'])

        test_time_start = time.time()

        # Test loop
        with torch.no_grad():
            for idx, (images, masks) in enumerate(test_loader):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                # Process outputs as needed for evaluation
                #masks = masks.numpy()
                pred_masks = torch.argmax(outputs, dim=1)

                meter.update_metrics(pred_label=pred_masks, gt_label=masks)

                #if idx > 50:
                #    break

        mIoU = meter.get_miou()
        test_time_end = time.time()
        hours = int((test_time_end - test_time_start) // 3600)
        minutes = int(((test_time_end - test_time_start) % 3600) // 60)
        print(f"******* Testing model {model_path} *******")
        print(f"Testing time [hh:mm]: {hours:02d}:{minutes:02d}")
        data = meter.print_table(class_ids)
        results[model_path] = data

        # Save results after each model
        with open(output_file, 'w') as file:
            yaml.dump(results, file)
            print(f"Results saved to {output_file}")

        print("******************************\n")

    print("Testing finished")

    print("Read out file TRIAL")
    with open(output_file, 'r') as file:
        results = yaml.load(file, Loader=yaml.SafeLoader)
        for model_path, metrics in results.items():
            model_name = model_path.split('/')[-1].replace('.pth', '')
            mIoU = metrics.get('mIoU', None)
            if mIoU is not None:
                print(f"{model_name}: {mIoU} [mIoU]")
