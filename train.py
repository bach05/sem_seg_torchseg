import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import os
from utils.SemSegDataset import SemSegDataset
import argparse
from utils.generic import load_config, print_config
from utils.metrics import SegmentationMetrics
from utils.visualization import save_tensorboard_images, get_images_with_mask, generate_distinguishable_colors
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import time
import numpy as np

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3_ResNet101_Weights

import torchseg

class Trainer:
    def __init__(self, file_list_path, base_path=None, transform=None, config=None):

        self.base_path = base_path
        self.config = config

        self.output_folder = os.path.join(self.config['output_folder'], self.config['dataset_id'])
        os.makedirs(self.output_folder, exist_ok=True)

        os.makedirs(os.path.join(self.output_folder, "tensorboard"), exist_ok=True)

        self.model_log_folder = os.path.join(self.output_folder, "checkpoints")
        os.makedirs(self.model_log_folder, exist_ok=True)

        # load models descriptions
        self.model_desc = load_config(self.config['model_desc'])

        # Define the model
        model_name = self.config['model']['id']

        # Create a list of colors, one for each class
        self.colors = generate_distinguishable_colors(self.config['num_classes'])

        # Get the parameters for the specified model name
        self.model_params = self.model_desc.get(model_name)

        # Load pre-trained weights
        if self.model_params:
            if self.config['checkpoints']:
                if self.model_params['head'] == 'Unet':
                    self.model = getattr(torchseg, self.model_params['head'])(
                        self.model_params['backbone'],
                        encoder_weights=None,
                        classes=self.config['num_classes_checkpoint'],
                        encoder_depth=self.model_params['encoder_depth'],
                        encoder_indices=self.model_params.get('encoder_indices', None),
                        head_upsampling=self.model_params.get('head_upsampling', 1),
                        decoder_channels=self.model_params['decoder_channels'],
                        encoder_params=self.model_params['encoder_params']
                    )
                elif self.model_params['head'] == 'FPN':
                    self.model = getattr(torchseg, self.model_params['head'])(
                        self.model_params['backbone'],
                        encoder_weights=None,
                        classes=self.config['num_classes_checkpoint'],
                        encoder_depth=self.model_params['encoder_depth'],
                        encoder_indices=self.model_params.get('encoder_indices', None),
                        decoder_pyramid_channels=self.model_params['decoder_pyramid_channels'],
                        decoder_segmentation_channels=self.model_params['decoder_segmentation_channels'],
                        upsampling=self.model_params.get('upsampling', 4),
                        encoder_params=self.model_params['encoder_params']
                    )

                self.model.load_state_dict(torch.load(self.config['checkpoints']))
                in_channels = self.model.segmentation_head[0].in_channels
            
                k_size = 3
                if self.model_params['head'] == 'FPN':
                    k_size = 1
                if self.model_params['head'] == 'Unet':
                    k_size = 3

                print(f"Using kernel size {k_size} for the segmentation head")
                self.model.segmentation_head = torchseg.base.SegmentationHead(in_channels, 
                                                                              self.config['num_classes'], 
                                                                              kernel_size=k_size,
                                                                              upsampling=self.model_params.get('upsampling', 4))
            else:

                if self.model_params['head'] == 'Unet':
                    self.model = getattr(torchseg, self.model_params['head'])(
                        self.model_params['backbone'],
                        encoder_weights=None,
                        classes=self.config['num_classes'],
                        encoder_depth=self.model_params['encoder_depth'],
                        encoder_indices=self.model_params.get('encoder_indices', None),
                        head_upsampling=self.model_params.get('head_upsampling', 1),
                        decoder_channels=self.model_params['decoder_channels'],
                        encoder_params=self.model_params['encoder_params']
                    )
                elif self.model_params['head'] == 'FPN':
                    self.model = getattr(torchseg, self.model_params['head'])(
                        self.model_params['backbone'],
                        encoder_weights=None,
                        classes=self.config['num_classes'],
                        encoder_depth=self.model_params['encoder_depth'],
                        encoder_indices=self.model_params.get('encoder_indices', None),
                        decoder_pyramid_channels=self.model_params['decoder_pyramid_channels'],
                        decoder_segmentation_channels=self.model_params['decoder_segmentation_channels'],
                        upsampling=self.model_params.get('upsampling', 4),
                        encoder_params=self.model_params['encoder_params']
                    )
                else:
                    print(f"Model head {self.model_params['head']} not supported")
                    exit(code=os.EX_CONFIG)

        else:
            print(f"Error: Model parameters for {model_name} not found.")
            exit(code=os.EX_CONFIG)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.config['model']['freeze_encoder']:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        if self.config['model']['freeze_decoder']:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        self.train_loader, self.val_loader = self.setup(file_list_path, transform)

        # Define loss function and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        if self.config['scheduler']:
            num_steps = len(self.train_loader) * self.config['epochs']
            warmup_steps = self.config['scheduler']['warmup_epochs'] * len(self.train_loader)
            if self.config['scheduler']['name'] == 'cosine':
                self.scheduler = OneCycleLR(self.optimizer,
                                       max_lr=self.config['scheduler']['base_lr'],
                                       total_steps=num_steps,
                                       pct_start=warmup_steps/num_steps,
                                       anneal_strategy='cos',
                                       div_factor=self.config['scheduler']['base_lr'] / self.config['scheduler']['warmup_lr'],
                                       final_div_factor=self.config['scheduler']['base_lr'] / self.config['scheduler']['min_lr'],
                                       cycle_momentum=False)

        if self.config['model']:
            self.train_id_string = f"{self.config['dataset_id']}_{self.config['model']['id']}"
            if self.config['model']['freeze_encoder']:
                self.train_id_string += "_freeze_encoder"
            if self.config['model']['freeze_decoder']:
                self.train_id_string += "_freeze_decoder"
        else:
            self.train_id_string = f"{self.config['dataset_id']}_deeplabv3_resnet101"

        self.val_meter = SegmentationMetrics(config['num_classes'])
        self.train_meter = SegmentationMetrics(config['num_classes'])

        self.class_ids = [i for i in range(self.config['num_classes'])]
        self.save_every = self.config['epochs'] // self.config['num_checkpoints']

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_folder, "tensorboard", self.train_id_string))


    def setup(self, file_list_path, transform=None):
        # Read file list from txt file
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()

        # Create dataset
        input_shape = self.model_params['input_size'] if self.config['model'] else (256, 256)
        dataset = SemSegDataset(file_list, base_path=self.base_path, transform=transform,
                                max_classes=self.config['num_classes'], mode="train_val",
                                resize=input_shape)

        # Split dataset into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Define data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                  num_workers=self.config['workers'])
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False,
                                num_workers=self.config['workers'])

        return train_loader, val_loader

    def train(self, epochs=10):
        best_val_loss = float('inf')
        iters = len(self.train_loader)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.model.train()
            running_loss = 0.0
            epoch_start = time.time()
            for i, (images, masks) in enumerate(self.train_loader):
                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                logits = outputs

                pred_mask = torch.argmax(logits, dim=1)
                loss = self.criterion(logits.squeeze(1), masks.squeeze(1).long())

                loss.backward()
                self.optimizer.step()
                if self.config['scheduler']:
                    self.scheduler.step()

                running_loss += loss.item()
                self.train_meter.update_metrics(pred_label=pred_mask, gt_label=masks)

                # if i > 100:
                #     break

            epoch_end = time.time()
            epoch_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            #log miou
            #self.writer.add_scalar('mIoU/train', self.train_meter.compute_miou())
            #self.train_meter.print_table(self.class_ids)
            #print(f"[TRAIN] mIoU: {self.train_meter.get_miou():.3f}")


            # save images and masks
            images_with_masks = get_images_with_mask(images, logits, colors=self.colors)
            save_tensorboard_images(images=images_with_masks,
                                    label='Pred/train',
                                    logger=self.writer,
                                    iters=epoch,
                                    nrow=4)
            #save images and gt masks
            gt_logits = torch.nn.functional.one_hot(masks.long(), num_classes=self.config["num_classes"]).permute(0,3,1,2) #one-hot encodinf of masks
            images_with_masks = get_images_with_mask(images, gt_logits, colors=self.colors)
            save_tensorboard_images(images=images_with_masks,
                                    label='GT/train',
                                    logger=self.writer,
                                    iters=epoch,
                                    nrow=4)

            time_seconds = (epoch_end - epoch_start)
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = int(time_seconds % 60)
            print(f"[Training] Loss: {epoch_loss}, Time [hh:mm:ss]: {hours:02d}:{minutes:02d}:{seconds:02d}")

            # Validation loop
            self.model.eval()
            val_loss = 0.0
            val_time_start = time.time()
            with torch.no_grad():
                for images, masks in self.val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)

                    outputs = self.model(images)
                    if self.config['model']:
                        logits = outputs
                    else:
                        logits = outputs['out']
                    pred_mask = torch.argmax(logits, dim=1)
                    loss = self.criterion(logits, masks.squeeze(1).long())

                    val_loss += loss.item()
                    self.val_meter.update_metrics(pred_label=pred_mask, gt_label=masks)

            val_time_end = time.time()
            val_loss /= len(self.val_loader)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('mIoU/val', self.val_meter.get_miou(), epoch)

            # save images and masks
            images_with_masks = get_images_with_mask(images, logits, colors=self.colors)
            save_tensorboard_images(images=images_with_masks,
                                    label='Pred/val',
                                    logger=self.writer,
                                    iters=epoch,
                                    nrow=4)
            # save images and gt masks
            gt_logits = torch.nn.functional.one_hot(masks.long(),
                                                    num_classes=self.config["num_classes"]).permute(0, 3, 1, 2)  # one-hot encodinf of masks
            images_with_masks = get_images_with_mask(images, gt_logits, colors=self.colors)
            save_tensorboard_images(images=images_with_masks,
                                    label='GT/val',
                                    logger=self.writer,
                                    iters=epoch,
                                    nrow=4)

            time_seconds = (val_time_end - val_time_start)
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = int(time_seconds % 60)
            print(f"[Validation] Loss: {val_loss}, Time [hh:mm:ss]: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print(f"[Validation] mIoU: {self.val_meter.get_miou():.3f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_log_folder,
                                                                 self.train_id_string+"_best_model.pth"))
                print(f">>>>>>>>>>>> Update best model at epoch {epoch}")

            # save checkpoint
            if epoch % self.save_every == 0 and epoch > 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_log_folder,
                                                                 f"{self.train_id_string}_epoch_{epoch}.pth"))

        self.writer.close()
        self.val_meter.print_table(self.class_ids)
        print('Finished Training')

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument('--config', type=str, default='config/COCO_train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_desc', type=str, default='config/models.yaml',
                        help='Path to model description file')
    parser.add_argument('--vis_images', action='store_true', help='Visualize images with pseudo masks')

    args = parser.parse_args()

    config = load_config(args.config)
    config['model_desc'] = args.model_desc

    print_config(config)


    trainer = Trainer(file_list_path=config["file_list_path"],
                      base_path=config["base_path"],
                      transform=None,
                      config=config,
                      )

    trainer.train(epochs=config["epochs"])