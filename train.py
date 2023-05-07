import os
import sys
import random
from typing import Optional
from PIL import Image
import numpy as np
import torch
import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

from nn import VGG16, TransformerNet, custom_denormalize, gram_matrix, style_transform, train_transform
from settings import DotEnvControl, Settings
from tg_sender import TgSender



class StyleTrasferTrainer():

    def __init__(self, params: Settings, env: DotEnvControl, tg_sender: Optional[TgSender] = None):
        self.params = params
        self.env = env
        self.style_name = params.style_image.split("/")[-1].split(".")[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() and env.GPU else "cpu")
        self.train_dataset = datasets.ImageFolder(
            params.dataset_path, train_transform(params.image_size)
        )
        self.dataloader = DataLoader(self.train_dataset, batch_size=params.batch_size)
        os.makedirs(f"images/outputs/{self.style_name}-training", exist_ok=True)
        os.makedirs(f"checkpoints", exist_ok=True)

        # Defines networks
        self.transformer = TransformerNet().to(self.device)
        self.vgg = VGG16(requires_grad=False).to(self.device)

        # Load checkpoint model
        if params.checkpoint_model:
            self.transformer.load_state_dict(torch.load(params.checkpoint_model))

        # Define optimizer and loss
        self.optimizer = Adam(self.transformer.parameters(), params.lr)
        self.l2_loss = torch.nn.MSELoss().to(self.device)

        # Load style image
        style = style_transform(params.style_size)(Image.open(params.style_image))
        style = style.repeat(params.batch_size, 1, 1, 1).to(self.device)

        # Extract style features
        features_style = self.vgg(style)
        self.gram_style = [gram_matrix(y) for y in features_style]

        # Sample 8 images for visual evaluation of the model
        image_samples = []
        for path in random.sample(glob.glob(f"{params.dataset_path}/*/*.jpg"), 8):
            image_samples += [style_transform(params.image_size)(Image.open(path))]
        self.image_samples = torch.stack(image_samples)

        self.tg_sender = tg_sender

        

    def save_sample(self, batches_done):
        """ Evaluates the model and saves image samples """
        self.transformer.eval()
        with torch.no_grad():
            output = self.transformer(self.image_samples.to(self.device))
        image_grid = custom_denormalize(torch.cat((self.image_samples.cpu(), output.cpu()), 2))
        save_image(image_grid, f"images/outputs/{self.style_name}-training/{batches_done}.jpg", nrow=4)
        self.transformer.train()


    def train(self):
        self.epoch_metrics = {"content": [], "style": [], "total": []}
        for epoch in range(self.params.epochs):
            for batch_i, (images, _) in enumerate(self.dataloader):
                print(f"start batch {batch_i}\n")
                self.optimizer.zero_grad()

                images_original = images.to(self.device)
                images_transformed = self.transformer(images_original)

                # Extract features
                features_original = self.vgg(images_original)
                features_transformed = self.vgg(images_transformed)

                # Compute content loss as MSE between features
                content_loss = self.params.lambda_content * \
                    self.l2_loss(features_transformed.relu2_2, features_original.relu2_2)

                # Compute style loss as MSE between gram matrices
                style_loss = 0
                for ft_y, gm_s in zip(features_transformed, self.gram_style):
                    gm_y = gram_matrix(ft_y)
                    style_loss += self.l2_loss(gm_y, gm_s[: images.size(0), :, :])
                style_loss *= self.params.lambda_style

                total_loss = content_loss + style_loss
                total_loss.backward()
                self.optimizer.step()

                self.epoch_metrics["content"] += [content_loss.item()]
                self.epoch_metrics["style"] += [style_loss.item()]
                self.epoch_metrics["total"] += [total_loss.item()]

                out_learn_proc = \
                "\r[Epoch %d/%d] [Batch %d/%d] [Content lose: %.2f (%.2f) Style lose: %.2f (%.2f) Total lose: %.2f (%.2f)]" % (
                        epoch + 1,
                        self.params.epochs,
                        batch_i * self.params.batch_size,
                        len(self.train_dataset),
                        content_loss.item(),
                        np.mean(self.epoch_metrics["content"]),
                        style_loss.item(),
                        np.mean(self.epoch_metrics["style"]),
                        total_loss.item(),
                        np.mean(self.epoch_metrics["total"]),
                    )

                sys.stdout.write(
                    out_learn_proc
                )

                batches_done = epoch * len(self.dataloader) + batch_i + 1
                if batches_done % self.params.sample_interval == 0:
                    self.save_sample(batches_done)
                    if self.tg_sender is not None:
                        self.tg_sender.send_file_to_tg(checkpoint_filename, checkpoint_filename)

                if self.params.checkpoint_interval > 0 and batches_done % self.params.checkpoint_interval == 0:
                    style_name = os.path.basename(self.params.style_image).split(".")[0]
                    checkpoint_filename = f"checkpoints/{style_name}_{batches_done}.pth"
                    torch.save(self.transformer.state_dict(), f"checkpoints/{style_name}_{batches_done}.pth")
                    if self.tg_sender is not None:
                        self.tg_sender.send_file_to_tg(checkpoint_filename, checkpoint_filename)
