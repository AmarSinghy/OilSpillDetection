import torch
from torch import nn
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        """
        Initialize the DoubleConv block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Perform forward pass through the DoubleConv block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.conv(x)


class UNET(nn.Module):
    """UNet architecture for image segmentation."""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        Initialize the UNet model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            features (list): List of feature channels for each encoder block.
        """
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.apply_weight_decay()

    def forward(self, x):
        """
        Perform forward pass through the UNet model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def apply_weight_decay(self, weight_decay=1e-5):
        """
        Apply weight decay to the convolutional layers.

        Args:
            weight_decay (float): Weight decay value.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.weight_decay = weight_decay


def test():
    """Perform a simple test on the UNet model."""
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()

    
import torch
from torch import nn
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

class BinaryUNetModel(pl.LightningModule):
    """
    PyTorch Lightning module for binary image segmentation using UNet architecture.
    """

    def __init__(self, pretrained=True, in_channels=3, num_classes=1, lr=3e-4):
        """
        Initialize the BinaryUNetModel.

        Args:
            pretrained (bool): Whether to use a pretrained UNet model.
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            lr (float): Learning rate for the optimizer.
        """
        super(BinaryUNetModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        self.model = UNET(self.in_channels, self.num_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')

        self.outputs = []

    def forward(self, x):
        """
        Forward pass through the BinaryUNetModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (Tuple): Input batch containing images and labels.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Loss tensor.
        """
        x, y = batch
        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        self.train_acc(preds, y)

        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc.compute(), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (Tuple): Input batch containing images and labels.
            batch_idx (int): Batch index.
        """
        x, y = batch
        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        self.val_acc(preds, y)

        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc.compute(), on_epoch=True)

    def on_train_epoch_end(self):
        """
        Perform operations at the end of each training epoch.
        """
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        train_acc = self.trainer.callback_metrics['train_acc'].item()

        print('Epoch Training Loss:', train_loss)
        print('Epoch Training Accuracy:', train_acc)

    def on_validation_epoch_end(self):
        """
        Perform operations at the end of each validation epoch.
        """
        val_loss = self.trainer.callback_metrics['val_loss'].item()
        val_acc = self.trainer.callback_metrics['val_acc'].item()

        print('Epoch Validation Loss:', val_loss)
        print('Epoch Validation Accuracy:', val_acc)
