#from model import UNet3D
from model_adaptable import UNet3D_adaptable
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from utils import DiceScore, DiceLoss

class Segmenter_2labels(pl.LightningModule):
    def __init__(self, depth, start_filts):
        super().__init__()

        #self.hparams = hparams
        self.depth = depth
        self.start_filts = start_filts

        self.save_hyperparameters("depth", "start_filts")

        self.model=UNet3D_adaptable(num_classes=2, in_channels=1, depth=self.depth, start_filts=self.start_filts)
        #self.model=UNet3D()
        #self.optimizer = torch.optim.Adadelta(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        #self.loss = torch.nn.CrossEntropyLoss()
        #self.loss = DiceScore(num_classes = 8, ignore_back=0, mean_loss=True)
        """if loss_funct == "Cross":
            self.loss = torch.nn.CrossEntropyLoss()
        if loss_funct == "Dice":
            self.loss = DiceLoss(nCls=8, average="micro")"""

        self.loss = torch.nn.CrossEntropyLoss()
        
        #self.dice = DiceScore()

        #self.log("batch_size", batch_size)

        

    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        img = batch["CT"]["data"].to(dtype = torch.float32)
        mask = batch["Label"]["data"][:,0]
        mask = mask.long()

        ##APLICAMOS EL MAPEO PARA EVITAR PROBLEAS CON LOS RANGOS DE LAS ETIQUETAS
        #mask = mask.apply_(aplicar_mapeo)

        pred = self(img)
        #pred_loss = torch.softmax(pred, dim=1)
        #pred_loss = torch.argmax(pred_loss, dim=1)
        loss = self.loss(pred, mask)

        #Logs
        self.log("Train Loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        img = batch["CT"]["data"].to(dtype=torch.float32)
        mask = batch["Label"]["data"][:,0]
        mask = mask.long()

        ##APLICAMOS EL MAPEO PARA EVITAR PROBLEAS CON LOS RANGOS DE LAS ETIQUETAS
        #mask = mask.apply_(aplicar_mapeo)

        pred = self(img)
        #pred_loss = torch.softmax(pred, dim=1)
        #pred_loss = torch.argmax(pred_loss, dim=1)
        loss = self.loss(pred, mask)

        #Logs
        self.log("Val Loss", loss)
        self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Val")

        #Aqui vamos a añadir para loggear los hyperparámetros
        #self.logger.experiment.add_hparams(self.hparams, {"loss": loss}) ##TENEMOS QUE CREAR ESTOS. 

        return loss
    
    
    
    def log_images(self, img, pred, mask, name):
        results = []
        
        pred = torch.nn.Softmax(dim=1)(pred)
        pred = torch.argmax(pred, dim=1)

        #Siempre vamos a representar la slice número 150
        axial_slice = 80

        fig, axis = plt.subplots(1,2)
        axis[0].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        axis[0].imshow(pred[0][:,:,axial_slice], alpha=0.5, cmap="Reds")
        axis[0].set_title("Prediction")

        axis[1].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        axis[1].imshow(mask[0][:,:,axial_slice], alpha=0.5, cmap="Reds")
        axis[1].set_title("Ground Truth")

        self.logger.experiment.add_figure(f"{name} Prediction vs Label", fig, self.global_step)

    def configure_optimizers(self):
        return [self.optimizer] #HAY QUE HACERLO
