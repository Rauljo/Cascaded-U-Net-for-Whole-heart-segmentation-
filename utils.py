import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
import csv
import os
import torch.nn as nn
from torch.nn import init
from dataset import cambiar_tam, np_to_image
import SimpleITK as sitk
import torchio as tio
import nibabel as nib

import torch
from pathlib import Path
import torchio as tio
import argparse
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from dataset import cambiar_tam, np_to_image
import time

def create_trainer(train_loader, val_loader=None, max_epochs=100, gpus=1, save_dir="./logs", name="default"):
    checkpoint_callback = ModelCheckpoint(
    monitor = "Val Loss",
    save_top_k = 1,
    mode = "min"
    )

    if val_loader is None: #Guardamos checkpoints según train loss. Aunque igual lo mejor es coger el último y punto. Xa veremos. 
        checkpoint_callback = ModelCheckpoint(
        monitor = "Train Loss",
        save_top_k = 5,
        mode = "min"
    )

    if val_loader != None:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='Val Loss',
            patience=25,
            min_delta=0.00,
            mode='min',
            verbose=True
        )


    if val_loader is None:
        gpus = gpus
        trainer = pl.Trainer(accelerator='gpu', logger=TensorBoardLogger(save_dir = save_dir, name = name), log_every_n_steps=1, callbacks=checkpoint_callback, max_epochs=max_epochs, limit_val_batches=0)
        return trainer

    else:
        gpus = gpus
        trainer = pl.Trainer(accelerator='gpu', logger=TensorBoardLogger(save_dir = save_dir, name = name), log_every_n_steps=1, callbacks=[checkpoint_callback, early_stop_callback], max_epochs=max_epochs)
        return trainer

class DiceScore(torch.nn.Module):
    def __init__(self, num_classes=8, ignore_back=None, mean_loss=False):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_back = ignore_back
        self.mean_loss = mean_loss

    def forward(self, pred, target):
        clases = self.num_classes

        target = torch.flatten(target)
        pred = torch.flatten(pred)

        target_one_hot = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=8)
        pred_one_hot = torch.nn.functional.one_hot(pred.to(torch.int64), num_classes=8)

        dice = []
        for clase in range(clases):
            if clase == self.ignore_back:
                continue
            resultado = (target_one_hot[:, clase] * pred_one_hot[:, clase]).sum()
            denum = target_one_hot[:, clase].sum() + pred_one_hot[:, clase].sum()

            if denum == 0: #Si no hay predicho ni ground_truth
               dice.append(1)
               continue

            dice_indv = (2*resultado) / denum
            dice.append(dice_indv)

        if self.mean_loss:
            return 1 - torch.mean(torch.tensor(dice))
        
        return dice
    
def test_step(model, test_loader, path):
    dice = DiceScore(num_classes=8)

    scores = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(test_loader):
            img = batch["CT"]["data"].to(dtype=torch.float32)
            #img = img.cuda()
            mask = batch["Label"]["data"][:,0]
            #mask = mask.long().cuda()
            mask = mask.long()

            pred = model(img)
            loss = model.loss(pred, mask)

            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)

            scores.append(dice(pred, mask))
            #print(scores[-1])

    
    #return (loss, torch.mean(torch.stack([torch.Tensor(score) for score in scores]), dim=0))
    return (loss, [torch.Tensor(score) for score in scores])

def test_step_2labels(model, test_loader, path):
    dice = DiceScore(num_classes=2)

    scores = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(test_loader):
            img = batch["CT"]["data"].to(dtype=torch.float32)
            #img = img.cuda()
            mask = batch["Label"]["data"][:,0]
            #mask = mask.long().cuda()
            mask = mask.long()

            pred = model(img)
            loss = model.loss(pred, mask)

            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)

            #Primero convertimos la mascara en caja
            mask = mask.squeeze().cpu().numpy().transpose(2,1,0)
            mask_img = sitk.GetImageFromArray(mask)

            componente = sitk.ConnectedComponent(mask_img)
            sorted_component_image = sitk.RelabelComponent(componente, sortByObjectSize=True)
            largest_component = sorted_component_image == 1

            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(mask_img, largest_component)
            box = stats.GetBoundingBox(1)

            box_mask_image = sitk.Image(mask_img.GetSize(), sitk.sitkInt16)

            box_mask_array = sitk.GetArrayFromImage(box_mask_image)

            #Relleno la caja
            for i in range(box[0], box[1]):
                for j in range(box[2], box[3]):
                    for k in range(box[4], box[5]):
                        box_mask_array[k,j,i] = 1

            #Ahora convertimos la prediccion en caja
            pred = pred.squeeze().cpu().numpy().transpose(2,1,0)
            pred_img = sitk.GetImageFromArray(pred)

            componente = sitk.ConnectedComponent(pred_img)
            sorted_component_image = sitk.RelabelComponent(componente, sortByObjectSize=True)
            largest_component = sorted_component_image == 1

            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(pred_img, largest_component)
            box = stats.GetBoundingBox(1)

            box_pred_image = sitk.Image(pred_img.GetSize(), sitk.sitkInt16)

            box_pred_array = sitk.GetArrayFromImage(box_pred_image)

            #Relleno la caja
            for i in range(box[0], box[1]):
                for j in range(box[2], box[3]):
                    for k in range(box[4], box[5]):
                        box_pred_array[k,j,i] = 1

            #Calculo el dice
            box_mask_array = box_mask_array.flatten()
            box_pred_array = box_pred_array.flatten()

            intersection = np.logical_and(box_mask_array, box_pred_array)
            union = np.logical_or(box_mask_array, box_pred_array)

            iou = intersection.sum() / union.sum()

            scores.append(torch.tensor([iou]))

            #scores.append(dice(pred, mask))
            #print(scores[-1])

    
    #return (loss, torch.mean(torch.stack([torch.Tensor(score) for score in scores]), dim=0))
    return (loss, [torch.Tensor(score) for score in scores])

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def weight_init(m):
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


def log_results(dice, loss, name, archivo = "results.csv"):
   dices_medios = torch.mean(torch.stack(dice), dim=0)
   dices_desviacion = torch.std(torch.stack(dice), dim=0)
   loss_medio = torch.mean(torch.tensor(loss))
   
   archivo_existe = os.path.exists(archivo)
   
   with open(archivo, mode='a', newline='') as archivo_csv:
        campos = ['Nombre'] + [f'Dice{i}' for i in range(1, 9)] + ['Loss']
        writer = csv.DictWriter(archivo_csv, fieldnames=campos)

        if not archivo_existe:
            writer.writeheader()

        # Extraer valores del tensor

        valores_lista = list(zip(dices_medios.tolist(), dices_desviacion.tolist()))

        # Escribir la fila
        writer.writerow({'Nombre': name, **{f'Dice{i}': valor for i, valor in enumerate(valores_lista, start=1)}, 'Loss': loss_medio.item()})

def dice_loss(pred, ref, nCls, average, eps: float = 1e-8):
    # Input tensors will have shape (Batch, Class)
    # Dimension 0 = batch
    # Dimension 1 = class code or predicted logit
    
    # compute softmax over the classes axis to convert logits to probabilities
    pred_soft = torch.softmax(pred, dim=1)
    pred_soft = torch.argmax(pred_soft, dim=1)
    pred_soft = torch.nn.functional.one_hot(pred_soft, num_classes = nCls).permute(0, 4, 1, 2, 3)

    # create reference one hot tensors
    ref_one_hot = torch.nn.functional.one_hot(ref, num_classes = nCls).permute(0, 4, 1, 2, 3)

    # Calculate the dice loss
    if average == "micro":
        # Use dim=1 to aggregate results across all classes
        intersection = torch.sum(pred_soft[:,1:,:,:,:] * ref_one_hot[:,1:,:,:,:], dim=1)
        cardinality = torch.sum(pred_soft[:,1:,:,:,:] + ref_one_hot[:,1:,:,:,:], dim=1)
    else:
        # With no dim argument, will be calculated separately for each class
        intersection = torch.sum(pred_soft[:,1:,:,:,:] * ref_one_hot[:,1:,:,:,:])
        cardinality = torch.sum(pred_soft[:,1:,:,:,:] + ref_one_hot[:,1:,:,:,:])

    dice_score = 2.0 * intersection / (cardinality + eps)
    dice_loss = -dice_score + 1.0

    # reduce the loss across samples (and classes in case of `macro` averaging)
    dice_loss = torch.mean(dice_loss)

    return dice_loss



class DiceLoss(nn.Module):
    def __init__(self, nCls, average, eps: float = 1e-8) -> None:
        super().__init__()
        self.nCls = nCls
        self.average = average
        self.eps = eps

    def forward(self, pred, ref):
        return dice_loss(pred, ref, self.nCls, self.average, self.eps)

#Utils for prediction

def preprocesar_imagen(img, nuevo_tamanho, orient_final):

    ##Ahora no tenemos labels, entonces el primer paso es hacer la prediccion. 
    img_preprocesada = cambiar_tam(img, nuevo_tamanho, orient_final)
    img_array = sitk.GetArrayFromImage(img_preprocesada)
    img_array[img_array < -1024] = -1024
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    img_preprocesada = np_to_image(img_array, img_preprocesada.GetOrigin(), img_preprocesada.GetSpacing(), img_preprocesada.GetDirection(), sitk.sitkFloat32)

    return img_preprocesada

def prediccion_crop(img_preprocesada, img, model_crop, device, img_nib):

    subject = tio.Subject({"CT": tio.ScalarImage.from_sitk(img_preprocesada)})

    with torch.no_grad():
        model_crop.eval()
        datos = subject["CT"][tio.DATA].to(device).unsqueeze(0)
        tiempo_inicio = time.time()
        pred = model_crop(datos)
        tiempo_fin = time.time()
        #print(f"Tiempo de inferencia CROP: {tiempo_fin - tiempo_inicio}s")
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)

    pred = pred.squeeze().cpu().numpy().transpose(2,1,0)

    #pred = np.where(np.isin(pred, list(labels_crop.keys())), pred, -1)
    #pred = np.vectorize(labels_crop.get)(pred)

    img_inferida = np_to_image(pred, img_preprocesada.GetOrigin(), img_preprocesada.GetSpacing(), img_preprocesada.GetDirection(), sitk.sitkInt16)

    """if nib.aff2axcodes(img_nib.affine) != ("L", "A", "S"):
        img_inferida_final = cambiar_tam(img_inferida, img.GetSize(), "RAS", mask=True)
    else:
        img_inferida_final = cambiar_tam(img_inferida, img.GetSize(), "LAS", mask=True)"""

    #AÑADIDOOOOOOOO
    img_prueba = sitk.DICOMOrient(img, "LPS")

    img_inferida_final = cambiar_tam(img_inferida, img_prueba.GetSize(), "".join(list(nib.aff2axcodes(img_nib.affine))), mask=True)

    return img_inferida_final, (tiempo_fin - tiempo_inicio)

def crop_imagen(img_inferida_final, MARGEN_SEGURIDAD, img):
    centro = [0,0,0]
    box_ajustado = [0,0,0,0,0,0]
    boxes = []

    #Ahora tenemos las labels inferidas, vamos con el crop
    componente = sitk.ConnectedComponent(img_inferida_final)
    sorted_component_image = sitk.RelabelComponent(componente, sortByObjectSize=True)
    largest_component = sorted_component_image == 1

    stats = sitk.LabelStatisticsImageFilter()

    """if img.GetSize() != largest_component.GetSize():
        #largest_component = sitk.PermuteAxes(largest_component, [2,0,1])
        print("Largest: ", largest_component.GetSize())
        print("Imagen: ", img.GetSize())
        #largest_component = sitk.PermuteAxes(largest_component, [0,2,1])
        #En ese caso primero sacamos la matriz de largest_componente
        largest_component_array = sitk.GetArrayFromImage(largest_component)

        print("Largest array: ", largest_component_array.shape)

        #Ahora transponemos a 0 2 1 
        largest_component_array = largest_component_array.transpose(1,0,2)
        print("Largest array transpose: ", largest_component_array.shape)

        #Ahora creamos una imagen largest_componente con la misma informacion que img
        largest_component = sitk.GetImageFromArray(largest_component_array)
        largest_component.SetOrigin(img.GetOrigin())
        largest_component.SetSpacing(img.GetSpacing())
        largest_component.SetDirection(img.GetDirection())
        print("Largest: ", largest_component.GetSize())"""
        

    

    stats.Execute(img, largest_component)
    box = stats.GetBoundingBox(1)

    ## Añadimos un margen de seguridad al crop del 15%
    x = (box[1]-box[0]) * img.GetSpacing()[0] * (1+MARGEN_SEGURIDAD)
    x = x  / img.GetSpacing()[0] #Vemos cuanto es en pixeles
    x = x - (box[1]-box[0]) #Vemos la diferencia
    box_ajustado[0] = box[0] - x//2
    box_ajustado[1] = box[1] + x//2

    y = (box[3]-box[2]) * img.GetSpacing()[1] * (1+MARGEN_SEGURIDAD)
    y = y  / img.GetSpacing()[1] #Vemos cuanto es en pixeles
    y = y - (box[3]-box[2]) #Vemos la diferencia
    box_ajustado[2] = box[2] - y//2
    box_ajustado[3] = box[3] + y//2

    z = (box[5]-box[4]) * img.GetSpacing()[2] * (1+MARGEN_SEGURIDAD)
    z = z  / img.GetSpacing()[2] #Vemos cuanto es en pixeles
    z = z - (box[5]-box[4]) #Vemos la diferencia
    box_ajustado[4] = box[4] - z//2
    box_ajustado[5] = box[5] + z//2

    box_ajustado = [0 if i < 0 else i for i in box_ajustado]
    if (box_ajustado[1] > img.GetSize()[0]):
        box_ajustado[1] = img.GetSize()[0]

    if (box_ajustado[3] > img.GetSize()[1]):
        box_ajustado[3] = img.GetSize()[1]

    if (box_ajustado[5] > img.GetSize()[2]):
        box_ajustado[5] = img.GetSize()[2]

    for i in range(6):
        box_ajustado[i] = int(box_ajustado[i]) 

    imagen_cropeada = sitk.Crop(img, [box_ajustado[0], box_ajustado[2], box_ajustado[4]], [img.GetSize()[0]- box_ajustado[1], img.GetSize()[1] - box_ajustado[3], img.GetSize()[2] - box_ajustado[5]])

    return (imagen_cropeada, box_ajustado)

def prediccion(imagen_cropeada_preprocesada, model_seg, labels_normales, imagen_cropeada, device, img_nib):
    subject = tio.Subject({"CT": tio.ScalarImage.from_sitk(imagen_cropeada_preprocesada)})

    with torch.no_grad():
        datos = subject["CT"][tio.DATA].to(device).unsqueeze(0)
        tiempo_inicio = time.time()
        pred = model_seg(datos)
        tiempo_fin = time.time()
        #print(f"tiempo de inferencia SEG: {tiempo_fin - tiempo_inicio}s")
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)

    pred = pred.squeeze().cpu().numpy().transpose(2,1,0)

    pred = np.where(np.isin(pred, list(labels_normales.keys())), pred, -1)
    pred = np.vectorize(labels_normales.get)(pred)

    img_cropeada_inferida = np_to_image(pred, imagen_cropeada_preprocesada.GetOrigin(), imagen_cropeada_preprocesada.GetSpacing(), imagen_cropeada_preprocesada.GetDirection(), sitk.sitkInt16)

    """if nib.aff2axcodes(img_nib.affine) != ("L", "A", "S"):
        img_inferida_cropeada_final = cambiar_tam(img_cropeada_inferida, imagen_cropeada.GetSize(), "RAS", mask=True)
    else:
        img_inferida_cropeada_final = cambiar_tam(img_cropeada_inferida, imagen_cropeada.GetSize(), "LAS", mask=True)"""

    #AÑADIDOOOOOOOO
    img_cropeada_prueba = sitk.DICOMOrient(imagen_cropeada, "LPS")

    img_inferida_cropeada_final = cambiar_tam(img_cropeada_inferida, img_cropeada_prueba.GetSize(), "".join(list(nib.aff2axcodes(img_nib.affine))), mask=True)

    return img_inferida_cropeada_final , (tiempo_fin - tiempo_inicio)

def devolver_dimensiones(img, img_inferida_cropeada_final, box_ajustado):
    target = sitk.Image(img.GetSize(), sitk.sitkInt16)
    target.CopyInformation(img)

    target = sitk.Paste(target, img_inferida_cropeada_final, img_inferida_cropeada_final.GetSize(), [0,0,0], [box_ajustado[0], box_ajustado[2], box_ajustado[4]])

    return target