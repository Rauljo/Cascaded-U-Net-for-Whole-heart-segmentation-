from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
#from celluloid import Camera
#from IPython.display import HTML

import torch

from pathlib import Path

import torchio as tio
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import os
import re

from sklearn.model_selection import KFold

labels = [0, 205, 420, 500, 550, 600, 820, 850]

def resample_data(im, new_size, mask=False):
    """ Resample image or mask to a fixed XY size (keep same Z size as input image)"""

    reference_image = sitk.Image(new_size, im.GetPixelIDValue())
    reference_image.SetOrigin(im.GetOrigin())
    reference_image.SetDirection(im.GetDirection())
    reference_image.SetSpacing([
            sz * spc / nsz
            for nsz, sz, spc in zip(new_size, im.GetSize(), im.GetSpacing())])
    # transform that does nothing but required input for sitk.Resample()
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(im.GetDirection())
    if mask:
        return sitk.Resample(im, reference_image, transform, sitk.sitkNearestNeighbor)
    else:
        return sitk.Resample(im, reference_image, transform, sitk.sitkLinear)
    
def label_from_image(path):
    return path.with_name(path.name.replace("image", "label"))

def np_to_image(img_arr, origin, spacing, direction, pixel_type):
    """Save numpy array as itk image (volume) specifiying origin, spacing and direction desired
    Add also few metadata."""
    itk_img = sitk.GetImageFromArray(img_arr, isVector=False)
    itk_img = sitk.Cast(itk_img, pixel_type)    # reduce size by specifying minimum required pixel type, i.e. sitk.sitkUInt8 for masks, sitk.sitkInt16 for CTs, etc
    itk_img.SetSpacing(spacing)
    itk_img.SetOrigin(origin)
    itk_img.SetDirection(direction)
    return itk_img

def copy_all_metadata(im_ref, im_out, verbose=False):
    for k in im_ref.GetMetaDataKeys():
        v = im_ref.GetMetaData(k)
        if verbose:
            print(f'({k}) = = "{v}"')
        im_out.SetMetaData(k, v)
    return im_out

def cambiar_tam(img, new_size, orient_final="LAS", mask=False):
    img = sitk.DICOMOrient(img, "LPS")
    img = resample_data(img, new_size, mask=mask)
    img = sitk.DICOMOrient(img, orient_final)
    return img

# Función para extraer el número del nombre del archivo
def extraer_numero(path):
    # Usar una expresión regular para encontrar el número en el nombre del archivo
    match = re.search(r'Case(\d+)_image\.nii\.gz', path.name)
    if match:
        return int(match.group(1))
    else:
        return float('inf')

#Esta función ya recibe la dirección del directorio donde estan las imagenes ya estandarizadas y todo, lo que falta es hacer el resample y las augmentations. y crear el dataset
def create_dataset(data_dir, transform=None, cross_val=False, n_splits=5, batch_size=1, num_workers=4):
    #subjects_paths = sorted(list(data_dir.glob("*image*")), key = lambda x: int(x.name.split("_")[2]))
    subjects_paths = sorted(data_dir.glob("*image*"), key=lambda x: extraer_numero(x))    
    subjects = []

    for subject_path in subjects_paths:
        label_path = label_from_image(subject_path)

        ## VAMOS A LEER LAS IMAGENES CON SITK Y DESPUÉS HACER QUE SEAN SUBJECTS
        img = sitk.ReadImage(str(subject_path))
        label = sitk.ReadImage(str(label_path))

        #Cambiamos el tamaño
        #img = cambiar_tam(img, new_size, orient_final)
        #label = cambiar_tam(label, new_size, orient_final, mask=True)

        subject = tio.Subject({"CT":tio.ScalarImage.from_sitk(img), "Label": tio.LabelMap.from_sitk(label)})

        subjects.append(subject)


    #Ahora que ya tenemos la lista de subjects, creamos el dataset (NO HACEN FALTA COLAS AL NO TENER PATCH_SIZE)
        datasets = []
    if cross_val == False:
        train_dataset = tio.SubjectsDataset(subjects, transform = transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,shuffle=True)
        datasets.append((train_loader, None, None))
    else:
        kf = KFold(n_splits=n_splits, shuffle=False)
        #vamos a crear un data loader para cada split
        for i, (train_index, val_index) in enumerate(kf.split(subjects)):
            for j in range(2):
                train_subs = [subjects[i] for i in train_index]
                if j == 0:
                    val_subs = [subjects[i] for i in val_index][:len(val_index) // 2]
                    test_subs = [subjects[i] for i in val_index][len(val_index) // 2:]
                if j == 1:
                    val_subs = [subjects[i] for i in val_index][len(val_index) // 2:]
                    test_subs = [subjects[i] for i in val_index][:len(val_index) // 2]

                train_dataset = tio.SubjectsDataset(train_subs, transform = transform)
                val_dataset = tio.SubjectsDataset(val_subs) #No aplicamos transformaciones en la validación
                test_dataset = tio.SubjectsDataset(test_subs)

                #sampler = tio.data.GridSampler(patch_size=(128,128,128), patch_overlap=0)

                #train_patches_queue = tio.Queue(train_dataset, max_length=40, samples_per_volume=1, sampler=sampler, num_workers=4)
                #val_patches_queue = tio.Queue(val_dataset, max_length=40, samples_per_volume=1, sampler=sampler, num_workers=1)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers = 0, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers = 0, shuffle=False)

                datasets.append((train_loader, val_loader, test_loader))

    return datasets

def _estandarizar_imagenes(img_array):
    img_array [ img_array < -1024] = -1024
    img_array [ img_array > 3071 ] = 3071
    img_array = 2*((img_array-(-1024))/(3071-(-1024)))-1
    """img_array = img_array / 1024.0
    img_array[img_array < -1] = -1
    img_array[img_array > 1] = 1"""

    return img_array


def preprocesado_imagenes(data_lect, data_escrit, nuevo_tamanho, orient_final, num_labels):


    if not data_escrit.exists():
        data_escrit.mkdir(parents=True)
    else:
        return #If the estandardized folder already exists, continue, no need to do it again. 

    #archivos = list(data_lect.glob("*image*"))
    archivos = list(data_lect.glob("**/*image*"))

    #for archivo in sorted(archivos, key = lambda x: int(x.name.split("_")[2])):
    for archivo in archivos:
        img = sitk.ReadImage(str(archivo))
        label = sitk.ReadImage(str(label_from_image(archivo)))

        #Primero cambiamos los tamaños:
        img = cambiar_tam(img, nuevo_tamanho, orient_final)
        label = cambiar_tam(label, nuevo_tamanho, orient_final, mask=True)

        img_array = sitk.GetArrayFromImage(img)
        label_array = sitk.GetArrayFromImage(label)

        #Estandarizamos imagen:
        #img_array = _estandarizar_imagenes(img_array)

        img_array[img_array < -1024] = -1024

        #Ahora aplicamos z-normalization
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)

        #Cuando se van a segmentar las imágenes en un solo label como el corazón completo
        if num_labels == 1:
            label_array[label_array != 0] = 1

        else:
            #Convertimos a 0-7 labels:
            for i in range(len(labels)):
                label_array[label_array == labels[i]] = i

        #Ahora convertimos a imagen de nuevo y guardamos:
        img = np_to_image(img_array, img.GetOrigin(), img.GetSpacing(), img.GetDirection(), sitk.sitkFloat32)
        label = np_to_image(label_array, label.GetOrigin(), label.GetSpacing(), label.GetDirection(), sitk.sitkInt8)

        sitk.WriteImage(img, str(data_escrit / archivo.name))
        sitk.WriteImage(label, str(data_escrit / label_from_image(archivo).name))



