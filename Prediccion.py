from segmenter import Segmenter
from segmenter_2labels import Segmenter_2labels
import torch
from dataset import create_dataset, preprocesado_imagenes
from pathlib import Path
from utils import create_trainer, test_step, log_results, reset_weights, weight_init
import torchio as tio
from pytorch_lightning.utilities.parsing import AttributeDict
#from itertools import product
import argparse

from segmenter import Segmenter
import torch
from pathlib import Path
import torchio as tio
import argparse
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from dataset import cambiar_tam, np_to_image
from utils import devolver_dimensiones, prediccion, crop_imagen, prediccion_crop, preprocesar_imagen
import os
import time
from pytorch_lightning.utilities.model_summary import summarize

parser = argparse.ArgumentParser(description='Predict images using the trained UNET3D with or without crop')

parser.add_argument('--model_pred', help="Path to the model predicting the 7 labels", type=str, default="None")
parser.add_argument('--model_crop', help="Path to the model cropping the image", type=str, default="None")
parser.add_argument('--images_folder', help="Path to the folder containing the images to predict", type=str, default="None")
parser.add_argument('--destination_folder', help="Path to the folder were the predicted labels will be writen to. It has to be already created", type=str, default="None")
parser.add_argument('--tam_pred', help="Size of the images the model was trained at", type=int, default=128)
parser.add_argument('--tam_crop', help="Size of the images the model_crop was trained at", type=int, default=128)
#optional: CROP TYPE
parser.add_argument('--crop_margin', help="The margin on each dimension used to crop the single label predicted image", type=float, default=0.15)

args = parser.parse_args()

print(args)

tiempos_carga_seg = 0
tiempos_carga_crop = 0
tiempos_pred = []
tiempos_crop = []

#CARGAMOS LOS MODELOS ENVIADOS

#Verificamos si CUDA está disponible
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

if (args.model_crop == "None"):
    VERSION = "Solo segunda"
else:   
    VERSION = "2 fases"

    tiempo_carga_crop = time.time()
    #model_crop = Segmenter_2labels.load_from_checkpoint(args.model_crop)
    model_crop = Segmenter_2labels.load_from_checkpoint(args.model_crop)
    tiempo_carga_crop_end = time.time()

    #Mostramos estadisticas del modelo
    #print(f"Peso del modelo CROP: {os.path.getsize(args.model_crop) / (1024 * 1024)} MB")
    #print(summarize(model_crop))
    #print(f"Tiempo de carga CROP: {tiempo_carga_crop_end - tiempo_carga_crop}s")

    model_crop.eval()
    model_crop.to(device);

if (args.model_pred == "None"):
    print("ERROR: Se necesita proporcionar un modelo como mínimo para la predicción de la segunda etapa")
else:
    tiempos_carga_seg = time.time()
    model_seg = Segmenter.load_from_checkpoint(args.model_pred)
    tiempos_carga_seg_end = time.time() 

    #Mostramos estadísticas del modelo
    #print(f"Peso del modelo SEG: {os.path.getsize(args.model_pred) / (1024 * 1024)} MB")
    #print(summarize(model_seg))
    #print(f"Tiempo de carga SEG: {tiempos_carga_seg_end - tiempos_carga_seg}s")

    #Modo evaluacion
    model_seg.eval()
    model_seg.to(device);

#Paths

path_lectura = Path(args.images_folder)
path_escritura = Path(args.destination_folder)

#Margen de seguridad

MARGEN_SEGURIDAD = args.crop_margin

#TAMAÑO

#nuevo_tamanho = (args.tam_pred, args.tam_pred, args.tam)

tamanho_seg = (args.tam_pred, args.tam_pred, args.tam_pred)
tamanho_crop = (args.tam_crop, args.tam_crop, args.tam_crop)

#Definicion de variables

orient_final = "LAS"
#labels_crop = {0:0,1:1,2:1,3:1,4:1,5:1,6:1,7:1} ###### MODIFICAR ESTO DESPUES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
labels_normales = {0:0,1:205,2:420,3:500,4:550,5:600,6:820,7:850}

#Lectura de imágenes


archivos = list(path_lectura.rglob('*image*'))

for archivo in archivos: #VERIFICAR SI ESTO FUNCIONA CON LAS IMÁGENES DEL CHUS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(archivo.name)

    img = sitk.ReadImage(str(archivo))
    img_nib = nib.load(str(archivo))

    if (VERSION == "2 fases"):
        #Hacemos la predicción del corazón al completo. 

        #Preprocesado de la imagen
        img_preprocesada = preprocesar_imagen(img, tamanho_crop, orient_final)

        #Inferimos la imagen con un solo label. El corazon completo
        img_inferida_final, tiempo_crop = prediccion_crop(img_preprocesada, img, model_crop, device, img_nib)

        tiempos_crop.append(tiempo_crop)

        #Ahora cropeamos la imagen
        imagen_cropeada, box_ajustado = crop_imagen(img_inferida_final, MARGEN_SEGURIDAD, img)

        #Preprocesadmos la imagen cropeada
        imagen_cropeada_preprocesada = preprocesar_imagen(imagen_cropeada, tamanho_seg, orient_final)

        #Prediccion segunda fase
        img_inferida_cropeada_final, tiempo_pred = prediccion(imagen_cropeada_preprocesada, model_seg, labels_normales, imagen_cropeada, device, img_nib)

        tiempos_pred.append(tiempo_pred)

        #Ahora devolvemos a las dimesiones originales
        target = devolver_dimensiones(img, img_inferida_cropeada_final, box_ajustado)

        #Añadimos al path_escritura la ruta de la imagen
        path_escritura_file = Path(args.destination_folder) / archivo.parent.relative_to(path_lectura)

        #primero creamos la carpeta si no existe
        if not os.path.exists(path_escritura_file):
            os.makedirs(path_escritura_file)

        #Finalmente escribimos la imagen
        sitk.WriteImage(target, str(path_escritura_file / archivo.name.replace('image', 'pred')))
    else:
        #En este caso no hay dos modelos, solo se hace la predicción sin el crop
        img_preprocesada = preprocesar_imagen(img, tamanho_seg, orient_final)

        #Hacemos la prediccion con el segundo modelo solo
        img_inferida_final, tiempo = prediccion(img_preprocesada, model_seg, labels_normales, img, device, img_nib)

        tiempos_pred.append(tiempo)

        #primero creamos la carpeta si no existe
        if not os.path.exists(path_escritura):
            os.makedirs(path_escritura)

        #Finalmente escribimos la imagen
        sitk.WriteImage(img_inferida_final, str(path_escritura / archivo.name.replace('image', 'pred')))


#print(f"Tiempo medio de prediccion SEG: {np.mean(np.array(tiempos_pred))}+-{np.std(np.array(tiempos_pred))}")
#if len(tiempos_crop) != 0:
#    print(f"Tiempo medio de prediccion CROP: {np.mean(np.array(tiempos_crop))}+-{np.std(np.array(tiempos_crop))}")

