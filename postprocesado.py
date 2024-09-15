#Lo que hacemos es coger el componente conectado más grande de todas las estructuras menos la arteria pulmonar
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description='Posprocess images. Largest connected componente on each label except from PA')

parser.add_argument('--source_folder', help='Path to the folder containing the source images', type=str, default="None")
parser.add_argument('--largest_pulmonar', help='0: maintain everything on pulmonar artery. 1: maintain only largest componente on pulmonary artery', type=int, default=0)

args = parser.parse_args()

path = Path(args.source_folder)
#path_escritura = path.parent / (path.stem + "_postprocesado" + path.suffix)

tiempos_post = []

#Creo la carpeta de escritura
#path_escritura.mkdir(exist_ok=True)

#Leo todos los archivos que hay en la carpeta
archivos = list(path.rglob('*pred*'))


for archivo in archivos:
    print(archivo)
    #Leo la imagen
    img = sitk.ReadImage(str(archivo))

    #print(img.GetSize())

    tiempo_inicio = time.time()

    #Creo una nueva imagen con las mismas características que esta
    img_nueva = sitk.Image(img.GetSize(), sitk.sitkInt16)
    #img_nueva.CopyInformation(img)

    #Ahora convierto la imagen a un array
    img_array = sitk.GetArrayFromImage(img)

    #Ahora encuentro los valores de las etiquetas
    etiquetas = np.unique(img_array)

    #Ahora recorro las etiquetas y me quedo con la más grande
    for etiqueta in etiquetas:
        #print(etiqueta)
        if etiqueta == 0:
            continue
        if args.largest_pulmonar == 0:
            if etiqueta == etiquetas[-1]:
                array_label = np.where(img_array == etiqueta, etiqueta, 0)
                #En este caso mantengo la etiqueta tal cual, sin coger el componente más grande
                largest_component = sitk.GetImageFromArray(array_label)
                largest_component = sitk.Cast(largest_component, sitk.sitkInt16)
                img_nueva = sitk.Add(img_nueva, largest_component)
                continue
        array_label = np.where(img_array == etiqueta, etiqueta, 0)
        componente = sitk.ConnectedComponent(sitk.GetImageFromArray(array_label))
        sorted_component_image = sitk.RelabelComponent(componente, sortByObjectSize=True)
        largest_component = sorted_component_image == 1
        #largest_componente = sitk.Multiply(largest_component, etiqueta.astype(float))
        #Convertimos largest componente a tipo 16 bit signed integer
        largest_component = sitk.Cast(largest_component, sitk.sitkInt16)
        largest_component = largest_component * etiqueta
        #img_nueva = sitk.Paste(img_nueva, largest_component, largest_component.GetSize(), [0,0,0], [0,0,0])
        img_nueva = sitk.Add(img_nueva, largest_component)

    img_nueva.CopyInformation(img)

    tiempos_post.append(time.time() - tiempo_inicio)

    #Escribo la imagen

    #Escribo la imagen reescribiendo la original
    sitk.WriteImage(img_nueva, str(archivo))


#print(f"Tiempo medio de prediccion POSTPROCESADO: {np.mean(np.array(tiempos_post))}+-{np.std(np.array(tiempos_post))}")