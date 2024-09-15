#Este código se ejecuta sobre las imaǵenes de train para crear una version del dataset con los corazones cropeados para poder entrenar el segundo modelo
from pathlib import Path
import SimpleITK as sitk
import argparse
from dataset import label_from_image

parser = argparse.ArgumentParser(description='Create cropped dataset for training the second stage U-Net')

parser.add_argument('--source_folder', help='Path to the folder containing the source images', type=str, default="None")
parser.add_argument('--destination_folder', help="Path to the folder were the new dataset images will be contained", type=str, default="None")
#optional: CROP TYPE
#optional: MARGIN SIZE
parser.add_argument('--margin_value', help="How much margin do the cuboids have", type=float, default=0.15)
parser.add_argument('--crop_type', help="Wether to use variable or fixe type", type=str, default="variable")

args = parser.parse_args()

print(args)

MARGEN_SEGURIDAD = args.margin_value

path_lectura = Path(args.source_folder)
path_escritura = Path(args.destination_folder)

if (args.crop_type == "variable"):
    print(f"Using variable crop type with {MARGEN_SEGURIDAD} margin")

    box_positions = [0,0,0,0,0,0]
    centro = [0,0,0]
    box_pixel = [0,0,0]

    box_ajustado = [0,0,0,0,0,0]

    #archivos = list(path_lectura.glob("*image*"))
    archivos = list(path_lectura.glob("**/*image*"))

    #for archivo in sorted(archivos, key=lambda x: int(x.stem.split('_')[-2])):
    for archivo in archivos:
        print(archivo.name)
        img = sitk.ReadImage(str(archivo))
        label = sitk.ReadImage(str(label_from_image(archivo)))

        #Buscamos la componente conectada más grande
        componente = sitk.ConnectedComponent(label)
        sorted_component_image = sitk.RelabelComponent(componente, sortByObjectSize=True)
        largest_component = sorted_component_image == 1

        #Buscamos la bounding box
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(img, largest_component)
        box = stats.GetBoundingBox(1)

        print("Box original: ", box)

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

        print("Tamaño final: ", box_ajustado)

        print("Tamaño imagen: ", img.GetSize())

        print([img.GetSize()[0]- box_ajustado[1], img.GetSize()[1] - box_ajustado[3], int(img.GetSize()[2] - box_ajustado[5])])

        #Cropeamos imagen y labels
        imagen = sitk.Crop(img, [int(box_ajustado[0]), int(box_ajustado[2]), int(box_ajustado[4])], [int(img.GetSize()[0]- box_ajustado[1]), int(img.GetSize()[1] - box_ajustado[3]), int(img.GetSize()[2] - box_ajustado[5])])
        label = sitk.Crop(label, [box_ajustado[0], box_ajustado[2], box_ajustado[4]], [label.GetSize()[0]- box_ajustado[1], label.GetSize()[1] - box_ajustado[3], label.GetSize()[2] - box_ajustado[5]])
    
        relative_path = archivo.relative_to(path_lectura)
        destination_image_path = path_escritura / relative_path
        destination_label_path = str(path_escritura / relative_path).replace('image', 'label')

        destination_image_path.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(imagen, str(destination_image_path))
        sitk.WriteImage(label, str(destination_label_path))

        #Vamos a escribir la imagen
        #sitk.WriteImage(imagen, str(path_escritura / archivo.name))
        #sitk.WriteImage(label, str(path_escritura / archivo.name).replace('image', 'label'))

elif (args.crop_type == "fixed"):
    print(f"Using fixed crop type with {MARGEN_SEGURIDAD} margin")

    box = [162.524, 158.268, 168.208]
    print("TAMAÑO FINAL: ", box)
    #box = box * (1+MARGEN_SEGURIDAD)
    box_positions = [0,0,0,0,0,0]
    centro = [0,0,0]
    box_pixel = [0,0,0]

    archivos = list(path_lectura.glob("**/*image*"))

    #for archivo in sorted(archivos, key=lambda x: int(x.stem.split('_')[-2])):
    for archivo in archivos:
        print(archivo.name)
        img = sitk.ReadImage(str(archivo))
        label = sitk.ReadImage(str(label_from_image(archivo)))

        #Buscamos la componente conectada más grande
        componente = sitk.ConnectedComponent(label)
        sorted_component_image = sitk.RelabelComponent(componente, sortByObjectSize=True)
        largest_component = sorted_component_image == 1

        #Buscamos la bounding box
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(img, largest_component)
        caja_real = stats.GetBoundingBox(1)

        print("Bounding box: ", caja_real)

        #Ahora calculamos el centroide de la caja
        centro[0] = (caja_real[0]+caja_real[1]) / 2
        centro[1] = (caja_real[2]+caja_real[3]) / 2
        centro[2] = (caja_real[4]+caja_real[5]) / 2

        print("Centroide: ", centro)
        
        #Ahora tenemos el centroide.


        #Calculamos las posiciones de la caja
        box_positions[0] = int(centro[0] - box[0]/img.GetSpacing()[0]/2)
        box_positions[1] = int(centro[0] + box[0]/img.GetSpacing()[0]/2)

        box_positions[2] = int(centro[1] - box[1]/img.GetSpacing()[1]/2)
        box_positions[3] = int(centro[1] + box[1]/img.GetSpacing()[1]/2)

        box_positions[4] = int(centro[2] - box[2]/img.GetSpacing()[2]/2)
        box_positions[5] = int(centro[2] + box[2]/img.GetSpacing()[2]/2)

        box_positions = [0 if i < 0 else i for i in box_positions]
        if (box_positions[1] > img.GetSize()[0]):
            box_positions[1] = img.GetSize()[0]
        if (box_positions[3] > img.GetSize()[1]):
            box_positions[3] = img.GetSize()[1]
        if (box_positions[5] > img.GetSize()[2]):
            box_positions[5] = img.GetSize()[2]

        #print("Tamaño imagen: ", box_img.shape)
        print("Caja final: ", box_positions)
        print("Tamaño imagen: ", img.GetSize())

        #Ahora recortamos:
        imagen = sitk.Crop(img, [box_positions[0], box_positions[2], box_positions[4]], [img.GetSize()[0]- box_positions[1], img.GetSize()[1] - box_positions[3], img.GetSize()[2] - box_positions[5]])
        label = sitk.Crop(label, [box_positions[0], box_positions[2], box_positions[4]], [img.GetSize()[0]- box_positions[1], img.GetSize()[1] - box_positions[3], img.GetSize()[2] - box_positions[5]])

        relative_path = archivo.relative_to(path_lectura)
        destination_image_path = path_escritura / relative_path
        destination_label_path = str(path_escritura / relative_path).replace('image', 'label')

        destination_image_path.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(imagen, str(destination_image_path))
        sitk.WriteImage(label, str(destination_label_path))

        

        #Vamos a escribir la imagen
        #sitk.WriteImage(imagen, str(path_escritura / archivo.name))
        #sitk.WriteImage(label, str(path_escritura / archivo.name).replace('image', 'label'))
