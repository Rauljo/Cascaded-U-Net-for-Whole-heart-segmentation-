from segmenter import Segmenter
import torch
from dataset import create_dataset, preprocesado_imagenes
from pathlib import Path
from utils import create_trainer, test_step, test_step_2labels, log_results, reset_weights, weight_init
import torchio as tio
from pytorch_lightning.utilities.parsing import AttributeDict
from segmenter_2labels import Segmenter_2labels
#from itertools import product
import argparse

parser = argparse.ArgumentParser(description='Train UNET3D neural network on WHS dataset')

parser.add_argument('--depth', help='Depth of the network', type=int, default=4)
parser.add_argument('--start_filts', help='Number of filters in the first layer', type=int, default=32)
parser.add_argument('--cross_val', help="10 fold cross val, or without val set", type=int, default=0)
parser.add_argument('--max_epochs', help="Max number of epochs", type=int, default=125)

parser.add_argument('--img_path', help="Path to read images from", type=str, default="ct_train")

parser.add_argument('--logs_folder', help="Name of the logs folder", type=str, default="logs")

parser.add_argument('--tam', help="Size of the images", type=int, default=128)

parser.add_argument('--augment', help="Wether to use augmentations or not", type=int, default=1)

parser.add_argument('--num_labels', help="Wether to segment just one label as the whole heart or the 8 interest structures", type=int, default=8)

args = parser.parse_args()

print(args)

#path_base_lect = Path("/mnt/beegfs/home/raul.salgado/UNET3D/todos_labels_cropeado/datasets/")
#path_base_lect = Path("/mnt/netapp2/Store_uni/home/usc/ci/rsg/UNET3D/data")

#path_lect = path_base_lect / args.img_path

path_lect = Path(args.img_path)

path_base_escrit = Path("../Estandarizados")

#TAMANHO = (128,128,128)
TAMANHO = (args.tam, args.tam, args.tam)

if args.num_labels == 1:
    path_escrit = path_base_escrit / (args.img_path.split("/")[-1] + str(TAMANHO[0]) + "_SOLO1LABEL")
else:
    path_escrit = path_base_escrit / (args.img_path.split("/")[-1] + str(TAMANHO[0]))

print("PATH LECTURA: ", path_lect)
print("PATH ESCRITURA: ", path_escrit)

path_logs = args.logs_folder

print("PATH LOGS: ", path_logs)

torch.cuda.empty_cache()
#print(torch.cuda.memory_allocated())
#print(torch.cuda.memory_reserved())

N_SPLITS = 3 #Que en realidad son 10...
BATCH_SIZE = 1 #Reajustar al cluster
NUM_WORKERS = 4 #Reajustar al cluster
ORIENTATION = "LAS"
if args.cross_val == 0:
    CROSS_VAL = False
else:
    CROSS_VAL = True
#CROSS_VAL = args.cross_val
LR = 1e-4 #NO IMPLEMENTADO
MAX_EPOCHS = args.max_epochs

transform_intensity = {
    tio.RandomNoise(std =0.1, p=1.0),
    #tio.RescaleIntensity((0.9, 1.1), p=1.0),
}

transform_intensity = tio.Compose(transform_intensity, p=0.35)

AUGMENTATION = tio.Compose({
    tio.RandomAffine(scales = (0.8, 1.2), degrees=(-10,10), isotropic=True, p=0.5),
    tio.transforms.RandomElasticDeformation(num_control_points=(8, 8, 8), max_displacement=(10,10,10), locked_borders=2, image_interpolation = 'bspline', p=0.2), #AQUI HABRÍA QUE CAMBIAR LOS INTERPOLATIONS MODES, PERO NI IDEA DE QUE PONER
    #FALTA SHIFT INTENSITIES. 
    transform_intensity,
})

if args.augment == 0:
    AUGMENTATION = None
else:
    AUGMENTATION = AUGMENTATION

#AUGMENTATION = None



#Primero estandarizamos las imagenes una vez en la nueva carpeta
preprocesado_imagenes(path_lect, path_escrit, TAMANHO, orient_final="LAS", num_labels=args.num_labels) #CUANDO SE HACE UNA VEZ, YA NO HACE FALTA HACERLO MÁS. TARDA BASTANTE PORQUE GUARDA LAS IMAGENES EN LAS DIMENSIONES ORIGINALES, QUE SON MUY PESADAS. 

print("Imagenes estandarizadas")

#Ahora creamos el dataset:
datasets = create_dataset(path_escrit, transform = AUGMENTATION, cross_val=CROSS_VAL, n_splits=N_SPLITS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
print(len(datasets))
print("Dataset creado")

#augmentation_list = [AUGMENTATION]
#loss_list = ["Cross"]
#depth_list = [args.depth]
#start_filts_list = [args.start_filts]

#combinations = list(product( loss_list, depth_list, start_filts_list))

#print(combinations)

if CROSS_VAL:
    if args.num_labels == 1:
        name = "Depth_" + str(args.depth) + "_Start_filts_" + str(args.start_filts) + "_augment_" + str(args.augment) + "_dataset_" + args.img_path.split("/")[-1] + "_tam_" + str(args.tam) + "_SOLO1LABEL_"
        model = Segmenter_2labels(start_filts=args.start_filts, depth = args.depth)

    else:
        name = "Depth_" + str(args.depth) + "_Start_filts_" + str(args.start_filts) + "_augment_" + str(args.augment) + "_dataset_" + args.img_path.split("/")[-1] + "_tam_" + str(args.tam) + "_"
        model = Segmenter(start_filts=args.start_filts, depth = args.depth)


    print(
            "###############TRAINING WITH############################################################################\n",
            f"NUM LABELS TO PREDICT: {args.num_labels}\n",
            f"start_filts: {args.start_filts}\n",
            f"Depth: {args.depth}\n",
            f"Augment: {args.augment}\n",
            f"Tam: {args.tam}\n",
            f"CROSS_VALIDATION\n",
            "#######################################################################################################\n"
        )

    results_dice = []
    results_loss = []

    for i in range(len(datasets)):
        train_loader = datasets[i][0]
        val_loader = datasets[i][1]
        test_loader = datasets[i][2]

        if args.num_labels == 1:
            model = Segmenter_2labels(start_filts=args.start_filts, depth = args.depth)
        else:
            model = Segmenter(start_filts=args.start_filts, depth = args.depth)

        model.apply(reset_weights);
        model.apply(weight_init);

        #Ajustamos el nombre al fold
        name_fold = name + str(i)

        #Entrenamos el modelo
        trainer = create_trainer(train_loader, val_loader, max_epochs=MAX_EPOCHS, gpus=1, save_dir= str(path_logs), name=name_fold)
        trainer.fit(model, train_loader, val_loader)

        #Evaluamos el modelo
        if args.num_labels == 1:
            model = Segmenter_2labels.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
            result = test_step_2labels(model, test_loader, path_escrit)
        else:
            model = Segmenter.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
            result = test_step(model, test_loader, path_escrit)
        print(result)

        results_dice.extend(result[1])
        results_loss.append(result[0])

    print("RESULTADO DICE VALIDACION CRUZADA: ", results_dice)
    print("RESULTADO LOSS VALIDACION CRUZADA: ", results_loss)

    #Aqui imprimimos la media de dice y loss
    print("Dice medio: ", torch.mean(torch.stack(results_dice), dim=0))
    print("Loss medio: ", torch.mean(torch.tensor(results_loss)))

    log_results(results_dice, results_loss, name=name, archivo="resultados.csv")

    """for params in combinations:

        name = "Loss_" + params[0] + "_Depth_" + str(params[1]) + "_Start_filts_" + str(params[2]) + "_dataset_" + args.img_path
        print(
            "###############TRAINING WITH############################################################################\n",
            f"Loss function: {params[0]}\n",
            f"Depth: {params[1]}\n",
            "#######################################################################################################\n"
        )

        #Creamos el modelo
        model = Segmenter(start_filts=params[2], depth=params[1], loss_funct=params[0])

        #Entrenamos el modelo
        results_dice = []
        results_loss = []
        for i in range(len(datasets)):
            train_loader = datasets[i][0]
            val_loader = datasets[i][1]
            test_loader = datasets[i][2]

            model = Segmenter(start_filts=params[2], depth=params[1], loss_funct=params[0])
            model.apply(reset_weights);
            model.apply(weight_init);

            #model = Segmenter.load_from_checkpoint("/mnt/beegfs/home/raul.salgado/UNET3D/todos_labels/logs/Loss_Cross_Depth_4_Start_filts_32/version_5/checkpoints/epoch=99-step=1600.ckpt")

            #Entrenamos el modelo
            trainer = create_trainer(train_loader, val_loader, max_epochs=MAX_EPOCHS, gpus=1, save_dir= str(path_logs), name=name)
            trainer.fit(model, train_loader, val_loader)
            #Evaluamos el modelo
            print("Path mejor: ", trainer.checkpoint_callback.best_model_path) #IMPRIMO ESTO PARA VERIFICAR QUE SOLO BUSCA EN ESA VERSIÓN Y NO EN NINGUNA OTRA. 
            model = Segmenter.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
            result = test_step(model, test_loader, path_escrit)
            print(result)

            results_dice.append(result[1])
            results_loss.append(result[0])"""



    

else: #ENTRENAMIENTO COMPLETO
    if args.num_labels == 1:
        name = "Depth_" + str(args.depth) + "_Start_filts_" + str(args.start_filts) + "_augment_" + str(args.augment) + "_dataset_" + args.img_path.split("/")[-1] + "_tam_" + str(args.tam) + "_SOLO1LABEL_COMPLETE_TRAINING"
    else:
        name = "Depth_" + str(args.depth) + "_Start_filts_" + str(args.start_filts) + "_augment_" + str(args.augment) + "_dataset_" + args.img_path.split("/")[-1] + "_tam_" + str(args.tam) + "_COMPLETE_TRAINING"

    print(
            "###############TRAINING WITH############################################################################\n",
            f"NUM LABELS TO PREDICT: {args.num_labels}\n",
            f"start_filts: {args.start_filts}\n",
            f"Depth: {args.depth}\n",
            f"Augment: {args.augment}\n",
            f"Tam: {args.tam}\n",
            f"COMPLETE_TRAINING\n",
            "#######################################################################################################\n"
        )

    if args.num_labels == 1:
        model = Segmenter_2labels(start_filts=args.start_filts, depth = args.depth)
    else:
        model = Segmenter(start_filts=args.start_filts, depth = args.depth)

    train_loader = datasets[0][0]

    model.apply(reset_weights);
    model.apply(weight_init);

    #Entrenamos el modelo
    #torch.cuda.memory_summary()
    trainer = create_trainer(train_loader, max_epochs=MAX_EPOCHS, gpus=1, save_dir= str(path_logs), name=name)
    #torch.cuda.memory_summary()
    trainer.fit(model, train_loader)

    print("Entrenamiento completado")

    #Ahora guardamos el modelo
    #torch.save(model.state_dict(), f'{name}_final.ckpt')

    #Ahora guardamos el modelo
    trainer.save_checkpoint(f'{name}_final.ckpt')

    """for params in combinations:
        name = "Loss_" + "Cross" + "_Depth_" + str(args.depth) + "_Start_filts_" + str(args.start_filts) + "_dataset_" + args.img_path + "_tam_" + str(TAMANHO[0]) + "_COMPLETE_TRAINING"
        print(
                "###############TRAINING WITH############################################################################\n",
                f"Loss function: {params[0]}\n",
                f"Depth: {params[1]}\n",
                "#######################################################################################################\n"
        )
        #Creamos el modelo
        model = Segmenter(start_filts=params[2], depth=params[1], loss_funct=params[0])

        train_loader = datasets[0][0]

        model.apply(reset_weights);
        model.apply(weight_init);

        #Entrenamos el modelo
        torch.cuda.memory_summary()
        trainer = create_trainer(train_loader, max_epochs=MAX_EPOCHS, gpus=1, save_dir= str(path_logs), name=name)
        torch.cuda.memory_summary()
        trainer.fit(model, train_loader)

        print("Entrenamiento completado")"""

        
