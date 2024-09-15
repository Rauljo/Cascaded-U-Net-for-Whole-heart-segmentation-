# ejecutar_comandos.py
import subprocess

def ejecutar_comando(comando):
    """
    Ejecuta un comando en el sistema y maneja su salida.
    """
    print(f"Ejecutando: {' '.join(comando)}...")
    try:
        result = subprocess.run(comando, check=True, capture_output=True, text=True)
        print(f"Salida del comando:\n{result.stdout}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e}")
        print(f"Salida de error:\n{e.stderr}")
        # Opcional: Puedes detener la ejecución si ocurre un error
        raise

def ejecutar_script():
    # Lista de comandos a ejecutar secuencialmente
    comandos = [
        [
            "python", "./CMT/Prediccion.py",
            "--model_pred=./CMT/Depth_5_Start_filts_64_augment_1_dataset_todo_train_crop_variable_margen_tam_192_COMPLETE_TRAINING_final.ckpt",
            "--images_folder=/input/",
            "--destination_folder=/output/",
            "--model_crop=./CMT/Depth_4_Start_filts_32_augment_1_dataset_todo_train_tam_128_SOLO1LABEL_COMPLETE_TRAINING_final.ckpt",
            "--tam_pred=192",
            "--tam_crop=128",
            "--crop_margin=0.15"
        ],
        [
            "python", "./CMT/postprocesado.py",
            "--source_folder=/output/",
            "--largest_pulmonar=1"
        ]
    ]

    for comando in comandos:
        ejecutar_comando(comando)

if __name__ == "__main__":
    ejecutar_script()
