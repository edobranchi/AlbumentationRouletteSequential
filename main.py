import os
import random
import albumentations as A
import cv2
import time

# Lista delle trasformazioni
common_transformations = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomGamma(p=0.5),
    A.CLAHE(p=0.5),
    A.RandomCrop(width=200, height=200, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
]

#funzione principale per trasformazioni
def generate_transformed_images(image, n, saveImgs):

    transformed_images_local = []

    for _ in range(n):
        # scelgo a caso un numero di trasf da applicare tra 1 e 5
        num_transforms = random.randint(1, 5)

        # le scelgo a caso dalla lista
        selected_transforms = random.sample(common_transformations, num_transforms)

        # le compongo in una pipeline
        transform = A.Compose(selected_transforms)

        # le applico
        transformed = transform(image=image)

        if saveImgs:
            transformed_images_local.append(transformed["image"])
    if saveImgs:
        return transformed_images_local
    else:
        return


if __name__ == "__main__":

    #True -> Salva le immagini
    #False -> non salva le immagini
    saveImgs = True
    #Numero di immagini da generare
    n = 5

    # Carico l'immagine e converto in RGB
    image = cv2.imread("input_image/paesaggio-grande.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    start_time=time.time()
    #Chiamo la funzione di trasformazione
    if saveImgs:
        transformed_images = generate_transformed_images(image, n,saveImgs)
    else:
        generate_transformed_images(image, n, saveImgs)

    end_time=time.time()
    exec_time=end_time-start_time
    print("Tempo di esecuzione: " ,exec_time)


    # Salvo le immagini
    if saveImgs:
        os.makedirs("output_image", exist_ok=True)
        for i, transformed_image in enumerate(transformed_images):
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"output_image/transformed_image_{i + 1}.jpg", transformed_image)
