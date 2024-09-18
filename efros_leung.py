import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from matplotlib import pyplot
import random

def loadImage(path):
    return Image.open(path)

def SSD_patches(a, b, mask):
    assert a.shape == b.shape == mask.shape

    diff = (a - b) * mask
    return np.sum(diff ** 2)

def getPixelToFill(input:np.ndarray, patchSize:tuple[int, int], filled:np.ndarray) -> tuple[int, int]:
    """
    input est l'image de la texture initiale,
    patchSize est la taille du patch que l'on prend depuis l'input et que l'on colle dans l'output,
    filled est un tableau de la taille de l'output qui contient 1 si le pixel est déjà rempli, 0 sinon.
    """
    (patchX, patchY) = patchSize
    kernel = np.ones(patchSize)
    candidates:list[tuple[int, int]] = []
    maxNeighborsNumber = 0

    for i in range(0, filled.shape[0] - patchX):
        for j in range(0, filled.shape[1] - patchY):
            if filled[i, j] == 1:
                continue

            # On détermine le voisinage du pixel (i, j)
            startX = max(0, i - patchX // 2)
            endX = min(filled.shape[0], i + patchX // 2)
            startY = max(0, j - patchY // 2)
            endY = min(filled.shape[1], j + patchY // 2)
            neighbors = filled[startX:endX, startY:endY]

            # Nombre de voisins déjà remplis
            neighborsNumber = np.sum(neighbors)

            # On vérifie que le voisinage contient au moins neighborsNumber pixels remplis
            if neighborsNumber < maxNeighborsNumber:
                continue

            # Si le nombre de voisins est identique, on ajoute le pixel aux candidats
            if neighborsNumber == maxNeighborsNumber:
                candidates.append((i, j))
            
            # Si le nombre de voisins est supérieur, on met à jour la liste des candidats
            else:
                candidates = [(i, j)]
                maxNeighborsNumber = neighborsNumber
    
    return random.choice(candidates)

def getDistancesArray(input:np.ndarray, pixel:tuple[int, int], outputImage:np.ndarray, patchSize:tuple[int, int], filled:np.ndarray) -> np.ndarray:
    """
    input est l'image de la texture initiale,
    patchToFill est le patch que l'on doit remplir.
    """
    out = np.zeros((input.shape[0], input.shape[1]), dtype=np.float32)

    patchX = patchSize[0]
    patchY = patchSize[1]

    # On détermine le voisinage du pixel à remplir
    startX = max(0, pixel[0] - patchX // 2)
    endX = min(filled.shape[0], pixel[0] + patchX // 2)
    startY = max(0, pixel[1] - patchY // 2)
    endY = min(filled.shape[1], pixel[1] + patchY // 2)
    mask = filled[startX:endX, startY:endY]
    patchToFill = outputImage[startX:endX, startY:endY]

    for i in range(0, input.shape[0] - patchX):
        for j in range(0, input.shape[1] - patchY):
            out[i, j] = SSD_patches(input[i:i+patchX, j:j+patchY], patchToFill, mask)
    
    return out


texture = np.array(loadImage("dataSynthese/text0.png"))
texture_size = texture.shape[:2]
patch_size = 7
l = patch_size * 2
overlap = 3
# On prend un patch initial aléatoirement dans Ismp
init_patch_pos_start = (random.randrange(0, texture_size[0] - l - 1), random.randrange(0, texture_size[1] - l - 1))
init_patch_pos_end = (init_patch_pos_start[0] + l, init_patch_pos_start[1] + l)

output = np.zeros((texture_size[0] * 2, texture_size[1] * 2, 3), dtype=np.uint8)
output_size = output.shape[:2]
output_px_number = output_size[0] * output_size[1]
# En haut à gauche de l'image finale, on place un premier patch de taille l*l
output[0:l, 0:l] = texture[init_patch_pos_start[0]:init_patch_pos_end[0], init_patch_pos_start[1]:init_patch_pos_end[1]]

# Tableau des pixels remplis
filledPixels = np.zeros((output_size[0], output_size[1]), dtype=np.uint8)
filledPixels[0:l, 0:l] = 1

# Tant qu'il reste des pixels à remplir
while np.sum(filledPixels) < output_px_number:
    # On récupère le pixel à remplir
    px = getPixelToFill(texture, (patch_size, patch_size), filledPixels)

    # On calcule la distance entre le patch à remplir et les patches de l'image source
    distances = getDistancesArray(texture, px, output, (patch_size, patch_size), filledPixels)



    filledPixelsNb = np.sum(filledPixels)
    if filledPixelsNb % int(output_px_number * 0.01) == 0:
        percentage = round(filledPixelsNb / output_px_number * 100)
        print(percentage, "%")
        if percentage >= 5 and percentage < 10:
            Image.fromarray(output).show()



Image.fromarray(output).show()