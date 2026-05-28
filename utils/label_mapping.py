import numpy as np


IGNORE_INDEX = 19


def get_coco_to_cityscapes_mapping():
    """
    Return a lookup table that maps selected COCO class IDs to Cityscapes train IDs.

    Classes that do not have a reasonable Cityscapes equivalent are mapped to
    IGNORE_INDEX. The returned array can be directly indexed with predicted COCO
    class IDs.

    Cityscapes train IDs:
        0  road
        1  sidewalk
        2  building
        3  wall
        4  fence
        6  traffic light
        7  traffic sign
        8  vegetation
        9  terrain
        10 sky
        11 person
        13 car
        14 truck
        15 bus
        16 train
        17 motorcycle
        18 bicycle
    """
    
    coco_to_cityscapes = np.full(256, IGNORE_INDEX, dtype=np.uint8)

    coco_to_cityscapes[0] = 11    # person -> person
    coco_to_cityscapes[1] = 18    # bicycle -> bicycle
    coco_to_cityscapes[2] = 13    # car -> car
    coco_to_cityscapes[3] = 17    # motorcycle -> motorcycle
    coco_to_cityscapes[5] = 15    # bus -> bus
    coco_to_cityscapes[6] = 16    # train -> train
    coco_to_cityscapes[7] = 14    # truck -> truck
    coco_to_cityscapes[9] = 6     # traffic light -> traffic light
    coco_to_cityscapes[11] = 7    # stop sign -> traffic sign

    coco_to_cityscapes[82] = 2    # bridge -> building
    coco_to_cityscapes[90] = 9    # gravel -> terrain
    coco_to_cityscapes[91] = 2    # house -> building
    coco_to_cityscapes[100] = 0   # road -> road
    coco_to_cityscapes[101] = 2   # roof -> building
    coco_to_cityscapes[102] = 9   # sand -> terrain
    coco_to_cityscapes[109] = 3   # wall-brick -> wall
    coco_to_cityscapes[110] = 3   # wall-stone -> wall
    coco_to_cityscapes[111] = 3   # wall-tile -> wall
    coco_to_cityscapes[112] = 3   # wall-wood -> wall
    coco_to_cityscapes[116] = 8   # tree-merged -> vegetation
    coco_to_cityscapes[117] = 4   # fence-merged -> fence
    coco_to_cityscapes[119] = 10  # sky-other-merged -> sky
    coco_to_cityscapes[123] = 1   # pavement-merged -> sidewalk
    coco_to_cityscapes[125] = 8   # grass-merged -> vegetation
    coco_to_cityscapes[126] = 9   # dirt-merged -> terrain
    coco_to_cityscapes[129] = 2   # building-other-merged -> building
    coco_to_cityscapes[131] = 3   # wall-other-merged -> wall

    return coco_to_cityscapes