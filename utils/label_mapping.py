import numpy as np
import torch


VOID_INDEX = 19


def get_coco_to_cityscapes_mapping():
    """
    Return a lookup table that maps selected COCO class IDs to Cityscapes train IDs.

    Classes that do not have a reasonable Cityscapes equivalent are mapped to
    VOID_INDEX. The returned array can be directly indexed with predicted COCO
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
    
    coco_to_cityscapes = np.full(256, VOID_INDEX, dtype=np.uint8)

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


def aggregate_coco_scores_to_cityscapes(semantic_scores):
    H, W = semantic_scores.shape[1], semantic_scores.shape[2]
    device_scores = semantic_scores.device
    aggregated_scores = torch.zeros((19, H, W), dtype=semantic_scores.dtype, device=device_scores)
    
    aggregated_scores[0]  = semantic_scores[100]          # road -> road
    aggregated_scores[1]  = semantic_scores[123]          # pavement -> sidewalk
    aggregated_scores[2]  = semantic_scores[[82, 91, 101, 129]].sum(dim=0)  # bridge, house, building -> building
    aggregated_scores[3]  = semantic_scores[[109, 110, 111, 112, 131]].sum(dim=0) # All different walls -> wall
    aggregated_scores[4]  = semantic_scores[117]          # fence -> fence
    aggregated_scores[6]  = semantic_scores[9]            # traffic light -> traffic light
    aggregated_scores[7]  = semantic_scores[11]           # stop sign -> traffic sign
    aggregated_scores[8]  = semantic_scores[[116, 125]].sum(dim=0) # tree, grass -> vegetation
    aggregated_scores[9]  = semantic_scores[[90, 102, 126]].sum(dim=0) # gravel, sand, dirt -> terrain
    aggregated_scores[10] = semantic_scores[119]          # sky -> sky
    aggregated_scores[11] = semantic_scores[0]            # person -> person
    aggregated_scores[13] = semantic_scores[2]            # car -> car
    aggregated_scores[14] = semantic_scores[7]            # truck -> truck
    aggregated_scores[15] = semantic_scores[5]            # bus -> bus
    aggregated_scores[16] = semantic_scores[6]            # train -> train
    aggregated_scores[17] = semantic_scores[3]            # motorcycle -> motorcycle
    aggregated_scores[18] = semantic_scores[1]            # bicycle -> bicycle
    
    return aggregated_scores
