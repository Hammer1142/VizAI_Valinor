# maps VisDrone classes to COCO classes
CLASS_MAPPING = {
    0: 0, # pedestrian -> person, in gt all class 0 have score 0, so are not relevant, we still keep them here for completeness
    1: 0, # people -> person
    2: 1, # bicycle -> bicycle
    3: 2, # car -> car
    4: 2, # van -> car as there is no van class in COCO
    5: 7, # truck -> truck
    8: 5, # bus -> bus
    9: 3 # motor -> motorcycle
    # tricycle (6) and awning-tricycle (7) from the VisDrone Dataset have no proper mappings to COCO classes
}

COCO_NAMES = {
    0: "person", 
    1: "bicycle", 
    2: "car", 
    3: "motorcycle", 
    5: "bus", 
    7: "truck"
}