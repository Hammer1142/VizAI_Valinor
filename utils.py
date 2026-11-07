# maps VisDrone classes to COCO classes
CLASS_MAPPING = {
    0: 0, # pedestrian -> person
    1: 0, # people -> person
    2: 1, # bicycle -> bicycle
    3: 2, # car -> car
    4: 2, # van -> car as there is no van class in COCO
    5: 7, # truck -> truck
    8: 5, # bus -> bus
    9: 3 # motor -> motorcycle
    # tricycle and awning-tricycle from the VisDrone Dataset have no proper mappings to COCO classes
}