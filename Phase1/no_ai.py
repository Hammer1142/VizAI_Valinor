from pathlib import Path
import sys
from ultralytics import YOLO
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
    
from utils import CLASS_MAPPING

COCO_CLASSES = sorted(set(CLASS_MAPPING.values()))
n_input_images = 548 # hard coded for the val Dataset

model = YOLO(str(ROOT_DIR/"model/yolo11n.pt"))

# first test image size with 640 pixels resolution for the larger side, the other side will be calculated to match the original ratio 
# (yolo automatically downsamples to that size, original images are of larger sizes, e.g. 1920x1080), later could try higher resolution
imgsz = 640
conf = 0.1 # for better analysis later on, we allow also lower confidence
# iou is default value 0.7
# max_det is equal to 300 maximum detection per default, could be adjusted if needed, however tests showed that 0 images reached that 300 detections bound
device = "mps" # switch to cuda, cpu, whatever fits for the system

results = model.predict(source=str(ROOT_DIR/"Data/VisDrone2019-DET-val/images"),
                batch=8,
                stream=True,
                imgsz = imgsz,
                conf = conf,
                classes = COCO_CLASSES, # filter predictions to the COCO mappings of our VisDrone classes
                device = device,
                verbose = False
                )

records = []
# Process results (https://docs.ultralytics.com/modes/predict)
for result in results:
    boxes = result.boxes # bounding boxes with all information
    image_key = Path(result.path).stem # get image name
    
    n_det = len(boxes)
    
    h_orig, w_orig = result.orig_shape
    coords = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    
    if n_det == 0:
        records.append({
            "image": image_key,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 0.0,
            "y2": 0.0,
            "class": 99, # does not map to any COCO class, just for our info that no detection was found
            "conf": 0.0,
            "width": w_orig,
            "height": h_orig
        })
    
    else:
        for i in range(n_det): # each detection gets an own record entry
            records.append({
                "image": image_key,
                "x1": coords[i, 0],
                "y1": coords[i, 1],
                "x2": coords[i, 2],
                "y2": coords[i, 3],
                "class": int(classes[i]),
                "conf": confs[i],
                "width": w_orig,
                "height": h_orig
            })
    
df = pd.DataFrame(records)
result_folder_path = Path(ROOT_DIR/"Results/Phase1")
result_folder_path.mkdir(parents=True, exist_ok=True) # make sure the Results/Phase1 folder exists

df.to_csv(str(ROOT_DIR/f"Results/Phase1/no_ai_detections_imgsz-{imgsz}_conf-{conf}.csv"), header=True, sep=";", index=False)

unique_img_paths = df["image"].nunique() # returns the number of unique image names
assert unique_img_paths == n_input_images, "There is not at least one entry for each input image in the output dataframe!"
