import torch
import numpy as np
import cv2 as cv
import os
from pathlib import Path
import datetime
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class GradCam_Yolo11:

    def __init__(self, yolo, target_layer):
        self.model = yolo.model.eval()

        for param in self.model.parameters():
            param.requires_grad = True

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_grad_cam(self, x):
        self.model.zero_grad()

        with torch.enable_grad():
            output = self.model(x)
            preds = output[0]  # [B, 4+nc, N]
            scores = preds[:, 4:, :]  # class scores
            target = scores.sum()

            if target.item() == 0:
                return None  # nothing detected

            target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture activations or gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[2:],
            mode="bilinear", align_corners=False)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return cam[0, 0].cpu().numpy()

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv.COLORMAP_JET) -> np.ndarray:
    """ Overlays the cam mask on the image as a heatmap """

    heatmap = cv.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image must be np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)

def apply_grad_cam(frame_path, target_layer_num=16, device="cpu") -> np.ndarray:
    yolo = YOLO("../model/yolo11n.pt")
    yolo.to(device)

    img = cv.imread(frame_path)
    img = cv.resize(img, (640, 640))

    img_f = np.float32(img) / 255
    img_rgb = img_f[:, :, [2, 1, 0]] # bgr to rgb

    img_tensor = torch.from_numpy(img_rgb)
    img_chw = img_tensor.permute(2, 0, 1) # hwc to chw
    x = img_chw.unsqueeze(0)
    x.to(device)
    x.requires_grad_(True)

    target_layer = yolo.model.model[target_layer_num]
    grad_cam = GradCam_Yolo11(yolo, target_layer)
    grad_mask = grad_cam.generate_grad_cam(x)
    cam_img = show_cam_on_image(img_f, grad_mask)
    cam_img = np.uint8(255 * cam_img)

    return cam_img

def save_grad_cam(data_path, result_path, target_layer_num=16, device="cpu"):
    frames = sorted(os.listdir(data_path))
    result_name = "grad_cam_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join(result_path, result_name)
    os.makedirs(result_dir, exist_ok=True)

    for frame in frames[0:10]:
        frame_path = os.path.join(data_path, frame)
        cam_img = apply_grad_cam(frame_path, target_layer_num, device)
        cv.imwrite(os.path.join(result_dir, frame), cam_img)

if __name__ == "__main__":
    data_path = PROJECT_ROOT / "Data" / "VisDrone2019-DET-val" / "images"
    result_path = PROJECT_ROOT / "Results"

    print("Data path exists:", data_path.exists())
    print("Resolved Data Path:", data_path)
    target_layer_num = 16
    device = "cpu"

    save_grad_cam(data_path, result_path, target_layer_num, device)