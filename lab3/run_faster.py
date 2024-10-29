import matplotlib
import torch
from PIL import Image
import torchvision

from faster import FasterRCNNwithFPN
from fpn import FeaturePyramidNetwork
from resnet import resnet50
from utils import normalize_img
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.utils import draw_bounding_boxes 
import numpy as np
import matplotlib.pyplot as plt

def generate_colormap(n):
    colormap = matplotlib.cm.get_cmap("jet", n)
    st0 = np.random.get_state()
    np.random.seed(44)
    perm = np.random.permutation(n)
    np.random.set_state(st0)
    cmap = np.round(colormap(np.arange(n)) * 255).astype(np.uint8)[:,:3]
    cmap = cmap[perm, :]
    return cmap


if __name__ == '__main__':
    backbone = resnet50(
        norm_layer=torchvision.ops.misc.FrozenBatchNorm2d
    )
    overwrite_eps(backbone, 0.0)

    fpn = FeaturePyramidNetwork(
        input_keys_list=['res2', 'res3', 'res4', 'res5'],
        input_channels_list=[256, 512, 1024, 2048],
        output_channels=256,
        norm_layer=None,
        pool=False
    )

    model = FasterRCNNwithFPN(
        backbone,
        fpn,
        num_classes=91
    )
    model.load_state_dict(torch.load('data/faster_params.pth'))
    model.eval()

    img = Image.open('bb44.png')
    img_pth = torch.from_numpy(np.array(img))
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_std = torch.tensor([0.229, 0.224, 0.225])
    img_pth = normalize_img(img_pth, image_mean, image_std)
    categories = _COCO_CATEGORIES
    colormap = generate_colormap(len(categories))
    boxes, scores, labels = model(img_pth)

    boxes_to_draw = []
    annotations = []
    for box, label, score in zip(boxes, labels, scores):
        if score.item() > 0.95:
            boxes_to_draw.append(box)
            annotation = f"label: {_COCO_CATEGORIES[label]}, score: {score*100:.2f}%"
            annotations.append(annotation)
    
    boxes_to_draw = torch.stack(boxes_to_draw)

    img = draw_bounding_boxes(image=torch.from_numpy(np.array(img)).permute(2,0,1),
                              boxes=boxes_to_draw,
                              labels=annotations,
                              width = 5,
                              fill=False)

    img = torchvision.transforms.ToPILImage()(img) 

    img.show() 