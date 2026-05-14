from typing import List, Dict, Union

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from ultralytics import YOLO

from mineru.utils.enum_class import CategoryId


class YOLORedactionModel:
    def __init__(
        self,
        weight: str,
        device: str = "cpu",
        imgsz: int = 1280,
        conf: float = 0.5,
        iou: float = 0.45,
    ):
        self.device = torch.device(device)
        self.model = YOLO(weight).to(self.device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def _parse_prediction(self, prediction) -> List[Dict]:
        results = []
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return results

        for xyxy, conf, cls in zip(
            prediction.boxes.xyxy.cpu(),
            prediction.boxes.conf.cpu(),
            prediction.boxes.cls.cpu(),
        ):
            coords = list(map(int, xyxy.tolist()))
            xmin, ymin, xmax, ymax = coords
            results.append({
                "category_id": CategoryId.Redaction,
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
                "redaction_subtype": int(cls.item()),
            })
        return results

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        prediction = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )[0]
        return self._parse_prediction(prediction)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4,
    ) -> List[List[Dict]]:
        results = []
        with tqdm(total=len(images), desc="Redaction Predict") as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                predictions = self.model.predict(
                    batch,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                    device=self.device,
                )
                for pred in predictions:
                    results.append(self._parse_prediction(pred))
                pbar.update(len(batch))
        return results
