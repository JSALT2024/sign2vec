import cv2
import numpy as np
from vitpose import VitPose
from typing import Dict
from detectron2.config import LazyConfig
from detect import DefaultPredictor_Lazy

class ViTPoseExtractor():

    def __init__(self):
        self.cpm = self.initialize_cpm()
        self.detector = self.initialize_detector()

    def initialize_cpm(self) -> VitPose:
        cpm = VitPose()
        return cpm
    
    def initialize_detector(self) -> DefaultPredictor_Lazy:
        cfg_path = '/configs/cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector

    def get_pose(self, img_path: str) -> Dict[np.ndarray]:

        # Load image
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = self.detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        vitposes_out = self.cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        people = []

        # Use hands based on hand keypoint detections
        for person_id, vitposes in enumerate(vitposes_out):
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            is_left_valid = 0
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

            people.append({
                'person_id': person_id,
                'bbox': bbox[-1],
                'left_hand': left_hand_keyp,
                'right_hand': right_hand_keyp,
                'pose': vitposes['keypoints'][:-42],
            }) 
        
        return people


if __name__ == '__main__':
    extractor = ViTPoseExtractor()
    # get img_paths from video
    img_paths = []
    cap = cv2.VideoCapture('data/video.mp4')
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_path = 'data/temp_{idx}.jpg'
        cv2.imwrite(img_path, frame)
        img_paths.append(img_path)
        idx += 1

    for img_path in img_paths:
        pose = extractor.get_pose(img_path)
        print(pose)
