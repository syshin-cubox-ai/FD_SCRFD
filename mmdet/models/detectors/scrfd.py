import torch
from mmcv.runner import auto_fp16

from mmdet.core import bbox2result
from .single_stage import SingleStageDetector
from ..builder import DETECTORS


@DETECTORS.register_module()
class SCRFD(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SCRFD, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypointss=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypointss, gt_bboxes_ignore)
        return losses

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if 'force_onnx_export' in kwargs.keys():
            force_onnx_export = kwargs['force_onnx_export']
        else:
            force_onnx_export = False

        if torch.onnx.is_in_onnx_export() or force_onnx_export:
            params = (img, img_metas, return_loss)
            return self.forward_onnx(*params)

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_onnx(
        self,
        img: torch.Tensor,
        conf_thres: torch.Tensor,
        iou_thres: torch.Tensor,
    ) -> torch.Tensor:
        # Forward
        x = self.extract_feat(img)
        x = self.bbox_head(x, onnx_export=True)  # scrfd_head.py의 forward_single() 참조
        if self.bbox_head.use_kps:
            pred = x[0] + x[1] + x[2]  # cls_score, bbox_pred, kps_pred
        else:
            pred = x[0] + x[1]  # cls_score, bbox_pred
        # pred 각 원소의 뜻은 아래와 같다.
        # ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32'] 또는
        # ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']

        # Post-forward
        bbox_list = []
        conf_list = []
        kps_list = []
        # 이 코드는 다중 배치여도 이미지 1장만 처리함
        for idx, stride in enumerate([8, 16, 32]):
            # Create anchor grid (앵커 개수=2)
            height = img.shape[2] // stride
            width = img.shape[3] // stride
            anchor_centers = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy')
            anchor_centers = torch.stack(anchor_centers, dim=-1)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = torch.stack([anchor_centers] * 2, dim=1).reshape((-1, 2)).to(torch.float32)

            # Post-process bbox, conf, kps
            bbox = pred[idx + 3][0] * stride
            bbox = self._distance2bbox(anchor_centers, bbox)
            bbox_list.append(bbox)
            conf = pred[idx + 0][0]
            conf_list.append(conf)
            if len(pred) == 9:
                kps = pred[idx + 6][0] * stride
                kps = self._distance2kps(anchor_centers, kps)
                kps_list.append(kps)

        bbox = torch.vstack(bbox_list)
        conf = torch.vstack(conf_list)
        pred = torch.hstack((bbox, conf))
        if len(kps_list) > 0:
            kps = torch.vstack(kps_list)
            pred = torch.hstack((pred, kps))
        order = conf.ravel().argsort(descending=True)
        pred = pred[order, :]

        # NMS
        pred = self._nms(pred, conf_thres, iou_thres)
        return pred

    def _distance2bbox(self, points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _distance2kps(self, points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        kps_coords = []
        for i in range(0, 10, 2):
            kps_coords.append(points[:, 0] + distance[:, i])
            kps_coords.append(points[:, 1] + distance[:, i + 1])
        return torch.stack(kps_coords, dim=-1)

    def _nms(self, pred: torch.Tensor, conf_thres: torch.Tensor, iou_thres: torch.Tensor) -> torch.Tensor:
        # Remove items below the confidence threshold.
        keep = torch.where(pred[:, 4] >= conf_thres)[0]
        pred = pred[keep]

        x1 = pred[:, 0]
        y1 = pred[:, 1]
        x2 = pred[:, 2]
        y2 = pred[:, 3]
        scores = pred[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)

        keep = []
        while order.shape[0] > 0:
            i = order[0]
            keep.append(i)
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = torch.maximum(torch.zeros(1), xx2 - xx1 + 1)
            h = torch.maximum(torch.zeros(1), yy2 - yy1 + 1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # i번째 항목(박스)을 기준으로, iou 임계값 이하인 iou를 가지는 항목만 남김
            idx = torch.where(iou <= iou_thres)[0]
            order = order[idx + 1]

        pred = pred[keep, :]
        return pred

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def feature_test(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs
