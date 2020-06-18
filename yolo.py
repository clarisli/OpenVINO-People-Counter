import logging as log
from math import exp as exp


#log.basicConfig(level=log.info)
class YoloDetector:
    
    class Params:
        # ------------------------------------------- Extracting layer parameters ------------------------------------------
        # Magic numbers are copied from Yolo samples
        def __init__(self, param, side):
            self.num = 3 if 'num' not in param else int(param['num'])
            self.coords = 4 if 'coords' not in param else int(param['coords'])
            self.classes = 80 if 'classes' not in param else int(param['classes'])
            self.side = side
            self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                            198.0,
                            373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

            self.isYoloV3 = False

            if param.get('mask'):
                mask = [int(idx) for idx in param['mask'].split(',')]
                self.num = len(mask)

                maskedAnchors = []
                for idx in mask:
                    maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                self.anchors = maskedAnchors

                self.isYoloV3 = True # Weak way to determine but the only one.

        def log_params(self):
            #params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
            #[log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]
            pass


    @staticmethod
    def _entry_index(side, coord, classes, location, entry):
        side_power_2 = side ** 2
        n = location // side_power_2
        loc = location % side_power_2
        return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

    @staticmethod
    def _scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
        xmin = int((x - w / 2) * w_scale)
        ymin = int((y - h / 2) * h_scale)
        xmax = int(xmin + w * w_scale)
        ymax = int(ymin + h * h_scale)

        #xcentroid, ycentroid = (xmin+xmax)/2, (ymin+ymax)/2
        return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)#, xcentroid=xcentroid, ycentroid=ycentroid)

    @staticmethod
    def _parse_Yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
        # ------------------------------------------ Validating output parameters ------------------------------------------
        _, _, out_blob_h, out_blob_w = blob.shape
        assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                        "be equal to width. Current height = {}, current width = {}" \
                                        "".format(out_blob_h, out_blob_w)

        # ------------------------------------------ Extracting layer parameters -------------------------------------------
        orig_im_h, orig_im_w = original_im_shape
        resized_image_h, resized_image_w = resized_image_shape
        objects = list()
        predictions = blob.flatten()
        side_square = params.side * params.side

        # ------------------------------------------- Parsing Yolo Region output -------------------------------------------
        for i in range(side_square):
            row = i // params.side
            col = i % params.side
            for n in range(params.num):
                obj_index = YoloDetector._entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
                scale = predictions[obj_index]
                if scale < threshold:
                    continue
                box_index = YoloDetector._entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
                # Network produces location predictions in absolute coordinates of feature maps.
                # Scale it to relative coordinates.
                x = (col + predictions[box_index + 0 * side_square]) / params.side
                y = (row + predictions[box_index + 1 * side_square]) / params.side
                # Value for exp is very big number in some cases so following construction is using here
                try:
                    w_exp = exp(predictions[box_index + 2 * side_square])
                    h_exp = exp(predictions[box_index + 3 * side_square])
                except OverflowError:
                    continue
                # Depends on topology we need to normalize sizes by feature maps (up to Yolov3) or by input shape (Yolov3)
                w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
                h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
                for j in range(params.classes):
                    class_index = YoloDetector._entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                            params.coords + 1 + j)
                    confidence = scale * predictions[class_index]
                    if confidence < threshold:
                        continue
                    bbox = YoloDetector._scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                            h_scale=orig_im_h, w_scale=orig_im_w)
                    objects.append(bbox)
        return objects

    @staticmethod
    def intersection_over_union(box_1, box_2):
        width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
        height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
        box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union
    
    @staticmethod
    def _get_detected_objects(output, network, prob_threshold, resized_image_shape, original_image_shape):
        objects = list()
        for layer_name, out_blob in output.items():
            out_blob = out_blob.reshape(network.network.layers[layer_name].out_data[0].shape)
            layer_params = YoloDetector.Params(network.network.layers[layer_name].params, out_blob.shape[2])
            layer_params.log_params()
            objects += YoloDetector._parse_Yolo_region(out_blob, resized_image_shape,
                                        original_image_shape, layer_params,
                                        prob_threshold)
        return objects
    
    @staticmethod
    def _filter_overlapping_boxes(objects, iou_threshold):
        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if YoloDetector.intersection_over_union(objects[i], objects[j]) > iou_threshold:
                    objects[j]['confidence'] = 0 
        return objects

    @staticmethod
    def get_qualified_objects(output, network, prob_threshold, iou_threshold, resized_image_shape, original_image_shape):
        objects = YoloDetector._get_detected_objects(output, network, prob_threshold, resized_image_shape, original_image_shape)
        objects = YoloDetector._filter_overlapping_boxes(objects, iou_threshold)
        objects = [obj for obj in objects if obj['confidence'] >= prob_threshold and obj['class_id'] == 0] # 0 is the class id for person
        return objects

