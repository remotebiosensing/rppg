import tensorflow as tf
import numpy as np
import math
class Facedetect:
    def __init__(self):
        #blazenet input
        self.INPUT_H = 128
        self.INPUT_W = 128
        #blazenet hyper param
        self.NUM_BOXES = 896
        self.NUM_COORDS = 16
        self.BYTE_SIZE_OF_FLOAT = 4
        self.strides = [8,16,16,16]
        self.ASPECT_RATIOS_SIZE = 1
        self.MIN_SCALE = 0.3
        self.MAX_SCALE = 0.8
        self.ANCHOR_OFFSET_X = 0.5
        self.ANCHOR_OFFSET_Y = 0.5
        self.X_SCALE = 128
        self.Y_SCALE = 128
        self.W_SCALE = 128
        self.H_SCALE = 128
        self.MIN_SUPPRESSION_THRESHOLD = 0.3

        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path="face_detection_front.tflite")
        self.interpreter.allocate_tensors()
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.anchors = self.generateAnchors()
        self.floating_model = False
        if self.input_details[0]['dtype'] == np.float32:
            self.floating_model = True


    def calscale(self, min_scale, max_scale, stride_index, num_strides):
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

    def intersectionArea(self, rect1, rect2):
        x1 = max(min(rect1[0], rect1[2]), min(rect2[0], rect2[2]))
        y1 = max(min(rect1[1], rect1[3]), min(rect2[1], rect2[3]))
        x2 = min(max(rect1[0], rect1[2]), max(rect2[0], rect2[2]))
        y2 = min(max(rect1[1], rect1[3]), max(rect2[1], rect2[3]))

        return (x2 - x1) * (y2 - y1)

    def overlapSimilarity(self, rect1, rect2):
        area = self.intersectionArea(rect1, rect2)
        if area <= 0:
            return 0.0
        norm = (rect1[3] - rect1[1]) * (rect1[2] - rect1[0]) + (rect2[3] - rect2[1]) * (rect2[2] - rect2[0]) - area

        if norm > 0.0:
            return norm / area
        else:
            return 0.0

    def generateAnchors(self):
        anchors = []
        layer_id = 0

        while layer_id < len(self.strides):
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []

            last_same_stride_layer = layer_id
            while (last_same_stride_layer < len(self.strides) and self.strides[last_same_stride_layer] == self.strides[layer_id]):
                scale = self.calscale(self.MIN_SCALE, self.MAX_SCALE, last_same_stride_layer, len(self.strides))
                for aspect_ratio_id in range(self.ASPECT_RATIOS_SIZE):
                    aspect_ratios.append(1.0)
                    scales.append(scale)
                if len(self.strides) - 1 == last_same_stride_layer:
                    scale_next = 1.0
                else:
                    scale_next = self.calscale(self.MIN_SCALE, self.MAX_SCALE, last_same_stride_layer + 1, len(self.strides))
                scales.append(math.sqrt(scale * scale_next))
                aspect_ratios.append(1.0)
                last_same_stride_layer = last_same_stride_layer + 1

                for i in range(len(aspect_ratios)):
                    ratio_sqrts = math.sqrt(aspect_ratios[i])
                    anchor_height.append(scales[i] / ratio_sqrts)
                    anchor_width.append(scales[i] * ratio_sqrts)

                stride = self.strides[layer_id];
                feature_map_height = math.ceil(1.0 * self.INPUT_H / stride);
                feature_map_width = math.ceil(1.0 * self.INPUT_W / stride);

                for y in range(feature_map_height):
                    for x in range(feature_map_width):
                        for anchor_id in range(len(anchor_height)):
                            tmp = []
                            x_center = (x + self.ANCHOR_OFFSET_X) * 1.0 / feature_map_width
                            y_center = (y + self.ANCHOR_OFFSET_Y) * 1.0 / feature_map_height

                            tmp.append(x_center)
                            tmp.append(y_center)
                            tmp.append(1.0)
                            tmp.append(1.0)

                            anchors.append(tmp)
                layer_id = last_same_stride_layer
        return anchors

    def weightedNonMaxSuppression(self, indexed_scores, detections):

        remained_indexed_scores = indexed_scores
        output_locations = []

        while len(remained_indexed_scores):
            detection = detections[remained_indexed_scores[0][0]]

            if len(detection) < 5:
                break

            if detection[4] < -1:
                break
            remained = []
            candidates = []
            location = detection#detection[0:4]
            del location[4]


            for i in range(len(indexed_scores)):
                rest_location = detections[indexed_scores[i][0]][0:4]
                sim = self.overlapSimilarity(rest_location, location)
                if sim > 0.3:
                    candidates.append(indexed_scores[i])
                else:
                    remained.append(indexed_scores[i])

            weighted_location = location
            if len(candidates) == 0:
                w_xmin = 0.0
                w_ymin = 0.0
                w_xmax = 0.0
                w_ymax = 0.0
                total_score = 0.0
                for i in range(len(candidates)):
                    total_score += candidates[i][1]
                    bbox = detections[candidates[i][0]][0:4]

                    w_xmin += bbox[0] * candidates[i][1]
                    w_ymin += bbox[1] * candidates[i][1]
                    w_xmax += bbox[2] * candidates[i][1]
                    w_ymax += bbox[3] * candidates[i][1]

                weighted_location[0] = w_xmin / total_score * self.INPUT_W
                weighted_location[1] = w_ymin / total_score * self.INPUT_H
                weighted_location[2] = w_xmax / total_score * self.INPUT_W
                weighted_location[3] = w_ymax / total_score * self.INPUT_H

            remained_indexed_scores = remained

            output_locations.append(weighted_location)
        return output_locations

    def detect(self, img):
        input_data = np.expand_dims(img,axis=0)
        if self.floating_model:
            input_data = np.float32(input_data)/127.5-1
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()
        detected_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        detected_scores = self.interpreter.get_tensor(self.output_details[1]['index'])

        detections = []
        for i in range(self.NUM_BOXES):
            score = detected_scores[0][i][0]
            if score < -100:
                score = -100
            if score > 100:
                score = 100
            score = 1/(1+math.exp(-1*score))
            if score <= 0.4:
                continue

            width_test = 1200;
            height_test = 800;
            x_center = detected_boxes[0][i][0] / self.X_SCALE * self.anchors[i][2] + self.anchors[i][0]
            x_center = x_center * width_test
            y_center = detected_boxes[0][i][1] / self.Y_SCALE * self.anchors[i][3] + self.anchors[i][1]
            y_center = y_center * height_test
            w = detected_boxes[0][i][2] / self.X_SCALE * self.anchors[i][2]
            w = w * width_test
            h = detected_boxes[0][i][3] / self.Y_SCALE * self.anchors[i][3]
            h = h * height_test
            # 추가 적인것 6개의 좌표임 /keypoint 6개
            box = []

            ymin = int(round(y_center - h / 2))
            xmin = int(round(x_center - w / 2))
            ymax = int(round(y_center + h / 2))
            xmax = int(round(x_center + w / 2))

            box.append(xmin)
            box.append(ymin)
            box.append(xmax)
            box.append(ymax)
            box.append(score)

            for j in range(4, 16, 2):
                pos_x = detected_boxes[0][i][j] / self.X_SCALE * self.anchors[i][2] + self.anchors[i][0]
                pos_x = pos_x * width_test
                pos_y = detected_boxes[0][i][j+1] / self.Y_SCALE * self.anchors[i][3] + self.anchors[i][1]
                pos_y = pos_y * height_test
                box.append(pos_x)
                box.append(pos_y)

            detections.append(box)

        #print(detections)

        index_list = []
        for i in range(len(detections)):
            index = []
            index.append(i)
            index.append(detections[i][4])
            index_list.append(index)

        index_list.sort(key=lambda x: x[1])
        index_list.reverse()

        out_result = self.weightedNonMaxSuppression(index_list, detections)

        # print(out_result)

        return out_result