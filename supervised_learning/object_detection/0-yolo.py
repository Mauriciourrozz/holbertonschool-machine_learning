#!/usr/bin/env python3
"""
0-yolo.py
"""
import tensorflow as tf


class Yolo:
    """
    Yolo class uses the YOLO v3 algorithm to perform object detection.

    Attributes:
        model (keras.Model): The Darknet Keras model loaded from model_path.
        class_names (list): List of class names for the model.
        class_t (float): Box score threshold for the initial filtering step.
        nms_t (float): Intersection over Union (IoU) threshold for non-max
            suppression.
        anchors (np.ndarray): Anchor boxes used by the model.
            Shape is (outputs, anchor_boxes, 2), where:
            - outputs: number of output layers of the Darknet model
            - anchor_boxes: number of anchor boxes per output
            - 2: width and height of each anchor box.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo object detector.

        Parameters:
        - model_path (str): path to the Darknet Keras model.
        - classes_path (str): path to the file containing class
            names used by the model.
        - class_t (float): box score threshold for the initial filtering step.
        - nms_t (float): IOU threshold for non-max suppression.
        - anchors (np.ndarray): array of shape (outputs, anchor_boxes, 2)
            with anchor box dimensions.
            outputs: number of output predictions made by the model
            anchor_boxes: number of anchor boxes per prediction
            2: [anchor_box_width, anchor_box_height]
        """

        self.model = tf.keras.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
