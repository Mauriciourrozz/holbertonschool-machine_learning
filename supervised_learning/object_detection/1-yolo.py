#!/usr/bin/env python3
"""
0-yolo.py
"""
import tensorflow as tf
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """
        Procesa las salidas del modelo YOLO para obtener:
        - bounding boxes ajustadas al tamaño de la imagen,
        - confianzas de objeto por celda,
        - probabilidades por clase.

        Parámetros:
        - outputs: lista de arrays de salida del modelo YOLO.
        - image_size: tupla (altura, ancho) de la imagen original.

        Devuelve:
        - boxes: lista de arrays con las cajas transformadas por escala.
        - box_confidences: lista de arrays con las confianzas.
        - box_class_probs: lista de arrays con las probabilidades por clase.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Preparamos el array donde se guardarán las bounding boxes
            box = np.zeros_like(output[:, :, :, :4])

            # Extraemos las predicciones crudas del modelo
            tx = output[:, :, :, 0]
            ty = output[:, :, :, 1]
            tw = output[:, :, :, 2]
            th = output[:, :, :, 3]

            # Creamos las grillas de coordenadas (meshgrid)
            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # Les damos forma para que puedan sumarse a tx y ty
            grid_x = grid_x[:, :, np.newaxis]  # (grid_h, grid_w, 1)
            grid_y = grid_y[:, :, np.newaxis]  # (grid_h, grid_w, 1)

            # Obtenemos los anchors correspondientes a esta escala
            anchor_w = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))

            # Input size del modelo
            input_shape = self.model.input.shape.as_list()
            input_h = input_shape[1]
            input_w = input_shape[2]

            # Calculamos las coordenadas normalizadas de los centros
            bx = (1 / (1 + np.exp(-tx)) + grid_x) / grid_w
            by = (1 / (1 + np.exp(-ty)) + grid_y) / grid_h

            # Calculamos el ancho y alto normalizados de las cajas
            bw = (np.exp(tw) * anchor_w) / input_w
            bh = (np.exp(th) * anchor_h) / input_h

            # Convertimos de centro-ancho-alto a (x1, y1, x2, y2)
            box[:, :, :, 0] = (bx - bw / 2) * image_w  # x1
            box[:, :, :, 1] = (by - bh / 2) * image_h  # y1
            box[:, :, :, 2] = (bx + bw / 2) * image_w  # x2
            box[:, :, :, 3] = (by + bh / 2) * image_h  # y2

            # Agregamos la caja a la lista
            boxes.append(box)

            # Calculamos la confianza de objeto con sigmoid
            confidence = 1 / (1 + np.exp(-output[:, :, :, 4]))
            box_confidences.append(
                confidence.reshape((grid_h, grid_w, anchor_boxes, 1)))

            # Calculamos la probabilidad por clase también con sigmoid
            class_probs = 1 / (1 + np.exp(-output[:, :, :, 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
