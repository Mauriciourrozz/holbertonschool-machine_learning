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
        Processes the outputs of the YOLO model.
        - outputs: list of prediction arrays (one per scale)
        - image_size: actual image size (height, width)
        Returns: processed boxes, confidences, and class probabilities.
        """
        image_h, image_w = image_size

        # Inicializamos las listas que vamos a devolver
        boxes = []                # Lista de cajas (x1, y1, x2, y2) por cada salida
        box_confidences = []      # Lista de confianzas por cada salida
        box_class_probs = []      # Lista de probabilidades por clase por cada salida

        # Extraemos el alto y ancho reales de la imagen original
        ih, iw = image_size

        # Recorremos cada una de las salidas del modelo YOLO
        for i, output in enumerate(outputs):
            # Obtenemos dimensiones de esta salida: alto, ancho de la grilla, cantidad de anchors
            gh, gw, anchorBoxes, _ = output.shape

            # Creamos una matriz vacía para guardar las coordenadas finales de las cajas
            box = np.zeros(output[:, :, :, :4].shape)

            # Extraemos las predicciones del modelo para tx, ty, tw, th
            tx = output[:, :, :, 0]
            ty = output[:, :, :, 1]
            tw = output[:, :, :, 2]
            th = output[:, :, :, 3]

            # Obtenemos los tamaños de las cajas ancla (anchor boxes) correspondientes a esta salida
            pwTotal = self.anchors[:, :, 0]  # anchos de anchors
            phTotal = self.anchors[:, :, 1]  # altos de anchors

            # Adaptamos los anchors para que coincidan con el tamaño de la grilla
            pw = np.tile(pwTotal[i], gw).reshape(gw, 1, len(pwTotal[i]))
            ph = np.tile(phTotal[i], gh).reshape(gh, 1, len(phTotal[i]))

            # Creamos las coordenadas de la grilla (cx, cy) para posicionar las cajas
            cx = np.tile(np.arange(gw), gh).reshape(gw, gw, 1)  # coordenada x por celda
            cy = np.tile(np.arange(gw), gh).reshape(gh, gh).T.reshape(gh, gh, 1)  # coordenada y por celda

            # Calculamos el centro de cada caja (normalizado entre 0 y 1)
            bx = (1 / (1 + np.exp(-tx)) + cx) / gw
            by = (1 / (1 + np.exp(-ty)) + cy) / gh

            # Calculamos el ancho y alto de cada caja (normalizado)
            bw = (np.exp(tw) * pw) / self.model.input.shape[1]
            bh = (np.exp(th) * ph) / self.model.input.shape[2]

            # Convertimos de centro (bx, by) y tamaño (bw, bh) a (x1, y1, x2, y2)
            # Y lo escalamos al tamaño real de la imagen
            box[:, :, :, 0] = (bx - (bw / 2)) * iw  # x1
            box[:, :, :, 1] = (by - (bh / 2)) * ih  # y1
            box[:, :, :, 2] = (bx + (bw / 2)) * iw  # x2
            box[:, :, :, 3] = (by + (bh / 2)) * ih  # y2

            # Guardamos las cajas ya procesadas
            boxes.append(box)

            # Calculamos la confianza de que haya un objeto (sigmoid del canal 4)
            temp = output[:, :, :, 4]
            sigmoid = (1 / (1 + np.exp(-temp)))
            box_confidences.append(sigmoid.reshape(gh, gw, anchorBoxes, 1))

            # Calculamos las probabilidades de clase (sigmoid del resto)
            temp = output[:, :, :, 5:]
            box_class_probs.append((1 / (1 + np.exp(-temp))))

        # Devolvemos las listas con todas las salidas procesadas
        return boxes, box_confidences, box_class_probs

