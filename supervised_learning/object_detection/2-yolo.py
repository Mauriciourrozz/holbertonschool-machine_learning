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
        Process the outputs of the YOLO model to obtain:
        - bounding boxes adjusted to the image size,
        - object trusts per cell,
        - probabilities per class.

        Parameters:
        - outputs: list of output arrays of the YOLO model.
        - image_size: tuple (height, width) of the original image.

        Returns:
        - boxes: list of arrays with the boxes transformed by scale.
        - box_confidences: list of arrays with the trusts.
        - box_class_probs: list of arrays with the probabilities per class.
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters bounding boxes based on confidence scores and class
        probabilities.

        Parameters:
        - boxes: list of numpy.ndarrays
            Each array has shape (grid_height, grid_width, anchor_boxes, 4)
            containing the bounding boxes predicted by the model for each
            output.

        - box_confidences: list of numpy.ndarrays
            Each array has shape (grid_height, grid_width, anchor_boxes, 1)
            containing the confidence scores for each bounding box.

        - box_class_probs: list of numpy.ndarrays
            Each array has shape (grid_height, grid_width, anchor_boxes,
            classes)
            containing the class probabilities for each bounding box.

        Returns:
        - filtered_boxes: numpy.ndarray
            Array of shape (?, 4) containing all bounding boxes that passed
            the confidence threshold.

        - box_classes: numpy.ndarray
            Array of shape (?,) containing the class indices for each filtered
            bounding box.

        - box_scores: numpy.ndarray
            Array of shape (?,) containing the scores for each filtered
            bounding box,
            calculated as confidence score multiplied by the highest
            class probability.
        """

        filtered_boxes = []
        box_classes = []
        box_scores = []

        # Recorremos cada salida del modelo (cada escala)
        for box, confidence, class_prob in zip(boxes, box_confidences,
                                               box_class_probs):
            # Calculamos: confianza * probabilidad por clase
            b_score = confidence * class_prob

            # Obtenemos la puntuación máxima y la clase
            # correspondiente para cada caja
            b_class_score = np.max(b_score, axis=-1)
            box_class_indices = np.argmax(b_score, axis=-1)

            # Aplicamos el filtro usando el umbral
            mask = b_class_score >= self.class_t

            # Aplanamos las cajas,clases y scores para poder unir todo al final
            filtered_boxes.append(box[mask])
            box_classes.append(box_class_indices[mask])
            box_scores.append(b_class_score[mask])

        # Concatenamos todo en arrays finales
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
