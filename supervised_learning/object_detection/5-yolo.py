#!/usr/bin/env python3
"""
0-yolo.py
"""
import tensorflow as tf
import numpy as np
import cv2
import os


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

    def sigmoid(self, x):
        """Apply the sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the YOLO model.

        Parameters:
        - outputs: list of arrays (one for each scale of the model)
        - image_size: actual image size [height, width]

        Returns:
        - boxes: coordinates (x1, y1, x2, y2) rescaled to the original image
        - box_confidences: object confidence for each box
        - box_class_probs: class probabilities for each box
        """
        image_h, image_w = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = output.shape[:3]

            # Separamos los datos del output
            t_xy = self.sigmoid(output[..., 0:2])  # tx, ty
            t_wh = output[..., 2:4]  # tw, th
            # confianza de que hay un objeto
            objectness = self.sigmoid(output[..., 4:5])
            # probabilidades por clase
            class_probs = self.sigmoid(output[..., 5:])

            # Creamos coordenadas (cx, cy) de la grilla
            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            cx, cy = np.meshgrid(grid_x, grid_y)
            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            # Sumamos la posición de la celda para calcular bx, by
            bx = (t_xy[..., 0] + cx) / grid_w
            by = (t_xy[..., 1] + cy) / grid_h

            # Ancho y alto usando anchor boxes
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            bw = (np.exp(t_wh[..., 0]) * anchor_w) / self.model.input.shape[1]
            bh = (np.exp(t_wh[..., 1]) * anchor_h) / self.model.input.shape[2]

            # Convertimos a (x1, y1, x2, y2)
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            # Apilamos las coordenadas en el último eje
            box = np.stack([x1, y1, x2, y2], axis=-1)

            # Guardamos los resultados
            boxes.append(box)
            box_confidences.append(objectness)
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
            Array of shape (?,) containing the class index for each filtered
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
            box_class_index = np.argmax(b_score, axis=-1)

            # Aplicamos el filtro usando el umbral
            mask = b_class_score >= self.class_t

            # Aplanamos las cajas,clases y scores para poder unir todo al final
            filtered_boxes.append(box[mask])
            box_classes.append(box_class_index[mask])
            box_scores.append(b_class_score[mask])

        # Concatenamos todo en arrays finales
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Performs Non-Maximum Suppression (NMS) on the filtered bounding boxes
        to reduce overlapping boxes for each detected object class.

        Parameters:
        - filtered_boxes (numpy.ndarray): Array of shape (?, 4) containing all
        filtered bounding boxes (x1, y1, x2, y2).
        - box_classes (numpy.ndarray): Array of shape (?,) containing the class
        indices corresponding to each box.
        - box_scores (numpy.ndarray): Array of shape (?) containing the score
        for each box.

        Returns:
        - box_predictions (numpy.ndarray): Array of shape (?, 4) containing the
        final bounding boxes after NMS.
        - predicted_box_classes (numpy.ndarray): Array of shape (?,) containing
        class indices for the boxes in box_predictions.
        - predicted_box_scores (numpy.ndarray): Array of shape (?) containing
        the scores for the boxes in box_predictions.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            # Ordenamos las cajas por score descendente
            sorted_indices = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            selected_boxes = []
            selected_scores = []

            while len(cls_boxes) > 0:
                # Seleccionamos la caja con mayor score
                box = cls_boxes[0]
                score = cls_scores[0]
                selected_boxes.append(box)
                selected_scores.append(score)

                # Calculamos IoU entre la caja seleccionada y las restantes
                rest_boxes = cls_boxes[1:]

                ious = self._iou(box, rest_boxes)

                # Filtramos las cajas cuyo IoU es menor que el umbral
                keep_indices = np.where(ious < self.nms_t)[0]

                # Actualizamos las listas manteniendo solo las cajas "keep"
                cls_boxes = rest_boxes[keep_indices]
                cls_scores = cls_scores[1:][keep_indices]

            selected_classes = np.full(len(selected_boxes), cls)

            box_predictions.append(np.array(selected_boxes))
            predicted_box_classes.append(selected_classes)
            predicted_box_scores.append(np.array(selected_scores))

        predict = np.concatenate(box_predictions, axis=0)
        predictBoxClasses = np.concatenate(predicted_box_classes, axis=0)
        predictBoxScores = np.concatenate(predicted_box_scores, axis=0)

        return predict, predictBoxClasses, predictBoxScores

    def _iou(self, box, boxes):
        """
        Calculate the IoU between one box and several boxes
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box_area + boxes_area - inter_area

        return inter_area / union_area

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a specified folder.

        Parameters:
        - folder_path (str): The path to the folder containing images to load.

        Returns:
        - images (list of numpy.ndarray): A list containing the loaded images
            as numpy arrays.
        - image_paths (list of str): A list containing the full paths to the
            loaded images.

        """
        images = []
        imagesPath = []

        for imagen in os.listdir(folder_path):
            path = os.path.join(folder_path, imagen)
            image = cv2.imread(path)

            if image is not None:
                images.append(image)
                imagesPath.append(path)

        return images, imagesPath

    def preprocess_images(self, images):
        """
        Preprocesses a list of images for input into the YOLO model.

        Each image is resized to the model's expected input dimensions and
        normalized to have pixel values between 0 and 1.

        Parameters:
        - images (list of numpy.ndarray): List of input images to preprocess.

        Returns:
        - pimages (numpy.ndarray): Array of preprocessed images, resized and
            normalized.
        - image_shape (numpy.ndarray): Array of original image shapes in the
            format (height, width)
        for each image in the input list.
        """
        pimages = []
        image_shapes = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i in images:
            resize = cv2.resize(i, (input_w, input_h),
                                interpolation=cv2.INTER_CUBIC)
            normalization = resize / 255.0

            image_shapes.append(i.shape[:2])
            pimages.append(normalization)
        
        return np.array(pimages, dtype=np.float32), np.array(image_shapes)
