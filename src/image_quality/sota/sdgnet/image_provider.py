import os
from sklearn.model_selection import train_test_split
import numpy as np


class ImageProvider:
    def __init__(self, image_folder, saliency_folder, image_mos_file):
        self.image_folder = image_folder
        self.saliency_folder = saliency_folder
        self.image_mos_file = image_mos_file

    def get_images_scores(self):
        image_files = []
        saliency_image_files = []
        scores = []
        with open(self.image_mos_file, 'r+') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split(',')
                image_files.append(os.path.join(self.image_folder, content[0].replace('"', '')))
                saliency_image_files.append(os.path.join(self.saliency_folder, content[0].replace('"', '')))
                score = float(content[-3])
                scores.append(score)
        return image_files, saliency_image_files, scores

    def generate_images(self):
        image_files, saliency_image_files, scores = self.get_images_scores()
        train_image_files, test_image_files, train_saliency_image_files, test_saliency_image_files, train_scores, test_scores = train_test_split(image_files, saliency_image_files, scores, test_size=0.1, random_state=42)
        return train_image_files, test_image_files, train_saliency_image_files, test_saliency_image_files, train_scores, test_scores


