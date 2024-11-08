import cv2
import numpy as np
import os
import pickle

class FeatureMatcher:
    def __init__(self, desc_dir, detector_type='SIFT', good_match_threshold=0.75, match_ratio_threshold=0.7):
        """
        инициализация

        :param desc_dir: путь к каталогу с дескрипторами классов
        :param detector_type: тип детектора ('SIFT' или 'ORB')

        :param good_match_threshold: для фильтрации хороших совпадений с помощью метода Lowe's Ratio Test
        для каждой пары совпадающих дескрипторов алгоритм сопоставления возвращает два ближайших совпадения (k=2) по расстоянию
        только те совпадения, у которых расстояние до первого ближайшего соседа существенно меньше (меньше 75% расстояния до второго совпадения), считаются хорошими совпадениями

        :param match_ratio_threshold: порог для окончательного определения, является ли деталь приближенной к идеальной или имеет дефекты
        после фильтрации "хороших" совпадений вычисляется общее количество хороших совпадений, это число делится на общее количество дескрипторов в новом изображении,
        и получается соотношение совпадений, которое показывает, насколько хорошо новое изображение совпадает с эталоном
        """
        self.desc_dir = desc_dir
        self.good_match_threshold = good_match_threshold
        self.match_ratio_threshold = match_ratio_threshold
        self.detector = self._initialize_detector(detector_type) 
        self.matcher = self._initialize_matcher()

    def _initialize_detector(self, detector_type):
        """
        инициализация детектора
        
        :param detector_type: тип детектора ('SIFT' или 'ORB')
        :return: инициализированный детектор
        """
        if detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif detector_type == 'ORB':
            return cv2.ORB_create()
        else:
            raise ValueError("Неподдерживаемый тип детектора. Используйте 'SIFT' или 'ORB'.")

    def _initialize_matcher(self):
        """
        инициализация FLANN-based matcher
        
        :return: FLANN-based matcher

        FLANN-based matcher - метод поиска и сопоставления дескрипторов на основе быстрого поиска ближайших соседей
        index_params: определяет алгоритм и его параметры
        algorithm=1 и trees=5 - использование K-мерных деревьев с 5 деревьями
        search_params:  определяет, сколько итераций будет выполнено при поиске ближайших соседей для каждого дескриптора
        """
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    def load_class_desc(self, class_number):
        """
        загрузка дескрипторов указанного класса
        
        :param class_number: номер класса для загрузки дескрипторов
        :return: список дескрипторов всех изображений данного класса
        """
        class_dir = os.path.join(self.desc_dir, str(class_number))
        desc = []
        for desc_file in os.listdir(class_dir):
            desc_path = os.path.join(class_dir, desc_file)
            with open(desc_path, 'rb') as f:
                desc = pickle.load(f)
                desc.append(desc)
        return desc

    def get_image_desc(self, image_path):
        """
        получение дескрипторов нового изображения
        
        :param image_path: путь к новому изображению
        :return: дескрипторы
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, desc = self.detector.detectAndCompute(image, None)
        return desc

    def is_perfect_match(self, new_desc, class_desc):
        """
        оценка качества совпадений для определения идеальности детали
        
        :param new_desc: дескрипторы нового изображения
        :param class_desc: список дескрипторов эталонного класса
        :return: True, если деталь идеальная; False, если нет
        """
        good_matches_total = 0

        for ref_desc in class_desc:
            # KNN (k=2)
            matches = self.matcher.knnMatch(new_desc, ref_desc, k=2)
            # фильтрация для поиска хороших совпадений
            good_matches = [m for m, n in matches if m.distance < self.good_match_threshold * n.distance]
            good_matches_total += len(good_matches)
        
        # соотношение совпадений
        match_ratio = good_matches_total / len(new_desc)
        return match_ratio > self.match_ratio_threshold
    
    def evaluate_image(self, class_number, image_path):
        """
        оценка нового изображения для проверки наличия дефектов
        
        :param class_number: номер класса для сравнения
        :param image_path: путь к изображению для оценки
        :return: результат проверки ("идеальная" или "с дефектами")
        """
        class_desc = self.load_class_desc(class_number)
        new_desc = self.get_image_desc(image_path)
        
        if new_desc is None:
            return "Не удалось извлечь дескрипторы для изображения"
        
        if self.is_perfect_match(new_desc, class_desc):
            return "Деталь идеальная"
        else:
            return "Деталь имеет дефекты"


if __name__ == "__main__":
    desc_dir = "./desc/"
    class_number = "1"  
    new_image_path = r"D:\Desktop\ref_training\processed_details\1\5.jpg"

    matcher = FeatureMatcher(desc_dir, detector_type='SIFT')
    result = matcher.evaluate_image(class_number, new_image_path)
    print(result)
