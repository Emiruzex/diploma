# neural_analysis.py
import cv2
import numpy as np
import logging
import torch
import torchvision
from ultralytics import YOLO
# import psutil # Не используется напрямую в этом файле после удаления из process_clahe
import os
import ultralytics as ultralytics_module # Для логгирования версии
import json
import threading
from typing import List, Dict, Optional, Any, Tuple # Tuple был, но не используется явно, можно убрать если что
from config import CONFIG
from detection_models import DetectionResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Версия PyTorch: {torch.__version__}")
logger.info(f"Версия Torchvision: {torchvision.__version__}")
logger.info(f"Версия Ultralytics: {ultralytics_module.__version__}")

model_yolo11: Optional[YOLO] = None
model_yolo8: Optional[YOLO] = None
models_loaded_event = threading.Event() # Событие для сигнализации о загрузке моделей

# Глобальные словари классов (заполняются при загрузке)
CLASSES_YOLO11: Dict[str, str] = {} # {class_id_str: coco_normalized_display_name}
CLASSES_YOLO8: Dict[str, Dict[str, str]] = {} # {class_id_str: {"displayName": "...", "labelName": "/m/...", "normalizedDisplayName": "..."}}
CLASS_MAPPING: Dict[str, Optional[str]] = {} # {oiv7_normalized_display_name: coco_normalized_display_name_or_None}

# Карты для OIV7 (заполняются из CLASSES_YOLO8)
OIV7_CLASS_ID_TO_LABEL_NAME: Dict[str, str] = {}       # {class_id_str: /m/object_id}
OIV7_LABEL_NAME_TO_DISPLAY_NAME: Dict[str, str] = {} # {/m/object_id: oiv7_normalized_display_name}
OIV7_DISPLAY_NAME_TO_LABEL_NAME: Dict[str, str] = {} # {oiv7_normalized_display_name: /m/object_id}

# Карты иерархии (ключи и значения - oiv7_normalized_display_name)
HIERARCHY_PARENT_CHILD: Dict[str, List[str]] = {} # {parent_norm_disp_name: [child1_norm_disp_name, ...]}
HIERARCHY_CHILD_PARENT: Dict[str, List[str]] = {} # {child_norm_disp_name: [parent1_norm_disp_name, ...]}
HIERARCHY_OBJECT_PARTS: Dict[str, List[str]] = {} # {object_norm_disp_name: [part1_norm_disp_name, ...]}
HIERARCHY_PART_IS_PART_OF: Dict[str, List[str]] = {} # {part_norm_disp_name: [object1_norm_disp_name, ...]}


def _normalize_name(name: Optional[str]) -> Optional[str]:
    """Нормализует имя: заменяет '_' на пробел и удаляет лишние пробелы по краям."""
    if name is None:
        return None
    return name.replace('_', ' ').strip()

def _parse_hierarchy_node(node_data: Dict, current_parent_label_name: Optional[str] = None):
    """Рекурсивно парсит узел иерархии OIV7."""
    node_label_name = node_data.get("LabelName") # /m/ ID
    # Проверяем, есть ли такой LabelName в наших картах (т.е. был ли он в classes.json)
    if not node_label_name or node_label_name not in OIV7_LABEL_NAME_TO_DISPLAY_NAME:
        return 
    
    node_display_name_norm = OIV7_LABEL_NAME_TO_DISPLAY_NAME[node_label_name] # Уже нормализован

    if current_parent_label_name:
        current_parent_display_name_norm = OIV7_LABEL_NAME_TO_DISPLAY_NAME.get(current_parent_label_name)
        if current_parent_display_name_norm:
            # Отношения "Подкатегория" (Subcategory)
            HIERARCHY_PARENT_CHILD.setdefault(current_parent_display_name_norm, []).append(node_display_name_norm)
            HIERARCHY_CHILD_PARENT.setdefault(node_display_name_norm, []).append(current_parent_display_name_norm)
    
    # Рекурсивный вызов для подкатегорий текущего узла
    if "Subcategory" in node_data:
        for sub_node in node_data["Subcategory"]:
            _parse_hierarchy_node(sub_node, current_parent_label_name=node_label_name) # Родитель - текущий узел
            
    # Обработка частей (Part) текущего узла
    if "Part" in node_data:
        HIERARCHY_OBJECT_PARTS.setdefault(node_display_name_norm, [])
        for part_node_data in node_data["Part"]:
            part_label_name = part_node_data.get("LabelName")
            if part_label_name and part_label_name in OIV7_LABEL_NAME_TO_DISPLAY_NAME:
                part_display_name_norm = OIV7_LABEL_NAME_TO_DISPLAY_NAME[part_label_name]
                
                HIERARCHY_OBJECT_PARTS[node_display_name_norm].append(part_display_name_norm)
                HIERARCHY_PART_IS_PART_OF.setdefault(part_display_name_norm, []).append(node_display_name_norm)
                
                # Части также могут иметь свои подкатегории
                if "Subcategory" in part_node_data:
                     for sub_part_node in part_node_data["Subcategory"]:
                         _parse_hierarchy_node(sub_part_node, current_parent_label_name=part_label_name)


def load_oiv7_hierarchy():
    """Загружает и парсит иерархию OIV7 из JSON-файла."""
    global HIERARCHY_PARENT_CHILD, HIERARCHY_CHILD_PARENT, HIERARCHY_OBJECT_PARTS, HIERARCHY_PART_IS_PART_OF

    if not OIV7_LABEL_NAME_TO_DISPLAY_NAME: # Проверка, что базовые карты OIV7 загружены
        logger.error("OIV7_LABEL_NAME_TO_DISPLAY_NAME пуст. Иерархия не может быть построена.")
        return

    hierarchy_file_path = CONFIG.get("hierarchy_file", "hierarchy.json")
    try:
        with open(hierarchy_file_path, 'r', encoding='utf-8') as f:
            hierarchy_data_root = json.load(f) # Загружаем корневой элемент иерархии
        
        # Иерархия может начинаться с "Subcategory" или "Part" на верхнем уровне
        if isinstance(hierarchy_data_root, dict):
            if "Subcategory" in hierarchy_data_root:
                 for top_level_node in hierarchy_data_root["Subcategory"]:
                    _parse_hierarchy_node(top_level_node) # Родителя нет на верхнем уровне
            
            # Обработка частей, если они есть на верхнем уровне (маловероятно для полного hierarchy.json, но для полноты)
            if "Part" in hierarchy_data_root: 
                 root_label_name_for_parts = hierarchy_data_root.get("LabelName") # Если есть LabelName у корня
                 parent_for_root_parts = root_label_name_for_parts if root_label_name_for_parts in OIV7_LABEL_NAME_TO_DISPLAY_NAME else None
                 for part_node in hierarchy_data_root["Part"]:
                     _parse_hierarchy_node(part_node, current_parent_label_name=parent_for_root_parts)

        parents_count = sum(len(v) for v in HIERARCHY_PARENT_CHILD.values())
        parts_count = sum(len(v) for v in HIERARCHY_OBJECT_PARTS.values())
        logger.info(f"Иерархия OIV7 загружена: {parents_count} родительских связей, {parts_count} связей частей.")

    except FileNotFoundError:
        logger.error(f"Файл иерархии {hierarchy_file_path} не найден.")
    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON в файле иерархии: {hierarchy_file_path}.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке иерархии OIV7: {e}", exc_info=True)


def load_class_dictionaries():
    """Загружает все словари классов (COCO, OIV7) и карту их сопоставления."""
    global CLASSES_YOLO11, CLASSES_YOLO8, CLASS_MAPPING
    global OIV7_CLASS_ID_TO_LABEL_NAME, OIV7_LABEL_NAME_TO_DISPLAY_NAME, OIV7_DISPLAY_NAME_TO_LABEL_NAME
    
    try:
        with open('coco_classes.json', 'r', encoding='utf-8') as f:
            raw_coco_classes = json.load(f)
        # Нормализуем отображаемые имена COCO сразу при загрузке
        CLASSES_YOLO11 = {k: _normalize_name(v) for k, v in raw_coco_classes.items() if _normalize_name(v)}
        logger.info(f"Классы COCO (YOLOv11) загружены: {len(CLASSES_YOLO11)} записей.")
    except Exception as e_coco:
        logger.error(f"Ошибка загрузки coco_classes.json: {e_coco}", exc_info=True); raise

    try:
        with open('classes.json', 'r', encoding='utf-8') as f: # OIV7 classes
            CLASSES_YOLO8 = json.load(f) # Загружаем как есть {class_id_str: {"displayName": "...", "labelName": "/m/..."}}
        
        # Заполняем карты OIV7 с нормализованными displayName
        for class_id_str, data_dict in CLASSES_YOLO8.items():
            if isinstance(data_dict, dict) and "displayName" in data_dict and "labelName" in data_dict:
                raw_display_name = data_dict["displayName"] # Может содержать '_'
                normalized_display_name = _normalize_name(raw_display_name)
                label_name_mid = data_dict["labelName"] # Это /m/ ID, его не нормализуем
                
                if normalized_display_name and label_name_mid:
                    OIV7_CLASS_ID_TO_LABEL_NAME[class_id_str] = label_name_mid
                    OIV7_LABEL_NAME_TO_DISPLAY_NAME[label_name_mid] = normalized_display_name
                    OIV7_DISPLAY_NAME_TO_LABEL_NAME[normalized_display_name] = label_name_mid
                    # Добавляем нормализованное имя в исходный словарь для удобства
                    CLASSES_YOLO8[class_id_str]['normalizedDisplayName'] = normalized_display_name
                else:
                    logger.warning(f"Пропущена некорректная запись для class_id {class_id_str} в classes.json: {data_dict}")
        logger.info(f"Карты классов OIV7 (YOLOv8) созданы: {len(OIV7_LABEL_NAME_TO_DISPLAY_NAME)} записей.")
    except Exception as e_oiv7:
        logger.error(f"Ошибка загрузки или обработки classes.json (OIV7): {e_oiv7}", exc_info=True); raise

    try:
        with open('class_mapping.json', 'r', encoding='utf-8') as f:
            raw_mapping_data = json.load(f)['mapping']
        # Ключи (OIV7 displayName с '_') и значения (COCO displayName с '_')
        # Нормализуем и ключи, и значения при загрузке
        CLASS_MAPPING = {_normalize_name(k): _normalize_name(v) 
                         for k, v in raw_mapping_data.items() if _normalize_name(k)}
        logger.info(f"Сопоставление классов OIV7->COCO загружено: {len(CLASS_MAPPING)} записей.")
    except Exception as e_map:
        logger.error(f"Ошибка загрузки class_mapping.json: {e_map}", exc_info=True); raise

    load_oiv7_hierarchy() # Загружаем иерархию после того, как OIV7 карты готовы

# Потокобезопасная загрузка моделей с использованием threading.Lock
_models_load_lock = threading.Lock()

def ensure_models_loaded():
    """Гарантирует, что нейросетевые модели YOLO загружены (потокобезопасно)."""
    global model_yolo11, model_yolo8
    
    if models_loaded_event.is_set(): # Если модели уже загружены
        return

    with _models_load_lock: # Блокировка, чтобы только один поток загружал модели
        if models_loaded_event.is_set(): # Повторная проверка внутри блокировки
            return

        logger.info("Начало загрузки нейросетевых моделей YOLO...")
        model_name_yolo11_cfg = CONFIG.get("model_name", "yolo11x.pt")
        model_name_yolo8_cfg = CONFIG.get("oiv7_model_name", "yolov8x-oiv7.pt")
        
        try:
            if not os.path.exists(model_name_yolo11_cfg):
                raise FileNotFoundError(f"Файл модели {model_name_yolo11_cfg} не найден.")
            if not os.path.exists(model_name_yolo8_cfg):
                raise FileNotFoundError(f"Файл модели {model_name_yolo8_cfg} не найден.")

            model_yolo11 = YOLO(model_name_yolo11_cfg)
            model_yolo8 = YOLO(model_name_yolo8_cfg)

            use_gpu_flag = CONFIG.get("use_gpu", True)
            if use_gpu_flag and torch.cuda.is_available():
                logger.info("CUDA доступна. Перемещение моделей на GPU и использование FP16 (half precision).")
                model_yolo11.to('cuda')
                # Проверка на наличие model.half() для совместимости
                if hasattr(model_yolo11, 'model') and hasattr(model_yolo11.model, 'half'): model_yolo11.model.half()
                
                model_yolo8.to('cuda')
                if hasattr(model_yolo8, 'model') and hasattr(model_yolo8.model, 'half'): model_yolo8.model.half()
            else:
                device_msg = "CUDA не доступна (или use_gpu=False)." if not torch.cuda.is_available() and use_gpu_flag else ""
                logger.info(f"Модели будут использовать CPU. {device_msg}")

            models_loaded_event.set() # Устанавливаем событие, сигнализируя о загрузке
            logger.info(f"Модели {model_name_yolo11_cfg} и {model_name_yolo8_cfg} успешно загружены.")

        except FileNotFoundError as e_fnf:
            logger.error(f"Ошибка загрузки моделей: {str(e_fnf)}")
            # Не устанавливаем models_loaded_event, чтобы при следующем вызове была попытка загрузки
            raise # Передаем исключение выше
        except Exception as e_load:
            logger.error(f"Критическая ошибка при инициализации моделей: {str(e_load)}", exc_info=True)
            raise


def классификация_объектов(
    изображение_bgr: np.ndarray, 
    confidence_threshold: Optional[float] = None, 
    min_area_pixels: Optional[int] = None,
    model_source_filter: Optional[str] = None  # "YOLOv11" или "YOLOv8" для фильтрации
) -> List[DetectionResult]:
    """Выполняет детекцию объектов на изображении с использованием указанной модели или обеих."""
    
    # Установка порогов из конфига, если не переданы напрямую
    conf_thresh_val = confidence_threshold if confidence_threshold is not None else CONFIG.get("confidence_threshold", 0.2)
    min_area_val = min_area_pixels if min_area_pixels is not None else CONFIG.get("min_area", 300)

    ensure_models_loaded() # Гарантирует, что модели загружены перед использованием

    logger.info(f"Начало классификации. Фильтр модели: {model_source_filter or 'Обе'}.")
    all_raw_detections: List[DetectionResult] = []
    
    try:
        if изображение_bgr is None or изображение_bgr.size == 0:
            logger.error("Ошибка: входное изображение для классификации пустое или None.")
            return []

        # --- Детекция моделью YOLOv11 (COCO) ---
        if model_yolo11 and (model_source_filter is None or model_source_filter == "YOLOv11"):
            try:
                # verbose=False для уменьшения вывода в консоль от YOLO
                yolo11_preds = model_yolo11.predict(изображение_bgr, conf=conf_thresh_val, verbose=False)
                for result_item in yolo11_preds: # result_item это ultralytics.engine.results.Results
                    for box in result_item.boxes: # box это ultralytics.engine.results.Boxes
                        x_center, y_center, w, h = box.xywh[0].cpu().numpy().astype(int)
                        conf = float(box.conf.cpu().numpy())
                        class_id_str = str(int(box.cls.cpu().numpy()))
                        
                        # Получаем нормализованное имя класса COCO
                        label_name_norm = CLASSES_YOLO11.get(class_id_str, f"COCO_Unknown_{class_id_str}")
                        
                        if (w * h) >= min_area_val:
                            det = DetectionResult(
                                x=int(x_center - w / 2), y=int(y_center - h / 2), w=w, h=h,
                                original_label_raw=label_name_norm, # Уже нормализовано
                                confidence=conf, model_source="YOLOv11"
                            )
                            all_raw_detections.append(det)
                logger.debug(f"YOLOv11 обнаружил {len(yolo11_preds[0].boxes) if yolo11_preds else 0} объектов (до фильтра площади).")
            except Exception as e_y11_pred:
                logger.error(f"Ошибка при предсказании YOLOv11: {e_y11_pred}", exc_info=True)

        # --- Детекция моделью YOLOv8 (OIV7) ---
        if model_yolo8 and (model_source_filter is None or model_source_filter == "YOLOv8"):
            try:
                yolo8_preds = model_yolo8.predict(изображение_bgr, conf=conf_thresh_val, verbose=False)
                for result_item in yolo8_preds:
                    for box in result_item.boxes:
                        x_center, y_center, w, h = box.xywh[0].cpu().numpy().astype(int)
                        conf = float(box.conf.cpu().numpy())
                        class_id_str = str(int(box.cls.cpu().numpy()))
                        
                        # Получаем нормализованное displayName из CLASSES_YOLO8
                        oiv7_display_label_norm = f"OIV7_Unknown_{class_id_str}" # Фоллбэк
                        if class_id_str in CLASSES_YOLO8 and 'normalizedDisplayName' in CLASSES_YOLO8[class_id_str]:
                            oiv7_display_label_norm = CLASSES_YOLO8[class_id_str]['normalizedDisplayName']
                        else:
                             logger.warning(f"Не найден class_id {class_id_str} или 'normalizedDisplayName' для YOLOv8.")

                        if (w * h) >= min_area_val:
                            det = DetectionResult(
                                x=int(x_center - w / 2), y=int(y_center - h / 2), w=w, h=h,
                                original_label_raw=oiv7_display_label_norm, # Уже нормализовано
                                confidence=conf, model_source="YOLOv8"
                            )
                            all_raw_detections.append(det)
                logger.debug(f"YOLOv8 обнаружил {len(yolo8_preds[0].boxes) if yolo8_preds else 0} объектов (до фильтра площади).")
            except Exception as e_y8_pred:
                logger.error(f"Ошибка при предсказании YOLOv8: {e_y8_pred}", exc_info=True)
        
        logger.info(f"Классификация завершена. Общее число 'сырых' детекций: {len(all_raw_detections)}")
        return all_raw_detections

    except Exception as e_classify_main: # Обработка ошибок ensure_models_loaded или других общих
        logger.error(f"Критическая ошибка в процессе классификации: {e_classify_main}", exc_info=True)
        return [] # Возвращаем пустой список в случае серьезной ошибки

# Загружаем словари классов и иерархию при импорте модуля
# Это должно быть сделано один раз при запуске приложения.
try:
    load_class_dictionaries()
except Exception as e_init_dicts:
    logger.critical(f"Не удалось загрузить словари классов при инициализации модуля neural_analysis: {e_init_dicts}", exc_info=True)
    # Приложение, скорее всего, не сможет нормально работать без этих словарей.
    # Можно здесь либо reraise, либо установить флаг ошибки, который проверится в app.py.
    # Пока просто логгируем критическую ошибку.