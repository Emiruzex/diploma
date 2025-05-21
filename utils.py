# utils.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
import os
import time # Not directly used, but was in imports. Can be removed if truly unused.
import colorsys
import re
from typing import List, Tuple, Dict, Any, Set, Optional
from copy import deepcopy

from config import CONFIG
from detection_models import DetectionResult
from neural_analysis import (
    CLASS_MAPPING, 
    OIV7_LABEL_NAME_TO_DISPLAY_NAME, 
    OIV7_DISPLAY_NAME_TO_LABEL_NAME, 
    HIERARCHY_PARENT_CHILD,    
    HIERARCHY_CHILD_PARENT,    
    HIERARCHY_OBJECT_PARTS,    
    HIERARCHY_PART_IS_PART_OF  
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {}

COCO_PERSON_LABEL_NORM: Optional[str] = None
OIV7_GENDER_LABELS_NORM: List[str] = [] 
OIV7_GENERIC_PERSON_LABEL_NORM: Optional[str] = None 

def _normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    return label.replace('_', ' ').strip()

def _get_oiv7_person_related_labels(
    class_mapping_norm: Dict[str, Optional[str]], 
    oiv7_display_name_to_label_name_norm: Dict[str, str], 
    hierarchy_child_parent_norm: Dict[str, List[str]], 
    hierarchy_parent_child_norm: Dict[str, List[str]] 
) -> Tuple[Optional[str], List[str], Optional[str]]:
    target_coco_person_label = _normalize_label("Человек")
    
    oiv7_generic_person: Optional[str] = None
    candidate_oiv7_generic_persons: List[str] = []

    for oiv7_disp_norm, mapped_coco_norm in class_mapping_norm.items():
        if mapped_coco_norm == target_coco_person_label:
            if oiv7_disp_norm == target_coco_person_label: 
                oiv7_generic_person = oiv7_disp_norm
                break
            candidate_oiv7_generic_persons.append(oiv7_disp_norm)
    
    if not oiv7_generic_person and candidate_oiv7_generic_persons: 
        oiv7_generic_person = candidate_oiv7_generic_persons[0]
    elif not oiv7_generic_person and target_coco_person_label in oiv7_display_name_to_label_name_norm:
        oiv7_generic_person = target_coco_person_label

    oiv7_gender_labels: Set[str] = set()
    potential_gender_strings_norm = [_normalize_label(s) for s in ["Женщина", "Мужчина", "Девочка", "Мальчик"]]

    for gender_candidate_norm in potential_gender_strings_norm:
        if class_mapping_norm.get(gender_candidate_norm) == target_coco_person_label:
            oiv7_gender_labels.add(gender_candidate_norm)
        elif oiv7_generic_person and gender_candidate_norm in hierarchy_parent_child_norm.get(oiv7_generic_person, []):
            oiv7_gender_labels.add(gender_candidate_norm)
        elif gender_candidate_norm in oiv7_display_name_to_label_name_norm and \
             (class_mapping_norm.get(gender_candidate_norm) is None or class_mapping_norm.get(gender_candidate_norm) == target_coco_person_label) and \
             gender_candidate_norm != oiv7_generic_person :
            oiv7_gender_labels.add(gender_candidate_norm)

    if oiv7_generic_person and oiv7_generic_person in oiv7_gender_labels:
        oiv7_gender_labels.remove(oiv7_generic_person)

    return target_coco_person_label, sorted(list(oiv7_gender_labels)), oiv7_generic_person


def initialize_class_colors(
    classes_yolo11_norm: Dict[str, str],       
    class_mapping_norm: Dict[str, Optional[str]], 
    oiv7_label_to_display_norm: Dict[str, str] 
):
    global CLASS_COLORS, COCO_PERSON_LABEL_NORM, OIV7_GENDER_LABELS_NORM, OIV7_GENERIC_PERSON_LABEL_NORM
    CLASS_COLORS.clear()
    
    unique_display_names_for_colors: Set[str] = set()
    unique_display_names_for_colors.update(filter(None, classes_yolo11_norm.values()))

    for oiv7_mid, oiv7_norm_name in oiv7_label_to_display_norm.items():
        mapped_coco_name = class_mapping_norm.get(oiv7_norm_name)
        if mapped_coco_name: 
            unique_display_names_for_colors.add(mapped_coco_name)
        else: 
            unique_display_names_for_colors.add(oiv7_norm_name)
    
    for mapped_value in class_mapping_norm.values():
        if mapped_value: unique_display_names_for_colors.add(mapped_value)

    sorted_unique_names = sorted(list(filter(None, unique_display_names_for_colors)))

    if not sorted_unique_names:
        logger.warning("Не найдено уникальных имен классов для генерации цветов. Используется цвет по умолчанию.")
        CLASS_COLORS["Неизвестно"] = (128, 128, 128)
    else:
        for idx, class_name in enumerate(sorted_unique_names):
            hue = (idx * 0.618033988749895) % 1.0 
            saturation = 0.85 + (idx % 3) * 0.05 
            value = 0.9 - (idx % 4) * 0.05       
            rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
            CLASS_COLORS[class_name] = tuple(int(c * 255) for c in rgb_float)

    person_label_for_color = _normalize_label("Человек")
    if person_label_for_color in CLASS_COLORS:
        CLASS_COLORS[person_label_for_color] = (255, 0, 0) 
    elif "Неизвестно" not in CLASS_COLORS: 
        CLASS_COLORS["Неизвестно"] = (128, 128, 128)

    COCO_PERSON_LABEL_NORM, OIV7_GENDER_LABELS_NORM, OIV7_GENERIC_PERSON_LABEL_NORM = \
        _get_oiv7_person_related_labels(
            CLASS_MAPPING, 
            OIV7_DISPLAY_NAME_TO_LABEL_NAME, 
            HIERARCHY_CHILD_PARENT, 
            HIERARCHY_PARENT_CHILD 
        )
    logger.info(f"Сгенерировано {len(CLASS_COLORS)} цветов. Персоны/гендеры: "
                f"COCO_Person='{COCO_PERSON_LABEL_NORM}', "
                f"OIV7_Genders='{OIV7_GENDER_LABELS_NORM}', "
                f"OIV7_Generic_Person='{OIV7_GENERIC_PERSON_LABEL_NORM}'")


def load_and_resize_image(file_path: str, max_size: Optional[int] = None) -> Optional[np.ndarray]:
    logger.info(f"Загрузка изображения: {file_path}")
    max_dim = max_size if max_size is not None else CONFIG.get("max_image_size", 1000)
    try:
        image_np_array = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

        if image is None: 
            pil_image = Image.open(file_path).convert('RGB')
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if image is None: 
             logger.error(f"Не удалось загрузить изображение {file_path} ни OpenCV, ни PIL.")
             return None

        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0: 
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                logger.warning(f"Масштабирование {file_path} привело к некорректным размерам ({new_w}x{new_h}). Используется оригинал.")
        return image
    except Exception as e:
        logger.error(f"Ошибка при загрузке/изменении размера {file_path}: {str(e)}", exc_info=True)
        return None

def get_contrast_color(background_rgb: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    r, g, b = background_rgb
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return (0, 0, 0, 255) if luminance > 0.5 else (255, 255, 255, 255)

def draw_results(image_bgr: np.ndarray, results_to_draw: List[DetectionResult]) -> np.ndarray:
    if not CLASS_COLORS: 
        logger.warning("CLASS_COLORS не инициализированы в draw_results. Результаты могут быть без корректных цветов.")
        
    if image_bgr is None:
        logger.error("draw_results: получено пустое изображение (None).")
        return np.zeros((CONFIG.get("max_display_size", 500), CONFIG.get("max_display_size", 500), 3), dtype=np.uint8)

    try:
        if len(image_bgr.shape) == 2: 
            pil_img_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB))
        elif image_bgr.shape[2] == 4: 
            pil_img_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB))
        elif image_bgr.shape[2] == 3: 
            pil_img_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        else:
            logger.error(f"draw_results: неподдерживаемый формат изображения с shape {image_bgr.shape}")
            return image_bgr.copy() 
        pil_img_rgba = pil_img_rgb.convert('RGBA') 
    except Exception as e_conv:
        logger.error(f"Ошибка конвертации NumPy в PIL: {e_conv}", exc_info=True)
        return image_bgr.copy()

    if not results_to_draw: 
        return image_bgr.copy()

    draw_ctx = ImageDraw.Draw(pil_img_rgba)
    font_size = max(12, min(18, int(pil_img_rgba.height / 45))) 
    font: Optional[ImageFont.FreeTypeFont] = None
    
    font_paths_to_try = ["arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    if os.name == 'nt': font_paths_to_try.insert(0, "C:/Windows/Fonts/arial.ttf")

    for fp in font_paths_to_try:
        try:
            if os.path.exists(fp): font = ImageFont.truetype(fp, font_size); break
        except IOError: continue
    if not font:
        try: font = ImageFont.load_default(); logger.warning("Системные шрифты не найдены, используется шрифт по умолчанию.")
        except Exception as e_font_def: logger.error(f"Не удалось загрузить шрифт по умолчанию: {e_font_def}. Текст не будет отображен."); font = None

    for det in results_to_draw:
        x, y, w, h = det.x, det.y, det.w, det.h
        x1, y1 = max(0, x), max(0, y) 
        x2, y2 = min(pil_img_rgba.width - 1, x + w), min(pil_img_rgba.height - 1, y + h) 

        if x2 <= x1 or y2 <= y1: continue 

        display_label_base = det.final_display_label or det.original_label_raw
        confidence_text = f"({int(det.confidence * 100)}%)"
        
        color_key = det.mapped_label_for_logic or _normalize_label(det.original_label_raw) or "Неизвестно"
        box_rgb_color = CLASS_COLORS.get(color_key, CLASS_COLORS.get("Неизвестно", (0, 255, 0))) 

        outline_rgba = box_rgb_color + (255,) 
        draw_ctx.rectangle((x1, y1, x2, y2), fill=None, outline=outline_rgba, width=2)

        if font:
            full_text_label = f"{display_label_base} {confidence_text}"
            
            try: text_box_coords = draw_ctx.textbbox((0,0), full_text_label, font=font) 
            except AttributeError: text_box_coords = (0,0) + draw_ctx.textsize(full_text_label, font=font) 
            
            text_w = text_box_coords[2] - text_box_coords[0]
            text_h = text_box_coords[3] - text_box_coords[1]
            padding = 3 

            text_bg_y0 = y1 - text_h - padding * 2
            text_y_pos_for_render = y1 - text_h - padding 

            if text_bg_y0 < 0: 
                text_bg_y0 = y1 + padding
                text_y_pos_for_render = y1 + padding + (padding // 2 if padding > 2 else 1)

            text_bg_x0 = x1
            text_bg_width_actual = min(text_w + padding * 2, pil_img_rgba.width - text_bg_x0) 
            text_bg_height_actual = text_h + padding * 2

            text_background_fill_rgba = box_rgb_color + (190,) 
            draw_ctx.rectangle(
                [text_bg_x0, text_bg_y0,
                 text_bg_x0 + text_bg_width_actual, text_bg_y0 + text_bg_height_actual],
                fill=text_background_fill_rgba
            )
            text_color_for_render = get_contrast_color(box_rgb_color)
            draw_ctx.text((x1 + padding, text_y_pos_for_render), full_text_label, font=font, fill=text_color_for_render)

    return cv2.cvtColor(np.array(pil_img_rgba.convert('RGB')), cv2.COLOR_RGB2BGR)


def calculate_iou(box1_xywh: Tuple[int,int,int,int], box2_xywh: Tuple[int,int,int,int]) -> Tuple[float, int]:
    x1, y1, w1, h1 = box1_xywh
    x2, y2, w2, h2 = box2_xywh

    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_width = max(0, xi_max - xi_min)
    inter_height = max(0, yi_max - yi_min)
    intersection_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou, intersection_area

def _prepare_detections_for_merging(raw_detections: List[DetectionResult]) -> List[DetectionResult]:
    prepared_list: List[DetectionResult] = []
    for det_raw in raw_detections:
        det = deepcopy(det_raw) 
        
        det.original_label_raw = _normalize_label(det.original_label_raw) or "UnknownRaw"
        
        mapped_coco_label = CLASS_MAPPING.get(det.original_label_raw)
        if mapped_coco_label: 
            det.mapped_label_for_logic = mapped_coco_label
        else: 
            det.mapped_label_for_logic = det.original_label_raw
            
        det.reason = None 
        det.is_subordinate = False 
        prepared_list.append(det)
    return prepared_list


def _refine_y11_with_y8(
    y11_detections: List[DetectionResult], 
    y8_detections: List[DetectionResult]  
) -> Tuple[List[DetectionResult], Set[str]]: 
    refined_y11_list = deepcopy(y11_detections) 
    used_y8_ids: Set[str] = set() 

    iou_strong = CONFIG.get("refine_iou_threshold_strong", 0.55)
    iou_weak_contain = CONFIG.get("refine_iou_threshold_weak_containment", 0.20)
    y8_in_y11_contain_thresh = CONFIG.get("refine_containment_y8_in_y11_threshold", 0.75)
    min_y8_conf_for_refine = CONFIG.get("refine_min_y8_confidence", 0.15)
    y8_pref_conf_factor = CONFIG.get("refine_y8_preference_conf_factor", 0.80) 
    y8_gender_pref_conf_factor = CONFIG.get("refine_y8_gender_preference_conf_factor", 0.60) 

    w_iou, w_conf, w_spec, w_contain = (
        CONFIG.get("refine_score_weight_iou", 0.4), CONFIG.get("refine_score_weight_conf", 0.3),
        CONFIG.get("refine_score_weight_specifity", 0.2), CONFIG.get("refine_score_weight_containment", 0.1)
    )
    
    oiv7_person_parts: List[str] = []
    if OIV7_GENERIC_PERSON_LABEL_NORM: 
        oiv7_person_parts = HIERARCHY_OBJECT_PARTS.get(OIV7_GENERIC_PERSON_LABEL_NORM, [])
        # Пример добавления конкретной части, если она не всегда в иерархии как Part:
        # human_face_label_norm = _normalize_label("Человеческое лицо")
        # if human_face_label_norm and human_face_label_norm not in oiv7_person_parts:
        #     oiv7_person_parts.append(human_face_label_norm)


    for det11 in refined_y11_list:
        best_y8_refiner: Optional[DetectionResult] = None
        highest_refinement_score = -1.0

        for det8 in y8_detections:
            if det8.unique_id in used_y8_ids or det8.confidence < min_y8_conf_for_refine:
                continue 

            iou, inter_area = calculate_iou(det11.box_xywh, det8.box_xywh)
            containment_of_y8_in_y11 = (inter_area / det8.area) if det8.area > 0 else 0.0

            is_geometrically_linked = (iou > iou_strong) or \
                                      (iou > iou_weak_contain and containment_of_y8_in_y11 > y8_in_y11_contain_thresh)

            if is_geometrically_linked:
                can_refine = False
                specifity_score_bonus = 0.0
                current_y8_conf_factor_for_comparison = y8_pref_conf_factor
                
                is_det8_human_part_candidate = False
                if det11.mapped_label_for_logic == COCO_PERSON_LABEL_NORM and \
                   det8.original_label_raw in oiv7_person_parts:
                    is_det8_human_part_candidate = True
                
                if not is_det8_human_part_candidate: 
                    if COCO_PERSON_LABEL_NORM and OIV7_GENDER_LABELS_NORM and \
                       det11.mapped_label_for_logic == COCO_PERSON_LABEL_NORM and \
                       det8.original_label_raw in OIV7_GENDER_LABELS_NORM:
                        can_refine = True
                        specifity_score_bonus = 0.70 # ВЫСОКИЙ БОНУС ДЛЯ ГЕНДЕРА
                        current_y8_conf_factor_for_comparison = y8_gender_pref_conf_factor
                    
                    elif OIV7_GENERIC_PERSON_LABEL_NORM and \
                         det11.mapped_label_for_logic == COCO_PERSON_LABEL_NORM and \
                         OIV7_GENERIC_PERSON_LABEL_NORM in HIERARCHY_CHILD_PARENT.get(det8.original_label_raw, []) and \
                         det8.original_label_raw not in OIV7_GENDER_LABELS_NORM:
                        can_refine = True
                        specifity_score_bonus = 0.25 

                    elif det8.mapped_label_for_logic == det11.mapped_label_for_logic and \
                         det8.original_label_raw != det11.original_label_raw:
                        if not (det11.mapped_label_for_logic == COCO_PERSON_LABEL_NORM and \
                                det8.original_label_raw == OIV7_GENERIC_PERSON_LABEL_NORM):
                            can_refine = True
                            specifity_score_bonus = 0.15
                
                if can_refine and det8.confidence >= det11.confidence * current_y8_conf_factor_for_comparison:
                    containment_score_val = containment_of_y8_in_y11 * w_contain if containment_of_y8_in_y11 > y8_in_y11_contain_thresh else 0
                    candidate_total_score = (iou * w_iou) + \
                                           (det8.confidence * w_conf) + \
                                           (specifity_score_bonus * w_spec) + \
                                           containment_score_val
                    
                    if candidate_total_score > highest_refinement_score:
                        highest_refinement_score = candidate_total_score
                        best_y8_refiner = det8
        
        if best_y8_refiner:
            logger.info(f"УТОЧНЕНИЕ: Y11 '{det11.original_label_raw}' ({det11.unique_id[:4]}) -> Y8 '{best_y8_refiner.original_label_raw}' ({best_y8_refiner.unique_id[:4]}) с очками {highest_refinement_score:.2f}")
            det11.original_label_raw = best_y8_refiner.original_label_raw
            det11.mapped_label_for_logic = best_y8_refiner.mapped_label_for_logic 
            det11.confidence = max(det11.confidence, best_y8_refiner.confidence) 
            det11.model_source = "YOLOv11+Y8" 
            reason_text = f"Уточнено Y8: {best_y8_refiner.original_label_raw} (conf {best_y8_refiner.confidence:.2f})"
            det11.reason = (det11.reason + "; " + reason_text) if det11.reason else reason_text
            used_y8_ids.add(best_y8_refiner.unique_id)
            
    return refined_y11_list, used_y8_ids

def _identify_and_mark_parts(
    detections: List[DetectionResult],
    iou_part_containment_thresh: float 
) -> List[DetectionResult]:
    eligible_detections = [d for d in detections if d.model_source in ["YOLOv8", "YOLOv11+Y8"]]

    for i in range(len(eligible_detections)):
        potential_whole_obj = eligible_detections[i]
        if potential_whole_obj.is_subordinate: continue 

        whole_label_norm = potential_whole_obj.original_label_raw 
        possible_parts_for_whole = HIERARCHY_OBJECT_PARTS.get(whole_label_norm, [])
        if not possible_parts_for_whole: continue 

        for j in range(len(eligible_detections)):
            if i == j: continue
            potential_part_obj = eligible_detections[j]
            if potential_part_obj.is_subordinate: continue 

            part_label_norm = potential_part_obj.original_label_raw 

            if part_label_norm in possible_parts_for_whole: 
                iou, inter_area = calculate_iou(potential_whole_obj.box_xywh, potential_part_obj.box_xywh)
                part_area = potential_part_obj.area
                containment_of_part_in_whole = (inter_area / part_area) if part_area > 0 else 0.0

                if containment_of_part_in_whole > iou_part_containment_thresh:
                    for det_in_main_list in detections:
                        if det_in_main_list.unique_id == potential_part_obj.unique_id:
                            if not det_in_main_list.is_subordinate: 
                                det_in_main_list.is_subordinate = True
                                reason_text = f"Часть '{part_label_norm}' от '{whole_label_norm}'"
                                det_in_main_list.reason = (det_in_main_list.reason + "; " + reason_text) if det_in_main_list.reason else reason_text
                                logger.debug(f"'{part_label_norm}' ({potential_part_obj.unique_id[:4]}) помечена как часть '{whole_label_norm}' ({potential_whole_obj.unique_id[:4]})")
                            break 
    return detections


def _apply_nms_rules_between_two_objects(
    obj_P: DetectionResult, 
    obj_Q: DetectionResult, 
    iou: float,
    suppressed_q_ids: Set[str], 
    config_thresh: Dict[str, float] 
) -> bool: 
    if obj_Q.unique_id in suppressed_q_ids or obj_Q.is_subordinate: 
        return False

    if obj_P.original_label_raw == obj_Q.original_label_raw and \
       iou > config_thresh["iou_thresh_duplicate_same_original_label"]:
        obj_Q.reason = (obj_Q.reason or "") + f"; Дубль '{obj_P.original_label_raw}' (IoU {iou:.2f} с P)"
        suppressed_q_ids.add(obj_Q.unique_id)
        return False

    if obj_P.model_source == "YOLOv11+Y8" and obj_Q.model_source != "YOLOv11+Y8":
        if (obj_P.mapped_label_for_logic == obj_Q.mapped_label_for_logic or \
            obj_P.original_label_raw == obj_Q.original_label_raw) and \
           iou > config_thresh["merge_iou_threshold_same_class"]:
            obj_Q.reason = (obj_Q.reason or "") + f"; Подавлен уточненным P='{obj_P.original_label_raw}' (IoU {iou:.2f})"
            suppressed_q_ids.add(obj_Q.unique_id)
            return False

    is_P_gender = COCO_PERSON_LABEL_NORM and OIV7_GENDER_LABELS_NORM and \
                  obj_P.mapped_label_for_logic == COCO_PERSON_LABEL_NORM and \
                  obj_P.original_label_raw in OIV7_GENDER_LABELS_NORM
    is_Q_generic_person = COCO_PERSON_LABEL_NORM and \
                          obj_Q.mapped_label_for_logic == COCO_PERSON_LABEL_NORM and \
                          (not OIV7_GENDER_LABELS_NORM or obj_Q.original_label_raw not in OIV7_GENDER_LABELS_NORM)

    if is_P_gender and is_Q_generic_person and iou > config_thresh["gender_refinement_iou_threshold"]:
        obj_Q.reason = (obj_Q.reason or "") + f"; Подавлен гендерным P='{obj_P.original_label_raw}' (IoU {iou:.2f})"
        suppressed_q_ids.add(obj_Q.unique_id)
        return False

    if obj_P.mapped_label_for_logic == obj_Q.mapped_label_for_logic and \
       obj_P.original_label_raw != obj_Q.original_label_raw and \
       iou > config_thresh["merge_iou_threshold_same_class"]:
        obj_Q.reason = (obj_Q.reason or "") + f"; NMS в классе '{obj_P.mapped_label_for_logic}' (IoU {iou:.2f} с P)"
        suppressed_q_ids.add(obj_Q.unique_id)
        return False

    is_P_oiv7_source = obj_P.model_source == "YOLOv8" or \
                      (obj_P.model_source == "YOLOv11+Y8" and obj_P.original_label_raw in OIV7_DISPLAY_NAME_TO_LABEL_NAME)
    is_Q_oiv7_source = obj_Q.model_source == "YOLOv8" or \
                      (obj_Q.model_source == "YOLOv11+Y8" and obj_Q.original_label_raw in OIV7_DISPLAY_NAME_TO_LABEL_NAME)

    if is_P_oiv7_source and is_Q_oiv7_source:
        p_oiv7_norm = obj_P.original_label_raw
        q_oiv7_norm = obj_Q.original_label_raw

        if q_oiv7_norm in HIERARCHY_PARENT_CHILD.get(p_oiv7_norm, []):
            if obj_Q.confidence > obj_P.confidence * config_thresh["hierarchy_spec_preference_conf_factor"] and \
               iou > config_thresh["merge_iou_threshold_same_class"]:
                obj_P.reason = (obj_P.reason or "") + f"; Подавлен более специфичным и уверенным Q='{q_oiv7_norm}' (IoU {iou:.2f})"
                return True 
            elif iou > config_thresh["merge_almost_full_containment_threshold"] and \
                 not (q_oiv7_norm in HIERARCHY_OBJECT_PARTS.get(p_oiv7_norm, [])): 
                if not obj_Q.is_subordinate: 
                    obj_Q.is_subordinate = True
                    obj_Q.reason = (obj_Q.reason or "") + f"; Подкатегория '{q_oiv7_norm}' в '{p_oiv7_norm}' (IoU {iou:.2f})"
            return False 

        elif p_oiv7_norm in HIERARCHY_PARENT_CHILD.get(q_oiv7_norm, []):
            if iou > config_thresh["merge_iou_threshold_same_class"]: 
                obj_Q.reason = (obj_Q.reason or "") + f"; Подавлен более специфичным P='{p_oiv7_norm}' (IoU {iou:.2f})"
                suppressed_q_ids.add(obj_Q.unique_id)
            return False

    if obj_P.mapped_label_for_logic != obj_Q.mapped_label_for_logic:
        iou_threshold_for_diff_class = config_thresh["merge_diff_class_strong_overlap_iou_threshold"] \
                                       if iou > 0.75 else config_thresh["merge_iou_threshold_different_class"]
        if iou > iou_threshold_for_diff_class:
            obj_Q.reason = (obj_Q.reason or "") + f"; Межклассовый NMS с '{obj_P.mapped_label_for_logic}' (IoU {iou:.2f})"
            suppressed_q_ids.add(obj_Q.unique_id)
        return False
        
    return False 

def _run_main_nms_loop(
    detections: List[DetectionResult],
    config_thresh: Dict[str, float]
) -> List[DetectionResult]:
    detections.sort(key=lambda d: (
        -d.confidence,
        0 if d.model_source == "YOLOv11+Y8" else (1 if d.model_source == "YOLOv11" else 2), 
        -d.area
    ))

    kept_detections: List[DetectionResult] = []
    suppressed_ids_this_pass: Set[str] = set() 

    idx = 0
    while idx < len(detections):
        current_P = detections[idx]
        if current_P.unique_id in suppressed_ids_this_pass:
            idx += 1
            continue

        is_P_already_in_kept = any(k.unique_id == current_P.unique_id for k in kept_detections)
        if not is_P_already_in_kept:
             kept_detections.append(current_P)
        
        p_was_suppressed_or_replaced = False
        j = idx + 1
        while j < len(detections):
            other_Q = detections[j]
            if other_Q.unique_id == current_P.unique_id or other_Q.unique_id in suppressed_ids_this_pass :
                j += 1
                continue

            iou_val, _ = calculate_iou(current_P.box_xywh, other_Q.box_xywh)
            
            p_suppressed_by_q = _apply_nms_rules_between_two_objects(
                current_P, other_Q, iou_val, suppressed_ids_this_pass, config_thresh
            )

            if p_suppressed_by_q:
                kept_detections = [k for k in kept_detections if k.unique_id != current_P.unique_id]
                detections[idx] = other_Q 
                suppressed_ids_this_pass.add(current_P.unique_id) 
                current_P = other_Q 
                
                if not any(k.unique_id == current_P.unique_id for k in kept_detections):
                     kept_detections.append(current_P)
                
                p_was_suppressed_or_replaced = True
                break 
            
            if other_Q.is_subordinate and other_Q.unique_id not in suppressed_ids_this_pass and \
               not any(k.unique_id == other_Q.unique_id for k in kept_detections):
                kept_detections.append(other_Q)
            j += 1
        
        if p_was_suppressed_or_replaced:
            continue 
        idx += 1
    
    final_list_after_nms = [
        det for det in kept_detections
        if not (det.unique_id in suppressed_ids_this_pass and not det.is_subordinate)
    ]
    
    unique_output_list: List[DetectionResult] = []
    seen_ids_for_final_output: Set[str] = set()
    for det in final_list_after_nms:
        if det.unique_id not in seen_ids_for_final_output:
            unique_output_list.append(det)
            seen_ids_for_final_output.add(det.unique_id)
            
    return unique_output_list


def _nms_for_grouped_subordinates(
    subordinate_group: List[DetectionResult],
    iou_thresh_sub_duplicate: float
) -> List[DetectionResult]:
    if not subordinate_group: return []
    
    subordinate_group.sort(key=lambda d: -d.confidence) 
    kept_subs_in_this_group: List[DetectionResult] = []
    
    for sub_P in subordinate_group:
        is_duplicate_of_kept = False
        for sub_Q_kept in kept_subs_in_this_group:
            iou_sub, _ = calculate_iou(sub_P.box_xywh, sub_Q_kept.box_xywh)
            if iou_sub > iou_thresh_sub_duplicate:
                is_duplicate_of_kept = True
                break
        if not is_duplicate_of_kept:
            kept_subs_in_this_group.append(sub_P)
            
    return kept_subs_in_this_group


def _finalize_and_numerate_results(
    detections_after_main_nms: List[DetectionResult]
) -> List[DetectionResult]:
    
    iou_thresh_sub_duplicate_cfg = CONFIG.get("iou_thresh_duplicate_subordinate", 0.75)
    containment_for_parent_search_cfg = CONFIG.get("merge_almost_full_containment_threshold", 0.90)

    main_objects = [d for d in detections_after_main_nms if not d.is_subordinate]
    subordinate_objects = [d for d in detections_after_main_nms if d.is_subordinate]

    grouped_subs_by_label: Dict[str, List[DetectionResult]] = {}
    for sub_obj in subordinate_objects:
        grouped_subs_by_label.setdefault(sub_obj.original_label_raw, []).append(sub_obj)

    final_kept_subordinates: List[DetectionResult] = []
    for _, group in grouped_subs_by_label.items():
        final_kept_subordinates.extend(
            _nms_for_grouped_subordinates(group, iou_thresh_sub_duplicate_cfg)
        )

    all_results_for_numeration = main_objects + final_kept_subordinates
    
    all_results_for_numeration.sort(key=lambda d: (
        d.mapped_label_for_logic,
        d.original_label_raw,
        d.is_subordinate, 
        -d.confidence,
        -d.area 
    ))

    instance_counters: Dict[str, int] = {} 
    final_output_list_with_labels: List[DetectionResult] = []

    for det_obj in all_results_for_numeration:
        base_display_name = det_obj.original_label_raw 
        
        if det_obj.mapped_label_for_logic and \
           det_obj.mapped_label_for_logic != det_obj.original_label_raw:
            base_display_name = f"{det_obj.original_label_raw} ({det_obj.mapped_label_for_logic})"
        
        counter_key_for_numeration = base_display_name
        if det_obj.is_subordinate: 
            counter_key_for_numeration += " [часть]"

        instance_counters[counter_key_for_numeration] = instance_counters.get(counter_key_for_numeration, 0) + 1
        instance_num = instance_counters[counter_key_for_numeration]
        
        det_obj.final_display_label = f"{base_display_name} {instance_num}"
        if det_obj.is_subordinate:
            det_obj.final_display_label += " [часть]"

            if not det_obj.reason or "Часть от" not in det_obj.reason: 
                best_parent: Optional[DetectionResult] = None
                best_parent_score = 0.0 

                possible_whole_labels = HIERARCHY_PART_IS_PART_OF.get(det_obj.original_label_raw, [])
                for main_obj_candidate in main_objects: 
                    if main_obj_candidate.original_label_raw in possible_whole_labels:
                        iou_with_main, inter_area = calculate_iou(det_obj.box_xywh, main_obj_candidate.box_xywh)
                        containment_of_part = (inter_area / det_obj.area) if det_obj.area > 0 else 0.0
                        current_score = max(iou_with_main, containment_of_part) 
                        if current_score > best_parent_score and current_score > 0.1: 
                            best_parent_score = current_score
                            best_parent = main_obj_candidate
                
                if not best_parent and det_obj.area > 0:
                    for main_obj_candidate in main_objects:
                        if main_obj_candidate.area < det_obj.area * 1.2: continue 
                        
                        _, inter_area = calculate_iou(det_obj.box_xywh, main_obj_candidate.box_xywh)
                        containment_ratio = inter_area / det_obj.area
                        if containment_ratio > containment_for_parent_search_cfg and containment_ratio > best_parent_score:
                            best_parent_score = containment_ratio
                            best_parent = main_obj_candidate
                
                if best_parent:
                    parent_info = f"'{best_parent.final_display_label or best_parent.original_label_raw}' ({best_parent.unique_id[:4]})"
                    new_reason = f"Часть от {parent_info} (связь {best_parent_score:.2f})"
                    det_obj.reason = (det_obj.reason + "; " + new_reason) if det_obj.reason else new_reason
                elif not det_obj.reason : 
                    det_obj.reason = "Подчиненный объект (родитель не определен)"
        
        final_output_list_with_labels.append(det_obj)
        
    return final_output_list_with_labels


def merge_results(raw_detections_input: List[DetectionResult]) -> List[DetectionResult]:
    logger.info(f"Запуск объединения для {len(raw_detections_input)} сырых детекций.")
    if not raw_detections_input:
        return []

    all_prepared_dets = _prepare_detections_for_merging(raw_detections_input)

    y11_for_refinement = sorted(
        [d for d in all_prepared_dets if d.model_source == "YOLOv11"],
        key=lambda d: -d.confidence
    )
    y8_for_refinement = sorted(
        [d for d in all_prepared_dets if d.model_source == "YOLOv8"],
        key=lambda d: -d.confidence
    )

    refined_y11_list, used_y8_ids_in_refinement = _refine_y11_with_y8(
        y11_for_refinement, y8_for_refinement
    )
    logger.info(f"После уточнения: {len(refined_y11_list)} Y11-based. Использовано {len(used_y8_ids_in_refinement)} Y8 для уточнения.")

    remaining_y8_list = [d8 for d8 in y8_for_refinement if d8.unique_id not in used_y8_ids_in_refinement]
    detections_for_main_nms = refined_y11_list + remaining_y8_list
    logger.info(f"Кандидатов для основного NMS: {len(detections_for_main_nms)}")

    iou_part_contain_thresh_cfg = CONFIG.get("iou_thresh_part_whole_containment", 0.85)
    detections_with_parts_marked = _identify_and_mark_parts(detections_for_main_nms, iou_part_contain_thresh_cfg)
    
    nms_config_thresholds = {
        "iou_thresh_duplicate_same_original_label": CONFIG.get("iou_thresh_duplicate_same_original_label", 0.85),
        "merge_iou_threshold_same_class": CONFIG.get("merge_iou_threshold_same_class", 0.60),
        "merge_diff_class_strong_overlap_iou_threshold": CONFIG.get("merge_diff_class_strong_overlap_iou_threshold", 0.50),
        "merge_iou_threshold_different_class": CONFIG.get("merge_iou_threshold_different_class", 0.70),
        "merge_almost_full_containment_threshold": CONFIG.get("merge_almost_full_containment_threshold", 0.90),
        "hierarchy_spec_preference_conf_factor": CONFIG.get("hierarchy_spec_preference_conf_factor", 0.95),
        "gender_refinement_iou_threshold": CONFIG.get("gender_refinement_iou_threshold", 0.60),
    }
    final_candidates_after_nms = _run_main_nms_loop(detections_with_parts_marked, nms_config_thresholds)
    logger.info(f"После основного NMS: {len(final_candidates_after_nms)} кандидатов.")

    final_results_with_labels = _finalize_and_numerate_results(final_candidates_after_nms)

    logger.info(f"Объединение завершено. Итоговых объектов: {len(final_results_with_labels)}")
    return final_results_with_labels