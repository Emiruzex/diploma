# config.py
CONFIG = {
    "app_title": "Object Detection (YOLOv8x-OIV7 + YOLOv11x)",
    "theme": "flatly",
    "window_size": "1200x800",
    "max_display_size": 500,
    "min_memory_mb": 500,
    "max_image_size": 1000,

    "gamma_value": 0.8,
    "blur_size": 7,
    "clahe_path_clip_limit": 2.0, # Используется в process_clahe, если оно будет снова активно

    "confidence_threshold": 0.2, # Общий порог для αρχinitial детекций
    "min_area": 300,
    "model_name": "yolo11x.pt",
    "oiv7_model_name": "yolov8x-oiv7.pt",
    "hierarchy_file": "hierarchy.json",

    "use_gpu": True,
    "memory_pause_threshold_mb": 200,

    # Пороги для _refine_y11_with_y8
    "refine_iou_threshold_strong": 0.55,
    "refine_iou_threshold_weak_containment": 0.20, # Если IoU слабое, но содержание хорошее
    "refine_containment_y8_in_y11_threshold": 0.75, # % площади y8 внутри y11 для связи
    "refine_min_y8_confidence": 0.15, # Минимальная уверенность y8 для рассмотрения как уточнителя
    "refine_y8_preference_conf_factor": 0.80, # Насколько y8 может быть менее уверенным, чем y11, чтобы все еще уточнить (общий случай)
    "refine_y8_gender_preference_conf_factor": 0.60, # Более мягкий порог для гендерного уточнения
    "refine_score_weight_iou": 0.4,
    "refine_score_weight_conf": 0.3,
    "refine_score_weight_specifity": 0.2,
    "refine_score_weight_containment": 0.1,

    # Пороги для _apply_nms_rules и основного NMS
    "iou_thresh_duplicate_same_original_label": 0.85, # Для подавления точных дубликатов
    "merge_iou_threshold_same_class": 0.60, # NMS внутри одного mapped_label (более агрессивный)
    "merge_diff_class_strong_overlap_iou_threshold": 0.50, # NMS для разных классов при сильном перекрытии
    "merge_iou_threshold_different_class": 0.70, # NMS для разных классов при умеренном перекрытии
    "merge_almost_full_containment_threshold": 0.90, # Для определения, что один объект почти полностью содержит другой (для is_subordinate)
    "hierarchy_spec_preference_conf_factor": 0.95, # Насколько специфичный OIV7 должен быть увереннее общего для замены
    "iou_thresh_part_whole_containment": 0.85, # Для маркировки частей (_identify_and_mark_parts)
    "gender_refinement_iou_threshold": 0.60, # IoU для правила "Человек" vs "М/Ж" в NMS (используется в _apply_nms_rules для остаточных конфликтов)
    "gender_preference_conf_factor": 0.75,   # Насколько М/Ж должен быть уверен по сравнению с "Человек" в NMS (используется в _apply_nms_rules)

    # Пороги для _finalize_and_numerate_results
    "iou_thresh_duplicate_subordinate": 0.75, # NMS для дублирующихся подчиненных объектов

    # Общие настройки для отображения и фильтра
    "final_confidence_threshold_high": 0.4, # Для авто-выбора чекбоксов в фильтре (логика А)
    "final_confidence_threshold_low_fallback": 0.2, # Для авто-выбора в "пустых" областях (логика Б)

    "temp_dir": "temp_images",
}