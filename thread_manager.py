# thread_manager.py
import threading
import logging
import psutil
import cv2
import os
import time
import torch
from typing import List, Dict, Any, Optional, Tuple
import tkinter as tk # Используется для after
import numpy as np

from image_processing import preprocess_image
from neural_analysis import (
    классификация_объектов,
    ensure_models_loaded,
    CLASS_MAPPING # Для _display_intermediate_results и merge_results
    # Иерархия и другие словари импортируются в utils.py, который вызывается для merge_results
)
from utils import draw_results, merge_results # merge_results - основной обработчик
from detection_models import DetectionResult
from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingState:
    """Хранит состояние и промежуточные результаты конвейера обработки."""
    def __init__(self):
        self.raw_y11_orig: Optional[List[DetectionResult]] = None
        self.raw_y8_orig: Optional[List[DetectionResult]] = None
        self.processed_image: Optional[np.ndarray] = None
        self.raw_y11_proc: Optional[List[DetectionResult]] = None
        self.raw_y8_proc: Optional[List[DetectionResult]] = None
        
        self.merged_y11: Optional[List[DetectionResult]] = None
        self.merged_y8: Optional[List[DetectionResult]] = None
        self.final_merged: Optional[List[DetectionResult]] = None
        
        self.error_message: Optional[str] = None
        self.current_progress: int = 0

class ThreadManager:
    def __init__(self, app_instance: Any): # app_instance это ObjectDetectionApp
        self.app = app_instance
        self.active_pipeline_thread: Optional[threading.Thread] = None
        # self.temp_dir = CONFIG.get("temp_dir", "temp_images") # Не используется напрямую в ThreadManager после удаления сохранения слоев
        # os.makedirs(self.temp_dir, exist_ok=True)

    def _update_gui_progress(self, increment: int = 0, absolute: Optional[int] = None, status_text: Optional[str] = None):
        """Потокобезопасное обновление GUI."""
        def update():
            if hasattr(self, 'processing_state'): # Убедимся, что state уже есть
                if absolute is not None:
                    current_val = absolute
                elif increment > 0:
                    current_val = self.processing_state.current_progress + increment
                else: # Если increment == 0 и absolute is None, берем текущее значение
                    current_val = self.processing_state.current_progress

                self.app.gui.progress['value'] = current_val
                self.processing_state.current_progress = current_val
            
            if status_text:
                self.app.gui.status_label.config(text=status_text)
            self.app.window.update_idletasks()

        self.app.window.after(0, update)


    def check_memory(self, required_mb: Optional[float] = None, task_name: str = "операции") -> bool:
        required_mb_val = required_mb if required_mb is not None else CONFIG.get("memory_pause_threshold_mb", 200)
        available_memory = psutil.virtual_memory().available / (1024 ** 2)
        if available_memory < required_mb_val:
            warning_text = f"Низкий уровень памяти: {available_memory:.2f} MB (требуется ~{required_mb_val}MB для '{task_name}'), пауза..."
            logger.warning(warning_text)
            self._update_gui_progress(status_text=warning_text)
            time.sleep(2)
            available_memory = psutil.virtual_memory().available / (1024 ** 2)
            if available_memory < required_mb_val:
                error_text = f"Памяти все еще недостаточно ({available_memory:.2f} MB) для '{task_name}'."
                logger.error(error_text)
                if hasattr(self, 'processing_state'):
                    self.processing_state.error_message = error_text
                return False
        return True

    def _task_ensure_models_loaded(self, state: ProcessingState) -> bool:
        self._update_gui_progress(absolute=0, status_text="Загрузка моделей...")
        try:
            ensure_models_loaded()
            self._update_gui_progress(absolute=5, status_text="Модели загружены.")
            return True
        except Exception as e_load_model:
            logger.error(f"Ошибка загрузки моделей: {e_load_model}", exc_info=True)
            state.error_message = f"Ошибка загрузки моделей: {str(e_load_model)}"
            return False

    def _task_detect_y11_orig(self, image: np.ndarray, state: ProcessingState):
        if state.error_message: return
        logger.info("Детекция YOLOv11 на оригинале...")
        self._update_gui_progress(status_text="YOLOv11 (оригинал): детекция...")
        if not self.check_memory(300, "YOLOv11 (оригинал)"): return

        try:
            state.raw_y11_orig = классификация_объектов(image, model_source_filter="YOLOv11")
            count = len(state.raw_y11_orig or [])
            logger.info(f"YOLOv11 (оригинал) найдено: {count}")
            self._update_gui_progress(increment=15, status_text=f"YOLOv11 (оригинал): {count} объектов.")
            
            if state.raw_y11_orig:
                 self._display_intermediate_results(
                    image.copy(), 
                    state.raw_y11_orig, 
                    f"YOLOv11 (на оригинале): {count} объектов",
                    is_temp_labeling_needed=True
                )
        except Exception as e:
            logger.error(f"Ошибка YOLOv11 (оригинал): {e}", exc_info=True)
            state.error_message = f"YOLOv11 (оригинал): {str(e)}"
        finally:
            if CONFIG.get("use_gpu",True) and torch.cuda.is_available(): torch.cuda.empty_cache()
            import gc; gc.collect()

    def _task_detect_y8_orig(self, image: np.ndarray, state: ProcessingState):
        if state.error_message: return
        logger.info("Детекция YOLOv8 на оригинале...")
        self._update_gui_progress(status_text="YOLOv8 (оригинал): детекция...")
        if not self.check_memory(300, "YOLOv8 (оригинал)"): return

        try:
            state.raw_y8_orig = классификация_объектов(image, model_source_filter="YOLOv8")
            count = len(state.raw_y8_orig or [])
            logger.info(f"YOLOv8 (оригинал) найдено: {count}")
            self._update_gui_progress(increment=15, status_text=f"YOLOv8 (оригинал): {count} объектов.")
        except Exception as e:
            logger.error(f"Ошибка YOLOv8 (оригинал): {e}", exc_info=True)
            state.error_message = f"YOLOv8 (оригинал): {str(e)}"
        finally:
            if CONFIG.get("use_gpu",True) and torch.cuda.is_available(): torch.cuda.empty_cache()
            import gc; gc.collect()

    def _task_preprocess(self, image: np.ndarray, state: ProcessingState):
        if state.error_message: return
        logger.info("Предобработка изображения...")
        self._update_gui_progress(status_text="Предобработка изображения...")
        if not self.check_memory(400, "Предобработка"): return

        try:
            processed_img_result = preprocess_image(image.copy())
            if processed_img_result is not None:
                state.processed_image = processed_img_result
                self.app.processed_image = state.processed_image # Обновляем в app
                logger.info("Предобработка завершена.")
                self._update_gui_progress(increment=20, status_text="Предобработка завершена.")
                if state.raw_y11_orig and self.app.processed_image is not None:
                    self._display_intermediate_results(
                        self.app.processed_image.copy(),
                        state.raw_y11_orig,
                        f"YOLOv11 (с оригинала) на обработанном: {len(state.raw_y11_orig)}",
                        is_temp_labeling_needed=True
                    )
            else:
                logger.warning("preprocess_image вернула None. Используется копия оригинала как обработанное.")
                state.processed_image = image.copy() 
                self.app.processed_image = state.processed_image
                # Не считаем это фатальной ошибкой для всего процесса, но сообщаем
                self._update_gui_progress(increment=20, status_text="Ошибка предобработки, используется оригинал.")
        except Exception as e:
            logger.error(f"Ошибка предобработки: {e}", exc_info=True)
            state.processed_image = image.copy() # Фоллбэк
            self.app.processed_image = state.processed_image
            state.error_message = f"Предобработка: {str(e)}" # Сообщаем об ошибке, но пытаемся продолжить
        finally:
            import gc; gc.collect()

    def _task_detect_y11_proc_and_merge(self, image_proc: Optional[np.ndarray], state: ProcessingState):
        if state.error_message and not image_proc: return # Если уже была ошибка И нет обработанного изображения
        if image_proc is None:
            logger.error("YOLOv11 (обработка): обработанное изображение отсутствует.")
            # Не устанавливаем error_message здесь, если уже есть от предобработки
            # или если это единственная ошибка, то пусть будет.
            if not state.error_message: state.error_message = "YOLOv11 (обработка): нет изображения."
            return
            
        logger.info("Детекция YOLOv11 на обработанном и слияние Y11...")
        self._update_gui_progress(status_text="YOLOv11 (обработка): детекция...")
        if not self.check_memory(300, "YOLOv11 (обработка)"): return

        try:
            state.raw_y11_proc = классификация_объектов(image_proc, model_source_filter="YOLOv11")
            count_proc = len(state.raw_y11_proc or [])
            logger.info(f"YOLOv11 (обработка) найдено: {count_proc}")
            
            self._update_gui_progress(status_text="YOLOv11: слияние результатов...")
            y11_to_merge = (state.raw_y11_orig or []) + (state.raw_y11_proc or [])
            
            if y11_to_merge:
                state.merged_y11 = merge_results(y11_to_merge)
                merged_count = len(state.merged_y11 or [])
                logger.info(f"Слияние YOLOv11 завершено, итого: {merged_count}")
                self.app.merged_yolo11_results = state.merged_y11
                if self.app.display_mode == "YOLOv11": # Обновляем GUI, если этот режим активен
                    self.app.switch_active_results_for_filter_and_refresh("YOLOv11")
            else:
                state.merged_y11 = [] # Гарантируем, что это список
                logger.info("Нет результатов YOLOv11 для слияния.")

            self._update_gui_progress(increment=20, status_text="YOLOv11: слияние завершено.")

        except Exception as e:
            logger.error(f"Ошибка YOLOv11 (обработка/слияние): {e}", exc_info=True)
            state.error_message = state.error_message or f"YOLOv11 (обработка/слияние): {str(e)}"
        finally:
            if CONFIG.get("use_gpu",True) and torch.cuda.is_available(): torch.cuda.empty_cache()
            import gc; gc.collect()

    def _task_detect_y8_proc_and_merge(self, image_proc: Optional[np.ndarray], state: ProcessingState):
        if state.error_message and not image_proc: return
        if image_proc is None:
            logger.error("YOLOv8 (обработка): обработанное изображение отсутствует.")
            if not state.error_message: state.error_message = "YOLOv8 (обработка): нет изображения."
            return

        logger.info("Детекция YOLOv8 на обработанном и слияние Y8...")
        self._update_gui_progress(status_text="YOLOv8 (обработка): детекция...")
        if not self.check_memory(300, "YOLOv8 (обработка)"): return
        try:
            state.raw_y8_proc = классификация_объектов(image_proc, model_source_filter="YOLOv8")
            count_proc = len(state.raw_y8_proc or [])
            logger.info(f"YOLOv8 (обработка) найдено: {count_proc}")

            self._update_gui_progress(status_text="YOLOv8: слияние результатов...")
            y8_to_merge = (state.raw_y8_orig or []) + (state.raw_y8_proc or [])

            if y8_to_merge:
                state.merged_y8 = merge_results(y8_to_merge)
                merged_count = len(state.merged_y8 or [])
                logger.info(f"Слияние YOLOv8 завершено, итого: {merged_count}")
                self.app.merged_yolo8_results = state.merged_y8
                if self.app.display_mode == "YOLOv8":
                     self.app.switch_active_results_for_filter_and_refresh("YOLOv8")
            else:
                state.merged_y8 = []
                logger.info("Нет результатов YOLOv8 для слияния.")

            self._update_gui_progress(increment=20, status_text="YOLOv8: слияние завершено.")
        except Exception as e:
            logger.error(f"Ошибка YOLOv8 (обработка/слияние): {e}", exc_info=True)
            state.error_message = state.error_message or f"YOLOv8 (обработка/слияние): {str(e)}"
        finally:
            if CONFIG.get("use_gpu",True) and torch.cuda.is_available(): torch.cuda.empty_cache()
            import gc; gc.collect()

    def _task_final_merge(self, state: ProcessingState):
        if state.error_message and not (state.merged_y11 or state.merged_y8): # Пропускаем, если уже есть ошибка И нет данных для слияния
            logger.warning("Финальное слияние пропущено из-за предыдущих ошибок и отсутствия данных.")
            state.final_merged = [] # Убедимся, что это пустой список
            return
            
        logger.info("Финальное слияние YOLOv11 и YOLOv8...")
        self._update_gui_progress(status_text="Финальное слияние всех результатов...")
        
        all_to_merge_finally = (state.merged_y11 or []) + (state.merged_y8 or [])

        if not all_to_merge_finally:
            logger.warning("Нет результатов для финального слияния.")
            state.final_merged = []
            self._update_gui_progress(increment=5, status_text="Нет данных для финального слияния.")
            return

        try:
            state.final_merged = merge_results(all_to_merge_finally)
            merged_count = len(state.final_merged or [])
            logger.info(f"Финальное слияние завершено, итого: {merged_count}")
            self.app.final_merged_results = state.final_merged
            if self.app.display_mode == "Слияние": # Это основной режим по умолчанию
                 self.app.switch_active_results_for_filter_and_refresh("Слияние")

            self._update_gui_progress(increment=5, status_text="Финальное слияние завершено.")
        except Exception as e:
            logger.error(f"Ошибка финального слияния: {e}", exc_info=True)
            state.error_message = state.error_message or f"Финальное слияние: {str(e)}"
            # Даже если здесь ошибка, предыдущие merged_y11/y8 могут быть полезны
            if state.final_merged is None: state.final_merged = []


    def _display_intermediate_results(self, base_img_to_draw_on: np.ndarray, 
                                      detections: List[DetectionResult], 
                                      status_text: str,
                                      is_temp_labeling_needed: bool = False):
        """Отображает промежуточные результаты (вызывается из GUI потока через after)."""
        
        def display_task():
            if base_img_to_draw_on is None:
                logger.warning(f"Пропуск отображения '{status_text}': нет базового изображения.")
                return
            
            # Копируем detections, если нужна временная метка, чтобы не изменять state
            detections_to_draw = [DetectionResult(**det.__dict__) for det in detections] if is_temp_labeling_needed else detections

            if is_temp_labeling_needed:
                temp_counters = {}
                for det in sorted(detections_to_draw, key=lambda d: (d.original_label_raw, -d.confidence)):
                    norm_orig_label = det.original_label_raw.replace("_", " ").strip()
                    
                    # Установка mapped_label_for_logic для корректного цвета
                    det.mapped_label_for_logic = CLASS_MAPPING.get(norm_orig_label, norm_orig_label)
                    
                    temp_counters[norm_orig_label] = temp_counters.get(norm_orig_label, 0) + 1
                    instance_num = temp_counters[norm_orig_label]
                    
                    base_display = norm_orig_label
                    # Отображаем маппинг, если он есть и отличается
                    if det.mapped_label_for_logic and det.mapped_label_for_logic != norm_orig_label:
                         base_display = f"{norm_orig_label} ({det.mapped_label_for_logic})"
                    det.final_display_label = f"{base_display} {instance_num}"
            
            if not detections_to_draw: # Если detections пуст или стал пуст после фильтрации (хотя здесь нет фильтрации)
                logger.info(f"Отображение '{status_text}': нет детекций, показываем базовое изображение.")
                self.app.result_image = base_img_to_draw_on.copy()
            else:
                self.app.result_image = draw_results(base_img_to_draw_on.copy(), detections_to_draw)
            
            self.app.display_image(status_text) # display_image сама обработает, если result_image None
            self.app.gui.status_label.config(text=status_text)

        self.app.window.after(0, display_task)


    def run_processing_pipeline(self, base_image: np.ndarray):
        """Основной конвейер обработки, запускается в отдельном потоке."""
        self.processing_state = ProcessingState()
        state = self.processing_state

        if not self._task_ensure_models_loaded(state):
            self._handle_pipeline_completion(state)
            return

        threads_stage1_2 = [
            threading.Thread(target=self._task_detect_y11_orig, args=(base_image, state), name="Y11Orig"),
            threading.Thread(target=self._task_detect_y8_orig, args=(base_image, state), name="Y8Orig"),
            threading.Thread(target=self._task_preprocess, args=(base_image, state), name="Preprocess")
        ]
        for t in threads_stage1_2: t.start()
        for t in threads_stage1_2: t.join()
        
        # Проверяем критическую ошибку после предобработки
        # state.processed_image будет оригиналом, если предобработка упала, но это не фатально для детекции на нем.
        # Фатально, если state.processed_image остался None. _task_preprocess должен это предотвращать.
        current_processed_image = state.processed_image # Берем из state, т.к. self.app.processed_image мог еще не обновиться
        if current_processed_image is None and state.error_message:
            logger.error("Критическая ошибка: нет изображения для дальнейшей обработки.")
            self._handle_pipeline_completion(state)
            return
        # Если current_processed_image is None, но ошибки нет - это странно, но следующие таски сами проверят.

        threads_stage3 = [
            threading.Thread(target=self._task_detect_y11_proc_and_merge, args=(current_processed_image, state), name="Y11ProcMerge"),
            threading.Thread(target=self._task_detect_y8_proc_and_merge, args=(current_processed_image, state), name="Y8ProcMerge")
        ]
        for t in threads_stage3: t.start()
        for t in threads_stage3: t.join()

        self._task_final_merge(state)
        self._handle_pipeline_completion(state)


    def _handle_pipeline_completion(self, state: ProcessingState):
        logger.info("Завершение конвейера обработки.")
        error_occurred = bool(state.error_message)
        
        self.app.raw_results_yolo11_orig = state.raw_y11_orig
        self.app.raw_results_yolo8_orig = state.raw_y8_orig
        self.app.processed_image = state.processed_image # Уже должно быть установлено, но на всякий случай
        self.app.raw_results_y11_proc = state.raw_y11_proc
        self.app.raw_results_yolo8_proc = state.raw_y8_proc
        self.app.merged_yolo11_results = state.merged_y11
        self.app.merged_yolo8_results = state.merged_y8
        self.app.final_merged_results = state.final_merged

        # Устанавливаем активный набор результатов для фильтра и GUI
        # display_mode уже должен быть актуален в self.app
        self.app.switch_active_results_for_filter_and_refresh(self.app.display_mode, initial_setup=True)

        if error_occurred:
            final_status_text = state.error_message or "Произошла неизвестная ошибка."
            self.app.window.after(0, lambda: self.app.handle_error(final_status_text))
            self._update_gui_progress(status_text=final_status_text) # Устанавливаем текст ошибки в статус
        # Если нет ошибки, статус будет обновлен через self.app.update_selected_objects(),
        # который вызывается из finish_processing.

        # Прогресс должен быть 100, если нет ошибки или есть какие-то результаты.
        # Если ошибка и нет результатов вообще, можно оставить текущий прогресс или сбросить.
        final_progress = 100
        if error_occurred and not (state.final_merged or state.merged_y11 or state.merged_y8):
             final_progress = state.current_progress # Оставляем как есть или можно сбросить, например, на 0

        self._update_gui_progress(absolute=final_progress) # Обновляем только значение прогресс-бара

        self.app.window.after(0, lambda: self.app.finish_processing(error_occurred=error_occurred))
        self.active_pipeline_thread = None
        logger.info("Конвейер обработки завершен, поток сброшен.")


    def start_processing(self, base_image_for_processing: np.ndarray):
        logger.info("Запрос на запуск конвейера обработки.")
        if self.active_pipeline_thread and self.active_pipeline_thread.is_alive():
            logger.warning("Конвейер обработки уже запущен. Новый запуск отменен.")
            self.app.window.after(0, lambda: self.app.gui.status_label.config(text="Обработка уже выполняется..."))
            return

        # Сброс предыдущих состояний и результатов в app
        self.app.is_processing = True # Устанавливаем флаг сразу
        self.app.raw_results_yolo11_orig = None
        self.app.raw_results_yolo8_orig = None
        self.app.processed_image = None
        self.app.raw_results_y11_proc = None
        self.app.raw_results_yolo8_proc = None
        self.app.merged_yolo11_results = None
        self.app.merged_yolo8_results = None
        self.app.final_merged_results = None
        self.app.active_results_for_filter = []
        self.app.object_vars.clear()
        self.app.class_vars.clear()
        if self.app.gui.filter_window and self.app.gui.filter_window.winfo_exists():
            self.app.gui.update_checkboxes("", self.app.gui.filter_window) # Очищаем фильтр
        
        # Обновляем GUI перед стартом потока
        self.app.gui.set_buttons_state(processing=True)
        self._update_gui_progress(absolute=0, status_text="Подготовка к обработке...")


        self.active_pipeline_thread = threading.Thread(
            target=self.run_processing_pipeline, 
            args=(base_image_for_processing,), 
            name="ProcessingPipelineThread",
            daemon=True
        )
        self.active_pipeline_thread.start()

    def cleanup(self):
        logger.info("Очистка менеджера потоков.")
        if self.active_pipeline_thread and self.active_pipeline_thread.is_alive():
            logger.warning(f"Поток {self.active_pipeline_thread.name} активен при cleanup.")
        
        if CONFIG.get("use_gpu",True) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA кэш очищен.")
            except Exception as e_cuda_clean:
                logger.error(f"Ошибка при очистке CUDA кэша: {e_cuda_clean}")
        
        import gc
        gc.collect()
        logger.info("Менеджер потоков очищен.")