# app.py
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import psutil
import os
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import List, Dict, Optional, Set

from thread_manager import ThreadManager
from utils import load_and_resize_image, draw_results, initialize_class_colors, calculate_iou
from config import CONFIG
from gui import ObjectDetectionGUI
# from ttkbootstrap.constants import DISABLED, NORMAL # ttkbootstrap constants used directly via ttk.DISABLED etc.
from detection_models import DetectionResult
from neural_analysis import (
    CLASSES_YOLO11,
    CLASS_MAPPING,
    OIV7_LABEL_NAME_TO_DISPLAY_NAME
    # HIERARCHY_PARENT_CHILD and HIERARCHY_OBJECT_PARTS are used by utils.py
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ObjectDetectionApp:
    def __init__(self, window):
        logger.info("Инициализация приложения ObjectDetectionApp")
        self.window = window
        self.window.title(CONFIG.get("app_title", "Object Detection App"))

        try:
            initialize_class_colors(
                CLASSES_YOLO11,
                CLASS_MAPPING,
                OIV7_LABEL_NAME_TO_DISPLAY_NAME
            )
        except Exception as e_colors:
            logger.error(f"Ошибка инициализации цветов классов: {e_colors}", exc_info=True)

        self.temp_dir = CONFIG.get("temp_dir", "temp_images")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.window.protocol("WM_DELETE_WINDOW", self.cleanup)

        self.image_path: Optional[str] = None
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None # Изображение с нарисованными результатами

        # Сырые и обработанные результаты
        self.raw_results_yolo11_orig: Optional[List[DetectionResult]] = None
        self.raw_results_yolo8_orig: Optional[List[DetectionResult]] = None
        self.raw_results_yolo11_proc: Optional[List[DetectionResult]] = None
        self.raw_results_yolo8_proc: Optional[List[DetectionResult]] = None
        
        # Результаты после индивидуального слияния для каждой модели
        self.merged_yolo11_results: Optional[List[DetectionResult]] = None
        self.merged_yolo8_results: Optional[List[DetectionResult]] = None
        
        # Финальный результат после слияния merged_yolo11 и merged_yolo8
        self.final_merged_results: Optional[List[DetectionResult]] = None

        self.active_results_for_filter: List[DetectionResult] = [] # Текущий набор для фильтра и отображения
        self.object_vars: Dict[str, tk.BooleanVar] = {} # Состояния чекбоксов для индивидуальных объектов
        self.class_vars: Dict[str, tk.BooleanVar] = {}  # Состояния чекбоксов для классов/групп

        self.is_processing = False
        self.show_processed = True # Показывать обработанное изображение (True) или оригинал (False)
        self.display_mode = "Слияние" # "Слияние", "YOLOv11", "YOLOv8"

        self.thread_manager = ThreadManager(self)
        self.last_resize_time = 0
        self.resize_debounce_ms = 100 # Задержка для обработки изменения размера окна
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 3.0

        self.gui = ObjectDetectionGUI(self)
        logger.info("Инициализация ObjectDetectionApp завершена")

    def load_dialog(self, event=None):
        if self.is_processing:
            messagebox.showwarning("Предупреждение", "Обработка уже выполняется. Дождитесь завершения.")
            return
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            self.load_image(file_path)

    def handle_drop(self, event):
        if self.is_processing:
            messagebox.showwarning("Предупреждение", "Обработка уже выполняется. Дождитесь завершения.")
            return
        file_path = event.data
        # Удаляем фигурные скобки, если файл перетаскивается из некоторых файловых менеджеров
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        
        if os.path.isfile(file_path): # Проверяем, что это действительно файл
            self.load_image(file_path)
        else:
            logger.warning(f"Перетащенный элемент не является файлом: {file_path}")
            messagebox.showwarning("Предупреждение", "Перетащенный элемент не является файлом.")


    def load_image(self, file_path: str):
        logger.info(f"Загрузка изображения: {file_path}")
        self.image_path = file_path
        
        available_memory = psutil.virtual_memory().available / (1024 ** 2)
        min_memory_req = CONFIG.get("min_memory_mb", 500)
        if available_memory < min_memory_req:
            messagebox.showerror("Ошибка", f"Недостаточно памяти для обработки (доступно {available_memory:.2f} MB, требуется {min_memory_req} MB)")
            # Не устанавливаем is_processing, т.к. обработка не начнется
            return

        # Состояние GUI обновляется в ThreadManager.start_processing
        # Но базовые сбросы и начальные установки делаем здесь
        self.gui.progress['value'] = 0
        self.gui.status_label.config(text="Загрузка изображения...")
        self.window.config(cursor="wait")
        self.gui.canvas.delete("placeholder")
        self.gui.image_label.config(text="Детекция объектов...")
        self.gui.set_buttons_state(processing=True) # Блокируем кнопки до начала потока

        # Сброс предыдущих результатов
        self.original_image = None
        self.processed_image = None
        self.result_image = None
        self.raw_results_yolo11_orig = None; self.raw_results_yolo8_orig = None
        self.raw_results_yolo11_proc = None; self.raw_results_yolo8_proc = None
        self.merged_yolo11_results = None; self.merged_yolo8_results = None
        self.final_merged_results = None
        self.active_results_for_filter = []
        self.object_vars.clear(); self.class_vars.clear()
        
        if self.gui.filter_window and self.gui.filter_window.winfo_exists():
            self.gui.update_checkboxes("", self.gui.filter_window) # Очистить окно фильтра

        try:
            loaded_image = load_and_resize_image(file_path)
            if loaded_image is None:
                raise ValueError("Не удалось загрузить или изменить размер изображения.")

            self.original_image = loaded_image
            # Отображаем оригинал сразу, пока идет обработка
            self.result_image = self.original_image.copy() 
            self.display_image("Подготовка к обработке...")

            # Запуск основного конвейера обработки в отдельном потоке
            self.thread_manager.start_processing(self.original_image.copy())

        except Exception as e:
            logger.error(f"Ошибка при загрузке или первичной обработке: {str(e)}", exc_info=True)
            # Сообщение об ошибке будет показано, is_processing сбросится в finish_processing
            self.finish_processing(error_occurred=True, error_message=f"Ошибка загрузки: {str(e)}")
        finally:
            import gc; gc.collect()


    def _apply_empty_area_activation_logic(self,
                                           high_conf_thresh: float, 
                                           low_fallback_thresh: float) -> int:
        """
        Активирует объекты с меньшей уверенностью в областях, не покрытых высокоуверенными.
        Работает с self.active_results_for_filter и self.object_vars.
        Возвращает количество дополнительно активированных объектов.
        """
        activated_count = 0
        if not self.active_results_for_filter:
            return activated_count

        # Объекты, уже выбранные Логикой А (высокая уверенность, не подчиненные)
        current_active_boxes = [
            det.box_xywh for det in self.active_results_for_filter
            if self.object_vars.get(det.unique_id) and self.object_vars[det.unique_id].get()
        ]
        
        # Кандидаты для Логики Б: не выбраны, уверенность >= low_fallback, не подчиненные
        candidates_for_logic_b = sorted(
            [det for det in self.active_results_for_filter
             if (not self.object_vars.get(det.unique_id) or not self.object_vars[det.unique_id].get()) and \
                det.confidence >= low_fallback_thresh and not det.is_subordinate],
            key=lambda x: -x.confidence # Сначала более уверенные из кандидатов
        )

        iou_overlap_threshold = 0.05 # Небольшое перекрытие считается "покрытой областью"

        for candidate_obj in candidates_for_logic_b:
            # Если unique_id не существует (не должно быть, но для безопасности)
            if candidate_obj.unique_id not in self.object_vars:
                self.object_vars[candidate_obj.unique_id] = tk.BooleanVar(value=False)
            
            is_covered = any(
                calculate_iou(candidate_obj.box_xywh, active_box_coords)[0] > iou_overlap_threshold
                for active_box_coords in current_active_boxes
            )
            
            if not is_covered:
                self.object_vars[candidate_obj.unique_id].set(True)
                current_active_boxes.append(candidate_obj.box_xywh) # Добавляем в активные для следующих проверок
                activated_count += 1
        
        return activated_count

    def switch_active_results_for_filter_and_refresh(self, new_mode: str, initial_setup: bool = False):
        logger.info(f"Переключение активного набора для фильтра: {new_mode}")
        self.display_mode = new_mode # Сохраняем текущий режим

        # Выбор набора результатов в зависимости от режима
        if new_mode == "YOLOv11":
            self.active_results_for_filter = self.merged_yolo11_results or []
        elif new_mode == "YOLOv8":
            self.active_results_for_filter = self.merged_yolo8_results or []
        elif new_mode == "Слияние":
            self.active_results_for_filter = self.final_merged_results or []
        else: # Фоллбэк на случай неизвестного режима
            logger.warning(f"Неизвестный режим отображения '{new_mode}'. Используется 'Слияние'.")
            self.active_results_for_filter = self.final_merged_results or []
            self.display_mode = "Слияние"
        
        self.gui.display_mode_var.set(self.display_mode) # Обновляем комбобокс в GUI

        # Очистка и инициализация состояний чекбоксов
        self.object_vars.clear()
        self.class_vars.clear()

        if self.active_results_for_filter:
            high_conf_thresh = CONFIG.get("final_confidence_threshold_high", 0.4)
            low_fallback_thresh = CONFIG.get("final_confidence_threshold_low_fallback", 0.2)

            # Логика А: Инициализация object_vars на основе высокой уверенности
            for det_obj in self.active_results_for_filter:
                # Подчиненные по умолчанию не выбраны, основные - если уверенность высокая
                is_initially_selected = (not det_obj.is_subordinate and 
                                         det_obj.confidence >= high_conf_thresh)
                self.object_vars[det_obj.unique_id] = tk.BooleanVar(value=is_initially_selected)

            # Логика Б: Активация объектов в "пустых" областях
            activated_by_logic_b = self._apply_empty_area_activation_logic(
                 high_conf_thresh, low_fallback_thresh
            )
            if activated_by_logic_b > 0:
                logger.info(f"Логика 'пустых областей' для '{new_mode}': активировано {activated_by_logic_b} объектов.")

        if not initial_setup: # initial_setup=True используется при завершении обработки из ThreadManager
            self.update_selected_objects() # Обновляет изображение и текст статуса
            if self.gui.filter_window and self.gui.filter_window.winfo_exists():
                current_search_text = self.gui.search_entry.get() if self.gui.search_entry else ""
                self.gui.update_checkboxes(current_search_text, self.gui.filter_window)

    def display_image(self, label_text="Обнаружение объектов"):
        image_to_show: Optional[np.ndarray] = None

        # Приоритет отображения: result_image -> processed_image (если show_processed) -> original_image
        if self.result_image is not None: # result_image - это то, что с рамками
            image_to_show = self.result_image
        elif self.show_processed and self.processed_image is not None:
            image_to_show = self.processed_image
        elif self.original_image is not None:
            image_to_show = self.original_image
        
        if image_to_show is None:
            logger.warning("display_image: Нет изображения для отображения.")
            self.gui.image_label.config(text="Нет изображения")
            if not self.is_processing: self.gui.center_placeholder_text()
            return

        max_display_dim = CONFIG.get("max_display_size", 500)
        try:
            if not isinstance(image_to_show, np.ndarray):
                logger.error(f"Изображение для отображения имеет неверный тип: {type(image_to_show)}")
                image_to_show = self.original_image if self.original_image is not None else \
                                np.zeros((max_display_dim,max_display_dim,3), dtype=np.uint8) # Фоллбэк

            h, w = image_to_show.shape[:2]
            if h == 0 or w == 0:
                logger.error("Изображение для отображения имеет нулевые размеры.")
                self.gui.image_label.config(text="Ошибка: изображение повреждено"); return

            # Масштабирование для отображения
            scale_factor = (max_display_dim / max(h, w)) * self.zoom_factor
            disp_w = max(1, int(w * scale_factor))
            disp_h = max(1, int(h * scale_factor))
            
            resized_display_img = cv2.resize(image_to_show, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            # Конвертация цвета для PIL/Tkinter
            if len(resized_display_img.shape) == 2: # Grayscale
                img_rgb = cv2.cvtColor(resized_display_img, cv2.COLOR_GRAY2RGB)
            elif resized_display_img.shape[2] == 3: # BGR
                img_rgb = cv2.cvtColor(resized_display_img, cv2.COLOR_BGR2RGB)
            elif resized_display_img.shape[2] == 4: # BGRA
                img_rgb = cv2.cvtColor(resized_display_img, cv2.COLOR_BGRA2RGB)
            else:
                logger.error(f"Неподдерживаемое количество каналов: {resized_display_img.shape}")
                self.gui.image_label.config(text="Ошибка формата каналов"); return

            pil_img = Image.fromarray(img_rgb)
            self.photo = ImageTk.PhotoImage(pil_img) # Сохраняем ссылку на PhotoImage

            self.gui.canvas.delete("all") # Очищаем холст
            canvas_w = self.gui.canvas.winfo_width()
            canvas_h = self.gui.canvas.winfo_height()
            # Центрирование изображения на холсте
            x_pos = (canvas_w - disp_w) // 2
            y_pos = (canvas_h - disp_h) // 2
            self.gui.canvas.create_image(x_pos, y_pos, image=self.photo, anchor='nw')
            self.gui.image_label.config(text=label_text)

        except Exception as e:
            logger.error(f"Ошибка отображения изображения: {str(e)}", exc_info=True)
            self.gui.image_label.config(text="Ошибка: не удалось отобразить")


    def update_selected_objects(self, filter_window_ref=None):
        logger.info("Обновление выбранных объектов и отображения.")
        
        base_image_for_drawing: Optional[np.ndarray] = None
        if self.show_processed and self.processed_image is not None:
            base_image_for_drawing = self.processed_image
        elif self.original_image is not None:
            base_image_for_drawing = self.original_image

        if base_image_for_drawing is None:
            logger.error("Нет базового изображения для отрисовки рамок.")
            self.result_image = None # Сбрасываем result_image, если нет основы
            self.display_image("Ошибка: нет базового изображения")
            self.gui.status_label.config(text="Ошибка: нет базового изображения")
            return

        if not self.active_results_for_filter:
            # Если нет активных результатов (например, после неудачной обработки или если режим не дал результатов)
            self.result_image = base_image_for_drawing.copy() # Показываем чистое базовое изображение
            status_text_empty = f"Ничего не обнаружено ({self.display_mode})"
            self.display_image(status_text_empty)
            self.gui.status_label.config(text=status_text_empty)
            # Обновляем чекбоксы, чтобы показать, что их нет
            if self.gui.filter_window and self.gui.filter_window.winfo_exists():
                 self.gui.update_checkboxes(self.gui.search_entry.get() if self.gui.search_entry else "", self.gui.filter_window)
            return

        selected_detections: List[DetectionResult] = [
            det for det in self.active_results_for_filter
            if self.object_vars.get(det.unique_id) and self.object_vars[det.unique_id].get()
        ]
        
        logger.info(f"Выбрано для отображения ({self.display_mode}): {len(selected_detections)} объектов.")
        self.result_image = draw_results(base_image_for_drawing.copy(), selected_detections)

        # Формирование текста для image_label и status_label
        status_prefix = f"Обнаружено ({self.display_mode})"
        current_label_text: str
        if selected_detections:
            # Показываем первые N меток
            label_parts = [f"{det.final_display_label} ({int(det.confidence*100)}%)" 
                           for det in selected_detections[:3]]
            current_label_text = f"{status_prefix}: " + ", ".join(label_parts)
            if len(selected_detections) > 3:
                current_label_text += f" и еще {len(selected_detections) - 3}..."
        else:
            # Если есть активные результаты, но ничего не выбрано чекбоксами
            current_label_text = f"Ничего не выбрано ({self.display_mode})"

        self.display_image(current_label_text)
        self.gui.status_label.config(text=current_label_text)

        # Обновление чекбоксов в окне фильтра, если оно открыто
        active_filter_win = filter_window_ref if filter_window_ref and filter_window_ref.winfo_exists() else \
                            (self.gui.filter_window if self.gui.filter_window and self.gui.filter_window.winfo_exists() else None)
        if active_filter_win:
            try:
                search_text = self.gui.search_entry.get() if self.gui.search_entry else ""
                self.gui.update_checkboxes(search_text, filter_window_ref=active_filter_win)
            except Exception as e_cb_update:
                logger.error(f"Ошибка обновления чекбоксов: {str(e_cb_update)}", exc_info=True)

    def toggle_image(self):
        if self.original_image is None and self.processed_image is None:
            messagebox.showwarning("Предупреждение", "Нет изображений для переключения.")
            return

        self.show_processed = not self.show_processed
        logger.info(f"Переключение изображения: {'показать обработанное' if self.show_processed else 'показать оригинал'}")

        button_text = "Показать оригинал" if self.show_processed else "Показать обработанное"
        
        # Обработка случаев, когда одно из изображений отсутствует
        if self.show_processed and self.processed_image is None:
            messagebox.showinfo("Информация", "Обработанное изображение отсутствует. Показывается оригинал.")
            self.show_processed = False # Возвращаемся к оригиналу
            button_text = "Показать обработанное"
        elif not self.show_processed and self.original_image is None:
             messagebox.showinfo("Информация", "Оригинальное изображение отсутствует. Показывается обработанное (если есть).")
             self.show_processed = self.processed_image is not None # Показываем обработанное, если оно есть
             button_text = "Показать оригинал" if self.show_processed else "Показать обработанное"
        
        self.gui.toggle_button.config(text=button_text)
        self.update_selected_objects() # Перерисовываем с учетом нового базового изображения

    def handle_error(self, message: str):
        """Обработка и отображение сообщения об ошибке в GUI."""
        logger.error(f"Обработка ошибки GUI: {message}")
        messagebox.showerror("Ошибка", f"Произошла ошибка: {message}")
        # Дополнительно можно обновить статус-бар
        if hasattr(self.gui, 'status_label'):
            self.gui.status_label.config(text=f"Ошибка: {message[:100]}...")

    def finish_processing(self, error_occurred: bool = False, error_message: Optional[str] = None):
        logger.info(f"Завершение сессии обработки. Ошибка: {error_occurred}, Сообщение: {error_message}")
        self.is_processing = False # Важно сбросить флаг
        self.window.config(cursor="") # Возвращаем стандартный курсор

        # Обновляем состояние кнопок
        self.gui.set_buttons_state(
            processing=False,
            has_results=bool(self.active_results_for_filter or \
                             self.final_merged_results or \
                             self.merged_yolo11_results or \
                             self.merged_yolo8_results),
            has_original=(self.original_image is not None),
            has_processed=(self.processed_image is not None)
        )
        
        current_status_text = self.gui.status_label.cget("text")
        if error_message: # Если передано сообщение об ошибке из load_image
            self.handle_error(error_message) # Покажет messagebox
            current_status_text = error_message # Обновит статус в handle_error

        if error_occurred:
            # Если была ошибка, но result_image (с рамками) не сформировался,
            # пытаемся показать хотя бы оригинал или обработанное.
            if self.result_image is None:
                if self.original_image is not None:
                    self.result_image = self.original_image.copy()
                    self.display_image(current_status_text) # Показываем оригинал со статусом ошибки
                else: # Если и оригинала нет (крайний случай)
                    self.gui.image_label.config(text=current_status_text or "Ошибка обработки")
                    self.gui.center_placeholder_text()
            else: # Если result_image есть (например, ошибка на позднем этапе), просто отображаем его
                self.display_image(current_status_text)
        else:
            # Если нет ошибки, update_selected_objects уже должен был быть вызван
            # из switch_active_results_for_filter_and_refresh (в ThreadManager._handle_pipeline_completion)
            # Но на всякий случай, если что-то пошло не так, вызовем еще раз
            self.update_selected_objects() 

        # Устанавливаем прогресс
        final_progress = self.gui.progress['value'] # Текущее значение
        if error_occurred and not self.active_results_for_filter: # Если ошибка и нет вообще результатов
             final_progress = 0 # или оставить текущий
        else:
             final_progress = 100
        self.gui.progress['value'] = final_progress
        
        import gc; gc.collect()

    def save_results(self):
        logger.info("Сохранение результатов.")
        if self.result_image is None:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения.")
            return

        original_filename = os.path.basename(self.image_path or "detected_image")
        name, ext = os.path.splitext(original_filename)
        # Добавляем текущий режим отображения в имя файла
        mode_suffix = self.display_mode.lower().replace(" ", "_")
        default_save_name = f"{name}_detected_{mode_suffix}.png"

        save_path = filedialog.asksaveasfilename(
            title="Сохранить результаты как...",
            initialfile=default_save_name,
            defaultextension=".png",
            filetypes=[("PNG файлы", "*.png"), ("JPEG файлы", "*.jpg")]
        )
        if save_path:
            try:
                img_to_save_bgr = self.result_image # result_image уже в BGR
                if not isinstance(img_to_save_bgr, np.ndarray):
                    raise ValueError("result_image не является numpy массивом для сохранения.")
                
                success = cv2.imwrite(save_path, img_to_save_bgr)
                if not success:
                    raise Exception(f"cv2.imwrite не удалось сохранить файл в {save_path}")
                messagebox.showinfo("Успех", f"Результаты успешно сохранены: {save_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении результатов: {str(e)}", exc_info=True)
                messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {str(e)}")

    def handle_resize(self, event=None):
        # Игнорируем события от других виджетов, если они случайно попали сюда
        if event and hasattr(event, 'widget') and event.widget != self.window:
            return
            
        current_time = time.time() * 1000 # мс
        if current_time - self.last_resize_time < self.resize_debounce_ms:
            return # Слишком часто, пропускаем
        self.last_resize_time = current_time

        # Перерисовываем изображение, если оно есть, или плейсхолдер
        if self.result_image is not None or self.processed_image is not None or self.original_image is not None:
            self.display_image(self.gui.image_label.cget("text")) # Используем текущий текст метки
        elif not self.is_processing: # Если нет изображений и не идет обработка
             self.gui.center_placeholder_text()

    def redraw_image(self):
        """Принудительная перерисовка текущего изображения (например, при изменении масштаба)."""
        logger.debug("Принудительная перерисовка изображения (redraw_image)")
        self.display_image(self.gui.image_label.cget("text"))

    def cleanup(self):
        logger.info("Очистка ресурсов и завершение работы приложения.")
        try:
            self.thread_manager.cleanup() # Завершение потоков и очистка CUDA, если используется
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Временная директория {self.temp_dir} удалена.")
        except Exception as e_cleanup:
            logger.error(f"Ошибка при очистке: {str(e_cleanup)}", exc_info=True)
        finally:
            logger.info("Закрытие главного окна приложения.")
            self.window.destroy()