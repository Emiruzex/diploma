# gui.py
import tkinter as tk
from tkinter import ttk as tkinter_ttk
from ttkbootstrap import Style
from ttkbootstrap.constants import *
import ttkbootstrap as ttk
from tkinterdnd2 import DND_FILES
import logging
import re
from fuzzywuzzy import fuzz
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from config import CONFIG

if TYPE_CHECKING:
    from app import ObjectDetectionApp

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

class ObjectDetectionGUI:
    def __init__(self, app_ref: 'ObjectDetectionApp'):
        self.app = app_ref
        self.window = self.app.window
        try:
            theme_to_use = CONFIG.get("theme", "flatly")
            self.style = Style(theme=theme_to_use)
            logger.info(f"Тема ttkbootstrap '{theme_to_use}' применена.")
        except Exception as e_theme:
            logger.warning(f"Не удалось применить тему ttkbootstrap '{CONFIG.get('theme', 'flatly')}': {e_theme}. Используется стиль по умолчанию.")
            self.style = Style()

        self.search_entry: Optional[ttk.Entry] = None
        self.checkbox_frames: Dict[str, ttk.LabelFrame] = {}
        self.scrollable_frame_for_filters: Optional[ttk.Frame] = None
        self.filter_window: Optional[tk.Toplevel] = None
        self.display_mode_var: Optional[tk.StringVar] = None # Инициализируем здесь

        self.setup_ui()

    def set_buttons_state(self, processing: bool, has_results: bool = False,
                          has_original: bool = False, has_processed: bool = False):
        load_state = ttk.DISABLED if processing else ttk.NORMAL
        save_state = ttk.NORMAL if not processing and self.app.result_image is not None else ttk.DISABLED
        toggle_state = ttk.NORMAL if not processing and has_original and has_processed else ttk.DISABLED
        filter_state = ttk.NORMAL if not processing and has_results else ttk.DISABLED
        
        display_mode_state = 'readonly' if not processing and \
                                          (self.app.merged_yolo11_results or \
                                           self.app.merged_yolo8_results or \
                                           self.app.final_merged_results) else ttk.DISABLED

        if hasattr(self, 'load_button'): self.load_button.config(state=load_state)
        if hasattr(self, 'save_button'): self.save_button.config(state=save_state)
        if hasattr(self, 'toggle_button'): self.toggle_button.config(state=toggle_state)
        if hasattr(self, 'filter_button'): self.filter_button.config(state=filter_state)
        if hasattr(self, 'display_mode_selector'): self.display_mode_selector.config(state=display_mode_state)

    def setup_ui(self):
        logger.info("Инициализация пользовательского интерфейса (GUI)")
        self.main_container = ttk.Frame(self.window, padding=(15, 10))
        self.main_container.pack(fill=BOTH, expand=True)

        self.toolbar = ttk.Frame(self.main_container)
        self.toolbar.pack(fill=X, pady=5)

        self.load_button = ttk.Button(self.toolbar, text="Загрузить", command=self.app.load_dialog, style='primary.TButton')
        self.load_button.pack(side=LEFT, padx=5)

        self.save_button = ttk.Button(self.toolbar, text="Сохранить", command=self.app.save_results, style='success.TButton', state=DISABLED)
        self.save_button.pack(side=LEFT, padx=5)

        self.toggle_button = ttk.Button(
            self.toolbar, text="Показать обработанное", command=self.app.toggle_image,
            style='secondary.TButton', state=DISABLED
        )
        self.toggle_button.pack(side=LEFT, padx=5)

        self.filter_button = ttk.Button(
            self.toolbar, text="Фильтр", command=self.open_filter_window,
            style='info.TButton', state=DISABLED
        )
        self.filter_button.pack(side=LEFT, padx=5)

        ttk.Label(self.toolbar, text="Режим отображения:").pack(side=LEFT, padx=(10, 2))
        self.display_mode_var = tk.StringVar(value=self.app.display_mode) 
        self.display_mode_selector = ttk.Combobox(
            self.toolbar, textvariable=self.display_mode_var,
            values=["Слияние", "YOLOv11", "YOLOv8"], state=DISABLED, width=10
        )
        self.display_mode_selector.pack(side=LEFT, padx=2)
        self.display_mode_selector.bind("<<ComboboxSelected>>", self._on_display_mode_changed)

        self.image_frame = ttk.LabelFrame(self.main_container, text="Изображение", padding=10)
        self.image_frame.pack(fill=BOTH, expand=True, padx=5, pady=(0,5))
        
        # --- ИСПРАВЛЕНО ПОЛУЧЕНИЕ ЦВЕТА ФОНА ---
        canvas_bg = 'white' 
        if hasattr(self, 'style') and self.style and hasattr(self.style, 'colors') and self.style.colors:
            try:
                canvas_bg = self.style.colors.bg 
            except AttributeError:
                logger.warning("Атрибут 'bg' не найден в style.colors. Используется 'white' для фона холста.")
        else:
            logger.warning("Атрибут 'style' или 'style.colors' не инициализирован. Используется 'white' для фона холста.")
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        self.canvas = tk.Canvas(self.image_frame, bg=canvas_bg, highlightthickness=1, highlightbackground='gray')
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<Button-1>", self.on_canvas_click_placeholder)
        self.canvas.bind("<MouseWheel>", self.zoom_image_event)
        self.canvas.bind("<Button-4>", self.zoom_image_event)
        self.canvas.bind("<Button-5>", self.zoom_image_event)
        self.center_placeholder_text()

        self.zoom_controls_frame = ttk.Frame(self.image_frame)
        self.zoom_controls_frame.pack(pady=5)
        self.zoom_out_button = ttk.Button(self.zoom_controls_frame, text="-", command=lambda: self.adjust_zoom(-0.1), width=3)
        self.zoom_out_button.pack(side=LEFT, padx=(0,2))
        self.zoom_scale = ttk.Scale(
            self.zoom_controls_frame, from_=self.app.min_zoom, to_=self.app.max_zoom,
            orient=HORIZONTAL, command=self.update_zoom_from_scale, length=150
        )
        self.zoom_scale.set(self.app.zoom_factor)
        self.zoom_scale.pack(side=LEFT, padx=2, fill=X, expand=True)
        self.zoom_in_button = ttk.Button(self.zoom_controls_frame, text="+", command=lambda: self.adjust_zoom(0.1), width=3)
        self.zoom_in_button.pack(side=LEFT, padx=(2,0))

        max_width_for_label = CONFIG.get("max_display_size", 500) + 100
        self.image_label = ttk.Label(self.image_frame, text="", wraplength=max_width_for_label, anchor="center")
        self.image_label.pack(fill=X, padx=5, pady=(0,5))

        self.status_frame = ttk.Frame(self.window, padding=(0, 5))
        self.status_frame.pack(fill=X, padx=10, pady=(0,5))
        self.progress = ttk.Progressbar(self.status_frame, mode='determinate', length=200, maximum=100)
        self.progress.pack(side=LEFT, padx=(0, 5))
        self.status_label = ttk.Label(self.status_frame, text="Готово", anchor='w')
        self.status_label.pack(fill=X, expand=True, padx=(5,0))

        try:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind('<<Drop>>', self.app.handle_drop)
        except tk.TclError as e_dnd:
            logger.warning(f"Не удалось зарегистрировать цель для Drag and Drop: {e_dnd}. Перетаскивание может не работать.")

        self.window.geometry(CONFIG.get("window_size", "1200x800"))
        self.window.minsize(600, 400)

    def _on_display_mode_changed(self, event=None):
        if self.app.is_processing:
            if self.display_mode_var: self.display_mode_var.set(self.app.display_mode)
            messagebox.showwarning("Внимание", "Дождитесь окончания обработки перед сменой режима.")
            return

        if self.display_mode_var:
            new_mode = self.display_mode_var.get()
            if new_mode != self.app.display_mode:
                logger.info(f"Пользователь изменил режим отображения на: {new_mode}")
                self.app.switch_active_results_for_filter_and_refresh(new_mode)
                if self.filter_window and self.filter_window.winfo_exists():
                    self.filter_window.title(f"Фильтр (режим: {self.app.display_mode})")

    def on_canvas_click_placeholder(self, event):
        is_placeholder_visible = bool(self.canvas.find_withtag("placeholder"))
        if is_placeholder_visible and self.app.original_image is None and not self.app.is_processing:
            self.app.load_dialog(event)

    def on_canvas_configure(self, event=None):
        if self.app.original_image is None and not self.app.is_processing:
            self.center_placeholder_text()
        elif self.app.original_image is not None : 
            self.app.handle_resize(event)

    def center_placeholder_text(self, event=None):
        self.canvas.delete("placeholder")
        self.canvas.update_idletasks()
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1: canvas_width = CONFIG.get("max_display_size", 500) 
        if canvas_height <= 1: canvas_height = CONFIG.get("max_display_size", 500)
            
        self.canvas.create_text(
            canvas_width / 2, canvas_height / 2,
            text="Перетащите или кликните, чтобы выбрать изображение",
            font=("Arial", 14), tags="placeholder", anchor="center", fill="grey"
        )

    def adjust_zoom(self, delta: float):
        if self.app.original_image is None: return
        new_zoom = self.app.zoom_factor + delta
        self.app.zoom_factor = max(self.app.min_zoom, min(self.app.max_zoom, new_zoom))
        self.zoom_scale.set(self.app.zoom_factor)
        self.app.redraw_image()

    def zoom_image_event(self, event):
        if self.app.original_image is None: return

        zoom_factor_change = 1.1
        if hasattr(event, 'delta') and event.delta != 0 : # Поддержка event.delta для Windows/macOS
             if event.delta > 0: self.app.zoom_factor *= zoom_factor_change
             else: self.app.zoom_factor /= zoom_factor_change
        elif hasattr(event, 'num'): # Поддержка event.num для Linux
            if event.num == 4: self.app.zoom_factor *= zoom_factor_change # Scroll up
            elif event.num == 5: self.app.zoom_factor /= zoom_factor_change # Scroll down
        else:
            return

        self.app.zoom_factor = max(self.app.min_zoom, min(self.app.max_zoom, self.app.zoom_factor))
        self.zoom_scale.set(self.app.zoom_factor)
        self.app.redraw_image()

    def update_zoom_from_scale(self, value_str: str):
        if self.app.original_image is None: return
        try:
            new_zoom_val = float(value_str)
            if self.app.zoom_factor != new_zoom_val:
                self.app.zoom_factor = new_zoom_val
                self.app.redraw_image()
        except ValueError:
            logger.warning(f"Некорректное значение от ползунка масштаба: {value_str}")

    def update_checkboxes(self, search_text: str, filter_window_ref: Optional[tk.Toplevel] = None):
        active_filter_win = filter_window_ref if filter_window_ref and filter_window_ref.winfo_exists() else self.filter_window
        
        if not self.scrollable_frame_for_filters or not active_filter_win or not active_filter_win.winfo_exists():
            logger.debug("Окно фильтра или его компоненты не инициализированы.")
            return

        for widget in self.scrollable_frame_for_filters.winfo_children():
            widget.destroy()
        self.checkbox_frames.clear()

        if not self.app.active_results_for_filter:
            ttk.Label(self.scrollable_frame_for_filters, text=f"Нет объектов для фильтрации (режим: {self.app.display_mode}).").pack(padx=10, pady=10)
            self._configure_scroll_region_for_filter_canvas()
            return

        logger.debug(f"Обновление чекбоксов (режим '{self.app.display_mode}', поиск: '{search_text}')")
        search_text_lower = search_text.lower().strip().replace('_', ' ')

        grouped_by_mapped_label: Dict[str, List[Any]] = {}
        for det_obj in self.app.active_results_for_filter:
            group_key = det_obj.mapped_label_for_logic or det_obj.original_label_raw
            grouped_by_mapped_label.setdefault(group_key, []).append(det_obj)

        display_groups_ordered: List[Dict[str, Any]] = []
        for group_label, objects_in_group in grouped_by_mapped_label.items():
            display_label_for_search = group_label.lower().replace('_', ' ')
            passes_search, score = False, 0
            if not search_text_lower:
                passes_search, score = True, 100 
            else:
                if re.search(re.escape(search_text_lower), display_label_for_search):
                    passes_search, score = True, 100
                else:
                    score = fuzz.partial_ratio(search_text_lower, display_label_for_search)
                    if score > 60: passes_search = True
            
            if passes_search:
                display_groups_ordered.append({
                    'group_display_name': group_label, 
                    'score': score, 
                    'objects_in_group': objects_in_group
                })
        
        display_groups_ordered.sort(key=lambda x: (-x['score'], x['group_display_name']))

        for group_info in display_groups_ordered:
            group_name = group_info['group_display_name']
            objects = group_info['objects_in_group']

            class_frame = ttk.LabelFrame(self.scrollable_frame_for_filters, text=group_name, padding=5)
            class_frame.pack(fill=X, pady=3, padx=3, anchor='w')
            self.checkbox_frames[group_name] = class_frame

            selected_in_group = sum(
                1 for det in objects 
                if self.app.object_vars.get(det.unique_id) and self.app.object_vars[det.unique_id].get()
            )
            total_in_group = len(objects)

            class_var = self.app.class_vars.get(group_name)
            if not class_var:
                class_var = tk.BooleanVar()
                self.app.class_vars[group_name] = class_var
            
            class_cb_state_flags = []
            if total_in_group > 0:
                if selected_in_group == total_in_group: class_var.set(True); class_cb_state_flags = ['!alternate', 'selected']
                elif selected_in_group > 0: class_var.set(False); class_cb_state_flags = ['alternate']
                else: class_var.set(False); class_cb_state_flags = ['!alternate', '!selected']
            else: 
                class_var.set(False); class_cb_state_flags = ['!alternate', '!selected', DISABLED]

            class_cb_text = f"Все '{group_name}' ({selected_in_group}/{total_in_group})"
            class_cb_style = 'primary.Toolbutton' if 'alternate' in class_cb_state_flags else 'Toolbutton'
            
            class_cb = ttk.Checkbutton(
                class_frame, text=class_cb_text, variable=class_var,
                command=lambda gn=group_name: self.toggle_class_objects(gn, active_filter_win),
                style=class_cb_style
            )
            if total_in_group == 0: class_cb.config(state=DISABLED)
            
            try: class_cb.state(class_cb_state_flags)
            except tk.TclError:
                 if 'selected' in class_cb_state_flags : class_cb.state(['selected'])
                 else: class_cb.state(['!selected'])
            class_cb.pack(anchor='w', pady=(0, 3))

            for det_obj in sorted(objects, key=lambda d: d.original_label_raw):
                obj_id = det_obj.unique_id
                
                base_label = det_obj.final_display_label or det_obj.original_label_raw
                label_for_filter = re.sub(r'\s+\d+\s*(\[часть\])?$', '', base_label).strip()
                if not label_for_filter: label_for_filter = det_obj.original_label_raw
                
                confidence_percent = int(det_obj.confidence * 100)
                final_text = f"{label_for_filter} ({confidence_percent}%)"
                if det_obj.is_subordinate: final_text += " [часть]"

                obj_var = self.app.object_vars.get(obj_id)
                if not obj_var:
                    obj_var = tk.BooleanVar(value=False)
                    self.app.object_vars[obj_id] = obj_var

                individual_cb = ttk.Checkbutton(
                    class_frame, text=final_text, variable=obj_var,
                    command=lambda afw=active_filter_win: self.app.update_selected_objects(afw)
                )
                individual_cb.pack(anchor='w', padx=20, pady=1)

        self._configure_scroll_region_for_filter_canvas()

    def _configure_scroll_region_for_filter_canvas(self):
        if self.scrollable_frame_for_filters and self.scrollable_frame_for_filters.master:
            canvas_for_scroll = self.scrollable_frame_for_filters.master
            if isinstance(canvas_for_scroll, tk.Canvas):
                self.scrollable_frame_for_filters.update_idletasks()
                canvas_for_scroll.config(scrollregion=canvas_for_scroll.bbox("all"))

    def toggle_class_objects(self, group_display_name: str, filter_window_ref: Optional[tk.Toplevel]):
        logger.debug(f"Переключение объектов для группы '{group_display_name}' (режим {self.app.display_mode})")
        active_filter_win = filter_window_ref if filter_window_ref and filter_window_ref.winfo_exists() else self.filter_window
        if not self.app.active_results_for_filter or not active_filter_win or not active_filter_win.winfo_exists():
            return

        objects_in_group = [
            det for det in self.app.active_results_for_filter
            if (det.mapped_label_for_logic or det.original_label_raw) == group_display_name
        ]
        if not objects_in_group: return

        selected_count_before = sum(
            1 for obj in objects_in_group
            if self.app.object_vars.get(obj.unique_id) and self.app.object_vars[obj.unique_id].get()
        )
        new_state_for_individuals = not (selected_count_before == len(objects_in_group))

        if group_display_name in self.app.class_vars:
            self.app.class_vars[group_display_name].set(new_state_for_individuals)
        
        for det_obj in objects_in_group:
            if det_obj.unique_id in self.app.object_vars:
                self.app.object_vars[det_obj.unique_id].set(new_state_for_individuals)

        self.app.update_selected_objects(active_filter_win)

    def _mass_selection_filter_objects(self, select_all: bool):
        logger.debug(f"{'Выделение' if select_all else 'Снятие выделения'} всех видимых объектов в фильтре (режим: {self.app.display_mode})")
        active_filter_win = self.filter_window
        if not self.app.active_results_for_filter or \
           not active_filter_win or not active_filter_win.winfo_exists() or \
           not self.checkbox_frames:
            return

        for group_label, frame_widget in self.checkbox_frames.items():
            if frame_widget.winfo_exists(): 
                if group_label in self.app.class_vars:
                    self.app.class_vars[group_label].set(select_all)
                
                objects_in_this_visible_group = [
                    det for det in self.app.active_results_for_filter
                    if (det.mapped_label_for_logic or det.original_label_raw) == group_label
                ]
                for det_obj in objects_in_this_visible_group:
                    if det_obj.unique_id in self.app.object_vars:
                        self.app.object_vars[det_obj.unique_id].set(select_all)
        
        self.app.update_selected_objects(active_filter_win)

    def select_all_filter_objects(self):
        self._mass_selection_filter_objects(True)

    def deselect_all_filter_objects(self):
        self._mass_selection_filter_objects(False)

    def open_filter_window(self):
        if not self.app.active_results_for_filter:
            messagebox.showinfo("Информация", f"Нет объектов для фильтрации (режим: {self.app.display_mode}).")
            return
        if self.filter_window and self.filter_window.winfo_exists():
            self.filter_window.lift()
            return

        logger.info(f"Открытие окна фильтрации (режим: {self.app.display_mode})")
        self.filter_window = ttk.Toplevel(self.window)
        self.filter_window.title(f"Фильтр (режим: {self.app.display_mode})")
        self.filter_window.geometry("550x700")
        self.filter_window.transient(self.window)

        if not self.app.object_vars and self.app.active_results_for_filter:
            logger.warning("object_vars пусты, но active_results_for_filter есть. Попытка переинициализации для фильтра.")
            self.app.switch_active_results_for_filter_and_refresh(self.app.display_mode, initial_setup=True)

        search_frame = ttk.Frame(self.filter_window); search_frame.pack(fill=X, padx=10, pady=(10,5))
        ttk.Label(search_frame, text="Поиск:").pack(side=LEFT, padx=(0,5))
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=LEFT, fill=X, expand=True)
        self.search_entry.bind("<KeyRelease>", 
            lambda event: self.update_checkboxes(self.search_entry.get() if self.search_entry else "", self.filter_window))

        list_buttons_frame = ttk.Frame(self.filter_window); list_buttons_frame.pack(fill=X, padx=10, pady=5)
        ttk.Button(list_buttons_frame, text="Выделить всё", command=self.select_all_filter_objects).pack(side=LEFT, padx=(0,5))
        ttk.Button(list_buttons_frame, text="Снять всё", command=self.deselect_all_filter_objects).pack(side=LEFT)

        list_container_frame = ttk.Frame(self.filter_window)
        list_container_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        filter_canvas_bg = 'white' # Значение по умолчанию для Canvas фильтра
        if hasattr(self, 'style') and self.style and hasattr(self.style, 'colors') and self.style.colors:
            try: filter_canvas_bg = self.style.colors.inputbg # Используем inputbg или другой подходящий цвет для фона списка
            except AttributeError: pass # Оставляем 'white', если inputbg нет

        filter_canvas = tk.Canvas(list_container_frame, highlightthickness=0, bg=filter_canvas_bg)
        filter_scrollbar = ttk.Scrollbar(list_container_frame, orient=VERTICAL, command=filter_canvas.yview)
        
        self.scrollable_frame_for_filters = ttk.Frame(filter_canvas)
        self.scrollable_frame_for_filters.bind("<Configure>", 
            lambda e: self._configure_scroll_region_for_filter_canvas()) 
        
        filter_canvas.create_window((0, 0), window=self.scrollable_frame_for_filters, anchor="nw")
        filter_canvas.configure(yscrollcommand=filter_scrollbar.set)
        
        filter_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        filter_scrollbar.pack(side=RIGHT, fill=Y)

        filter_canvas.bind("<MouseWheel>", lambda event: self._on_mouse_wheel_for_specific_canvas(event, filter_canvas))
        filter_canvas.bind("<Button-4>", lambda event: self._on_mouse_wheel_for_specific_canvas(event, filter_canvas))
        filter_canvas.bind("<Button-5>", lambda event: self._on_mouse_wheel_for_specific_canvas(event, filter_canvas))

        self.update_checkboxes("", self.filter_window)

        apply_button_frame = ttk.Frame(self.filter_window); apply_button_frame.pack(fill=X, padx=10, pady=(5,10))
        self.apply_button = ttk.Button(apply_button_frame, text="Закрыть", command=self.on_filter_window_close)
        self.apply_button.pack(side=RIGHT)

        self.filter_window.protocol("WM_DELETE_WINDOW", self.on_filter_window_close)
        self.filter_window.bind("<Escape>", lambda e: self.on_filter_window_close())

    def _on_mouse_wheel_for_specific_canvas(self, event, canvas: tk.Canvas):
        scroll_units = 0
        if hasattr(event, 'delta') and event.delta != 0:
            scroll_units = -1 if event.delta > 0 else 1
        elif hasattr(event, 'num'):
            if event.num == 4: scroll_units = -1
            elif event.num == 5: scroll_units = 1
        
        if scroll_units != 0:
            canvas.yview_scroll(scroll_units, "units")

    def on_filter_window_close(self):
        logger.debug("Закрытие окна фильтрации.")
        if self.filter_window and self.filter_window.winfo_exists():
            self.filter_window.destroy()
        self.filter_window = None
        self.search_entry = None
        self.scrollable_frame_for_filters = None
        self.checkbox_frames.clear()