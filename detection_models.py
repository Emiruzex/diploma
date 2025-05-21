# detection_models.py
from dataclasses import dataclass, field
from typing import Tuple, Optional
import uuid

@dataclass
class DetectionResult:
    """
    Хранит информацию об одном обнаруженном объекте.
    """
    x: int
    y: int
    w: int
    h: int
    original_label_raw: str  # Исходная метка от модели
    confidence: float
    model_source: str        # Источник модели (например, "YOLOv11", "YOLOv8")
    
    # Поля, заполняемые после слияния и дополнительной обработки
    unique_id: str = field(default_factory=lambda: str(uuid.uuid4())) # Уникальный ID
    final_display_label: str = ""  # Финальная метка для GUI (может включать номер экземпляра)
    mapped_label_for_logic: str = "" # Метка для логики сопоставления (например, "Кот")
    reason: Optional[str] = None     # Причина подчиненности или фильтрации
    is_subordinate: bool = False     # Флаг подчиненного объекта (например, часть тела)

    def __post_init__(self):
        """Выполняется после инициализации основных полей."""
        if not self.final_display_label:
            # По умолчанию, финальная метка - исходная (может быть перезаписана).
            self.final_display_label = self.original_label_raw
        
        if not self.mapped_label_for_logic:
            # По умолчанию, для логики используется исходная метка (должна быть установлена позже).
             self.mapped_label_for_logic = self.original_label_raw

    @property
    def box_xywh(self) -> Tuple[int, int, int, int]:
        """Координаты рамки (x, y, ширина, высота)."""
        return (self.x, self.y, self.w, self.h)

    @property
    def area(self) -> int:
        """Площадь ограничивающей рамки."""
        return self.w * self.h