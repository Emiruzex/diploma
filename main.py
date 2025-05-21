import logging
from tkinterdnd2 import TkinterDnD
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

if __name__ == "__main__":
    """Точка входа в приложение."""
    logger.info("Запуск приложения")
    try:
        window = TkinterDnD.Tk()
        app = ObjectDetectionApp(window)
        window.mainloop()
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске приложения: {str(e)}", exc_info=True)
        raise
