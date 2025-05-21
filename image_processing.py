# image_processing.py
import cv2
import numpy as np
import logging
import os # Оставлен на случай, если вы решите раскомментировать сохранение финального изображения
import time
import psutil
from typing import Optional
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

def color_normalization(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Нормализация цвета изображения методом Gray World.
    """
    try:
        if image is None or image.size == 0:
            logger.error("color_normalization: изображение пустое.")
            return image # Возвращаем как есть, если это None, или пустое

        b, g, r = cv2.split(image)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)

        if b_mean == 0 or g_mean == 0 or r_mean == 0:
            logger.warning("color_normalization: один из каналов имеет нулевое среднее, пропуск.")
            return image

        gray_mean = (b_mean + g_mean + r_mean) / 3
        b_scale, g_scale, r_scale = gray_mean / b_mean, gray_mean / g_mean, gray_mean / r_mean

        max_scale, min_scale = 2.0, 0.5
        b_scale = min(max_scale, max(min_scale, b_scale))
        g_scale = min(max_scale, max(min_scale, g_scale))
        r_scale = min(max_scale, max(min_scale, r_scale))

        b_normalized = np.clip(b * b_scale, 0, 255).astype(np.uint8)
        g_normalized = np.clip(g * g_scale, 0, 255).astype(np.uint8)
        r_normalized = np.clip(r * r_scale, 0, 255).astype(np.uint8)

        normalized_image = cv2.merge([b_normalized, g_normalized, r_normalized])
        logger.debug(f"Нормализация цвета: B_scale={b_scale:.2f}, G_scale={g_scale:.2f}, R_scale={r_scale:.2f}")
        return normalized_image
    except Exception as e:
        logger.error(f"Ошибка в color_normalization: {str(e)}", exc_info=True)
        return image # Возвращаем оригинальное в случае ошибки

def histogram_equalization(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Адаптивная гистограммная эквализация (CLAHE) с защитой от пересвета.
    """
    try:
        if image is None or image.size == 0:
            logger.error("histogram_equalization: изображение пустое.")
            return image

        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
        
        mean_brightness = np.mean(y_channel)
        overexposure_percent = np.sum(y_channel > 200) / y_channel.size * 100
        
        clip_limit = 2.0
        if mean_brightness < 50: clip_limit = 3.0
        elif mean_brightness > 150 or overexposure_percent > 20: clip_limit = 1.0
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        y_equalized = clahe.apply(y_channel)
        
        blend_weight = 0.7 if overexposure_percent > 20 else 0.8
        y_blended = cv2.addWeighted(y_equalized, blend_weight, y_channel, 1.0 - blend_weight, 0)
        
        ycrcb_equalized_image = cv2.merge([y_blended, cr_channel, cb_channel])
        result_image = cv2.cvtColor(ycrcb_equalized_image, cv2.COLOR_YCrCb2BGR)
        
        logger.debug(f"Адаптивная гистограммная эквализация: clipLimit={clip_limit:.1f}, blend_weight={blend_weight:.1f}")
        return result_image
    except Exception as e:
        logger.error(f"Ошибка в histogram_equalization: {str(e)}", exc_info=True)
        return image

def preprocess_image(image_input: np.ndarray) -> Optional[np.ndarray]:
    """
    Основной путь предобработки изображения.
    Возвращает обработанное изображение или None в случае критической ошибки.
    """
    logger.info("Начало предобработки изображения (preprocess_image).")
    try:
        if image_input is None or image_input.size == 0:
            logger.error("preprocess_image: входное изображение пустое или None.")
            return None

        image_copy = image_input.copy()

        available_memory_mb = psutil.virtual_memory().available / (1024 ** 2)
        memory_pause_threshold = CONFIG.get("memory_pause_threshold_mb", 200)
        if available_memory_mb < memory_pause_threshold:
            logger.warning(f"Низкий уровень памяти ({available_memory_mb:.2f}MB < {memory_pause_threshold}MB) в preprocess_image, пауза.")
            time.sleep(2) # Даем системе время освободить ресурсы

        processed_image = color_normalization(image_copy)
        if processed_image is None: # color_normalization могла вернуть None или оригинальное изображение
            logger.error("preprocess_image: color_normalization не вернула ожидаемое изображение.")
            return image_input.copy() # Возвращаем копию оригинала как фоллбэк

        gray_after_color_norm = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        mean_brightness_after_color_norm = np.mean(gray_after_color_norm)
        overexposure_after_color_norm = np.sum(gray_after_color_norm > 200) / gray_after_color_norm.size * 100
        
        processed_image_eq = histogram_equalization(processed_image)
        if processed_image_eq is None:
            logger.error("preprocess_image: histogram_equalization не вернула ожидаемое изображение.")
            # Продолжаем с processed_image (после color_normalization)
        else:
            processed_image = processed_image_eq


        if mean_brightness_after_color_norm < 100: # Гамма и осветление на основе яркости *после* нормализации цвета
            gamma_factor = 2.0 if mean_brightness_after_color_norm < 50 else 1.5
            if overexposure_after_color_norm > 20: # Если после нормализации был пересвет
                gamma_factor = max(1.0, gamma_factor * 0.7)
            
            inv_gamma = 1.0 / (CONFIG.get("gamma_value", 0.8) * gamma_factor)
            lookup_table_gamma = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            processed_image = cv2.LUT(processed_image, lookup_table_gamma)
            
            beta_val_lighten = (120 - mean_brightness_after_color_norm) * 0.9 
            processed_image = cv2.convertScaleAbs(processed_image, alpha=1.0, beta=beta_val_lighten)

        gray_for_contrast_step = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        current_std_dev_step = np.std(gray_for_contrast_step)
        current_overexposure_step = np.sum(gray_for_contrast_step > 200) / gray_for_contrast_step.size * 100
        current_noise_level_step = np.var(gray_for_contrast_step)
        
        adaptive_clip_limit = 0.5 # Базовый clip_limit для финального CLAHE
        if mean_brightness_after_color_norm < 50: adaptive_clip_limit = 0.7
        elif current_std_dev_step < 30 or current_noise_level_step > 1000: adaptive_clip_limit = 0.6
        elif mean_brightness_after_color_norm < 100: adaptive_clip_limit = 0.3
        if current_overexposure_step > 20: adaptive_clip_limit = min(adaptive_clip_limit * 0.8, 0.3)

        if current_std_dev_step < 30 or mean_brightness_after_color_norm < 100 or \
           current_overexposure_step > 20 or current_noise_level_step > 1000:
            
            gauss_for_contrast = cv2.GaussianBlur(gray_for_contrast_step, (0, 0), sigmaX=10)
            contrast_weight = 1.5 if current_std_dev_step < 30 else 1.2
            if current_overexposure_step > 20 or current_noise_level_step > 1000:
                contrast_weight *= 0.7
            
            gray_enhanced_contrast = cv2.addWeighted(gray_for_contrast_step, contrast_weight, gauss_for_contrast, -0.5, 0)
            gray_enhanced_contrast = np.clip(gray_enhanced_contrast, 0, 255).astype(np.uint8)

            final_clahe_clip_limit_step = adaptive_clip_limit * (0.8 if current_overexposure_step > 20 else 1.0)
            clahe_contrast_obj = cv2.createCLAHE(clipLimit=final_clahe_clip_limit_step, tileGridSize=(8, 8))
            l_channel_final_clahe = clahe_contrast_obj.apply(gray_enhanced_contrast)
            
            lab_current_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
            _, a_ch, b_ch = cv2.split(lab_current_processed)
            lab_merged_final = cv2.merge((l_channel_final_clahe, a_ch, b_ch))
            processed_image = cv2.cvtColor(lab_merged_final, cv2.COLOR_LAB2BGR)

        processed_image = cv2.fastNlMeansDenoisingColored(
            processed_image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=15
        )
        
        processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Опциональное сохранение финального обработанного изображения для отладки
        # temp_dir = CONFIG.get("temp_dir", "temp_images")
        # os.makedirs(temp_dir, exist_ok=True)
        # timestamp = int(time.time() * 1000)
        # temp_path = os.path.join(temp_dir, f"final_preprocessed_{timestamp}.png")
        # try:
        #     cv2.imwrite(temp_path, processed_image)
        #     logger.debug(f"Финальное обработанное изображение сохранено в: {temp_path}")
        # except Exception as e_write:
        #     logger.error(f"Не удалось сохранить финальное обработанное изображение {temp_path}: {e_write}")

        logger.info("Предобработка изображения (preprocess_image) завершена успешно.")
        return processed_image

    except Exception as e:
        logger.error(f"Критическая ошибка в preprocess_image: {str(e)}", exc_info=True)
        return None # Возвращаем None в случае неустранимой ошибки
    finally:
        import gc
        gc.collect()