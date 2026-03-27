from ultralytics import YOLO

# Загружаем предобученную модель для сегментации
model = YOLO('yolov8n-seg.pt')

# Новое обучение
results = model.train(
    data='../../dataset/dataset_for_seal_detector/data.yaml',  # Исправленный путь
    epochs=50,  # Увеличиваем количество эпох
    imgsz=640,
    batch=8,  # Уменьшаем batch size для стабильности
    device='cpu',
    augment=True,  # Включаем аугментацию
    mosaic=0.5,  # Включаем mosaic
    mixup=0.1,  # Включаем mixup
    copy_paste=0.1,  # Включаем copy_paste
    auto_augment='randaugment',  # Включаем auto augmentation
    erasing=0.1,  # Включаем random erasing
    hsv_h=0.015,  # Небольшие изменения HSV
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,  # Горизонтальные перевороты
    flipud=0.0,  # Вертикальные перевороты отключаем для документов
    scale=0.5,  # Масштабирование
    translate=0.1,  # Сдвиги
    shear=0.0,  # Сдвиги отключаем для документов
    perspective=0.0,  # Перспективу отключаем для документов
    project='runs/segment',  # Относительный путь от текущей папки
    name='train',  # Стандартное имя
    save_period=10,  # Сохраняем каждые 10 эпох
    patience=15,  # Early stopping после 15 эпох без улучшений
    verbose=True
) 