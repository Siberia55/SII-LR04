import argparse
import json
import logging
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Подсчёт объектов выбранного класса в видео с помощью YOLO."
    )
    parser.add_argument("--source", required=True, help="Путь к видеофайлу")
    parser.add_argument("--class", dest="class_name", default="handbag",
                        help="Название класса COCO (по умолчанию handbag)")
    parser.add_argument("--stride", type=int, default=15,
                        help="Обрабатывать каждый N-й кадр (по умолчанию 15)")
    parser.add_argument("--model", default="yolo26n.pt",
                        help="Модель YOLO (по умолчанию yolo26n.pt)")
    parser.add_argument("--output", default="output",
                        help="Каталог результатов (по умолчанию output)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Порог уверенности детекции (по умолчанию 0.25)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stride < 1:
        raise SystemExit("Ошибка: --stride должен быть не меньше 1.")

    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("Ошибка: --conf должен быть в диапазоне от 0 до 1.")

    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"Видеофайл не найден: {source}")

    output_dir = Path(args.output)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    logger.info("Загрузка модели: %s", args.model)
    model = YOLO(args.model)

    names = model.names
    class_items = names.items() if isinstance(names, dict) else enumerate(names)
    class_ids = [
        class_id for class_id, name in class_items
        if name.lower() == args.class_name.lower()
    ]

    if not class_ids:
        raise SystemExit(
            f"Класс '{args.class_name}' отсутствует в модели. "
            f"Доступные классы: {names}"
        )

    class_id = class_ids[0]
    logger.info("Класс: %s (ID %s)", args.class_name, class_id)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {source}")

    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections_per_frame = []
    frame_index = 0
    frames_processed = 0
    frames_saved = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Индексы кадров начинаются с нуля: 0, stride, 2*stride...
            if frame_index % args.stride == 0:
                results = model.predict(
                    source=frame,
                    classes=[class_id],
                    conf=args.conf,
                    verbose=False,
                )

                result = results[0]
                count = len(result.boxes) if result.boxes is not None else 0
                detections_per_frame.append(count)
                frames_processed += 1

                # Сохраняем кадр с рамками и подписями YOLO.
                annotated = result.plot()
                frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"

                if not cv2.imwrite(str(frame_path), annotated):
                    logger.warning("Не удалось сохранить кадр: %s", frame_path)
                else:
                    frames_saved += 1

                if frames_processed % 10 == 0:
                    logger.info(
                        "Обработано: %d; пропущено: %d",
                        frames_processed,
                        frame_index + 1 - frames_processed,
                    )

            frame_index += 1
    finally:
        cap.release()

    elapsed = time.perf_counter() - started
    actual_frames_total = frame_index
    frames_skipped = actual_frames_total - frames_processed

    mean_count = (
        sum(detections_per_frame) / len(detections_per_frame)
        if detections_per_frame else 0.0
    )
    max_count = max(detections_per_frame, default=0)

    report = {
        "source": str(source),
        "class_name": args.class_name,
        "stride": args.stride,
        "frames_total": actual_frames_total,
        "frames_processed": frames_processed,
        "frames_skipped": frames_skipped,
        "detections_per_frame_mean": round(mean_count, 3),
        "detections_per_frame_max": max_count,
        "unique_tracks": None,
        "model_version": args.model,
        "elapsed_sec": round(elapsed, 3),
        "frames_saved": frames_saved,
        "confidence_threshold": args.conf,
        "tracker_used": False,
        "tracker_limitation": (
            "Трекер не используется: программа считает детекции на каждом "
            "обработанном кадре, но не связывает один и тот же объект "
            "между кадрами. Поэтому число детекций не равно числу уникальных сумок."
        ),
    }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Всего кадров: %d", actual_frames_total)
    logger.info("Обработано: %d", frames_processed)
    logger.info("Пропущено: %d", frames_skipped)
    logger.info("Среднее число сумок на кадр: %.3f", mean_count)
    logger.info("Максимум сумок на кадре: %d", max_count)
    logger.info("Сохранено кадров с рамками: %d", frames_saved)
    logger.info("Время выполнения: %.3f сек.", elapsed)
    logger.info("Отчёт: %s", report_path)


if __name__ == "__main__":
    main()
