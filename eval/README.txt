ПРОВЕРКА И ОБУЧЕНИЕ (как получены цифры в «Разборе проекта»)

Своих записей с камер у нас нет, поэтому всё делалось на открытых данных AUTO.RIA
(nomeroff.net.ua, лицензия CC BY 4.0). Нужен Python 3 + numpy, opencv-python,
onnxruntime; для обучения сети углов ещё torch (CPU) и onnx.

ДАННЫЕ (качаются только нужные файлы из больших zip — rzip.py читает zip по сети кусками)
  Разметку углов берём из того же архива autoriaNumberplateDataset-2023-03-06:
  train/via_region_data.json и val/via_region_data.json
  (сохранить как train_via_region_data.json и val_via_region_data.json).
  fetch2.py scenes train 0 6000 train_sc   — 6000 фото для обучения (окно вокруг номера)
  fetch2.py scenes val 700 1500 val_h      — 800 фото для итоговой проверки
                                             (не участвовали ни в настройке, ни в обучении)
  fetch2.py ocr train 8000 ocrtrain        — вырезки номеров для вклейки при обучении
  fetch2.py ocr test 3000 ocrtest          — вырезки номеров с известным текстом для проверки

ВАРИАНТ 2: СЕТЬ УГЛОВ
  synth_lib.py      — общая «фабрика» снимков: вклеить номер в 4 размеченных угла,
                      наклонить кадр (roll / side / both), уменьшить, положить в кадр
                      размером с камеру, сжать JPEG.
  gen_corner_ds.py  — обучающие примеры: на каждый снимок запускается НАСТОЯЩИЙ детектор
                      (detpy.py), сохраняется область вокруг его рамки и правильные углы.
                      Запуск в 2 потока: gen_corner_ds.py 0 2 3 и gen_corner_ds.py 1 2 3.
  train_corners.py  — обучение (30 эпох ≈ 25 мин на 2 ядрах) и выгрузка corners.onnx.
                      Положить в models/corners.onnx и пересобрать (или рядом с
                      программой в models\corners.onnx, или --corners файл.onnx).

ПРОВЕРКА
  gen_eval.py       — 2400 проверочных снимков (сцены val_h + вырезки ocrtest):
                      plain — почти прямо, roll — поворот 5–15°, side — вид сбоку.
  gen_real.py       — настоящие таблички без вклейки (сцены val_h): эталон — уверенное
                      прочтение крупного снимка без наклона, проверяются наклонённые копии.
  Прогон:  nomerok evalh -o out --no-open   (по умолчанию --rectify auto: все вырезки)
  evals2.py out evalh/labels.json   — точность: обычная / +углы по яркости / +сеть /
                      все три / оракул; сколько стало хуже и лучше; ошибка углов.
  evals_real.py out realh/labels.json — то же на настоящих табличках.

ВАРИАНТ 1 (0.2, по яркости): gen_synth.py, evals.py, rule.py, bias.py,
measure_layout.py — как настраивались пороги и поправки find_corners.
