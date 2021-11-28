# samara-ai-hack
Хакатон в Самаре 26 ноября - 28 ноября.

Для запуска необходимо
- clone repo
- `pip install -r requirements.txt`
- download weights from yandex drive [обученные веса моделей](https://disk.yandex.ru/d/PIzf6TVLlV62LA)
- cp weights_epoch_77.h5 ./
- cp best1.pt Detector/
- weights_epoch_63.h5

- Для предсказания использовать notebooks: make_detector_csv.ipynb, make_princess_csv.ipynb.
- Для тренировки модели
В качестве детектора использован yolov5m, обученный на 2 классах: леопард, тигр.
В качестве классификатора тигрицы принцессы использовались модели resnet50, resnet152 с выходом в виде embedding_vector, и дальнейшим обучением kNN классификатора.
Для обучения модели классификации "Принцесса - Не Принцесса" необходимо запустить
`train_resnet50.py $princess_dataset_labelled_path $odir`, указав пути к файлам

Файлы с предсказаниями
- labels.csv - предсказание модели "Тигры, Леопарды, другие", 1 - тигр, 2 - леопард, 3 - нет тигра, нет леопарда
- princess.csv - предсказание модели "Принцесса - Не Принцесса", 0 - не Принцесса, 1 - Принцесса

