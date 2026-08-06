best.pt y last.pt de esta carpeta NO se comitean (siguen ignorados por la
regla *.pt de .gitignore) -- pesarian 22MB cada uno y duplicarian lo que ya
esta comiteado.

El modelo final resultante de este entrenamiento (identico a este best.pt)
ya esta en models/yolov8_finetuned_armando.pt, que es el que usa el resto
del pipeline (config.YOLO_TRAINED). Si necesitas los pesos, usa ese archivo.

El resto de esta carpeta (curvas de entrenamiento, matriz de confusion,
results.csv, batches de muestra) si esta comiteado normalmente.
