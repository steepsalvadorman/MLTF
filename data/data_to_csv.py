import os
import pandas as pd

# Ruta al directorio 'data'
data_path = "./"

# Verifica si la ruta existe
if not os.path.exists(data_path):
    raise FileNotFoundError(f"La ruta especificada no existe: {data_path}")

data = []
print(f"Contenido de {data_path}: {os.listdir(data_path)}")  # Lista los elementos en data_path
for level in os.listdir(data_path):
    level_dir = os.path.join(data_path, level)
    if os.path.isdir(level_dir) and level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
        print(f"Procesando nivel: {level}")
        print(f"Contenido de {level_dir}: {os.listdir(level_dir)}")  # Lista los elementos en cada subdirectorio
        for text_file in os.listdir(level_dir):
            file_path = os.path.join(level_dir, text_file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                data.append({"text": text, "label": level})
                print(f"Archivo leído: {file_path}")
            except UnicodeDecodeError:
                print(f"Error decoding {file_path}")
            except Exception as e:
                print(f"Error al leer {file_path}: {e}")

print(f"Número total de documentos procesados: {len(data)}")

if data:
    df = pd.DataFrame(data)
    output_path = os.path.join(data_path, "cefr_leveled_texts.csv")
    df.to_csv(output_path, index=False)
    print(f"Datos guardados en {output_path}")
else:
    print("No se encontraron documentos para procesar.")
