import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Словарь для перевода названий животных
translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant": "elefante",
    "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca",
    "spider": "ragno", "squirrel": "scoiattolo"
}

# Путь к папке с отдельными папками животных
folder_path = "raw-img"

# Путь к папке, в которую будем копировать фотографии
output_folder_path = "animals_dataset"

# Создаем папку, если она не существует
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Проходимся по всем папкам в основной папке
for folder_name in os.listdir(folder_path):
    # Формируем полный путь к текущей папке
    current_folder_path = os.path.join(folder_path, folder_name)
    # Проверяем, является ли текущий элемент папкой
    if os.path.isdir(current_folder_path):
        # Получаем название животного из имени папки
        animal_name = folder_name
        # Проверяем, существует ли разметка для текущего животного
        if animal_name in translate:
            translated_name = translate[animal_name]
            # Создаем папку для текущего класса животного
            animal_folder_path = os.path.join(output_folder_path, translated_name)
            os.makedirs(animal_folder_path, exist_ok=True)
            # Проходимся по всем файлам в текущей папке
            for filename in os.listdir(current_folder_path):
                # Формируем полный путь к текущему файлу
                file_path = os.path.join(current_folder_path, filename)
                # Формируем путь к новому файлу с использованием разметки
                new_file_path = os.path.join(animal_folder_path, filename)
                # Попробуем скопировать файл и обработаем возможные ошибки
                try:
                    shutil.copy(file_path, new_file_path)
                    print(f"Скопирован файл: {file_path}")
                except Exception as e:
                    print(f"Ошибка при копировании файла {file_path}: {str(e)}")



# Создаем объект ImageDataGenerator для аугментации данных
data_generator = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Генерируем обучающие данные
train_data = data_generator.flow_from_directory(
    output_folder_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Генерируем валидационные данные
validation_data = data_generator.flow_from_directory(
    output_folder_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Выводим информацию о классах
print("Классы:", train_data.class_indices)
print("Количество классов:", len(train_data.class_indices))

# Теперь у вас есть набор данных TensorFlow, готовый для обучения модели