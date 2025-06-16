# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Устанавливаем seed для воспроизводимости результатов
np.random.seed(42)


# 1. Создаем синтетический набор данных для многоклассовой классификации
def generate_data(n_samples, n_classes):
    """
    Функция для генерации синтетического набора данных
    с тремя классами и двумя признаками
    """
    points = []
    labels = []

    # Генерируем точки для каждого класса
    for i in range(n_samples):
        # Выбираем случайный класс
        class_idx = np.random.randint(0, n_classes)

        # Создаем точку с координатами, зависящими от класса
        if class_idx == 0:
            # Класс 0 - кластер вокруг (2, 2)
            x = 2 + np.random.randn() * 0.5
            y = 2 + np.random.randn() * 0.5
        elif class_idx == 1:
            # Класс 1 - кластер вокруг (6, 2)
            x = 6 + np.random.randn() * 0.5
            y = 2 + np.random.randn() * 0.5
        else:
            # Класс 2 - кластер вокруг (4, 6)
            x = 4 + np.random.randn() * 0.5
            y = 6 + np.random.randn() * 0.5

        points.append([x, y])
        labels.append(class_idx)

    return np.array(points), np.array(labels)


# Генерируем данные
n_samples = 300
n_classes = 3
X, y = generate_data(n_samples, n_classes)

# 2. Разделяем данные на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразуем метки классов в формат one-hot encoding
y_train_one_hot = to_categorical(y_train, n_classes)
y_test_one_hot = to_categorical(y_test, n_classes)

print(f"Общее количество данных: {n_samples}")
print(f"Количество классов: {n_classes}")
print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# 3. Построение стандартной нейросетевой модели
model = Sequential([
    # Входной слой и первый скрытый слой
    Dense(10, activation='relu', input_shape=(2,)),
    # Второй скрытый слой
    Dense(8, activation='relu'),
    # Выходной слой с функцией активации softmax для многоклассовой классификации
    Dense(n_classes, activation='softmax')
])

# Компилируем модель
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Функция потерь для многоклассовой классификации
    metrics=['accuracy']  # Метрика качества - точность
)

# Выводим структуру модели
print("Структура модели:")
model.summary()

# 4. Обучение модели
history = model.fit(
    X_train, y_train_one_hot,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test_one_hot),
    verbose=1
)

# 5. Оценка качества модели
# Оцениваем модель на тестовых данных
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Потери на тестовой выборке: {loss:.4f}")
print(f"Точность на тестовой выборке: {accuracy:.4f}")

# Получаем предсказания для тестовой выборки
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Вычисляем метрики качества
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# Вычисляем матрицу ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(conf_matrix)

# 6. Визуализация результатов
# Визуализируем данные и границы решений
plt.figure(figsize=(12, 10))

# 6.1. Визуализация процесса обучения
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Точность (обучение)')
plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
plt.title('Динамика точности')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Потери (обучение)')
plt.plot(history.history['val_loss'], label='Потери (валидация)')
plt.title('Динамика функции потерь')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()

# 6.2. Визуализация данных и границ принятия решений
plt.subplot(2, 2, 3)
# Визуализируем точки из тестовой выборки
colors = ['red', 'green', 'blue']
for i in range(n_classes):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
                color=colors[i], label=f'Класс {i}', alpha=0.7)

plt.title('Тестовые данные')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()

# 6.3. Визуализация правильных и неправильных классификаций
plt.subplot(2, 2, 4)
# Визуализируем правильные прогнозы
correct = y_pred == y_test
plt.scatter(X_test[correct, 0], X_test[correct, 1],
            c='green', marker='o', alpha=0.7, label='Верно')
# Визуализируем неправильные прогнозы
incorrect = ~correct
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1],
            c='red', marker='x', s=100, alpha=0.7, label='Неверно')

plt.title('Результаты классификации')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()

# Добавляем границы принятия решений (для этого нужно создать сетку)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Получаем предсказания для каждой точки сетки
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

# Рисуем контуры границ принятия решений
plt.contourf(xx, yy, Z, alpha=0.2, colors=['red', 'green', 'blue'])

plt.tight_layout()
plt.savefig('classification_results.png')
plt.show()

print("\nЗадача многоклассовой классификации успешно решена!")