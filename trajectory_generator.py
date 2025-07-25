import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def generate_smooth_2d_trajectory(
    total_time=100.0,            # Общая длительность траектории (секунды)
    delta_t=0.01,               # Шаг по времени для дискретизации траектории (секунды)
    max_acceleration=100.0,       # Максимальная случайная величина ускорения (м/с^2)
    accel_key_points_per_sec=30, # Количество ключевых точек ускорения в секунду
    initial_position=(0.0, 0.0), # Начальное положение (м)
    initial_velocity=(0.0, 0.0)  # Начальная скорость (м/с)
):
    """
    Генерирует случайную плавную 2D-траекторию для объекта.

    Параметры:
    - total_time (float): Общая длительность генерируемой траектории.
    - delta_t (float): Шаг по времени для дискретизации траектории. Меньше = плавнее, но медленнее.
    - max_acceleration (float): Максимальная абсолютная величина случайного ускорения
                               для контрольных точек.
    - accel_key_points_per_sec (int): Количество ключевых точек ускорения,
                                      генерируемых каждую секунду.
    - initial_position (tuple): Кортеж (x, y) начального положения объекта.
    - initial_velocity (tuple): Кортеж (vx, vy) начальной скорости объекта.

    Возвращает:
    - numpy.ndarray: Массив временных точек.
    - numpy.ndarray: Массив положений (N, 2), где N - количество точек.
    - numpy.ndarray: Массив скоростей (N, 2).
    - numpy.ndarray: Массив ускорений (N, 2).
    """

    # Вычисляем интервал между ключевыми точками ускорения
    accel_control_interval = 1.0 / accel_key_points_per_sec
    if accel_control_interval < delta_t:
        print(f"Предупреждение: Интервал ключевых точек ускорения ({accel_control_interval:.4f} с) "
              f"меньше шага дискретизации траектории ({delta_t:.4f} с). "
              f"Рекомендуется, чтобы delta_t был меньше accel_control_interval.")

    # --- 1. Генерация контрольных точек ускорения ---
    # Создаем временные точки для контрольных точек ускорения
    # Добавляем небольшой запас, чтобы сплайн покрывал весь интервал total_time
    accel_time_points = np.arange(0, total_time + accel_control_interval, accel_control_interval)

    # Генерируем случайные значения ускорения для каждой оси (X и Y) в контрольных точках
    accel_control_x = np.random.uniform(-max_acceleration, max_acceleration, len(accel_time_points))
    accel_control_y = np.random.uniform(-max_acceleration, max_acceleration, len(accel_time_points))

    # --- 2. Создание кубических сплайнов для каждой компоненты ускорения ---
    spline_accel_x = CubicSpline(accel_time_points, accel_control_x)
    spline_accel_y = CubicSpline(accel_time_points, accel_control_y)

    # --- 3. Интегрирование для получения скорости и положения ---
    # Создаем массив временных точек для итоговой траектории
    time_points = np.arange(0, total_time, delta_t)

    # Инициализируем массивы для хранения результатов (для 2D)
    positions = np.zeros((len(time_points), 2))
    velocities = np.zeros((len(time_points), 2))
    accelerations = np.zeros((len(time_points), 2))

    # Устанавливаем начальные значения
    current_position = np.array(initial_position)
    current_velocity = np.array(initial_velocity)

    positions[0] = current_position
    velocities[0] = current_velocity

    for i in range(1, len(time_points)):
        current_time = time_points[i]

        # Получаем ускорение в текущей временной точке из сплайнов
        current_acceleration = np.array([
            spline_accel_x(current_time),
            spline_accel_y(current_time)
        ])
        accelerations[i] = current_acceleration

        # Обновляем скорость: V(t) = V(t-dt) + A(t)*dt
        current_velocity += current_acceleration * delta_t
        velocities[i] = current_velocity

        # Обновляем положение: P(t) = P(t-dt) + V(t)*dt
        current_position += current_velocity * delta_t
        positions[i] = current_position

    return time_points, positions, velocities, accelerations



import numpy as np

class PointTransformer:
    """
    A class to perform normalization and inverse transformation on 2D points.

    Normalization involves two steps:
    1. Centering: Subtracting the first point from all other points.
    2. Alignment: Applying an affine rotation so that the first and second
       points lie on the positive x-axis.

    The class also provides a method to reverse these operations.
    """
    def __init__(self):
        """
        Initializes the transformer with no stored translation vector or
        rotation matrix. These will be set during the normalization process.
        """
        self.translation_vector = None
        self.rotation_matrix = None

    def normalize_coordinates(self, points):
        """
        Normalizes a set of 2D points.

        The first point will be translated to the origin (0, 0), and the
        second point will be rotated to lie on the positive x-axis.

        Args:
            points (np.ndarray): A NumPy array of shape (N, 2) representing
                                 the input points. N must be at least 2.

        Returns:
            np.ndarray: The normalized points, also of shape (N, 2).

        Raises:
            ValueError: If the input points are not N x 2 or if N < 2.
        """
        if points.shape[1] != 2:
            raise ValueError("Input points must be of dimension N x 2.")
        if points.shape[0] < 2:
            raise ValueError("At least two points are required for normalization.")

        # Store the first point for centering and subtract it from all points.
        # This moves the first point to (0, 0).
        self.translation_vector = points[0, :]
        centered_points = points - self.translation_vector

        # Determine the rotation needed to align the second point with the
        # positive x-axis. The first point is now (0, 0).
        second_point_centered = centered_points[1, :]

        # Handle the edge case where the first two points are identical.
        # In this scenario, the second centered point is also (0,0),
        # so no rotation is needed.
        if np.allclose(second_point_centered, [0, 0]):
            self.rotation_matrix = np.eye(2) # Identity matrix (no rotation)
        else:
            # Calculate the angle of the vector from (0,0) to the second point.
            angle = np.arctan2(second_point_centered[1], second_point_centered[0])
            
            # Create a 2D rotation matrix that rotates by -angle to align
            # the vector with the positive x-axis.
            self.rotation_matrix = np.array([
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle),  np.cos(-angle)]
            ])

        # Apply the rotation to the centered points.
        # We use (self.rotation_matrix @ centered_points.T).T because
        # matrix multiplication in NumPy expects (M, K) @ (K, N) -> (M, N).
        # Our points are (N, 2), so we transpose to (2, N), multiply, then transpose back.
        normalized_points = (self.rotation_matrix @ centered_points.T).T
        return normalized_points

    def inverse_transform_coordinates(self, normalized_points):
        """
        Applies the inverse transformation to a set of normalized 2D points,
        recovering their original positions.

        Args:
            normalized_points (np.ndarray): A NumPy array of shape (N, 2)
                                             representing the normalized points.

        Returns:
            np.ndarray: The original (un-normalized) points, also of shape (N, 2).

        Raises:
            ValueError: If normalization has not been performed yet, or if
                        the input normalized points are not N x 2.
        """
        if self.translation_vector is None or self.rotation_matrix is None:
            raise ValueError(
                "Normalization must be performed first to set transformation parameters."
            )
        if normalized_points.shape[1] != 2:
            raise ValueError("Input normalized points must be of dimension N x 2.")

        # Inverse rotation: For orthogonal matrices like 2D rotation matrices,
        # the inverse is simply its transpose.
        inverse_rotation_matrix = self.rotation_matrix.T
        rotated_back_points = (inverse_rotation_matrix @ normalized_points.T).T

        # Inverse translation: Add the original translation vector back to
        # restore the points to their original positions.
        original_points = rotated_back_points + self.translation_vector
        return original_points



import torch

def create_training_data_from_long_trajectory_with_normalization(
    long_trajectory: np.ndarray, 
    input_seq_len: int = 7, 
    output_seq_len: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Создает обучающие тензоры X_train и y_train из одной длинной траектории,
    применяя нормализацию к каждой паре (X_sample, y_sample).

    Каждая строка X_train будет содержать 'input_seq_len' последовательных точек
    из длинной траектории, а соответствующая строка y_train будет содержать
    'output_seq_len' точек, следующих непосредственно за точками из X_train.

    Нормализация применяется к КАЖДОЙ паре (X_sample, y_sample):
    1. X_sample центрируется путем вычитания первой точки из X_sample.
    2. Затем X_sample и y_sample поворачиваются так, чтобы первая и вторая
       точки из X_sample лежали на положительной оси Ox.

    Args:
        long_trajectory (np.ndarray): NumPy массив формы (N, 2), где N - общая
                                      длина траектории (количество точек).
                                      N должно быть достаточно большим, чтобы
                                      сформировать хотя бы одну пару (N >= input_seq_len + output_seq_len).
        input_seq_len (int): Длина входной последовательности (количество точек в X). По умолчанию 7.
        output_seq_len (int): Длина выходной последовательности (количество точек в Y). По умолчанию 3.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Кортеж из двух тензоров PyTorch:
            - X_train (torch.Tensor): Тензор формы (num_samples, input_seq_len, 2).
            - y_train (torch.Tensor): Тензор формы (num_samples, output_seq_len, 2).
            num_samples - это количество обучающих пар, которое может быть сформировано.

    Raises:
        ValueError: Если длина long_trajectory недостаточна для формирования хотя бы одной пары.
    """
    
    if long_trajectory.ndim != 2 or long_trajectory.shape[1] != 2:
        raise ValueError("long_trajectory должна быть 2D NumPy массивом формы (N, 2).")

    min_required_len = input_seq_len + output_seq_len
    if long_trajectory.shape[0] < min_required_len:
        raise ValueError(
            f"Длина long_trajectory ({long_trajectory.shape[0]} точек) слишком мала. "
            f"Требуется как минимум {min_required_len} точек для формирования одной пары "
            f"(input_seq_len={input_seq_len}, output_seq_len={output_seq_len})."
        )
    
    # Check if input_seq_len is at least 2 for rotation logic
    if input_seq_len < 2:
        raise ValueError("input_seq_len must be at least 2 for normalization (to define orientation).")


    normalized_X_list = []
    normalized_y_list = []

    # Итерируемся по длинной траектории, создавая скользящие окна
    for i in range(long_trajectory.shape[0] - min_required_len + 1):
        x_sequence_original = long_trajectory[i : i + input_seq_len]
        y_sequence_original = long_trajectory[i + input_seq_len : i + input_seq_len + output_seq_len]

        # Создаем новый экземпляр трансформера для каждой обучающей пары
        # Это важно, так как параметры нормализации (сдвиг и поворот)
        # зависят от первых двух точек текущего окна X.
        transformer = PointTransformer()

        # Нормализуем X-последовательность
        # Внутри normalize_coordinates сохраняются векторы сдвига и поворота.
        x_sequence_normalized = transformer.normalize_coordinates(x_sequence_original)
        
        # Применяем ТЕ ЖЕ самые параметры сдвига и поворота к y-последовательности
        # Сначала сдвигаем y_sequence
        y_sequence_centered = y_sequence_original - transformer.translation_vector
        # Затем поворачиваем y_sequence
        y_sequence_normalized = (transformer.rotation_matrix @ y_sequence_centered.T).T

        normalized_X_list.append(x_sequence_normalized)
        normalized_y_list.append(y_sequence_normalized)

    # Преобразуем списки NumPy массивов в тензоры PyTorch
    X_train = torch.tensor(np.array(normalized_X_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(normalized_y_list), dtype=torch.float32)

    return X_train, y_train
