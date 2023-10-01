import matplotlib.pyplot as plt


def make_plot(valid_data, result_data, color_change=None):
    plt.figure(figsize=(8, 6))
    if color_change==None:
        plt.scatter(valid_data, result_data, c='b', edgecolors='k', s=100)
    else:
        plt.scatter(valid_data, result_data, c=color_change, cmap='viridis', edgecolors='k', s=100)
    plt.plot([0, 5], [0, 5], color='red', linestyle='--')  # Линия идеального соответствия (x=y)

    # Настройки графика
    plt.xlabel('Валидные значения')
    plt.ylabel('Экспериментальные значения')
    plt.title('Сравнение экспериментальных и валидных значений')
    plt.grid(True)
    plt.show()