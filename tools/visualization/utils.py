def generate_rgb_gradient(n):
    if n <= 0:
        return []

    gradient = []

    # 红->绿 和 绿->蓝 的步数
    steps_rg = (n + 1) // 2
    steps_gb = n // 2

    # 红 -> 绿
    for i in range(steps_rg):
        ratio = i / max(1, steps_rg - 1)
        gradient.append((
            round(1.0 - ratio, 5),
            round(ratio, 5),
            0.0
        ))

    # 绿 -> 蓝
    for i in range(steps_gb):
        ratio = (i + 1) / max(1, steps_gb)
        gradient.append((
            0.0,
            round(1.0 - ratio, 5),
            round(ratio, 5)
        ))

    return gradient


def generate_rainbow_gradient(n):
    if n <= 0:
        return []

    # ROYGBIV 关键颜色：红、橙、黄、绿、蓝、靛、紫
    key_colors = [
        (1.0, 0.0, 0.0),   # Red
        (1.0, 0.5, 0.0),   # Orange
        (1.0, 1.0, 0.0),   # Yellow
        (0.0, 1.0, 0.0),   # Green
        (0.0, 0.0, 1.0),   # Blue
        (0.3, 0.0, 0.5),   # Indigo
        (0.5, 0.0, 1.0),   # Violet
    ]

    gradient = []
    num_segments = len(key_colors) - 1
    colors_per_segment = (n - 1) // num_segments
    remainder = (n - 1) % num_segments

    for i in range(num_segments):
        start = key_colors[i]
        end = key_colors[i + 1]
        steps = colors_per_segment + (1 if i < remainder else 0)

        for j in range(steps + 1):
            ratio = j / steps
            r = round(start[0] + (end[0] - start[0]) * ratio, 5)
            g = round(start[1] + (end[1] - start[1]) * ratio, 5)
            b = round(start[2] + (end[2] - start[2]) * ratio, 5)
            gradient.append((r, g, b))

    # 确保长度正好是 n
    if len(gradient) > n:
        gradient = gradient[:n]
    elif len(gradient) < n:
        last_color = gradient[-1]
        while len(gradient) < n:
            gradient.append(last_color)

    return gradient


if __name__ == '__main__':
    print(generate_rgb_gradient(3))
