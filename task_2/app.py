import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import matplotlib.pyplot as plt
# configure page
st.set_page_config(
    page_title="Лабораторна робота 2 - Нейронні мережі",
    layout="wide",
)
# hide anchor
st.markdown(
    """
    <style>
    .header-anchor, [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# базові функції обробки та генерації зображень

def generate_digit(symbol: str, noise: float = 0.0) -> Image.Image:
    # генерує графічний образ 150x150, відцентрований під сітку 5x5.
    size = 150
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    if symbol == "0":
        draw.rounded_rectangle([25, 15, 125, 135], radius=25, outline=0, width=22)
    elif symbol == "1":
        draw.line([(75, 18), (75, 132)], fill=0, width=22)
        draw.line([(45, 45), (75, 18)], fill=0, width=20)
        draw.line([(35, 132), (115, 132)], fill=0, width=22)
    elif symbol == "2":
        draw.arc([25, 15, 125, 80], start=180, end=20, fill=0, width=22)
        draw.line([(120, 55), (30, 130)], fill=0, width=22)
        draw.line([(25, 130), (125, 130)], fill=0, width=22)

    arr = np.array(img)
    # додаємо випадковий шум
    if noise > 0:
        mask = np.random.rand(*arr.shape) < noise
        arr[mask] = 0
    return Image.fromarray(arr)


def extract_features(image: Image.Image, rows: int = 5, cols: int = 5, invert: bool = False):
    """
    Вилучає ознаки з малюнка:
    - абсолютний вектор (кількість темних пікселів у зоні);
    - нормований вектор (сума = 1.0);
    - біполярний вектор {-1, +1} для Хопфілда і Хеммінга.
    """
    gray = image.convert("L")
    arr = np.array(gray)
    binary = arr > 128 if invert else arr < 128

    h, w = binary.shape
    row_step, col_step = h / rows, w / cols

    abs_vec = []
    # calculate absolute vector by counting dark pixels in each cell
    for r in range(rows):
        for c in range(cols):
            r_start, r_end = int(r * row_step), int((r + 1) * row_step)
            c_start, c_end = int(c * col_step), int((c + 1) * col_step)
            cell = binary[r_start:r_end, c_start:c_end]
            abs_vec.append(int(np.sum(cell)))

    abs_vec = np.array(abs_vec, dtype=float)
    total = np.sum(abs_vec)
    # calculate normalized vector
    norm_vec = abs_vec / total if total > 0 else np.zeros_like(abs_vec)

    # calculate bipolar vector based on thresholding
    active_mean = np.mean(abs_vec[abs_vec > 0]) if np.any(abs_vec > 0) else 0.0
    threshold = max(25.0, 0.35 * active_mean)
    bipolar_vec = np.where(abs_vec > threshold, 1.0, -1.0)

    return abs_vec, norm_vec, bipolar_vec


def draw_grid(image: Image.Image, rows: int = 5, cols: int = 5) -> Image.Image:
    """Накладає червону сітку на малюнок."""
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    w, h = img_rgb.size
    for r in range(1, rows):
        y = int(r * (h / rows))
        draw.line([(0, y), (w, y)], fill="red", width=2)
    for c in range(1, cols):
        x = int(c * (w / cols))
        draw.line([(x, 0), (x, h)], fill="red", width=2)
    return img_rgb


def bipolar_to_image(bipolar_vec: np.ndarray, rows: int = 5, cols: int = 5) -> Image.Image:
    """Візуалізує біполярний вектор у вигляді матриці пікселів із сіткою."""
    grid = bipolar_vec.reshape((rows, cols))
    pixel_arr = np.where(grid == 1.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixel_arr, mode="L").resize((150, 150), resample=Image.NEAREST)
    return draw_grid(img, rows, cols)


# interface
st.title("Лабораторна робота №2: Моделювання нейронних мереж", anchor=False)

tab_perc, tab_hopfield, tab_hamming = st.tabs([
    "1. Перцептрон Розенблата (2 класи)",
    "2. Мережа Хопфілда (3 класи)",
    "3. Мережа Хеммінга (3 класи)",
])

# Вкладка 1: Перцептрон Розенблата
with tab_perc:
    st.header("1. Перцептрон Розенблата для класифікації двох класів", anchor=False)
    st.write("Розпізнавання класів: **Клас '0' ($d = -1$)** та **Клас '1' ($d = +1$)** за нормованими векторами ознак.")
    # ініціалізація ваг та зсуву
    if "p_weights" not in st.session_state:
        np.random.seed(42)
        st.session_state["p_weights"] = np.random.uniform(0.01, 0.1, size=25)
        st.session_state["p_bias"] = float(np.random.uniform(0.01, 0.1))
        st.session_state["p_history"] = []

    p_c1, p_c2 = st.columns([1, 2])

    with p_c1:
        st.subheader("Параметри навчання")
        lr = st.slider("Швидкість навчання (r)", 0.01, 0.5, 0.1, 0.01)
        epochs = st.number_input("Кількість епох", 1, 100, 25)

        # формування навчальної послідовності з 10 зразків (по 5 на кожен клас)
        train_samples = []
        for i in range(5):
            abs_0, norm_0, _ = extract_features(generate_digit("0", noise=0.012 * (i + 1)))
            train_samples.append((norm_0, -1.0, "0", abs_0))

            abs_1, norm_1, _ = extract_features(generate_digit("1", noise=0.012 * (i + 1)))
            train_samples.append((norm_1, 1.0, "1", abs_1))
        # вивід ознакових векторів навчальної послідовності
        with st.expander("Ознакові вектори навчальної послідовності (Завдання 1.2)"):
            for idx, (x_vec, target, label, abs_v) in enumerate(train_samples):
                st.write(f"**Зразок {idx+1} (Клас '{label}', d={target})**")
                c1, c2 = st.columns(2)
                c1.caption(f"Абсолютний: `{np.array2string(abs_v.astype(int), separator=', ')}`")
                c2.caption(f"Нормований: `{np.array2string(np.round(x_vec, 3), separator=', ')}`")
        # навчання перцептрона
        if st.button("Навчити перцептрон"):
            w = st.session_state["p_weights"].copy()
            b = st.session_state["p_bias"]
            history = []
            for _ in range(int(epochs)):
                errors = 0
                for x_vec, target, _, _ in train_samples:
                    net = np.dot(w, x_vec) - b
                    y = 1.0 if net > 0 else -1.0
                    if y != target:
                        w += lr * (target - y) * x_vec
                        b -= lr * (target - y)
                        errors += 1
                history.append(errors)
                if errors == 0:
                    break
            st.session_state["p_weights"] = w
            st.session_state["p_bias"] = b
            st.session_state["p_history"] = history
            st.success(f"Навчання завершено за {len(history)} епох! Помилок: {history[-1]}")

        if st.button("Скинути ваги"):
            np.random.seed(None)
            st.session_state["p_weights"] = np.random.uniform(0.01, 0.1, size=25)
            st.session_state["p_bias"] = float(np.random.uniform(0.01, 0.1))
            st.session_state["p_history"] = []
            st.rerun()

    with p_c2:
        st.subheader("Збіжність та ваги")
        # графік помилок по епохах
        if st.session_state["p_history"]:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(st.session_state["p_history"], marker='o', color='crimson')
            ax.set_ylabel("Помилки")
            ax.set_xlabel("Епоха")
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
        st.write(f"**Зсув (bias b):** `{st.session_state['p_bias']:.4f}`")
        with st.expander("Вектор синаптичних ваг W (25 компонент)"):
            st.code(np.array2string(np.round(st.session_state["p_weights"], 4), max_line_width=75))

    st.divider()
    st.subheader("Тестування розпізнавання перцептроном")
    tc1, tc2 = st.columns([1, 2])
    with tc1:
        u_p = st.file_uploader("Завантажте образ", type=["bmp", "png", "jpg", "jpeg"], key="u_p")
        choice_p = st.radio("Або оберіть контрольний зразок:", ["0", "1"], horizontal=True)
        noise_p = st.slider("Шум образу", 0.0, 0.06, 0.01, step=0.01)

        test_img_p = Image.open(u_p) if u_p is not None else generate_digit(choice_p, noise=noise_p)
        abs_t, norm_t, _ = extract_features(test_img_p)

        st.caption(f"Абсолютний вектор: `{np.array2string(abs_t.astype(int), separator=', ')}`")
        st.caption(f"Нормований вектор: `{np.array2string(np.round(norm_t, 4), separator=', ')}`")

    with tc2:
        st.image(draw_grid(test_img_p), caption="Вхідний образ із сіткою", width=130)
        net_val = np.dot(st.session_state["p_weights"], norm_t) - st.session_state["p_bias"]
        y_out = 1.0 if net_val > 0 else -1.0
        if y_out == 1.0:
            st.success("Результат: **Клас '1'** (активація +1)")
        else:
            st.info("Результат: **Клас '0'** (активація -1)")


# Вкладка 2: Мережа Хопфілда
with tab_hopfield:
    st.header("2. Штучна нейронна мережа Хопфілда (Автоасоціативна пам'ять)", anchor=False)
    st.write("Мережа зберігає 3 еталони ('0', '1', '2') розмірністю 25 біполярних бітів ({-1, +1}) та ітераційно відновлює образ.")

    classes_h = ["0", "1", "2"]
    cols_h = st.columns(len(classes_h))
    exemplars_hopfield = {}
    # відображення еталонних образів та їхніх біполярних векторів
    for i, cls in enumerate(classes_h):
        with cols_h[i]:
            st.write(f"**Еталон '{cls}'**")
            img_h = generate_digit(cls)
            st.image(draw_grid(img_h), width=110)
            abs_h, _, bip_vec = extract_features(img_h)
            exemplars_hopfield[cls] = bip_vec
            with st.expander("Вектори еталону"):
                st.caption(f"Абсолютний: `{np.array2string(abs_h.astype(int), separator=', ')}`")
                st.caption(f"Біполярний: `{np.array2string(bip_vec.astype(int), separator=', ')}`")

    # формування матриці ваг за правилом Хебба
    n_dim = 25
    W_hopfield = np.zeros((n_dim, n_dim))
    for vec in exemplars_hopfield.values():
        W_hopfield += np.outer(vec, vec)
    np.fill_diagonal(W_hopfield, 0.0)

    with st.expander("Переглянути вагову матрицю зв'язків W (25x25)"):
        st.dataframe(W_hopfield.astype(int))

    st.divider()
    st.subheader("Відновлення зашумленого образу")
    hc1, hc2 = st.columns([1, 2])

    with hc1:
        test_cls_h = st.selectbox("Образ для перевірки:", classes_h, key="sel_hop")
        noise_level_h = st.slider("Шум графічного образу", 0.0, 0.08, 0.02, step=0.01)

        raw_noisy_img = generate_digit(test_cls_h, noise=noise_level_h)
        _, _, in_bipolar_h = extract_features(raw_noisy_img)

        st.image(draw_grid(raw_noisy_img), caption="Вхідний зашумлений малюнок", width=125)
        st.image(bipolar_to_image(in_bipolar_h), caption="Дискретизований вхід Y(0)", width=125)

    with hc2:
        if st.button("Запустити релаксацію Хопфілда"):
            y_curr = in_bipolar_h.copy()
            history_states = [y_curr.copy()]
            for it in range(35):
                changed = False
                for node in range(n_dim):
                    s = np.dot(W_hopfield[node], y_curr)
                    if s > 0:
                        new_val = 1.0
                    elif s < 0:
                        new_val = -1.0
                    else:
                        new_val = y_curr[node]

                    if new_val != y_curr[node]:
                        y_curr[node] = new_val
                        changed = True
                history_states.append(y_curr.copy())
                # якщо стан не змінився, зупиняємо ітерації
                if not changed:
                    break

            st.write(f"Мережа стабілізувалася за **{len(history_states)-1} ітерацій**!")
            st.image(bipolar_to_image(y_curr), caption="Відновлений образ Y(t)", width=140)
            # визначення, який клас відновлено
            rec = None
            for cls, e_vec in exemplars_hopfield.items():
                if np.array_equal(y_curr, e_vec):
                    rec = cls
                    break
                elif np.array_equal(y_curr, -e_vec):
                    rec = f"Інверсний '{cls}'"
                    break

            if rec:
                st.success(f"Мережа успішно згадала та відновила: **Клас '{rec}'**")
            else:
                st.warning("Мережа потрапила у хибний стан (spurious state). Зменшіть рівень шуму образу.")


# Вкладка 3: Мережа Хеммінга
with tab_hamming:
    st.header("3. Штучна нейронна мережа Хеммінга", anchor=False)
    st.write("Двошаровий класифікатор: Перший шар розраховує ступінь схожості, другий (MAXNET) знаходить переможця.")

    m_classes = ["0", "1", "2"]
    m = len(m_classes)
    n = 25
    # ініціалізація ваг першого шару Хеммінга
    W_hamming = np.zeros((m, n))
    for k, cls in enumerate(m_classes):
        W_hamming[k] = exemplars_hopfield[cls] / 2.0
    b_hamming = np.full(m, n / 2.0)

    with st.expander("Вагові коефіцієнти першого шару Хеммінга"):
        st.dataframe(W_hamming)
        st.write(f"Пороги нейронів: `b_k = {n/2.0}`")

    st.divider()
    st.subheader("Класифікація вхідного образу")
    ham_c1, ham_c2 = st.columns([1, 2])

    with ham_c1:
        u_ham = st.file_uploader("Завантажте образ", type=["bmp", "png", "jpg", "jpeg"], key="u_ham")
        choice_ham = st.selectbox("Або виберіть еталон:", m_classes, key="s_ham")
        noise_ham = st.slider("Шум образу", 0.0, 0.08, 0.02, step=0.01, key="n_ham")

        in_img_ham = Image.open(u_ham) if u_ham is not None else generate_digit(choice_ham, noise=noise_ham)
        abs_ham, _, in_vec_ham = extract_features(in_img_ham)

        st.image(draw_grid(in_img_ham), caption="Досліджуваний образ", width=125)
        with st.expander("Вектори вхідного образу"):
            st.caption(f"Абсолютний: `{np.array2string(abs_ham.astype(int), separator=', ')}`")
            st.caption(f"Біполярний: `{np.array2string(in_vec_ham.astype(int), separator=', ')}`")

    with ham_c2:
        # розрахунок виходів першого шару Хеммінга
        s1 = np.dot(W_hamming, in_vec_ham) + b_hamming
        y_layer1 = s1.copy()

        st.write("### Виходи першого шару (оцінка схожості):")
        cols_l1 = st.columns(m)
        for idx, cls in enumerate(m_classes):
            cols_l1[idx].metric(f"Клас '{cls}'", f"{y_layer1[idx]:.1f}")
        # запуск MAXNET для визначення переможця
        eps = 0.25
        y_maxnet = y_layer1.copy()
        history_maxnet = [y_maxnet.copy()]

        for step in range(25):
            total_sum = np.sum(y_maxnet)
            # оновлення виходів MAXNET
            new_y = np.maximum(0.0, y_maxnet - eps * (total_sum - y_maxnet))
            history_maxnet.append(new_y.copy())
            # перевірка на зупинку якщо залишився один активний нейрон або всі значення близькі до попередніх
            if np.count_nonzero(new_y) <= 1 or np.allclose(new_y, y_maxnet):
                y_maxnet = new_y
                break
            y_maxnet = new_y
        # визначення переможця
        winner_idx = int(np.argmax(y_maxnet))
        winner_class = m_classes[winner_idx]

        st.write("### Результат конкуренції (MAXNET):")
        st.success(f"Переможець: **Клас '{winner_class}'** за {len(history_maxnet)-1} ітерацій!")
        # графік динаміки конкуренції в шарі MAXNET
        fig_h, ax_h = plt.subplots(figsize=(6, 2.8))
        h_arr = np.array(history_maxnet)
        for idx, cls in enumerate(m_classes):
            ax_h.plot(h_arr[:, idx], label=f"Клас '{cls}'", linewidth=2)
        ax_h.set_title("Динаміка конкуренції в шарі MAXNET")
        ax_h.set_xlabel("Ітерація")
        ax_h.set_ylabel("Активація")
        ax_h.legend()
        ax_h.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig_h)