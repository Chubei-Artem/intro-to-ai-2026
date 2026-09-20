import io
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

# Налаштування сторінки Streamlit та кастомні стилі
st.set_page_config(
    page_title="Лабораторна робота 1 - Система розпізнавання",
    layout="wide",
)

# Приховуємо автоматичні іконки посилань біля заголовків для чистого вигляду UI
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

# Модуль 1: Допоміжні функції аналізу та візуалізації
def extract_features(
    image: Image.Image,
    rows: int = 5,
    cols: int = 5,
    invert_colors: bool = False,
):
    # Виконує бінаризацію зображення, розбиття регулярною сіткою rows x cols
    # та обчислює абсолютний і нормований вектори ознак.
    # Переведення у відтінки сірого (Grayscale)
    gray = image.convert("L")
    arr = np.array(gray)

    # Бінаризація: визначаємо, які пікселі належать об'єкту (малюнку)
    if invert_colors:
        # Світлий об'єкт (значення > 128) на темному тлі
        binary = arr > 128
    else:
        # Темний об'єкт (значення < 128) на світлому тлі (стандарт для BMP/паперу)
        binary = arr < 128

    h, w = binary.shape
    row_step = h / rows
    col_step = w / cols

    abs_vec = []
    # Підрахунок кількості активних пікселів у кожній клітинці сітки
    for r in range(rows):
        for c in range(cols):
            r_start, r_end = int(r * row_step), int((r + 1) * row_step)
            c_start, c_end = int(c * col_step), int((c + 1) * col_step)
            cell = binary[r_start:r_end, c_start:c_end]
            abs_vec.append(int(np.sum(cell)))

    abs_vec = np.array(abs_vec, dtype=float)
    total = np.sum(abs_vec)

    # Нормування: ділення на загальну площу (суму пікселів об'єкта)
    norm_vec = abs_vec / total if total > 0 else np.zeros_like(abs_vec)
    return abs_vec, norm_vec


def draw_grid(
    image: Image.Image,
    rows: int = 5,
    cols: int = 5,
) -> Image.Image:
    # Накладає поверх вхідного зображення червону сітку розбиття для візуалізації.
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    w, h = img_rgb.size

    # Горизонтальні лінії
    for r in range(1, rows):
        y = int(r * (h / rows))
        draw.line([(0, y), (w, y)], fill="red", width=2)

    # Вертикальні лінії
    for c in range(1, cols):
        x = int(c * (w / cols))
        draw.line([(x, 0), (x, h)], fill="red", width=2)

    return img_rgb


def compute_distance(v1: np.ndarray, v2: np.ndarray, metric: str) -> float:
    # Обчислює відстань між двома векторами за обраною геометричною метрикою.
    if metric == "Евклідова":
        # L2-норма: пряма відстань між двома точками
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    elif metric == "Манхеттенська":
        # L1-норма: сума модулів різниць координат
        return float(np.sum(np.abs(v1 - v2)))
    elif metric == "Чебишева":
        # L_inf-норма: максимальна розбіжність за однією з координат
        return float(np.max(np.abs(v1 - v2)))
    return 0.0


def generate_digit(symbol: str, noise: float = 0.0) -> Image.Image:
    """
    Створює синтетичний ч/б образ цифри фіксованого розміру 140x140.
    Дозволяє тестувати систему автономно без зовнішніх файлів.
    """
    size = 140
    img = Image.new("L", (size, size), color=255)  # Біле полотно
    draw = ImageDraw.Draw(img)

    if symbol == "0":
        draw.ellipse([30, 20, 110, 120], outline=0, width=16)
    elif symbol == "1":
        draw.line([(75, 20), (75, 120)], fill=0, width=16)
        draw.line([(45, 45), (75, 20)], fill=0, width=16)
        draw.line([(40, 120), (105, 120)], fill=0, width=16)
    elif symbol == "2":
        draw.arc([30, 20, 110, 80], start=180, end=0, fill=0, width=16)
        draw.line([(110, 50), (30, 120)], fill=0, width=16)
        draw.line([(30, 120), (110, 120)], fill=0, width=16)
    elif symbol == "3":
        # Плавне накреслення цифри 3 через поєднання прямих та округлої дуги
        draw.line([(35, 25), (105, 25)], fill=0, width=16)
        draw.line([(105, 25), (65, 68)], fill=0, width=16)
        draw.arc([30, 50, 110, 125], start=270, end=140, fill=0, width=16)

    arr = np.array(img)
    # Додавання випадкового імпульсного шуму до зображення (імітація спотворень)
    if noise > 0:
        mask = np.random.rand(*arr.shape) < noise
        arr[mask] = 0

    return Image.fromarray(arr)

# Модуль 2: Графічний інтерфейс користувача (GUI)
st.title(
    "Лабораторна робота №1: Адаптивна система розпізнавання та кластеризації",
    anchor=False,
)

# Розбиття інтерфейсу на 3 вкладки відповідно до завдань методички
tab1, tab2, tab3 = st.tabs([
    "Завдання 1: Побудова векторів",
    "Завдання 2: Класифікація за еталонами",
    "Завдання 3: Кластеризація та навчання",
])

# Вкладка 1: Побудова векторів ознак за сіткою
with tab1:
    st.header(
        "1. Побудова абсолютного та нормованого векторів ознак",
        anchor=False,
    )
    col1, col2 = st.columns([1, 2])

    with col1:
        grid_dim = st.slider(
            "Розмірність сітки (N x N)",
            min_value=3,
            max_value=8,
            value=5,
            key="g1",
        )
        upload_1 = st.file_uploader(
            "Завантажте зображення",
            type=["bmp", "png", "jpg", "jpeg"],
            key="u1",
        )
        invert_1 = st.checkbox(
            "Інвертувати колір (якщо білий об'єкт на чорному фоні)"
        )
        sample_choice_1 = st.selectbox(
            "Або оберіть стандартний зразок:",
            ["2", "0", "1", "3"],
        )

        # Вибір джерела: завантажений файл або синтетичний зразок
        if upload_1 is not None:
            active_img = Image.open(upload_1)
        else:
            active_img = generate_digit(sample_choice_1)

    with col2:
        abs_vec, norm_vec = extract_features(
            active_img,
            grid_dim,
            grid_dim,
            invert_colors=invert_1,
        )

        # Виведення зображення із накладеною сіткою
        st.image(
            draw_grid(active_img, grid_dim, grid_dim),
            caption=f"Сітка {grid_dim}x{grid_dim}",
            width=200,
        )

        # Виведення абсолютного вектора
        st.write(f"**Абсолютний вектор ознак ({len(abs_vec)} компонент):**")
        st.code(
            np.array2string(
                abs_vec.astype(int),
                separator=", ",
                max_line_width=80,
            )
        )

        # Виведення нормованого вектора
        st.write(
            f"**Нормований вектор ознак (контрольна сума = {np.sum(norm_vec):.2f}):**"
        )
        st.code(
            np.array2string(
                np.round(norm_vec, 4),
                separator=", ",
                max_line_width=80,
            )
        )

# Вкладка 2: Класифікація за методом порівняння з еталоном
with tab2:
    st.header(
        "2. Класифікація за методом порівняння з еталоном",
        anchor=False,
    )

    metric_name = st.selectbox(
        "Метрика оцінки близькості:",
        ["Евклідова", "Манхеттенська", "Чебишева"],
    )

    st.subheader("Еталони класів")
    classes = ["0", "1", "2", "3"]
    et_cols = st.columns(len(classes))
    class_vectors_norm = {}
    class_vectors_abs = {}

    # Розрахунок та відображення еталонів для кожного класу
    for i, cls in enumerate(classes):
        with et_cols[i]:
            st.write(f"**Еталон '{cls}'**")
            e_img = generate_digit(cls)
            st.image(e_img, width=110)

            e_abs, e_norm = extract_features(e_img, 5, 5)
            class_vectors_abs[cls] = e_abs
            class_vectors_norm[cls] = e_norm

            # Детальний перегляд еталонних числових векторів
            with st.expander("Вектори еталону"):
                st.caption("Абсолютний:")
                st.text(np.array2string(e_abs.astype(int), max_line_width=35))
                st.caption("Нормований:")
                st.text(np.array2string(np.round(e_norm, 3), max_line_width=35))

    st.divider()
    st.subheader("Введення та розпізнавання невідомого образу")
    u2_c1, u2_c2 = st.columns([1, 2])

    with u2_c1:
        upload_2 = st.file_uploader(
            "Завантажте образ для перевірки",
            type=["bmp", "png", "jpg", "jpeg"],
            key="u2",
        )
        sample_choice_2 = st.radio(
            "Або протестуйте згенерований образ із шумом:",
            classes,
            horizontal=True,
        )
        invert_2 = st.checkbox(
            "Інвертувати колір вхідного образу",
            key="inv2",
        )

    # Вибір вхідного образу для розпізнавання
    test_img_2 = (
        Image.open(upload_2)
        if upload_2 is not None
        else generate_digit(sample_choice_2, noise=0.03)
    )

    test_abs_2, test_norm_2 = extract_features(
        test_img_2,
        5,
        5,
        invert_colors=invert_2,
    )

    with u2_c2:
        st.image(test_img_2, caption="Досліджуваний образ", width=120)

    # Порівняння вектора невідомого образу з кожним з еталонів
    distances = {
        cls: compute_distance(test_norm_2, vec, metric_name)
        for cls, vec in class_vectors_norm.items()
    }
    # Знаходження класу з мінімальною відстанню
    best_class = min(distances, key=distances.get)

    st.write("### Відстані до еталонів:")
    res_cols = st.columns(len(classes))
    for i, cls in enumerate(classes):
        with res_cols[i]:
            st.metric(f"Клас '{cls}'", f"{distances[cls]:.4f}")

    st.success(
        f"Об'єкт класифіковано як: **Клас '{best_class}'** "
        f"(мінімальна {metric_name.lower()} відстань)"
    )

# Вкладка 3: Кластеризація, навчання та статистична обробка
with tab3:
    st.header(
        "3. Навчання системи та статистична обробка кластерів",
        anchor=False,
    )

    cluster_mode = st.radio(
        "Оберіть варіант кластерного аналізу (згідно з методичкою):",
        [
            "Варіант 1: Геометричні центри кластерів (математичне сподівання)",
            "Варіант 2: Метод ортогональних областей (min / max меж компонент)",
        ],
    )

    # Блок навчання системи на серії з 10 зразків для кожного класу
    if st.button("Провести навчання системи"):
        centers = {}
        bounding_boxes = {}
        train_classes = ["0", "1", "2"]
        num_samples = 10  # Точно за методичкою: 10 зразків на клас

        st.subheader("Навчальні послідовності класів (по 10 зразків)")
        for cls in train_classes:
            st.write(f"**Клас '{cls}':**")
            sample_vectors = []
            cols = st.columns(num_samples)

            for j in range(num_samples):
                # Генерація варіацій накреслення зі зростаючим рівнем шуму
                s_img = generate_digit(cls, noise=0.012 * (j + 1))
                cols[j].image(s_img, width=55, caption=f"#{j+1}")
                _, s_norm = extract_features(s_img, 5, 5)
                sample_vectors.append(s_norm)

            vectors_arr = np.array(sample_vectors)

            # Варіант 1: Розрахунок геометричного центру (математичного сподівання)
            centers[cls] = np.mean(vectors_arr, axis=0)

            # Варіант 2: Обчислення гіперпрямокутника [min, max] для ортогональних областей
            bounding_boxes[cls] = {
                "min": np.min(vectors_arr, axis=0),
                "max": np.max(vectors_arr, axis=0),
            }

        # Зберігаємо навчені параметри в сесії Streamlit
        st.session_state["cluster_centers"] = centers
        st.session_state["cluster_boxes"] = bounding_boxes
        st.success(
            f"Навчання завершено! Оброблено по {num_samples} зразків для кожного класу."
        )

    # Блок класифікації на основі навчених кластерів
    if "cluster_centers" in st.session_state:
        st.divider()
        st.subheader("Класифікація невідомого образу")
        u3_c1, u3_c2 = st.columns([1, 2])

        with u3_c1:
            upload_3 = st.file_uploader(
                "Завантажте тестовий образ",
                type=["bmp", "png", "jpg", "jpeg"],
                key="u3",
            )
            eval_choice = st.selectbox(
                "Або оберіть контрольний зразок:",
                ["0", "1", "2"],
            )
            invert_3 = st.checkbox(
                "Інвертувати колір вхідного образу",
                key="inv3",
            )

        test_img_3 = (
            Image.open(upload_3)
            if upload_3 is not None
            else generate_digit(eval_choice, noise=0.04)
        )

        _, test_norm_3 = extract_features(
            test_img_3,
            5,
            5,
            invert_colors=invert_3,
        )

        with u3_c2:
            st.image(
                draw_grid(test_img_3, 5, 5),
                caption="Образ із накладеною сіткою",
                width=130,
            )

        # Класифікація за Варіантом 1 (відстань до математичного сподівання)
        if (
            cluster_mode
            == "Варіант 1: Геометричні центри кластерів (математичне сподівання)"
        ):
            c_dists = {
                cls: compute_distance(test_norm_3, c_vec, "Евклідова")
                for cls, c_vec in st.session_state["cluster_centers"].items()
            }
            predicted = min(c_dists, key=c_dists.get)

            st.write("### Евклідові відстані до центрів кластерів:")
            c_cols = st.columns(len(c_dists))
            for i, (cls, d_val) in enumerate(c_dists.items()):
                with c_cols[i]:
                    st.metric(f"Центр '{cls}'", f"{d_val:.4f}")

            st.success(
                f"Зразок класифіковано до кластера: **Клас '{predicted}'**"
            )

        # Класифікація за Варіантом 2 (метод ортогональних областей)
        else:
            st.write("### Результати перевірки ортогональних областей:")
            ortho_cols = st.columns(len(st.session_state["cluster_boxes"]))
            matched_classes = []

            for i, (cls, box) in enumerate(st.session_state["cluster_boxes"].items()):
                # Перевірка належності всім компонентам інтервалу [min, max] з невеликим допуском
                inside = np.all(
                    (test_norm_3 >= (box["min"] - 0.015))
                    & (test_norm_3 <= (box["max"] + 0.015))
                )
                # Відхилення від гіперпрямокутника
                below_min = np.maximum(0, box["min"] - test_norm_3)
                above_max = np.maximum(0, test_norm_3 - box["max"])
                deviation = float(np.sum(below_min + above_max))

                with ortho_cols[i]:
                    status = "Попадає в область" if inside else "Поза областю"
                    st.metric(f"Клас '{cls}'", f"Відхилення: {deviation:.4f}")
                    st.caption(f"Статус: **{status}**")

                if inside:
                    matched_classes.append(cls)

            # Формування вердикту розпізнавання
            if len(matched_classes) == 1:
                st.success(
                    f"Зразок однозначно належить області: **Клас '{matched_classes[0]}'**"
                )
            elif len(matched_classes) > 1:
                st.info(
                    "Зразок потрапив у зону перекриття областей класів: "
                    f"{', '.join(matched_classes)}"
                )
            else:
                # Якщо зразок поза межами, визначаємо найближчу область за сумою відхилень
                closest_cls = min(
                    st.session_state["cluster_boxes"].keys(),
                    key=lambda c: np.sum(
                        np.maximum(0, st.session_state["cluster_boxes"][c]["min"] - test_norm_3)
                        + np.maximum(0, test_norm_3 - st.session_state["cluster_boxes"][c]["max"])
                    ),
                )
                st.warning(
                    "Зразок знаходиться поза межами всіх навчених областей. "
                    f"Найближча область: **Клас '{closest_cls}'**"
                )