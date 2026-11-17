import json
import os
import time
from typing import Any, Dict, List
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Завантажуємо API-ключ із .env
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Актуальна швидка модель, перевірена в Лабі №3
MODEL_ID = "gemini-3.5-flash-lite"

# Базовий текст для завдань 1–4 (Предметна область: Deep Learning та оптимізація)
BASE_TEXT = (
    "Згорткові нейронні мережі (CNN) революціонізували галузь комп'ютерного"
    " зору завдяки механізму локальних рецептивних полів та спільних ваг, що"
    " забезпечує інваріантність до зсуву. Проте зі збільшенням глибини мереж"
    " виникає проблема згасання градієнтів, яка ускладнює навчання перших шарів"
    " архітектури. Для подолання цього бар'єра було впроваджено залишкове"
    " навчання (Residual Learning) в архітектурі ResNet через так звані"
    " skip-connections (пропускні зв'язки), що дозволяють градієнту"
    " безпосередньо протікати крізь шари. Додатково продуктивність оптимізації"
    " визначається вибором алгоритму: адаптивні методи, такі як Adam,"
    " поєднують розрахунок моментів першого та другого порядків, що забезпечує"
    " швидшу збіжність порівняно зі стандартним стохастичним градієнтним"
    " спуском (SGD). Водночас надмірне ускладнення моделі за обмеженого обсягу"
    " тренувальних даних призводить до перенавчання (overfitting), яке"
    " контролюється дропаутом (Dropout) та L2-регуляризацією."
)


def ask_gemini(
    prompt: str, config: types.GenerateContentConfig = None, retries: int = 6
) -> str:
    """Надсилає запит із паузою 5 сек (до 12 RPM, щоб не перевищити ліміт 15 RPM) та автоповтором."""
    for attempt in range(1, retries + 1):
        try:
            time.sleep(5.0)
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=config,
            )
            return response.text
        except errors.APIError as e:
            if e.code in (429, 503) and attempt < retries:
                wait_time = attempt * 10
                print(
                    f"[Ліміт або навантаження ({e.code}). Чекаємо {wait_time} сек..."
                    f" (спроба {attempt}/{retries})]"
                )
                time.sleep(wait_time)
            else:
                raise


def run_task_1():
    print("=" * 65)
    print("ЗАВДАННЯ 1: Звичайний prompt vs Просунутий prompt")
    print("=" * 65)

    simple_prompt = (
        f"Проаналізуй наведений текст та визнач його основні ідеї:\n\n{BASE_TEXT}"
    )

    advanced_prompt = (
        "Роль: Ти провідний дослідник у галузі Deep Learning та викладач"
        " університету.\nКонтекст: Матеріал призначений для студентів"
        " магістратури з комп'ютерних наук.\nЗавдання: Проаналізуй наведений"
        " текст, виконавши покрокову декомпозицію:\n1. Визнач генеральну тему"
        " тексту (1 речення).\n2. Сформулюй 3 ключові інженерні виклики та методи"
        " їх розв'язання.\n3. Склади перелік із 3 ключових термінів із коротким"
        " описом їхньої ролі.\n4. Зроби стислий підсумковий висновок про"
        " архітектурну еволюцію CNN (до 30 слів).\nОбмеження: Жодного вступного"
        " та підсумкового тексту; дотримуйся суворої нумерації"
        f" розділів.\n\nТекст:\n{BASE_TEXT}"
    )

    for label, prompt in [
        ("Варіант 1: Звичайний prompt", simple_prompt),
        ("Варіант 2: Просунутий prompt", advanced_prompt),
    ]:
        print(f"\n--- {label} ---")
        print(f"Відповідь:\n{ask_gemini(prompt)}\n")


def run_task_2():
    print("=" * 65)
    print("ЗАВДАННЯ 2: Декомпозиція складної задачі (Chain of Prompts)")
    print("=" * 65)

    # Підхід А: Один загальний запит
    single_prompt = (
        "За наведеним текстом склади 3 тестові запитання з варіантами відповідей"
        f" A, B, C, D та вкажи правильну відповідь:\n\n{BASE_TEXT}"
    )
    print("\n--- Підхід А: Один загальний prompt ---")
    print(f"Відповідь:\n{ask_gemini(single_prompt)}\n")

    # Підхід Б: Послідовна декомпозиція (Крок 1 -> Крок 2 -> Крок 3)
    print("\n--- Підхід Б: Покрокова декомпозиція ---")
    step_1 = (
        "Проаналізуй текст та виділи 3 ключові причинно-наслідкові зв'язки,"
        " навколо яких можна побудувати питання високого рівня складності (рівень"
        f" аналізу за Блумом):\n\n{BASE_TEXT}"
    )
    concepts = ask_gemini(step_1)
    print(f"[Крок 1 — Виділені концепції]:\n{concepts}\n")

    step_2 = (
        "На базі виділених 3 концепцій сформулюй ситуативні тестові запитання."
        " Для кожного запитання створи 4 варіанти (A, B, C, D) з реалістичними"
        f" дистракторами:\n\n{concepts}"
    )
    questions = ask_gemini(step_2)
    print(f"[Крок 2 — Згенеровані питання]:\n{questions}\n")

    step_3 = (
        "Для сформованих питань сформуй фінальний блок: вкажи правильну відповідь"
        " та надай коротке обґрунтування, чому правильний варіант вірний, а інші"
        f" три — хибні:\n\n{questions}"
    )
    final_test = ask_gemini(step_3)
    print(f"[Крок 3 — Валідація та пояснення дистракторів]:\n{final_test}\n")


def run_task_3():
    print("=" * 65)
    print("ЗАВДАННЯ 3: Імітація підходу ReAct (Reasoning + Acting)")
    print("=" * 65)

    react_prompt = (
        "Застосуй парадигму ReAct для аналізу наведеного тексту за схемою:\n"
        "[ЗАДАЧА] -> [ПЛАН ДІЙ] -> [ВИКОНАННЯ КРОКІВ] -> [САМОПЕРЕВІРКА /"
        " ВЕРИФІКАЦІЯ] -> [ПІДСУМКОВИЙ РЕЗУЛЬТАТ]\n\n"
        "Задача: Розподіли всі наведені в тексті твердження на дві групи: сухі"
        " технічні факти та авторські узагальнення. Визнач, чи містить текст"
        f" суперечності або необґрунтовані тези.\n\nТекст:\n{BASE_TEXT}"
    )

    print(f"Відповідь:\n{ask_gemini(react_prompt)}\n")


def parse_and_validate_llm_json(raw_response: str) -> Dict[str, Any]:
    cleaned_str = raw_response.strip()
    if cleaned_str.startswith("```json"):
        cleaned_str = cleaned_str[7:]
    if cleaned_str.startswith("```"):
        cleaned_str = cleaned_str[3:]
    if cleaned_str.endswith("```"):
        cleaned_str = cleaned_str[:-3]
    cleaned_str = cleaned_str.strip()

    data = json.loads(cleaned_str)
    required_keys = {"topic", "key_challenges", "keywords", "conclusion"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise KeyError(f"У JSON відсутні обов'язкові ключі: {missing_keys}")
    return data


def run_task_4():
    print("=" * 65)
    print("ЗАВДАННЯ 4: Структурований JSON та програмна валідація")
    print("=" * 65)

    json_prompt = (
        "Проаналізуй наведений текст про CNN та оптимізацію. Поверни результат"
        " виключно у форматі валідного JSON-об'єкта з полями: 'topic' (рядок),"
        " 'key_challenges' (масив об'єктів з ключами 'challenge' та 'solution'),"
        " 'keywords' (масив рядків) та 'conclusion' (рядок). Без форматування"
        f" Markdown і жодних коментарів.\n\nТекст:\n{BASE_TEXT}"
    )

    raw_json = ask_gemini(
        json_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    print(f"Сира відповідь моделі:\n{raw_json}\n")
    parsed_data = parse_and_validate_llm_json(raw_json)
    print("Результат валідації Python:")
    print(f"  -> Тема: {parsed_data['topic']}")
    print(f"  -> Кількість викликів: {len(parsed_data['key_challenges'])}")
    print(f"  -> Ключові слова: {', '.join(parsed_data['keywords'])}\n")


def chunk_text_sliding_window(
    text: str, chunk_size: int = 45, overlap: int = 12
) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def run_task_5():
    print("=" * 65)
    print("ЗАВДАННЯ 5: Context Engineering — Chunking та Sliding Window")
    print("=" * 65)

    long_document = (
        "Розділ 1. Класичні згорткові мережі (CNN) використовують індуктивне"
        " упередження локальності та еквіваріантності до зсуву, що дозволяє їм"
        " ефективно навчатися на невеликих вибірках зображень за допомогою"
        " фільтрів 3x3. "
        "Розділ 2. Архітектура Vision Transformers (ViT) відмовляється від"
        " згорток, розбиваючи зображення на фіксовані патчі розміром 16x16"
        " пікселів. Головним недоліком ViT є квадратична обчислювальна"
        " складність механізму Self-Attention O(N^2) відносно кількості патчів та"
        " відсутність вбудованого індуктивного упередження, через що модель"
        " потребує мільйонів зображень для попереднього навчання. "
        "Розділ 3. Сучасні гібридні архітектури, такі як CoAtNet, поєднують"
        " глибинні згортки на ранніх етапах із блоками Self-Attention на"
        " пізніх, а для стабілізації тренування застосовують оптимізатор AdamW та"
        " аугментацію Mixup."
    )

    # 1. Без chunking
    print("\n--- Без chunking (єдиний запит) ---")
    res_whole = ask_gemini(
        "Зроби стислий технічний конспект цього огляду (до 4"
        f" речень):\n\n{long_document}"
    )
    print(f"Відповідь:\n{res_whole}\n")

    # 2. Із chunking (Sliding Window)
    print("--- Із chunking (Sliding Window + агрегація) ---")
    chunks = chunk_text_sliding_window(long_document, chunk_size=45, overlap=12)
    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        res_chunk = ask_gemini(
            "Виділи ключові технічні факти та формули з цього фрагмента (1-2"
            f" речення):\n\n{chunk}"
        ).strip()
        summaries.append(f"Фрагмент {idx}: {res_chunk}")
        print(f"[Chunk {idx}]: {res_chunk}")

    combined_summaries = "\n".join(summaries)
    final_summary = ask_gemini(
        "Об'єднай ці локальні вижимки в єдиний структурований технічний"
        f" конспект без втрати формул та деталей:\n\n{combined_summaries}"
    )
    print(f"\n[Фінальне узагальнення після chunking]:\n{final_summary}\n")


def run_task_6():
    print("=" * 65)
    print("ЗАВДАННЯ 6: Дослідження параметрів generation (temperature та top_p)")
    print("=" * 65)

    slogan_prompt = (
        "Сформулюй коротке гасло (рівно 1 речення) для нового AI-стартапу, який"
        " розробляє систему виявлення хвороб рослин на полях за допомогою дронів"
        " та комп'ютерного зору."
    )

    print("\n--- 6.1. Вплив temperature (top_p = 0.95) ---")
    for temp in [0.0, 0.5, 1.0]:
        res = ask_gemini(
            slogan_prompt,
            config=types.GenerateContentConfig(temperature=temp, top_p=0.95),
        ).strip()
        print(f"temperature = {temp} -> {res}")

    print("\n--- 6.2. Вплив top_p (temperature = 0.7) ---")
    for top_p_val in [0.5, 0.8, 1.0]:
        res = ask_gemini(
            slogan_prompt,
            config=types.GenerateContentConfig(temperature=0.7, top_p=top_p_val),
        ).strip()
        print(f"top_p = {top_p_val} -> {res}")


if __name__ == "__main__":
    run_task_1()
    run_task_2()
    run_task_3()
    run_task_4()
    run_task_5()
    run_task_6()
