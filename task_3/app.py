import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Завантажуємо API-ключ із файлу .env
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Використовуємо актуальну швидку модель Gemini
MODEL_ID = "gemini-3.5-flash-lite"


def run_task_1():
    print("=" * 60)
    print("ЗАВДАННЯ 1: 5 типів запитів (Предметна область: ШІ та нейромережі)")
    print("=" * 60)

    queries = {
        "1. Пояснення": (
            "Поясни простими словами, що таке функція втрат (Loss Function) у"
            " машинному навчанні та навіщо вона потрібна."
        ),
        "2. Генерація тексту": (
            "Напиши короткий вступний абзац (до 60 слів) для статті про"
            " актуальність згорткових нейронних мереж (CNN) у комп'ютерному зорі."
        ),
        "3. Переклад": (
            "Переклади українською мовою з дотриманням технічної термінології: "
            "'Gradient descent is an iterative optimization algorithm used to"
            " minimize the loss function by updating parameters in the direction"
            " of steepest descent.'"
        ),
        "4. Узагальнення": (
            "Скороти та узагальни наступний текст до двох лаконічних тез: "
            "'Перенавчання (overfitting) виникає, коли модель занадто детально"
            " підлаштовується під тренувальні дані, запам'ятовуючи навіть"
            " випадковий шум та несуттєві деталі. Через це на навчальній вибірці"
            " вона показує майже стовідсоткову точність, проте під час роботи з"
            " новими даними припускається грубих помилок. Основними методами"
            " боротьби є регуляризація, шари Dropout та збільшення датасету.'"
        ),
        "5. Генерація списку": (
            "Сформуй список із 4 популярних оптимізаторів для навчання"
            " нейромереж. Для кожного вкажи головну особливість у форматі:"
            " 'Назва: особливість'."
        ),
    }

    for title, prompt in queries.items():
        print(f"\n--- {title} ---")
        print(f"Prompt: {prompt}\n")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        print(f"Відповідь:\n{response.text}\n")


def run_task_2():
    print("=" * 60)
    print("ЗАВДАННЯ 2: Порівняння Zero-shot та Few-shot prompt")
    print("=" * 60)

    zero_shot_prompt = (
        "Класифікуй гіперпараметр learning_rate за категорією "
        "(Оптимізація / Архітектура / Регуляризація) та вкажи тип даних."
    )

    few_shot_prompt = (
        "Класифікуй параметр за категорією (Оптимізація, Архітектура,"
        " Регуляризація) "
        "та типом даних за наведеним зразком:\n\n"
        "Вхід: num_layers\nВихід: num_layers | Архітектура | int\n\n"
        "Вхід: dropout_rate\nВихід: dropout_rate | Регуляризація | float\n\n"
        "Вхід: batch_size\nВихід: batch_size | Оптимізація | int\n\n"
        "Вхід: learning_rate\nВихід:"
    )

    for label, prompt in [
        ("Zero-shot", zero_shot_prompt),
        ("Few-shot", few_shot_prompt),
    ]:
        print(f"\n--- {label} ---")
        print(f"Prompt:\n{prompt}\n")
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        print(f"Відповідь:\n{response.text}\n")


def run_task_3():
    print("=" * 60)
    print("ЗАВДАННЯ 3: Удосконалення prompt")
    print("=" * 60)

    basic_prompt = "Розкажи про штучний інтелект."
    advanced_prompt = (
        "Роль: Викладач курсу «Вступ до штучного інтелекту» в технічному"
        " університеті.\nЦільова аудиторія: Студенти 1 курсу спеціальності"
        " «Комп'ютерні науки».\nЗавдання: Поясни різницю між термінами «Штучний"
        " інтелект», «Машинне навчання» та «Глибоке навчання».\nФормат: 3 пункти"
        " маркованого списку за принципом матрьошки (від ширшого до вужчого"
        " поняття), кожен пункт — рівно 2 речення.\nОбсяг: До 100 слів."
    )

    for label, prompt in [
        ("Початковий prompt", basic_prompt),
        ("Вдосконалений prompt", advanced_prompt),
    ]:
        print(f"\n--- {label} ---")
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        print(f"Відповідь:\n{response.text}\n")


def run_task_4():
    print("=" * 60)
    print("ЗАВДАННЯ 4: Використання системної ролі (System Instruction)")
    print("=" * 60)

    question = "Що таке градієнтний спуск?"
    roles = {
        "Викладач для початківців": (
            "Ти шкільний вчитель фізики та інформатики, який пояснює складні"
            " речі дітям через прості метафори та життєві приклади."
        ),
        "Senior ML-програміст": (
            "Ти Senior ML Engineer. Твій стиль — лаконічний, технічний,"
            " орієнтований на продакшн, формули, оптимізацію та код."
        ),
        "Академічний науковець": (
            "Ти професор математичного аналізу. Відповідай строго академічною"
            " мовою, використовуючи точні математичні терміни та векторний"
            " аналіз."
        ),
    }

    for role_name, sys_instruction in roles.items():
        print(f"\n--- Роль: {role_name} ---")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
            ),
        )
        print(f"Відповідь:\n{response.text}\n")


if __name__ == "__main__":
    run_task_1()
    run_task_2()
    run_task_3()
    run_task_4()
