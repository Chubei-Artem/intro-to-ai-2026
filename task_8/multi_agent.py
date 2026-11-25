import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

LLM_MODEL = "gemini-3.5-flash-lite"


def call_llm(
    system_prompt: str, user_prompt: str, temperature: float = 0.2, retries: int = 6
) -> str:
    for attempt in range(1, retries + 1):
        try:
            time.sleep(5.0)
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                ),
            )
            return response.text.strip()
        except errors.APIError as e:
            if e.code in (429, 503) and attempt < retries:
                time.sleep(attempt * 10)
            else:
                raise


# =====================================================================
# ЗАВДАННЯ 2: Спеціалізовані AI-агенти (Планувальник, Виконавець, Перевіряльник)
# =====================================================================
class PlannerAgent:
    """Агент-планувальник: декомпозує навчальну тему на чіткі підзадачі."""

    SYSTEM_PROMPT = (
        "Ти — Агент-планувальник навчальних курсів. Твоє завдання: отримати тему"
        " від користувача та розбити підготовку навчального матеріалу рівно на 3"
        " конкретні підзадачі: 1) Визначення ключових понять теми; 2) Пояснення"
        " принципу роботи з одним практичним прикладом; 3) Складання 2 тестових"
        " запитань із правильними відповідями. Поверни лише нумерований план."
    )

    def create_plan(self, topic: str) -> str:
        return call_llm(
            self.SYSTEM_PROMPT,
            f"Склади план підготовки навчального матеріалу за темою: «{topic}».",
            temperature=0.1,
        )


class ExecutorAgent:
    """Агент-виконавець: генерує навчальний матеріал строго за планом."""

    SYSTEM_PROMPT = (
        "Ти — Агент-виконавець (методист). Отримавши тему та план від"
        " Планувальника, лаконічно та структуровано виконай усі 3 пункти плану"
        " українською мовою. Уникай зайвої води, пиши чітко й академічно."
    )

    def execute(self, topic: str, plan: str) -> str:
        prompt = (
            f"Тема: «{topic}»\n\n"
            f"План від Агента-планувальника:\n{plan}\n\n"
            "Виконай усі пункти плану та сформуй чернетку навчального матеріалу."
        )
        return call_llm(self.SYSTEM_PROMPT, prompt, temperature=0.3)


class ReviewerAgent:
    """Агент-перевіряльник: аудитує чернетку, виправляє неточності та формує фінал."""

    SYSTEM_PROMPT = (
        "Ти — Агент-перевіряльник (науковий редактор). Перевір чернетку"
        " навчального матеріалу на точність термінів, повноту розкриття всіх 3"
        " пунктів плану та наявність правильних відповідей до тестів.\n"
        "Обов'язково поверни відповідь у форматі двох блоків:\n"
        "[ВИСНОВОК ПЕРЕВІРКИ]: коротка оцінка якості (бал від 1 до 10) та"
        " зауваження/підтвердження відсутності помилок.\n"
        "[ФІНАЛЬНИЙ РЕЗУЛЬТАТ]: відредагований та готовий до використання"
        " навчальний матеріал."
    )

    def review(self, topic: str, plan: str, draft: str) -> str:
        prompt = (
            f"Тема: «{topic}»\n\n"
            f"План:\n{plan}\n\n"
            f"Чернетка від Агента-виконавця:\n{draft}\n\n"
            "Виконай перевірку та сформуй фінальний результат."
        )
        return call_llm(self.SYSTEM_PROMPT, prompt, temperature=0.1)


# =====================================================================
# ЗАВДАННЯ 4: Оркестратор мультиагентної системи
# =====================================================================
def run_multi_agent_pipeline(topic: str) -> dict:
    start_time = time.time()
    planner = PlannerAgent()
    executor = ExecutorAgent()
    reviewer = ReviewerAgent()

    plan = planner.create_plan(topic)
    draft = executor.execute(topic, plan)
    reviewed = reviewer.review(topic, plan, draft)

    elapsed = round(time.time() - start_time, 2)
    return {
        "topic": topic,
        "plan": plan,
        "draft": draft,
        "reviewed": reviewed,
        "steps": 3,
        "elapsed_sec": elapsed,
    }


# =====================================================================
# ЗАВДАННЯ 6: Монолітний (один) AI-агент для порівняння
# =====================================================================
def run_single_agent(topic: str) -> dict:
    start_time = time.time()
    system_prompt = (
        "Ти — універсальний AI-помічник викладача. Самостійно створи навчальний"
        " матеріал за темою: визнач ключові поняття, дай пояснення з прикладом"
        " та склади 2 тестові запитання з відповідями."
    )
    output = call_llm(
        system_prompt,
        f"Підготуй навчальний матеріал за темою: «{topic}».",
        temperature=0.2,
    )
    elapsed = round(time.time() - start_time, 2)
    return {
        "topic": topic,
        "output": output,
        "steps": 1,
        "elapsed_sec": elapsed,
    }


def main():
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА №8 (Частина 1): Мультиагентна система (3 агенти)")
    print("=" * 80)

    # ЗАВДАННЯ 5: Тестування на 3 різних задачах
    test_topics = [
        "Градієнтний спуск у машинному навчанні",
        "Нормалізація реляційних баз даних (1НФ, 2НФ, 3НФ)",
        "Принцип роботи протоколу HTTPS та TLS-шифрування",
    ]

    for idx, topic in enumerate(test_topics, 1):
        print(f"\n{'=' * 80}")
        print(f"ЗАДАЧА №{idx}: {topic}")
        print("=" * 80)
        res = run_multi_agent_pipeline(topic)
        print(f"\n--- 1. [АГЕНТ-ПЛАНУВАЛЬНИК] ---\n{res['plan']}")
        print(f"\n--- 2. [АГЕНТ-ВИКОНАВЕЦЬ (Чернетка)] ---\n{res['draft']}")
        print(f"\n--- 3. [АГЕНТ-ПЕРЕВІРЯЛЬНИК] ---\n{res['reviewed']}")
        print(
            f"\n[Метрики мультиагентної системи: кроків = {res['steps']}, час = {res['elapsed_sec']} с]"
        )

    # ЗАВДАННЯ 6: Порівняння 1 агента та мультиагентної системи на Задачі №1
    print("\n" + "=" * 80)
    print("ЗАВДАННЯ 6: ПОРІВНЯННЯ ОДНОГО АГЕНТА ТА МУЛЬТИАГЕНТНОЇ СИСТЕМИ")
    print("=" * 80)
    compare_topic = test_topics[0]
    single_res = run_single_agent(compare_topic)
    print(f"Тема порівняння: «{compare_topic}»")
    print(
        f"Варіант 1 (Один агент): кроків = {single_res['steps']}, час = {single_res['elapsed_sec']} с,"
        f" довжина відповіді = {len(single_res['output'])} символів (без незалежної перевірки)."
    )
    print(f"\n--- Результат одного агента ---\n{single_res['output']}")


if __name__ == "__main__":
    main()
