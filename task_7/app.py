import ast
from datetime import datetime
import math
import operator
import os
import time
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

LLM_MODEL = "gemini-3.5-flash-lite"


# =====================================================================
# ЗАВДАННЯ 2: Зовнішні інструменти (Tools) для AI-агента
# =====================================================================
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "round": round,
    "abs": abs,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Ділення на нуль неможливе.")
        return _ALLOWED_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in _ALLOWED_FUNCS:
            args = [_eval_ast(arg) for arg in node.args]
            return float(_ALLOWED_FUNCS[func_name](*args))
    raise ValueError(f"Непідтримуваний вираз: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Безпечно обчислює математичний вираз (наприклад, '(12 + 18 + 25 + 30) / 4' або 'sqrt(144)')."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast(tree)
        return f"Результат обчислення ({expression}) = {round(result, 4)}"
    except ZeroDivisionError:
        return "Помилка калькулятора: ділення на нуль."
    except Exception as e:
        return f"Помилка обчислення виразу '{expression}': {e}"


def days_until_deadline(target_date: str) -> str:
    """Обчислює кількість днів від поточної дати (2026-11-19) до вказаної дати дедлайну у форматі YYYY-MM-DD."""
    try:
        current = datetime(2026, 11, 19)
        deadline = datetime.strptime(target_date.strip(), "%Y-%m-%d")
        delta = (deadline - current).days
        if delta < 0:
            return f"Дедлайн {target_date} вже минув ({abs(delta)} дн. тому)."
        return f"До дати {target_date} залишилося {delta} днів (відлік від 2026-11-19)."
    except ValueError:
        return f"Помилка формату дати '{target_date}'. Очікується формат YYYY-MM-DD."


# =====================================================================
# ЗАВДАННЯ 3, 4, 5: AI-агент із плануванням, Tool Calling та Memory
# =====================================================================
SYSTEM_INSTRUCTION = """Ти — навчальний AI-агент «Помічник студента».
Твої правила роботи:
1. ЗАВЖДИ починай свою фінальну відповідь із короткого нумерованого блоку:
[ПЛАН ДІЙ]
1. ...
2. ...
2. Після блоку [ПЛАН ДІЙ] пиши блок [ВІДПОВІДЬ] із чітким розв'язанням.
3. Для будь-яких арифметичних розрахунків обов'язково викликай інструмент `calculator`.
4. Для обчислення кількості днів до дати іспиту чи здачі лабораторної роботи обов'язково викликай `days_until_deadline`.
5. Якщо інструмент не потрібен (звичайне теоретичне запитання) — відповідай одразу після плану.
6. Уважно враховуй усю попередню історію діалогу (Memory), навіть якщо користувач не повторює тему або числа в наступному повідомленні.
7. Якщо користувач запитує про приватний розклад конкретного викладача або вираз із діленням на нуль — чесно повідом про обмеження або помилку інструмента."""


class StudyAssistantAgent:

    def __init__(self):
        self.memory: List[types.Content] = []
        self.tools = [calculator, days_until_deadline]

    def _generate_with_retry(self, retries: int = 6):
        for attempt in range(1, retries + 1):
            try:
                time.sleep(5.0)
                return client.models.generate_content(
                    model=LLM_MODEL,
                    contents=self.memory,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=self.tools,
                        temperature=0.1,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
            except errors.APIError as e:
                if e.code in (429, 503) and attempt < retries:
                    time.sleep(attempt * 10)
                else:
                    raise

    def ask(self, user_message: str) -> str:
        self.memory.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        used_tools = []
        while True:
            response = self._generate_with_retry()
            candidate = response.candidates[0]
            self.memory.append(candidate.content)

            function_calls = [
                part.function_call
                for part in candidate.content.parts
                if part.function_call
            ]

            if not function_calls:
                tool_info = (
                    ", ".join(used_tools) if used_tools else "не використовувався (LLM)"
                )
                return (
                    f"[Використаний інструмент: {tool_info}]\n{response.text.strip()}"
                )

            tool_response_parts = []
            for fc in function_calls:
                func_name = fc.name
                func_args = dict(fc.args) if fc.args else {}
                if func_name == "calculator":
                    output = calculator(**func_args)
                elif func_name == "days_until_deadline":
                    output = days_until_deadline(**func_args)
                else:
                    output = f"Невідомий інструмент: {func_name}"

                used_tools.append(f"{func_name}({func_args}) -> {output}")
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=func_name, response={"result": output}
                    )
                )

            self.memory.append(types.Content(role="user", parts=tool_response_parts))


def main():
    print("=" * 75)
    print("ЛАБОРАТОРНА РОБОТА №7: Розробка простого AI-агента (Помічник студента)")
    print("=" * 75)

    agent = StudyAssistantAgent()

    # =====================================================================
    # ЗАВДАННЯ 6: Тестування агента (6 сценаріїв, включно з Memory і помилкою)
    # =====================================================================
    test_prompts = [
        # 1. Просте питання без інструмента
        "Поясни коротко в 3 реченнях, що таке перенавчання (overfitting) у машинному навчанні.",
        # 2. Задача, що потребує калькулятора
        "Обчисли середнє арифметичне моїх балів за лабораторні: 14, 18, 20, 17, 19 та поясни результат.",
        # 3. Складне багатокрокове завдання (дата + калькулятор + план)
        "Скільки днів залишилося до іспиту 2026-12-04 і скільки тем на день мені треба вчити, якщо всього в курсі 45 тем? Склади короткий план.",
        # 4. Перевірка Memory (повідомлення 1 із ланцюжка)
        "Я зараз готуюся до заліку з курсу «Теорія ймовірностей» і вже вивчив 12 білетів із 30.",
        # 5. Перевірка Memory (повідомлення 2 — без назви курсу, опора на попередній контекст)
        "Скільки відсотків програми я вже пройшов і скільки білетів залишилося?",
        # 6. Перевірка Memory + граничний випадок / помилка (повідомлення 3)
        "А якщо я хочу закінчити решту білетів за 0 днів до заліку, скільки це вийде на день? Також підкажи, в якій аудиторії завтра приймає мій викладач.",
    ]

    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 75}")
        print(f"ТЕСТ №{idx}")
        print(f"Користувач: {prompt}")
        print("-" * 75)
        reply = agent.ask(prompt)
        print(reply)

    print("\n" + "=" * 75)
    print(
        f"Перевірка Memory: у пам'яті агента збережено {len(agent.memory)} повідомлень/кроків діалогу."
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
