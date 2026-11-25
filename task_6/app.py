import os
import time
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import numpy as np

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

LLM_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

CATEGORIES = [
    "Мережева інфраструктура",
    "Бази даних",
    "Безпека та доступи",
    "Прикладне ПЗ",
]


# =====================================================================
# ЗАВДАННЯ 1 та 5: Набір тестових даних (8 однозначних прикладів)
# =====================================================================
TEST_DATASET = [
    {
        "id": 1,
        "text": (
            "Під час виконання транзакції виникає взаємне блокування (deadlock)"
            " та повільне виконання SELECT-запитів через відсутність індексу."
        ),
        "expected": "Бази даних",
    },
    {
        "id": 2,
        "text": (
            "Користувачі філії втратили зв'язок із центральним сервером через"
            " обрив BGP-сесії на граничному маршрутизаторі."
        ),
        "expected": "Мережева інфраструктура",
    },
    {
        "id": 3,
        "text": (
            "Зафіксовано багаторазові невдалі спроби входу під обліковим"
            " записом адміністратора та спробу підбору пароля (brute-force)."
        ),
        "expected": "Безпека та доступи",
    },
    {
        "id": 4,
        "text": (
            "Після оновлення клієнтського модуля бухгалтерська програма"
            " аварійно завершує роботу при експорті звіту в Excel."
        ),
        "expected": "Прикладне ПЗ",
    },
    {
        "id": 5,
        "text": (
            "У системному журналі зафіксовано корпоративний код інциденту"
            " ERR-GW-409 на шлюзі віддаленого підключення співробітників."
        ),
        "expected": "Мережева інфраструктура",
    },
    {
        "id": 6,
        "text": (
            "Спрацював тригер внутрішнього аудиту POL-SEC-12 через підключення"
            " неавторизованого USB-накопичувача до робочої станції."
        ),
        "expected": "Безпека та доступи",
    },
    {
        "id": 7,
        "text": (
            "Репліка сховища відстає від майстер-вузла на 40 хвилин через"
            " переповнення журналу транзакцій WAL."
        ),
        "expected": "Бази даних",
    },
    {
        "id": 8,
        "text": (
            "У CRM-системі не відображається кнопка погодження договору після"
            " зміни масштабу інтерфейсу в браузері."
        ),
        "expected": "Прикладне ПЗ",
    },
]


# =====================================================================
# Допоміжні функції API (з rate-limit захистом)
# =====================================================================
def get_embedding(
    text: str, task_type: str = "RETRIEVAL_DOCUMENT", retries: int = 5
) -> List[float]:
    for attempt in range(1, retries + 1):
        try:
            time.sleep(0.8)
            res = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return res.embeddings[0].values
        except errors.APIError as e:
            if e.code in (429, 503) and attempt < retries:
                time.sleep(attempt * 5)
            else:
                raise


def ask_llm(prompt: str, retries: int = 6) -> str:
    for attempt in range(1, retries + 1):
        try:
            time.sleep(5.0)
            res = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return res.text.strip()
        except errors.APIError as e:
            if e.code in (429, 503) and attempt < retries:
                time.sleep(attempt * 10)
            else:
                raise


def normalize_category(raw_output: str) -> str:
    cleaned = raw_output.strip().strip(".").strip('"').strip("'")
    for cat in CATEGORIES:
        if cat.lower() in cleaned.lower():
            return cat
    return cleaned


# =====================================================================
# ЗАВДАННЯ 2: Підхід 1 — Prompt Engineering (Zero-shot)
# =====================================================================
def classify_prompt_engineering(text: str) -> str:
    prompt = (
        "Визнач категорію наведеного технічного інциденту.\n"
        "Дозволені категорії: Мережева інфраструктура, Бази даних, Безпека та"
        " доступи, Прикладне ПЗ.\n"
        "Поверни виключно назву однієї категорії без пояснень і розділових"
        " знаків.\n\n"
        f"Текст інциденту: {text}"
    )
    return normalize_category(ask_llm(prompt))


# =====================================================================
# ЗАВДАННЯ 3: Підхід 2 — RAG (База регламентів + Vector Store)
# =====================================================================
def build_rag_store(docs_dir: str = "docs") -> List[Dict[str, Any]]:
    os.makedirs(docs_dir, exist_ok=True)
    kb_docs = {
        "reg_1_network.txt": (
            "Категорія: Мережева інфраструктура. Сюди належать збої"
            " маршрутизації, протоколу BGP, комутаторів, DNS, падіння каналів"
            " зв'язку між філіями, а також корпоративний код інциденту"
            " ERR-GW-409 на шлюзі віддаленого підключення."
        ),
        "reg_2_databases.txt": (
            "Категорія: Бази даних. Сюди належать помилки СУБД (PostgreSQL,"
            " MySQL), взаємні блокування (deadlock), повільні SQL-запити,"
            " деградація індексів, проблеми реплікації та переповнення журналу"
            " транзакцій WAL."
        ),
        "reg_3_security.txt": (
            "Категорія: Безпека та доступи. Сюди належать інциденти інформаційної"
            " безпеки, спроби підбору паролів (brute-force), блокування облікових"
            " записів, керування правами доступу та спрацювання політики"
            " внутрішнього аудиту POL-SEC-12 щодо USB-носіїв."
        ),
        "reg_4_software.txt": (
            "Категорія: Прикладне ПЗ. Сюди належать помилки в роботі бізнес-програм,"
            " CRM-систем, бухгалтерського програмного забезпечення, збої"
            " інтерфейсу користувача та помилки експорту документів після оновлень."
        ),
    }

    store = []
    for filename, content in kb_docs.items():
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        emb = np.array(
            get_embedding(content, task_type="RETRIEVAL_DOCUMENT"), dtype=np.float32
        )
        store.append({"source": filename, "text": content, "embedding": emb})
    return store


def retrieve_top_doc(query: str, store: List[Dict[str, Any]]) -> Tuple[str, str, float]:
    q_vec = np.array(
        get_embedding(query, task_type="RETRIEVAL_QUERY"), dtype=np.float32
    )
    best_doc = None
    best_score = -1.0
    for item in store:
        d_vec = item["embedding"]
        score = float(
            np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
        )
        if score > best_score:
            best_score = score
            best_doc = item
    return best_doc["source"], best_doc["text"], best_score


def classify_rag(text: str, store: List[Dict[str, Any]]) -> Tuple[str, str, float]:
    source, context_text, sim = retrieve_top_doc(text, store)
    prompt = (
        "Використовуючи наведений регламент компанії, визнач категорію інциденту.\n"
        "Дозволені категорії: Мережева інфраструктура, Бази даних, Безпека та"
        " доступи, Прикладне ПЗ.\n"
        "Поверни виключно назву категорії без пояснень.\n\n"
        f"Регламент ({source}): {context_text}\n\n"
        f"Текст інциденту: {text}"
    )
    return normalize_category(ask_llm(prompt)), source, sim


# =====================================================================
# ЗАВДАННЯ 4: Підхід 3 — Імітація Fine-tuning (Few-shot prompting)
# =====================================================================
def classify_few_shot(text: str) -> str:
    prompt = (
        "Визнач категорію інциденту за наведеними зразками. Поверни лише назву"
        " категорії.\n\n"
        "Текст: Помилка підключення до пулу з'єднань PostgreSQL та пошкодження індексу таблиці.\n"
        "Категорія: Бази даних\n\n"
        "Текст: Втрата пакетів на магістральному комутаторі та недоступність VLAN офісу.\n"
        "Категорія: Мережева інфраструктура\n\n"
        "Текст: Виявлено фішингову розсилку та компрометацію токена доступу співробітника.\n"
        "Категорія: Безпека та доступи\n\n"
        "Текст: Помилка рендерингу форми замовлення в клієнтському вебдодатку ERP.\n"
        "Категорія: Прикладне ПЗ\n\n"
        "Текст: Зафіксовано код інциденту ERR-GW-409 під час ініціалізації тунелю.\n"
        "Категорія: Мережева інфраструктура\n\n"
        f"Текст: {text}\n"
        "Категорія:"
    )
    return normalize_category(ask_llm(prompt))


def main():
    print("=" * 75)
    print("ЛАБОРАТОРНА РОБОТА №6: Порівняння Prompt Engineering, RAG та Fine-tuning")
    print("=" * 75)

    print("\n[Ініціалізація бази регламентів для RAG...]")
    rag_store = build_rag_store("docs")
    print(f"Створено та векторизовано {len(rag_store)} документи у папці 'docs/'.\n")

    results = []
    correct_pe = 0
    correct_rag = 0
    correct_ft = 0

    for item in TEST_DATASET:
        idx = item["id"]
        text = item["text"]
        expected = item["expected"]

        print(f"--- Тест №{idx} (Еталон: {expected}) ---")
        print(f"Текст: {text}")

        pred_pe = classify_prompt_engineering(text)
        pred_rag, src, sim = classify_rag(text, rag_store)
        pred_ft = classify_few_shot(text)

        if pred_pe == expected:
            correct_pe += 1
        if pred_rag == expected:
            correct_rag += 1
        if pred_ft == expected:
            correct_ft += 1

        print(
            f"  1. Prompt Engineering : {pred_pe} {'[OK]' if pred_pe == expected else '[ERR]'}"
        )
        print(
            f"  2. RAG ({src}, sim={sim:.3f}) : {pred_rag} {'[OK]' if pred_rag == expected else '[ERR]'}"
        )
        print(
            f"  3. Імітація Fine-tuning: {pred_ft} {'[OK]' if pred_ft == expected else '[ERR]'}\n"
        )

        results.append((idx, expected, pred_pe, pred_rag, pred_ft))

    total = len(TEST_DATASET)
    acc_pe = (correct_pe / total) * 100
    acc_rag = (correct_rag / total) * 100
    acc_ft = (correct_ft / total) * 100

    print("=" * 75)
    print("ПІДСУМКОВА ПОРІВНЯЛЬНА ТАБЛИЦЯ (ЗАВДАННЯ 5)")
    print("=" * 75)
    print(
        f"{'№':<3} | {'Еталон':<24} | {'Prompt Eng.':<24} | {'RAG':<24} | {'Few-shot (FT)':<24}"
    )
    print("-" * 108)
    for idx, exp, pe, rag, ft in results:
        print(f"{idx:<3} | {exp:<24} | {pe:<24} | {rag:<24} | {ft:<24}")
    print("-" * 108)
    print(
        f"Точність (Accuracy): Prompt Engineering = {correct_pe}/{total} ({acc_pe:.1f}%)"
        f" | RAG = {correct_rag}/{total} ({acc_rag:.1f}%)"
        f" | Імітація Fine-tuning = {correct_ft}/{total} ({acc_ft:.1f}%)"
    )


if __name__ == "__main__":
    main()
