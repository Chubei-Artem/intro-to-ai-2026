import os
import time
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import numpy as np

# Завантажуємо API-ключ із .env
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Моделі згідно з твоїми квотами Google AI Studio
LLM_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"


# =====================================================================
# ЗАВДАННЯ 1: Підготовка набору документів (4 тематичні файли .txt)
# =====================================================================
def create_knowledge_base(docs_dir: str = "docs") -> List[str]:
    os.makedirs(docs_dir, exist_ok=True)

    documents = {
        "doc_1_cnn_basics.txt": (
            "Згорткові нейронні мережі (CNN) спеціалізуються на обробці"
            " візуальних даних. Головною операцією є дискретна згортка, де ядро"
            " (фільтр) розміром 3x3 або 5x5 ковзає по матриці пікселів із заданим"
            " кроком (stride). Завдяки механізму спільних ваг (weight sharing) та"
            " локальних рецептивних полів CNN досягають інваріантності до зсуву"
            " об'єктів та суттєво зменшують кількість параметрів порівняно з"
            " повнозв'язними мережами (MLP). Після згорткових шарів зазвичай"
            " застосовують операцію підвибірки Max Pooling 2x2 для зменшення"
            " просторової розмірності карт ознак."
        ),
        "doc_2_resnet_architectures.txt": (
            "При збільшенні глибини класичних нейронних мереж понад 20 шарів"
            " виникає проблема деградації точності та згасання градієнтів під час"
            " зворотного поширення помилки. У 2015 році дослідники Microsoft"
            " запропонували архітектуру ResNet (Residual Network). Її ключовою"
            " інновацією є залишкові блоки із пропускними зв'язками"
            " (skip-connections), які обчислюють функцію H(x) = F(x) + x. Це"
            " дозволяє градієнтам безпосередньо протікати крізь шари без"
            " експоненційного затухання, завдяки чому стало можливим успішно"
            " навчати мережі глибиною 50, 101 та 152 шари (ResNet-50, ResNet-152)."
        ),
        "doc_3_optimizers_adam.txt": (
            "Ефективність навчання глибоких мереж критично залежить від алгоритму"
            " оптимізації. Класичний стохастичний градієнтний спуск (SGD)"
            " використовує єдину фіксовану швидкість навчання (learning rate) для"
            " всіх параметрів. Натомість адаптивний оптимізатор Adam (Adaptive"
            " Moment Estimation) обчислює експоненційно зважені ковзні середні"
            " першого моменту (середнє градієнтів, параметр beta_1 = 0.9) та"
            " другого моменту (нецентрована дисперсія градієнтів, параметр beta_2"
            " = 0.999). Стандартне початкове значення швидкості навчання для Adam"
            " становить lr = 0.001, що забезпечує швидку збіжність на початкових"
            " етапах тренування."
        ),
        "doc_4_regularization_methods.txt": (
            "Перенавчання (overfitting) виникає, коли модель запам'ятовує"
            " випадковий шум навчальної вибірки і втрачає здатність до"
            " узагальнення на тестових даних. Для боротьби з цим явищем"
            " застосовують методи регуляризації. Метод Dropout під час кожної"
            " ітерації навчання випадково вимикає нейрони з імовірністю p"
            " (зазвичай p = 0.2 для вхідних шарів та p = 0.5 для прихованих"
            " повнозв'язних шарів). L2-регуляризація (Weight Decay) додає до"
            " функції втрат штраф, пропорційний сумі квадратів ваг моделі, що"
            " запобігає надмірному зростанню окремих коефіцієнтів."
        ),
    }

    created_files = []
    for filename, content in documents.items():
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        created_files.append(filepath)

    return created_files


# =====================================================================
# ЗАВДАННЯ 2: Завантаження документів і розбиття на chunks із метаданими
# =====================================================================
def load_and_chunk_documents(
    filepaths: List[str], chunk_size: int = 35, overlap: int = 8
) -> List[Dict[str, Any]]:
    chunks_data = []
    chunk_counter = 1

    for path in filepaths:
        source_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        words = text.split()
        start = 0
        step = chunk_size - overlap

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks_data.append(
                {
                    "chunk_id": f"chunk_{chunk_counter:02d}",
                    "source": source_name,
                    "text": chunk_text,
                }
            )
            chunk_counter += 1
            if end >= len(words):
                break
            start += step

    return chunks_data


# =====================================================================
# ЗАВДАННЯ 3: Створення Embeddings через Google GenAI API
# =====================================================================
def get_embedding(
    text: str, task_type: str = "RETRIEVAL_DOCUMENT", retries: int = 5
) -> List[float]:
    for attempt in range(1, retries + 1):
        try:
            time.sleep(0.8)  # Захист від спайків (ліміт embedding 100 RPM)
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return result.embeddings[0].values
        except errors.APIError as e:
            if e.code in (429, 503) and attempt < retries:
                wait_time = attempt * 5
                print(
                    f"[Embedding API зайнятий ({e.code}). Повтор через"
                    f" {wait_time} сек...]"
                )
                time.sleep(wait_time)
            else:
                raise


def ask_llm(prompt: str, retries: int = 6) -> str:
    for attempt in range(1, retries + 1):
        try:
            time.sleep(5.0)  # Захист ліміту 15 RPM для gemini-3.5-flash-lite
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return response.text.strip()
        except errors.APIError as e:
            if e.code in (429, 503) and attempt < retries:
                wait_time = attempt * 10
                print(
                    f"[LLM ліміт ({e.code}). Чекаємо {wait_time} сек..."
                    f" (спроба {attempt}/{retries})]"
                )
                time.sleep(wait_time)
            else:
                raise


# =====================================================================
# ЗАВДАННЯ 4: Векторне сховище (Vector Store) + Пошук (Retrieval)
# =====================================================================
class SimpleVectorStore:

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add(self, chunk_id: str, source: str, text: str, embedding: List[float]):
        self.records.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "text": text,
                "embedding": np.array(embedding, dtype=np.float32),
            }
        )

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def search(
        self, query_embedding: List[float], top_k: int = 2
    ) -> List[Tuple[Dict[str, Any], float]]:
        q_vec = np.array(query_embedding, dtype=np.float32)
        scored = []
        for rec in self.records:
            score = self._cosine_similarity(q_vec, rec["embedding"])
            scored.append((rec, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# =====================================================================
# ЗАВДАННЯ 6: Повний RAG-ланцюжок (Query -> Embedding -> Retrieval -> LLM)
# =====================================================================
def rag_answer(
    question: str, vector_store: SimpleVectorStore, top_k: int = 2
) -> Dict[str, Any]:
    q_emb = get_embedding(question, task_type="RETRIEVAL_QUERY")
    retrieved = vector_store.search(q_emb, top_k=top_k)

    context_blocks = []
    sources = []
    chunk_ids = []
    for rec, score in retrieved:
        context_blocks.append(
            f"[Джерело: {rec['source']} | {rec['chunk_id']}]: {rec['text']}"
        )
        if rec["source"] not in sources:
            sources.append(rec["source"])
        chunk_ids.append(f"{rec['chunk_id']} (sim={score:.3f})")

    retrieved_context = "\n".join(context_blocks)

    rag_prompt = (
        "Використовуючи виключно наведений нижче контекст, дай точну і лаконічну"
        " відповідь на запитання користувача.\n"
        "Якщо відповіді на запитання немає в контексті, прямо повідом про це і не"
        " вигадуй факти.\n\n"
        f"Контекст:\n{retrieved_context}\n\n"
        f"Запитання:\n{question}"
    )

    answer = ask_llm(rag_prompt)
    return {
        "question": question,
        "chunks": chunk_ids,
        "sources": sources,
        "retrieved_text": retrieved[0][0]["text"],
        "answer": answer,
    }


def main():
    print("=" * 70)
    print(
        "ЗАВДАННЯ 1–4: Створення бази документів, Chunking, Embeddings та Vector Store"
    )
    print("=" * 70)

    files = create_knowledge_base("docs")
    print(f"Створено {len(files)} документи у папці 'docs/'.")

    chunks = load_and_chunk_documents(files, chunk_size=35, overlap=8)
    print(f"Документи розбито на {len(chunks)} фрагментів (chunks):")
    for ch in chunks:
        print(f"  {ch['chunk_id']} -> {ch['source']} ({len(ch['text'].split())} слів)")

    vector_store = SimpleVectorStore()
    print("\nГенерація embeddings та збереження у Vector Store...")
    for ch in chunks:
        emb = get_embedding(ch["text"], task_type="RETRIEVAL_DOCUMENT")
        vector_store.add(ch["chunk_id"], ch["source"], ch["text"], emb)

    sample_vec = vector_store.records[0]["embedding"]
    print(
        f"Готово! У сховищі збережено {len(vector_store.records)} векторів."
        f" Розмірність кожного вектора: {len(sample_vec)}."
    )
    print(
        "Приклад початку вектора для chunk_01:"
        f" [{', '.join(f'{x:.4f}' for x in sample_vec[:5])}, ...]\n"
    )

    # =====================================================================
    # ЗАВДАННЯ 5 та 7: Перевірка пошуку (Retrieval) та відповідей RAG (5 запитів)
    # =====================================================================
    print("=" * 70)
    print("ЗАВДАННЯ 5–7: Перевірка роботи RAG-системи на 5 запитах")
    print("=" * 70)

    test_questions = [
        "Який розмір фільтрів та операцію підвибірки використовують у класичних CNN?",
        "Яку математичну функцію обчислюють залишкові блоки в архітектурі ResNet і скільки шарів вдалося навчити?",
        "Які стандартні значення параметрів beta_1, beta_2 та learning rate використовує оптимізатор Adam?",
        "З якою ймовірністю p зазвичай вимикають нейрони в методі Dropout для вхідних та прихованих шарів?",
        "Як L2-регуляризація запобігає перенавчанню нейронної мережі?",
    ]

    for idx, q in enumerate(test_questions, 1):
        res = rag_answer(q, vector_store, top_k=2)
        print(f"\n--- Запит №{idx} ---")
        print(f"Питання: {res['question']}")
        print(f"Знайдені чанки: {', '.join(res['chunks'])}")
        print(f"Джерела: {', '.join(res['sources'])}")
        print(f"Топ-фрагмент: {res['retrieved_text']}")
        print(f"Відповідь LLM (RAG): {res['answer']}")

    # =====================================================================
    # ЗАВДАННЯ 8: Дослідження обмежень RAG (запитання поза базою знань)
    # =====================================================================
    print("\n" + "=" * 70)
    print("ЗАВДАННЯ 8: Дослідження обмежень RAG (відсутня інформація в базі)")
    print("=" * 70)

    out_of_domain_q = (
        "Яка архітектура застосовується у великій мовній моделі GPT-4 та скільки"
        " в ній параметрів?"
    )
    res_ood = rag_answer(out_of_domain_q, vector_store, top_k=2)
    print(f"Питання поза базою: {res_ood['question']}")
    print(f"Знайдені чанки (найближчі за вектором): {', '.join(res_ood['chunks'])}")
    print(f"Відповідь LLM (RAG): {res_ood['answer']}\n")

    # =====================================================================
    # ДОДАТКОВЕ ЗАВДАННЯ*: Порівняння LLM без RAG та LLM із RAG
    # =====================================================================
    print("=" * 70)
    print("ДОДАТКОВЕ ЗАВДАННЯ*: Порівняння LLM без RAG та LLM із RAG")
    print("=" * 70)

    compare_q = (
        "Які конкретні значення параметрів beta_1, beta_2 та початкового"
        " learning rate рекомендовані в нашій документації для оптимізатора Adam"
        " і чому?"
    )
    no_rag_answer = ask_llm(compare_q)
    with_rag_res = rag_answer(compare_q, vector_store, top_k=2)

    print(f"Запитання: {compare_q}\n")
    print(f"[1. Відповідь LLM БЕЗ RAG]:\n{no_rag_answer}\n")
    print(
        f"[2. Відповідь LLM ІЗ RAG (джерело: {', '.join(with_rag_res['sources'])})]:\n{with_rag_res['answer']}\n"
    )


if __name__ == "__main__":
    main()
