import os
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("StudentAssistantMCP")


@mcp.resource("course://syllabus")
def get_course_syllabus() -> str:
    """Повертає програму та систему оцінювання навчального курсу з ШІ."""
    return (
        "Курс: Вступ до штучного інтелекту (3 курс).\n"
        "Лабораторні роботи №1–№8: максимум 100 балів.\n"
        "Шкала ECTS: 90–100 = A (Відмінно), 82–89 = B (Добре), 74–81 = C (Добре)."
    )


@mcp.tool()
def calculate_grade_stats(scores_csv: str) -> str:
    """Обчислює середній бал, суму та підсумкову оцінку ECTS за списком балів через кому (наприклад, '92, 88, 95, 90')."""
    try:
        scores = [float(x.strip()) for x in scores_csv.split(",") if x.strip()]
        if not scores:
            return "Помилка: список балів порожній."
        avg_score = round(sum(scores) / len(scores), 2)
        if avg_score >= 90:
            ects = "A (Відмінно)"
        elif avg_score >= 82:
            ects = "B (Добре)"
        elif avg_score >= 74:
            ects = "C (Добре)"
        else:
            ects = "D/E (Задовільно)"
        return (
            f"Кількість робіт: {len(scores)}, Середній бал: {avg_score}, "
            f"Мін/Макс: {min(scores)}/{max(scores)}, Оцінка ECTS: {ects}"
        )
    except ValueError:
        return f"Помилка формату вхідних даних: '{scores_csv}'."


@mcp.tool()
def save_study_note(filename: str, content: str) -> str:
    """Зберігає навчальний конспект у текстовий файл у директорії notes/."""
    os.makedirs("notes", exist_ok=True)
    safe_name = os.path.basename(filename)
    filepath = os.path.join("notes", safe_name)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Файл успішно збережено за шляхом: {filepath} ({len(content)} символів)."


if __name__ == "__main__":
    mcp.run(transport="stdio")
