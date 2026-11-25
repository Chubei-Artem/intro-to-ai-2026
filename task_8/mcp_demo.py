import asyncio
import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

LLM_MODEL = "gemini-3.5-flash-lite"


async def run_mcp_demo():
    print("=" * 80)
    print(
        "ЛАБОРАТОРНА РОБОТА №8 (Частина 2): Демонстрація Model Context Protocol (MCP)"
    )
    print("=" * 80)

    server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mcp_server.py"
    )
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", server_script],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Отримання списку інструментів (Tools) та ресурсів (Resources) від MCP-сервера
            tools_resp = await session.list_tools()
            resources_resp = await session.list_resources()

            print("\n[MCP Client -> MCP Server] Підключення встановлено!")
            print("Доступні MCP Tools:", [t.name for t in tools_resp.tools])
            print(
                "Доступні MCP Resources:",
                [str(r.uri) for r in resources_resp.resources],
            )

            # 2. Читання MCP-ресурсу course://syllabus
            syllabus_data = await session.read_resource("course://syllabus")
            syllabus_text = syllabus_data.contents[0].text
            print(f"\n[Отримано MCP Resource 'course://syllabus']:\n{syllabus_text}")

            # 3. Конвертація MCP Tools у декларації функцій для Gemini
            gemini_declarations = []
            for t in tools_resp.tools:
                gemini_declarations.append(
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=t.input_schema,
                    )
                )
            gemini_tools = [types.Tool(function_declarations=gemini_declarations)]

            user_query = (
                "Ось мої оцінки за 6 лабораторних робіт: 95, 92, 88, 96, 94, 91. "
                "Порахуй мою статистику та оцінку ECTS за допомогою доступного інструмента "
                "і збережи короткий звіт у файл 'my_grades_summary.txt'."
            )
            print(f"\n[Запит користувача до AI-моделі]:\n{user_query}")

            messages = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=f"Довідка з MCP-ресурсу:\n{syllabus_text}\n\nЗапит: {user_query}"
                        )
                    ],
                )
            ]

            # 4. Цикл взаємодії AI-моделі з MCP-сервером
            while True:
                time.sleep(5.0)
                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        tools=gemini_tools,
                        temperature=0.1,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
                candidate = response.candidates[0]
                messages.append(candidate.content)

                fn_calls = [
                    p.function_call for p in candidate.content.parts if p.function_call
                ]
                if not fn_calls:
                    print(f"\n[Фінальна відповідь AI-моделі]:\n{response.text.strip()}")
                    break

                fn_responses = []
                for fc in fn_calls:
                    args = dict(fc.args) if fc.args else {}
                    print(
                        f"\n[AI-модель -> MCP Client -> MCP Server] Виклик tool: {fc.name}({args})"
                    )
                    mcp_result = await session.call_tool(fc.name, arguments=args)
                    tool_output = mcp_result.content[0].text
                    print(f"[MCP Server -> MCP Client] Результат: {tool_output}")
                    fn_responses.append(
                        types.Part.from_function_response(
                            name=fc.name, response={"result": tool_output}
                        )
                    )
                messages.append(types.Content(role="user", parts=fn_responses))


if __name__ == "__main__":
    asyncio.run(run_mcp_demo())
