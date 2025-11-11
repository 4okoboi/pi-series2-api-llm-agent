import traceback
from datetime import datetime, timedelta
from typing import Optional
import uuid
from caldav import DAVClient, timezone
from dotenv import load_dotenv
import pytz
import os

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_cerebras import ChatCerebras
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from pydantic import ValidationError

from agent_models import PlanRequest

load_dotenv()

APPLE_ID = os.environ.get("APPLE_ID", "").strip()
ICLOUD_APP_PASSWORD = os.environ.get("APPLE_ID_PASSWORD", "").strip()
ICLOUD_CALENDAR_URL = os.environ.get("ICLOUD_CALENDAR_URL", "").strip()
ICLOUD_CALENDAR_NAME = os.environ.get("ICLOUD_CALENDAR_NAME", "").strip() or "Домашний"
LOCAL_TIMEZONE = os.environ.get("LOCAL_TIMEZONE", "").strip() or "UTC"
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "").strip()

LOCAL_TZ = pytz.timezone(LOCAL_TIMEZONE)

PARSE_PROMPT = """
Ты помощник-планировщик. Твоя задача - извлекать строгую структуру PlanRequest из команды пользователя.
Требования:
- Не придумывай то, чего нет.
- Если в команде есть URL, то помести его в поле location
- Поле duration_minutes по умолчанию 60, если явно не указано иное.
ВЕРНИ ТОЛЬКО JSON (без остальных символов), соответствующий схеме:
class PlanRequest(BaseModel):
    title: str = Field(..., description="Название события")
    event_datetime: str = Field(..., description="Дата и время начала события в ISO формате, например '2025-11-06T15:30:00'")
    duration_minutes: Optional[int] = Field(60, description="Длительность события в минутах")
    location: Optional[str] = Field(..., description="Местоположение события или ссылка на мероприятие")
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PARSE_PROMPT),
        ("human", "{user_text}")
    ]
)

llm = ChatCerebras(
    model='gpt-oss-120b',
    api_key=CEREBRAS_API_KEY
)

request_parser = JsonOutputParser(pydantic_object=PlanRequest)

extract_chain = prompt | llm | request_parser


def _get_icloud_calendar():
    """
    Функция для поиска календарей и получения требуемого.
    """
    client = DAVClient(
        url=ICLOUD_CALENDAR_URL,
        username=APPLE_ID,
        password=ICLOUD_APP_PASSWORD
    )
    principal = client.principal()
    calendars = principal.calendars()
    if not calendars:
        raise RuntimeError("Нет доступных календарей в iCloud.")
    if ICLOUD_CALENDAR_NAME:
        for c in calendars:
            name = c.name.strip()
            if name == ICLOUD_CALENDAR_NAME:
                return c

        avail = ", ".join(
            [str(cal.name) for cal in calendars]
        )
        raise RuntimeError(f"Календарь '{ICLOUD_CALENDAR_NAME}' не найден. Доступные: {avail}")

    return calendars[0]


def _build_ics(summary: str, dtstart: datetime, dtend: datetime, location: Optional[str]) -> str:
    """
    Функция для сборки объекта события из распаршенной информации
    """
    event_data = f"""
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:{uuid.uuid4()}
DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{dtstart.strftime('%Y%m%dT%H%M%SZ')}
DTEND:{dtend.strftime('%Y%m%dT%H%M%SZ')}
SUMMARY:{summary}
LOCATION:{location}
END:VEVENT
END:VCALENDAR
    """

    print(event_data)

    return event_data


@tool("add_to_icloud_calendar", return_direct=True)
def add_to_icloud_calendar(
        title: str,
        event_datetime: str,
        duration_minutes: int = 60,
        location: Optional[str] = None,
) -> str:
    """
    Добавляет событие в iCloud Calendar через CalDAV.
    Параметры:
      - title: название события
      - when_text: естественная фраза времени на русском
      - duration_minutes: длительность (мин)
      - location: место или ссылка на звонок/встречу
    Возвращает человеко-читаемое подтверждение с финальными датами/временем.
    """
    dt_start_local = datetime.fromisoformat(event_datetime).replace(tzinfo=LOCAL_TZ)
    dt_end_local = dt_start_local + timedelta(minutes=int(duration_minutes or 60))

    ics_data = _build_ics(title, dt_start_local, dt_end_local, location)

    calendar = _get_icloud_calendar()
    calendar.add_event(ics_data)

    start_str = dt_start_local.strftime("%Y-%m-%d %H:%M (%Z)")
    end_str = dt_end_local.strftime("%Y-%m-%d %H:%M (%Z)")
    loc_info = f"; место: {location}" if location else ""
    return f"Событие добавлено: «{title}» — {start_str} → {end_str}{loc_info}"



TOOLS = [add_to_icloud_calendar]


memory = MemorySaver()
agent_graph = create_agent(
    model=llm.bind_tools(TOOLS),
    tools=TOOLS,
    checkpointer=memory
)


def test_func(user_text: str) -> str:
    plan: dict = extract_chain.invoke({"user_text": user_text})
    print(plan)
    plan: PlanRequest = PlanRequest.model_validate(plan)
    confirmation = add_to_icloud_calendar.invoke({
        "title": plan.title,
        "event_datetime": plan.event_datetime,
        "duration_minutes": plan.duration_minutes or 60,
        "location": plan.location
    })
    return confirmation


if __name__ == "__main__":
    print("Пример: введите команду типа 'завтра в 15:00 звонок с Ильсуром https://zoom.us/abc123'")
    try:
        text = input("> ").strip() + f"Текущее время: {str(datetime.now())}"
        if not text:
            raise SystemExit(0)

        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=f"Добавь событие: {text}\n текущая дата и время: {str(datetime.now())}")]},
            config={"configurable": {"thread_id": "cli"}}
        )
        print(result["messages"][-1].content)

    except ValidationError as ve:
        print("Ошибка в структуре запроса:", ve)
    except Exception as e:
        print(traceback.format_exc())
        print("Ошибка:", e)
