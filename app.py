from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request as urlrequest

from ontology import LearningOntology
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "app.db"

app = Flask(__name__)
app.secret_key = "dev-secret-change-me"

SEED_CONTENT_PATH = BASE_DIR / "content" / "seed_content.json"


def load_seed_content() -> tuple[list[dict], list[dict], dict[str, str], dict]:
    """Загружает стартовый контент курса и онтологию из JSON-файла."""
    if not SEED_CONTENT_PATH.exists():
        return [], [], {}, {}
    try:
        raw_content = json.loads(SEED_CONTENT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], [], {}, {}

    topics = raw_content.get("topics", [])
    tasks = raw_content.get("tasks", [])
    materials = raw_content.get("materials", {})
    ontology = raw_content.get("ontology", {})

    if not isinstance(topics, list):
        topics = []
    if not isinstance(tasks, list):
        tasks = []
    if not isinstance(materials, dict):
        materials = {}
    if not isinstance(ontology, dict):
        ontology = {}

    return topics, tasks, {str(key): str(value) for key, value in materials.items()}, ontology


SEED_TOPICS, SEED_TASKS, SEED_MATERIALS, SEED_ONTOLOGY = load_seed_content()
LEARNING_ONTOLOGY = LearningOntology.from_seed_content(SEED_TOPICS, SEED_TASKS, SEED_ONTOLOGY)


def get_db_connection() -> sqlite3.Connection:
    """Возвращает подключение к SQLite с доступом к колонкам по именам."""
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Добавляет колонку в таблицу, если приложение обновилось, а база уже существует.
def ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {row[1] for row in columns}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


# Инициализация схемы БД и миграция недостающих колонок.
def init_db() -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                answers TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                created_by INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(created_by) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                task_type TEXT NOT NULL,
                due_date TEXT,
                content_json TEXT NOT NULL,
                created_by INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(topic_id) REFERENCES topics(id),
                FOREIGN KEY(created_by) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                task_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                answers TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(task_id) REFERENCES tasks(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptive_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                source_concepts TEXT NOT NULL DEFAULT '[]',
                content_json TEXT NOT NULL,
                llm_used INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptive_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                adaptive_task_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                answers TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(adaptive_task_id) REFERENCES adaptive_tasks(id)
            )
            """
        )
        # Мягкая миграция для уже существующей таблицы adaptive_tasks.
        ensure_column(connection, "adaptive_tasks", "created_by", "created_by INTEGER")
        ensure_column(connection, "adaptive_tasks", "reviewed_by", "reviewed_by INTEGER")
        ensure_column(connection, "adaptive_tasks", "assigned_at", "assigned_at TEXT")
        connection.commit()

    ensure_admin_user()
    seed_content()


def ensure_admin_user() -> None:
    """Создаёт учётную запись администратора по умолчанию, если её ещё нет."""
    with get_db_connection() as connection:
        admin = connection.execute(
            "SELECT id FROM users WHERE username = ?", ("admin",)
        ).fetchone()
        if admin is None:
            connection.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
                ("admin", generate_password_hash("admin")),
            )
            connection.commit()


def seed_content() -> None:
    """Первичное заполнение тем и заданий из seed-файла."""
    with get_db_connection() as connection:
        count = connection.execute("SELECT COUNT(*) AS total FROM topics").fetchone()
        if count and count["total"] > 0:
            return
        admin = connection.execute(
            "SELECT id FROM users WHERE username = ?", ("admin",)
        ).fetchone()
        if admin is None:
            return
        now = now_iso()
        for topic in SEED_TOPICS:
            connection.execute(
                """
                INSERT INTO topics (title, description, created_by, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (topic["title"], topic["description"], admin["id"], now),
            )
        connection.commit()
        topics = connection.execute("SELECT id, title FROM topics").fetchall()
        topic_map = {topic["title"]: topic["id"] for topic in topics}
        for task in SEED_TASKS:
            topic_id = topic_map.get(task["topic_title"])
            if not topic_id:
                continue
            content = json.dumps({"questions": task["questions"]}, ensure_ascii=False)
            connection.execute(
                """
                INSERT INTO tasks (
                    topic_id, title, description, task_type, due_date, content_json, created_by, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    topic_id,
                    task["title"],
                    task["description"],
                    task["task_type"],
                    None,
                    content,
                    admin["id"],
                    now,
                ),
            )
        connection.commit()


def log_activity(user_id: int | None, event_type: str, metadata: dict | None = None) -> None:
    if user_id is None:
        return
    payload = json.dumps(metadata or {}, ensure_ascii=False)
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO activity_logs (user_id, event_type, metadata, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, event_type, payload, now_iso()),
        )
        connection.commit()


def current_user() -> sqlite3.Row | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    with get_db_connection() as connection:
        return connection.execute(
            "SELECT id, username, is_admin FROM users WHERE id = ?", (user_id,)
        ).fetchone()


def load_task(task_id: int) -> sqlite3.Row | None:
    with get_db_connection() as connection:
        return connection.execute(
            """
            SELECT tasks.*, topics.title AS topic_title
            FROM tasks
            JOIN topics ON tasks.topic_id = topics.id
            WHERE tasks.id = ?
            """,
            (task_id,),
        ).fetchone()


def parse_questions(task_row: sqlite3.Row) -> list[dict]:
    raw = task_row["content_json"] if task_row else "{}"
    try:
        content = json.loads(raw)
    except json.JSONDecodeError:
        content = {}
    return content.get("questions", [])


def evaluate_quiz(questions: list[dict], form: dict) -> tuple[int, dict]:
    score = 0
    stored_answers: dict[str, str | None] = {}
    for index, question in enumerate(questions):
        answer_value = form.get(f"question-{index}")
        stored_answers[str(index)] = answer_value
        if answer_value == question.get("answer"):
            score += 1
    return score, stored_answers


def weak_concepts(area_stats: list[dict], threshold: float = 0.7) -> list[str]:
    """Возвращает концепты, по которым точность ниже заданного порога."""
    return [item["area"] for item in area_stats if item["accuracy"] < threshold]


# Последние адаптивные наборы, которые уже назначены студенту или завершены им.
def recent_adaptive_tasks(user_id: int, limit: int = 5) -> list[sqlite3.Row]:
    with get_db_connection() as connection:
        return connection.execute(
            """
            SELECT id, title, description, status, llm_used, created_at
            FROM adaptive_tasks
            WHERE user_id = ? AND status IN ('assigned', 'completed')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()


def wrong_question_examples(user_id: int, concepts: list[str], limit: int = 8) -> list[dict]:
    """Собирает примеры ошибок студента для таргетированной генерации заданий."""
    concept_set = set(concepts)
    examples: list[dict] = []
    with get_db_connection() as connection:
        attempts = connection.execute(
            """
            SELECT task_attempts.answers, tasks.content_json
            FROM task_attempts
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE task_attempts.user_id = ?
            ORDER BY task_attempts.created_at DESC
            LIMIT 20
            """,
            (user_id,),
        ).fetchall()

    for attempt in attempts:
        if len(examples) >= limit:
            break
        try:
            answers = json.loads(attempt["answers"])
        except json.JSONDecodeError:
            answers = {}
        try:
            questions = json.loads(attempt["content_json"]).get("questions", [])
        except json.JSONDecodeError:
            questions = []
        for index, question in enumerate(questions):
            concept = LEARNING_ONTOLOGY.concept_for_question(question, "Общее")
            if concept_set and concept not in concept_set:
                continue
            selected = answers.get(str(index))
            if selected == question.get("answer"):
                continue
            examples.append(
                {
                    "question": question.get("question", ""),
                    "correct": question.get("answer", ""),
                    "selected": selected or "нет ответа",
                    "concept": concept,
                }
            )
            if len(examples) >= limit:
                break
    return examples


def question_bank_for_concepts(concepts: list[str], limit: int = 12) -> list[dict]:
    """Формирует банк валидных вопросов по выбранным концептам из существующего курса."""
    concept_set = set(concepts)
    collected: list[dict] = []
    with get_db_connection() as connection:
        tasks = connection.execute(
            "SELECT content_json FROM tasks ORDER BY id"
        ).fetchall()

    for task in tasks:
        if len(collected) >= limit:
            break
        try:
            questions = json.loads(task["content_json"]).get("questions", [])
        except json.JSONDecodeError:
            questions = []
        for question in questions:
            concept = LEARNING_ONTOLOGY.concept_for_question(question, "Общее")
            if concept_set and concept not in concept_set:
                continue
            options = question.get("options", [])
            answer = question.get("answer")
            if not isinstance(options, list) or answer not in options:
                continue
            collected.append(
                {
                    "question": question.get("question", ""),
                    "options": options,
                    "answer": answer,
                    "concept": concept,
                }
            )
            if len(collected) >= limit:
                break
    return collected


def llm_generate_adaptive_questions(concepts: list[str], examples: list[dict]) -> list[dict] | None:
    """Пробует сгенерировать адаптивные вопросы через внешнюю LLM-модель."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Сгенерируй адаптивные учебные задания в формате JSON. "
                    "Нужен только JSON-массив объектов с полями question, options, answer, concept."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "concepts": concepts,
                        "wrong_examples": examples,
                        "constraints": {
                            "count": 5,
                            "single_correct_answer": True,
                            "options_count": 4,
                            "language": "ru",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    req = urlrequest.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    try:
        content = result["choices"][0]["message"]["content"]
        generated = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    return generated if isinstance(generated, list) else None


def fallback_adaptive_questions(concepts: list[str], examples: list[dict]) -> list[dict]:
    """Локальный генератор на случай, если LLM недоступна или вернула некорректный ответ."""
    questions: list[dict] = []
    pool = examples[:5] if examples else []
    for item in pool:
        distractors = [
            item["selected"],
            "Вариант A",
            "Вариант B",
            item["correct"],
        ]
        random.shuffle(distractors)
        questions.append(
            {
                "question": f"Выберите корректный ответ: {item['question']}",
                "options": list(dict.fromkeys(distractors))[:4],
                "answer": item["correct"],
                "concept": item["concept"],
            }
        )

    if not questions:
        bank = question_bank_for_concepts(concepts)
        for item in bank[:5]:
            options = list(item["options"])
            random.shuffle(options)
            questions.append(
                {
                    "question": item["question"],
                    "options": options,
                    "answer": item["answer"],
                    "concept": item["concept"],
                }
            )
    return questions


def normalize_generated_questions(raw_questions: list[dict], concepts: list[str]) -> list[dict]:
    """Нормализует и валидирует вопросы перед сохранением в базу."""
    normalized: list[dict] = []
    concept_default = concepts[0] if concepts else "Общее"
    for question in raw_questions:
        text = str(question.get("question", "")).strip()
        options = question.get("options", [])
        answer = str(question.get("answer", "")).strip()
        concept = str(question.get("concept", concept_default)).strip() or concept_default
        if not text or not isinstance(options, list):
            continue
        clean_options = [str(option).strip() for option in options if str(option).strip()]
        clean_options = list(dict.fromkeys(clean_options))
        if answer not in clean_options:
            if clean_options:
                answer = clean_options[0]
            else:
                continue
        if len(clean_options) < 2:
            continue
        normalized.append(
            {
                "question": text,
                "options": clean_options[:4],
                "answer": answer,
                "concept": concept,
            }
        )
    return normalized


def generate_adaptive_questions(user_id: int, concepts: list[str]) -> tuple[list[dict], bool]:
    """Генерирует адаптивный набор: сначала LLM, затем fallback-путь."""
    examples = wrong_question_examples(user_id, concepts)
    llm_questions = llm_generate_adaptive_questions(concepts, examples)
    if llm_questions:
        normalized = normalize_generated_questions(llm_questions, concepts)
        if normalized:
            return normalized, True
    fallback = normalize_generated_questions(fallback_adaptive_questions(concepts, examples), concepts)
    return fallback, False


def latest_attempt_concepts(user_id: int, limit: int = 3) -> list[str]:
    """Извлекает предметные концепты из последних попыток студента."""
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT tasks.content_json
            FROM task_attempts
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE task_attempts.user_id = ?
            ORDER BY task_attempts.created_at DESC
            LIMIT 5
            """,
            (user_id,),
        ).fetchall()
    concepts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        try:
            questions = json.loads(row["content_json"]).get("questions", [])
        except json.JSONDecodeError:
            questions = []
        for question in questions:
            concept = LEARNING_ONTOLOGY.concept_for_question(question, "Общее")
            if concept == "Общее" or concept in seen:
                continue
            seen.add(concept)
            concepts.append(concept)
            if len(concepts) >= limit:
                return concepts
    return concepts

# Черновики адаптивных наборов для модерации преподавателем.
def adaptive_drafts_for_user(user_id: int, limit: int = 10) -> list[sqlite3.Row]:
    with get_db_connection() as connection:
        return connection.execute(
            """
            SELECT id, title, status, llm_used, created_at
            FROM adaptive_tasks
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()



def get_monitoring_data(user_id: int) -> dict:
    with get_db_connection() as connection:
        last_activity = connection.execute(
            "SELECT MAX(created_at) AS last_seen FROM activity_logs WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        visit_count = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM activity_logs
            WHERE user_id = ? AND event_type IN ('view_topic', 'view_topics', 'view_task')
            """,
            (user_id,),
        ).fetchone()
        task_count = connection.execute("SELECT COUNT(*) AS total FROM tasks").fetchone()
        completed_count = connection.execute(
            """
            SELECT COUNT(DISTINCT task_id) AS total
            FROM task_attempts
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        avg_score = connection.execute(
            """
            SELECT AVG(CAST(score AS FLOAT) / NULLIF(total, 0)) AS avg_score
            FROM task_attempts
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        late_submissions = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM task_attempts
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE task_attempts.user_id = ?
              AND tasks.due_date IS NOT NULL
              AND task_attempts.created_at > tasks.due_date
            """,
            (user_id,),
        ).fetchone()

    last_seen_value = last_activity["last_seen"] if last_activity else None
    days_since_last = None
    if last_seen_value:
        last_seen_dt = datetime.fromisoformat(last_seen_value)
        days_since_last = (datetime.now(timezone.utc) - last_seen_dt).days
    total_tasks = task_count["total"] if task_count else 0
    completed_tasks = completed_count["total"] if completed_count else 0
    completion_rate = (completed_tasks / total_tasks) if total_tasks else 0
    average_score = avg_score["avg_score"] if avg_score else None
    return {
        "last_seen": last_seen_value,
        "days_since_last": days_since_last,
        "visits": visit_count["total"] if visit_count else 0,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "completion_rate": completion_rate,
        "average_score": average_score,
        "late_submissions": late_submissions["total"] if late_submissions else 0,
    }


def get_area_performance(user_id: int) -> list[dict]:
    areas: dict[str, dict[str, int]] = {}
    with get_db_connection() as connection:
        attempts = connection.execute(
            """
            SELECT task_attempts.answers, tasks.content_json
            FROM task_attempts
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE task_attempts.user_id = ?
            """,
            (user_id,),
        ).fetchall()

    for attempt in attempts:
        try:
            answers = json.loads(attempt["answers"])
        except json.JSONDecodeError:
            answers = {}
        try:
            content = json.loads(attempt["content_json"])
        except json.JSONDecodeError:
            content = {}
        questions = content.get("questions", [])
        for index, question in enumerate(questions):
            area = LEARNING_ONTOLOGY.concept_for_question(question, "Общее")
            areas.setdefault(area, {"correct": 0, "total": 0})
            areas[area]["total"] += 1
            if answers.get(str(index)) == question.get("answer"):
                areas[area]["correct"] += 1
    results = []
    for area, stats in areas.items():
        total = stats["total"]
        accuracy = stats["correct"] / total if total else 0
        results.append({"area": area, "accuracy": accuracy, "total": total})
    return sorted(results, key=lambda item: item["accuracy"])


def evaluate_risk(monitoring: dict) -> tuple[str, list[str]]:
    reasons = []
    if monitoring["days_since_last"] is None:
        return "Нет данных", ["Студент ещё не проявлял активности."]
    if monitoring["days_since_last"] >= 5:
        reasons.append("Не было входа в курс более 5 дней.")
    if monitoring["completion_rate"] < 0.5:
        reasons.append("Менее 50% заданий выполнено.")
    if monitoring["average_score"] is not None and monitoring["average_score"] < 0.6:
        reasons.append("Средний результат ниже 60%.")
    if len(reasons) >= 2:
        return "Высокий риск", reasons
    if reasons:
        return "Зона риска", reasons
    return "Норма", ["Риски не обнаружены."]


def material_hint(area_name: str) -> str:
    return SEED_MATERIALS.get(
        area_name,
        f"Повторите материалы по разделу «{area_name}» и выполните дополнительные упражнения.",
    )


def build_recommendations(
    monitoring: dict, area_stats: list[dict], risk_level: str
) -> list[str]:
    recommendations = []
    weak_concepts: list[str] = []
    for area in area_stats:
        if area["accuracy"] < 0.7:
            weak_concepts.append(area["area"])
            recommendations.append(f"{area['area']}: {material_hint(area['area'])}")

    for concept in LEARNING_ONTOLOGY.infer_support_concepts(weak_concepts):
        recommendations.append(
            f"Рекомендуется повторить опорный раздел «{concept}», связанный с текущими ошибками."
        )

    if risk_level in {"Высокий риск", "Зона риска"}:
        recommendations.append(
            "Рекомендуется консультация с преподавателем или наставником."
        )
    if monitoring["late_submissions"]:
        recommendations.append("Составьте план выполнения заданий с учётом сроков.")
    if not recommendations:
        recommendations.append("Продолжайте в том же темпе, результаты стабильные.")
    return recommendations


def optimize_plan(monitoring: dict) -> str:
    average_score = monitoring["average_score"] or 0
    completion_rate = monitoring["completion_rate"]
    if average_score < 0.6 and completion_rate > 0.7:
        return "Снизьте темп и уделите время повторению сложных тем."
    if average_score > 0.85 and completion_rate < 0.4:
        return "Можно ускорить прохождение разделов, добавив дополнительные задания."
    return "Сохраните текущий учебный план и периодически контролируйте прогресс."


def generate_comment_draft(
    username: str, monitoring: dict, risk_level: str, recommendations: list[str]
) -> str:
    average_score = monitoring["average_score"]
    avg_text = (
        f"Средний результат: {average_score:.0%}."
        if average_score is not None
        else "Нет оценок для расчёта среднего результата."
    )
    rec_text = " ".join(recommendations[:2])
    return (
        f"{username}, ваш текущий статус: {risk_level}. {avg_text} "
        f"Предлагаю обратить внимание на следующие шаги: {rec_text}"
    )


def support_report_for_user(user_id: int) -> dict:
    """Собирает единый аналитический отчёт для поддержки студента."""
    monitoring = get_monitoring_data(user_id)
    area_stats = get_area_performance(user_id)
    risk_level, reasons = evaluate_risk(monitoring)
    recommendations = build_recommendations(monitoring, area_stats, risk_level)
    plan_tip = optimize_plan(monitoring)
    return {
        "monitoring": monitoring,
        "area_stats": area_stats,
        "weak_concepts": weak_concepts(area_stats),
        "risk_level": risk_level,
        "risk_reasons": reasons,
        "recommendations": recommendations,
        "plan_tip": plan_tip,
    }


@app.context_processor
def inject_user() -> dict[str, sqlite3.Row | None]:
    return {"user": current_user()}


@app.route("/")
def index():
    user = current_user()
    return render_template("index.html", user=user)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Введите имя пользователя и пароль.")
            return redirect(url_for("register"))
        with get_db_connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                connection.commit()
            except sqlite3.IntegrityError:
                flash("Такой пользователь уже существует.")
                return redirect(url_for("register"))
        flash("Регистрация успешна, теперь войдите.")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db_connection() as connection:
            user = connection.execute(
                "SELECT id, password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Неверное имя пользователя или пароль.")
            return redirect(url_for("login"))
        session["user_id"] = user["id"]
        log_activity(user["id"], "login")
        flash("Вы вошли в систему.")
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Вы вышли из системы.")
    return redirect(url_for("index"))


@app.route("/topics")
def topics():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    with get_db_connection() as connection:
        topics_rows = connection.execute(
            """
            SELECT topics.id, topics.title, topics.description,
                   COUNT(tasks.id) AS task_count
            FROM topics
            LEFT JOIN tasks ON tasks.topic_id = topics.id
            GROUP BY topics.id
            ORDER BY topics.id
            """
        ).fetchall()
        progress = connection.execute(
            """
            SELECT tasks.topic_id, COUNT(DISTINCT task_attempts.task_id) AS completed
            FROM task_attempts
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE task_attempts.user_id = ?
            GROUP BY tasks.topic_id
            """,
            (user["id"],),
        ).fetchall()
    progress_map = {row["topic_id"]: row["completed"] for row in progress}
    log_activity(user["id"], "view_topics")
    return render_template(
        "topics.html", topics=topics_rows, progress_map=progress_map
    )


@app.route("/topics/<int:topic_id>")
def topic_detail(topic_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    with get_db_connection() as connection:
        topic = connection.execute(
            "SELECT id, title, description FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if topic is None:
            flash("Тема не найдена.")
            return redirect(url_for("topics"))
        tasks = connection.execute(
            """
            SELECT tasks.*, COUNT(task_attempts.id) AS attempts
            FROM tasks
            LEFT JOIN task_attempts
              ON task_attempts.task_id = tasks.id AND task_attempts.user_id = ?
            WHERE tasks.topic_id = ?
            GROUP BY tasks.id
            ORDER BY tasks.id
            """,
            (user["id"], topic_id),
        ).fetchall()
    log_activity(user["id"], "view_topic", {"topic_id": topic_id})
    return render_template("topic_detail.html", topic=topic, tasks=tasks)


@app.route("/tasks/<int:task_id>", methods=["GET", "POST"])
def task_detail(task_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    task = load_task(task_id)
    if task is None:
        flash("Задание не найдено.")
        return redirect(url_for("topics"))
    questions = parse_questions(task)

    if request.method == "POST":
        if is_owner and adaptive_task["status"] not in {"assigned", "completed"}:
            flash("Набор нельзя проходить до назначения преподавателем.")
            return redirect(url_for("support"))
        score, stored_answers = evaluate_quiz(questions, request.form)
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO task_attempts (user_id, task_id, score, total, answers, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user["id"],
                    task_id,
                    score,
                    len(questions),
                    json.dumps(stored_answers, ensure_ascii=False),
                    now_iso(),
                ),
            )
            connection.commit()
        log_activity(user["id"], "submit_task", {"task_id": task_id, "score": score})
        return render_template(
            "task_result.html",
            task=task,
            score=score,
            total=len(questions),
        )

    log_activity(user["id"], "view_task", {"task_id": task_id})
    return render_template("task_detail.html", task=task, questions=questions)


@app.route("/support")
def support():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    report = support_report_for_user(user["id"])
    comment_draft = generate_comment_draft(
        user["username"],
        report["monitoring"],
        report["risk_level"],
        report["recommendations"],
    )
    log_activity(user["id"], "view_support")
    return render_template(
        "support.html",
        report=report,
        comment_draft=comment_draft,
        for_user=user,
        adaptive_tasks=recent_adaptive_tasks(user["id"]),
    )


# Студент может запросить адаптивный набор, но назначает его преподаватель после проверки.
@app.route("/adaptive/generate", methods=["POST"])
def adaptive_generate():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))

    report = support_report_for_user(user["id"])
    concepts = [concept for concept in report["weak_concepts"] if concept != "Общее"]
    target_concepts = concepts[:3] or latest_attempt_concepts(user["id"])
    if not target_concepts:
        flash(
            "Пока недостаточно предметных данных для адаптивного набора. "
            "Сначала выполните хотя бы одно обычное задание по теме."
        )
        return redirect(url_for("support"))

    questions, llm_used = generate_adaptive_questions(user["id"], target_concepts)
    if not questions:
        flash(
            "Не удалось собрать качественный адаптивный набор по выбранным разделам. "
            "Попробуйте пройти ещё одно основное задание и повторить."
        )
        return redirect(url_for("support"))

    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO adaptive_tasks (
                user_id, title, description, source_concepts, content_json,
                llm_used, status, created_by, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?)
            """,
            (
                user["id"],
                f"Адаптивная практика: {', '.join(target_concepts)}",
                "Черновик набора сформирован автоматически и ждёт проверки преподавателя.",
                json.dumps(target_concepts, ensure_ascii=False),
                json.dumps({"questions": questions}, ensure_ascii=False),
                1 if llm_used else 0,
                user["id"],
                now_iso(),
            ),
        )
        connection.commit()
        adaptive_task_id = cursor.lastrowid

    log_activity(
        user["id"],
        "generate_adaptive_task",
        {
            "adaptive_task_id": adaptive_task_id,
            "concepts": target_concepts,
            "llm_used": llm_used,
            "status": "draft",
        },
    )
    flash("Черновик адаптивного набора отправлен преподавателю на проверку.")
    return redirect(url_for("support"))


@app.route("/adaptive/tasks/<int:adaptive_task_id>", methods=["GET", "POST"])
def adaptive_task_detail(adaptive_task_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))

    with get_db_connection() as connection:
        adaptive_task = connection.execute(
            """
            SELECT id, user_id, title, description, source_concepts, content_json, llm_used, status, created_at
            FROM adaptive_tasks
            WHERE id = ?
            """,
            (adaptive_task_id,),
        ).fetchone()
    if adaptive_task is None:
        flash("Адаптивное задание не найдено.")
        return redirect(url_for("support"))

    is_owner = adaptive_task["user_id"] == user["id"]
    is_admin = bool(user["is_admin"])
    if not is_owner and not is_admin:
        flash("Недостаточно прав для просмотра адаптивного задания.")
        return redirect(url_for("support"))

    if is_owner and adaptive_task["status"] == "draft":
        flash("Этот набор ещё не назначен преподавателем.")
        return redirect(url_for("support"))

    try:
        payload = json.loads(adaptive_task["content_json"])
    except json.JSONDecodeError:
        payload = {}
    questions = payload.get("questions", [])
    try:
        source_concepts = json.loads(adaptive_task["source_concepts"])
    except json.JSONDecodeError:
        source_concepts = []

    if request.method == "POST":
        if is_owner and adaptive_task["status"] not in {"assigned", "completed"}:
            flash("Набор нельзя проходить до назначения преподавателем.")
            return redirect(url_for("support"))
        score, stored_answers = evaluate_quiz(questions, request.form)
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO adaptive_attempts (user_id, adaptive_task_id, score, total, answers, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user["id"],
                    adaptive_task_id,
                    score,
                    len(questions),
                    json.dumps(stored_answers, ensure_ascii=False),
                    now_iso(),
                ),
            )
            connection.execute(
                "UPDATE adaptive_tasks SET status = 'completed' WHERE id = ?",
                (adaptive_task_id,),
            )
            connection.commit()
        log_activity(
            user["id"],
            "submit_adaptive_task",
            {"adaptive_task_id": adaptive_task_id, "score": score},
        )
        return render_template(
            "adaptive_task_result.html",
            adaptive_task=adaptive_task,
            score=score,
            total=len(questions),
            source_concepts=source_concepts,
        )

    return render_template(
        "adaptive_task_detail.html",
        adaptive_task=adaptive_task,
        questions=questions,
        source_concepts=source_concepts,
    )




@app.route("/ontology")
def ontology_export():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для просмотра онтологии.")
        return redirect(url_for("index"))
    return app.response_class(
        LEARNING_ONTOLOGY.to_turtle(),
        mimetype="text/turtle; charset=utf-8",
        headers={"Content-Disposition": "inline; filename=ontology.ttl"},
    )
@app.route("/admin/results")
def admin_results():
    return redirect(url_for("admin_users"))


@app.route("/admin/users")
def admin_users():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для просмотра результатов.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        users = connection.execute(
            """
            SELECT users.id, users.username,
                   COUNT(task_attempts.id) AS attempts,
                   MAX(task_attempts.created_at) AS last_attempt
            FROM users
            LEFT JOIN task_attempts ON task_attempts.user_id = users.id
            GROUP BY users.id
            ORDER BY users.username
            """
        ).fetchall()
    return render_template("admin_results.html", users=users)


@app.route("/admin/users/<int:user_id>")
def admin_user_profile(user_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для просмотра профилей.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        selected_user = connection.execute(
            "SELECT id, username, is_admin FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if selected_user is None:
            flash("Пользователь не найден.")
            return redirect(url_for("admin_users"))
        attempts_stats = connection.execute(
            """
            SELECT COUNT(*) AS attempts, MAX(created_at) AS last_attempt
            FROM task_attempts
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return render_template(
        "admin_user_profile.html",
        selected_user=selected_user,
        attempts_stats=attempts_stats,
    )


@app.route("/admin/users/<int:user_id>/results")
def admin_user_results(user_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для просмотра результатов.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        user_row = connection.execute(
            "SELECT id, username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if user_row is None:
            flash("Пользователь не найден.")
            return redirect(url_for("admin_users"))
        attempts = connection.execute(
            """
            SELECT task_attempts.id, task_attempts.score, task_attempts.total,
                   task_attempts.created_at, tasks.title AS task_title
            FROM task_attempts
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE task_attempts.user_id = ?
            ORDER BY task_attempts.created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return render_template(
        "admin_user_results.html",
        selected_user=user_row,
        attempts=attempts,
    )


@app.route("/admin/users/<int:user_id>/results/attempts/<int:result_id>")
def admin_attempt_detail(user_id: int, result_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для просмотра результатов.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        user_row = connection.execute(
            "SELECT id, username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if user_row is None:
            flash("Пользователь не найден.")
            return redirect(url_for("admin_users"))
        result = connection.execute(
            """
            SELECT task_attempts.id, task_attempts.score, task_attempts.total,
                   task_attempts.created_at, task_attempts.answers,
                   tasks.title AS task_title, tasks.content_json
            FROM task_attempts
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE task_attempts.id = ? AND task_attempts.user_id = ?
            """,
            (result_id, user_id),
        ).fetchone()
    if result is None:
        flash("Попытка не найдена.")
        return redirect(url_for("admin_user_results", user_id=user_id))
    try:
        answers = json.loads(result["answers"] or "{}")
    except json.JSONDecodeError:
        answers = {}
    try:
        content = json.loads(result["content_json"] or "{}")
    except json.JSONDecodeError:
        content = {}
    questions = content.get("questions", [])
    return render_template(
        "admin_attempt_detail.html",
        selected_user=user_row,
        result=result,
        questions=questions,
        answers=answers,
    )


@app.route("/admin/support")
def admin_support():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для просмотра отчётов.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        users = connection.execute(
            "SELECT id, username FROM users ORDER BY username"
        ).fetchall()
    reports = []
    for user_row in users:
        report = support_report_for_user(user_row["id"])
        reports.append({"user": user_row, "report": report})
    return render_template("admin_support.html", reports=reports)


@app.route("/admin/support/users/<int:user_id>")
def admin_support_detail(user_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для просмотра отчётов.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        user_row = connection.execute(
            "SELECT id, username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if user_row is None:
        flash("Пользователь не найден.")
        return redirect(url_for("admin_support"))
    report = support_report_for_user(user_id)
    comment_draft = generate_comment_draft(
        user_row["username"],
        report["monitoring"],
        report["risk_level"],
        report["recommendations"],
    )
    return render_template(
        "admin_support_detail.html",
        selected_user=user_row,
        report=report,
        comment_draft=comment_draft,
        adaptive_drafts=adaptive_drafts_for_user(user_id),
    )


# Преподаватель запускает генерацию черновика для выбранного студента.
@app.route("/admin/support/users/<int:user_id>/adaptive/generate", methods=["POST"])
def admin_generate_adaptive_for_user(user_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для генерации адаптивного набора.")
        return redirect(url_for("index"))

    report = support_report_for_user(user_id)
    concepts = [concept for concept in report["weak_concepts"] if concept != "Общее"]
    target_concepts = concepts[:3] or latest_attempt_concepts(user_id)
    if not target_concepts:
        flash("Недостаточно данных для генерации черновика.")
        return redirect(url_for("admin_support_detail", user_id=user_id))

    questions, llm_used = generate_adaptive_questions(user_id, target_concepts)
    if not questions:
        flash("Не удалось сформировать качественный черновик набора.")
        return redirect(url_for("admin_support_detail", user_id=user_id))

    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO adaptive_tasks (
                user_id, title, description, source_concepts, content_json,
                llm_used, status, created_by, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?)
            """,
            (
                user_id,
                f"Адаптивная практика: {', '.join(target_concepts)}",
                "Черновик сформирован автоматически. Проверьте и назначьте студенту.",
                json.dumps(target_concepts, ensure_ascii=False),
                json.dumps({"questions": questions}, ensure_ascii=False),
                1 if llm_used else 0,
                user["id"],
                now_iso(),
            ),
        )
        connection.commit()
    flash("Черновик адаптивного набора создан.")
    return redirect(url_for("admin_support_detail", user_id=user_id))


# Страница модерации: преподаватель проверяет вопросы и назначает набор студенту.
@app.route("/admin/adaptive/<int:adaptive_task_id>/review", methods=["GET", "POST"])
def admin_review_adaptive_task(adaptive_task_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для проверки адаптивных наборов.")
        return redirect(url_for("index"))

    with get_db_connection() as connection:
        adaptive_task = connection.execute(
            """
            SELECT adaptive_tasks.*, users.username
            FROM adaptive_tasks
            JOIN users ON users.id = adaptive_tasks.user_id
            WHERE adaptive_tasks.id = ?
            """,
            (adaptive_task_id,),
        ).fetchone()
    if adaptive_task is None:
        flash("Адаптивный набор не найден.")
        return redirect(url_for("admin_support"))

    try:
        payload = json.loads(adaptive_task["content_json"])
    except json.JSONDecodeError:
        payload = {"questions": []}

    if request.method == "POST":
        action = request.form.get("action", "save")
        questions_raw = request.form.get("questions_json", "")
        try:
            questions = json.loads(questions_raw)
            questions = normalize_generated_questions(questions, [])
        except json.JSONDecodeError:
            questions = []
        if not questions:
            flash("Сохранение невозможно: проверьте JSON вопросов.")
            return render_template(
                "admin_adaptive_review.html",
                adaptive_task=adaptive_task,
                questions_json=questions_raw,
                questions=payload.get("questions", []),
            )

        new_status = "draft" if action == "save" else "assigned"
        assigned_at = now_iso() if action == "assign" else adaptive_task["assigned_at"]
        with get_db_connection() as connection:
            connection.execute(
                """
                UPDATE adaptive_tasks
                SET content_json = ?, status = ?, reviewed_by = ?, assigned_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps({"questions": questions}, ensure_ascii=False),
                    new_status,
                    user["id"],
                    assigned_at,
                    adaptive_task_id,
                ),
            )
            connection.commit()

        if action == "assign":
            flash("Набор назначен студенту.")
        else:
            flash("Черновик сохранён.")
        return redirect(url_for("admin_review_adaptive_task", adaptive_task_id=adaptive_task_id))

    return render_template(
        "admin_adaptive_review.html",
        adaptive_task=adaptive_task,
        questions=payload.get("questions", []),
        questions_json=json.dumps(payload.get("questions", []), ensure_ascii=False, indent=2),
    )


@app.route("/admin/adaptive/<int:adaptive_task_id>/delete", methods=["POST"])
def admin_delete_adaptive_task(adaptive_task_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для удаления адаптивного набора.")
        return redirect(url_for("index"))

    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT id, user_id FROM adaptive_tasks WHERE id = ?",
            (adaptive_task_id,),
        ).fetchone()
        if row is None:
            flash("Набор уже удалён.")
            return redirect(url_for("admin_support"))
        connection.execute(
            "DELETE FROM adaptive_attempts WHERE adaptive_task_id = ?",
            (adaptive_task_id,),
        )
        connection.execute("DELETE FROM adaptive_tasks WHERE id = ?", (adaptive_task_id,))
        connection.commit()
    flash("Адаптивный набор удалён.")
    return redirect(url_for("admin_support_detail", user_id=row["user_id"]))


@app.route("/teacher/topics")
def teacher_topics():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для работы с темами.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        topics_rows = connection.execute(
            "SELECT id, title, description FROM topics ORDER BY id"
        ).fetchall()
    return render_template("teacher_topics.html", topics=topics_rows)


@app.route("/teacher/topics/new", methods=["GET", "POST"])
def teacher_new_topic():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для работы с темами.")
        return redirect(url_for("index"))
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        if not title or not description:
            flash("Заполните название и описание темы.")
            return redirect(url_for("teacher_new_topic"))
        with get_db_connection() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO topics (title, description, created_by, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (title, description, user["id"], now_iso()),
                )
                connection.commit()
            except sqlite3.IntegrityError:
                flash("Тема с таким названием уже существует.")
                return redirect(url_for("teacher_new_topic"))
        flash("Тема добавлена.")
        return redirect(url_for("teacher_topics"))
    return render_template("teacher_topic_form.html")


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
def admin_delete_user(user_id: int):
    """Удаление пользователя и всех связанных результатов."""
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для удаления пользователей.")
        return redirect(url_for("index"))
    if user_id == user["id"]:
        flash("Нельзя удалить текущего администратора.")
        return redirect(url_for("admin_users"))

    with get_db_connection() as connection:
        target = connection.execute(
            "SELECT id, username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if target is None:
            flash("Пользователь не найден.")
            return redirect(url_for("admin_users"))
        is_admin_target = connection.execute(
            "SELECT is_admin FROM users WHERE id = ?", (user_id,)
        ).fetchone()["is_admin"]
        if is_admin_target:
            flash("Удаление администратора запрещено.")
            return redirect(url_for("admin_users"))
        connection.execute("DELETE FROM task_attempts WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM adaptive_attempts WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM adaptive_tasks WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM activity_logs WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
        connection.commit()
    flash(f"Пользователь {target['username']} удалён.")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/results/attempts/<int:result_id>/delete", methods=["POST"])
def admin_delete_attempt(user_id: int, result_id: int):
    """Удаление отдельной попытки прохождения задания."""
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для удаления результатов.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        connection.execute(
            "DELETE FROM task_attempts WHERE id = ? AND user_id = ?",
            (result_id, user_id),
        )
        connection.commit()
    flash("Попытка удалена.")
    return redirect(url_for("admin_user_results", user_id=user_id))


@app.route("/admin/users/<int:user_id>/results/delete_all", methods=["POST"])
def admin_delete_all_attempts(user_id: int):
    """Удаление всех результатов студента."""
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для удаления результатов.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        connection.execute("DELETE FROM task_attempts WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM adaptive_attempts WHERE user_id = ?", (user_id,))
        connection.commit()
    flash("Все результаты пользователя удалены.")
    return redirect(url_for("admin_user_results", user_id=user_id))


@app.route("/teacher/topics/<int:topic_id>/delete", methods=["POST"])
def teacher_delete_topic(topic_id: int):
    """Удаление темы и связанных заданий/попыток."""
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для удаления тем.")
        return redirect(url_for("index"))

    with get_db_connection() as connection:
        task_rows = connection.execute(
            "SELECT id FROM tasks WHERE topic_id = ?", (topic_id,)
        ).fetchall()
        task_ids = [row["id"] for row in task_rows]
        if task_ids:
            placeholders = ",".join(["?"] * len(task_ids))
            connection.execute(
                f"DELETE FROM task_attempts WHERE task_id IN ({placeholders})",
                task_ids,
            )
            connection.execute(
                f"DELETE FROM tasks WHERE id IN ({placeholders})",
                task_ids,
            )
        connection.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
        connection.commit()
    flash("Тема и связанные задания удалены.")
    return redirect(url_for("teacher_topics"))


@app.route("/teacher/topics/<int:topic_id>/tasks/new", methods=["GET", "POST"])
def teacher_new_task(topic_id: int):
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))
    if not user["is_admin"]:
        flash("Недостаточно прав для работы с заданиями.")
        return redirect(url_for("index"))
    with get_db_connection() as connection:
        topic = connection.execute(
            "SELECT id, title FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
    if topic is None:
        flash("Тема не найдена.")
        return redirect(url_for("teacher_topics"))
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        due_date = request.form.get("due_date", "").strip() or None
        questions_raw = request.form.get("questions", "")
        if not title or not description or not questions_raw:
            flash("Заполните все поля задания и вопросы.")
            return redirect(url_for("teacher_new_task", topic_id=topic_id))
        try:
            questions = json.loads(questions_raw)
        except json.JSONDecodeError:
            flash("Вопросы должны быть в формате JSON.")
            return redirect(url_for("teacher_new_task", topic_id=topic_id))
        if not isinstance(questions, list) or not questions:
            flash("Добавьте хотя бы один вопрос.")
            return redirect(url_for("teacher_new_task", topic_id=topic_id))
        content = json.dumps({"questions": questions}, ensure_ascii=False)
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    topic_id, title, description, task_type, due_date, content_json, created_by, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    topic_id,
                    title,
                    description,
                    "quiz",
                    due_date,
                    content,
                    user["id"],
                    now_iso(),
                ),
            )
            connection.commit()
        flash("Задание добавлено.")
        return redirect(url_for("teacher_topics"))
    example = json.dumps(
        [
            {
                "question": "Пример вопроса",
                "options": ["Ответ 1", "Ответ 2", "Ответ 3"],
                "answer": "Ответ 1",
                "area": "Раздел/навык",
            }
        ],
        ensure_ascii=False,
        indent=2,
    )
    return render_template("teacher_task_form.html", topic=topic, example=example)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
