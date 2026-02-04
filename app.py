from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "app.db"

app = Flask(__name__)
app.secret_key = "dev-secret-change-me"

SEED_TOPICS = [
    {
        "title": "XV век: централизация Руси",
        "description": "Итоги борьбы за независимость и рост Москвы как центра",
    },
    {
        "title": "XVI век: формирование Русского царства",
        "description": "Реформы Ивана IV и расширение границ государства",
    },
    {
        "title": "XVII век: Смутное время и первые Романовы",
        "description": "Кризис власти, восстановление государства и реформы",
    },
    {
        "title": "XVIII век: реформы Петра I и империя",
        "description": "Модернизация страны, армия, флот и новые столицы",
    },
    {
        "title": "XIX век: реформы и общественные движения",
        "description": "Отечественная война 1812 года, реформы и новые идеи",
    },
]

SEED_TASKS = [
    {
        "topic_title": "XV век: централизация Руси",
        "title": "Мини-тест: завершение монгольской зависимости",
        "description": "Проверьте знания по ключевым событиям XV века.",
        "task_type": "quiz",
        "questions": [
            {
                "question": "В каком году произошло стояние на реке Угре?",
                "options": ["1480", "1380", "1497"],
                "answer": "1480",
                "area": "XV век",
            },
            {
                "question": "Какой князь присоединил Новгород к Москве?",
                "options": ["Иван III", "Василий II", "Дмитрий Донской"],
                "answer": "Иван III",
                "area": "XV век",
            },
            {
                "question": "Как называется первый общерусский Судебник?",
                "options": ["Судебник 1497", "Русская Правда", "Соборное уложение"],
                "answer": "Судебник 1497",
                "area": "XV век",
            },
        ],
    },
    {
        "topic_title": "XVI век: формирование Русского царства",
        "title": "Мини-тест: эпоха Ивана Грозного",
        "description": "События и реформы XVI века.",
        "task_type": "quiz",
        "questions": [
            {
                "question": "В каком году произошло венчание Ивана IV на царство?",
                "options": ["1547", "1552", "1565"],
                "answer": "1547",
                "area": "XVI век",
            },
            {
                "question": "Какой город был взят в ходе Казанского похода 1552 года?",
                "options": ["Казань", "Астрахань", "Псков"],
                "answer": "Казань",
                "area": "XVI век",
            },
            {
                "question": "Как называется политика, введённая Иваном IV в 1565 году?",
                "options": ["Опричнина", "Земщина", "Соборность"],
                "answer": "Опричнина",
                "area": "XVI век",
            },
        ],
    },
    {
        "topic_title": "XVII век: Смутное время и первые Романовы",
        "title": "Мини-тест: Смутное время",
        "description": "Проверьте знания о событиях начала XVII века.",
        "task_type": "quiz",
        "questions": [
            {
                "question": "Кто возглавил Второе народное ополчение 1612 года?",
                "options": [
                    "Кузьма Минин и Дмитрий Пожарский",
                    "Борис Годунов",
                    "Василий Шуйский",
                ],
                "answer": "Кузьма Минин и Дмитрий Пожарский",
                "area": "XVII век",
            },
            {
                "question": "В каком году Земский собор избрал Михаила Романова?",
                "options": ["1613", "1598", "1649"],
                "answer": "1613",
                "area": "XVII век",
            },
            {
                "question": "Как называется свод законов 1649 года?",
                "options": ["Соборное уложение", "Судебник 1550", "Табель о рангах"],
                "answer": "Соборное уложение",
                "area": "XVII век",
            },
        ],
    },
    {
        "topic_title": "XVIII век: реформы Петра I и империя",
        "title": "Мини-тест: реформы Петра I",
        "description": "Ключевые преобразования начала XVIII века.",
        "task_type": "quiz",
        "questions": [
            {
                "question": "В каком году была основана Санкт-Петербург?",
                "options": ["1703", "1721", "1682"],
                "answer": "1703",
                "area": "XVIII век",
            },
            {
                "question": "Какой документ ввёл служебную иерархию в 1722 году?",
                "options": ["Табель о рангах", "Жалованная грамота", "Соборное уложение"],
                "answer": "Табель о рангах",
                "area": "XVIII век",
            },
            {
                "question": "Как завершилась Северная война 1700–1721 гг.?",
                "options": ["Ништадтским миром", "Полтавской битвой", "Андрусовским перемирием"],
                "answer": "Ништадтским миром",
                "area": "XVIII век",
            },
        ],
    },
    {
        "topic_title": "XIX век: реформы и общественные движения",
        "title": "Мини-тест: XIX век",
        "description": "Главные события и реформы XIX века.",
        "task_type": "quiz",
        "questions": [
            {
                "question": "В каком году была отменена крепостная зависимость?",
                "options": ["1861", "1812", "1905"],
                "answer": "1861",
                "area": "XIX век",
            },
            {
                "question": "Как называется восстание 1825 года на Сенатской площади?",
                "options": ["Декабристов", "Пугачёвское", "Стрелецкое"],
                "answer": "Декабристов",
                "area": "XIX век",
            },
            {
                "question": "Кто командовал русской армией в Бородинском сражении?",
                "options": ["М.И. Кутузов", "А.В. Суворов", "П.С. Нахимов"],
                "answer": "М.И. Кутузов",
                "area": "XIX век",
            },
        ],
    },
]

MATERIALS = {
    "XV век": "Повторите тему: стояние на Угре и политика Ивана III.",
    "XVI век": "Повторите тему: реформы Избранной рады и опричнина.",
    "XVII век": "Повторите тему: Смутное время и первые Романовы.",
    "XVIII век": "Повторите тему: реформы Петра I и Северная война.",
    "XIX век": "Повторите тему: реформы Александра II и движение декабристов.",
}


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        connection.commit()

    ensure_admin_user()
    seed_content()


def ensure_admin_user() -> None:
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
            area = question.get("area", "Общее")
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


def build_recommendations(
    monitoring: dict, area_stats: list[dict], risk_level: str
) -> list[str]:
    recommendations = []
    for area in area_stats:
        if area["accuracy"] < 0.7:
            material = MATERIALS.get(area["area"], "Повторите тему и выполните дополнительные упражнения.")
            recommendations.append(f"{area['area']}: {material}")
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
    monitoring = get_monitoring_data(user_id)
    area_stats = get_area_performance(user_id)
    risk_level, reasons = evaluate_risk(monitoring)
    recommendations = build_recommendations(monitoring, area_stats, risk_level)
    plan_tip = optimize_plan(monitoring)
    return {
        "monitoring": monitoring,
        "area_stats": area_stats,
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
    )


@app.route("/admin/results")
def admin_results():
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


@app.route("/admin/results/users/<int:user_id>")
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
            return redirect(url_for("admin_results"))
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


@app.route("/admin/results/users/<int:user_id>/attempts/<int:result_id>")
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
            return redirect(url_for("admin_results"))
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
    )


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
                "area": "Тема",
            }
        ],
        ensure_ascii=False,
        indent=2,
    )
    return render_template("teacher_task_form.html", topic=topic, example=example)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
