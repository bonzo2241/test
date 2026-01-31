from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from flask import (Flask, flash, redirect, render_template, request, session,
                   url_for)
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "app.db"

app = Flask(__name__)
app.secret_key = "dev-secret-change-me"

QUESTIONS = [
    {
        "question": "Сколько будет 2 + 2?",
        "options": ["3", "4", "5"],
        "answer": "4",
    },
    {
        "question": "Столица Франции?",
        "options": ["Париж", "Лион", "Марсель"],
        "answer": "Париж",
    },
    {
        "question": "Какой цвет получается при смешивании синего и жёлтого?",
        "options": ["Зелёный", "Фиолетовый", "Оранжевый"],
        "answer": "Зелёный",
    },
]


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


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
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.commit()

    ensure_admin_user()


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


def current_user() -> sqlite3.Row | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    with get_db_connection() as connection:
        return connection.execute(
            "SELECT id, username, is_admin FROM users WHERE id = ?", (user_id,)
        ).fetchone()


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
        flash("Вы вошли в систему.")
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Вы вышли из системы.")
    return redirect(url_for("index"))


@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    user = current_user()
    if user is None:
        flash("Сначала войдите.")
        return redirect(url_for("login"))

    if request.method == "POST":
        answers = request.form
        score = 0
        for index, item in enumerate(QUESTIONS):
            if answers.get(f"question-{index}") == item["answer"]:
                score += 1
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO results (user_id, score, total, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user["id"], score, len(QUESTIONS), datetime.utcnow().isoformat()),
            )
            connection.commit()
        return render_template(
            "results.html",
            score=score,
            total=len(QUESTIONS),
        )

    return render_template("quiz.html", questions=QUESTIONS)


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
        results = connection.execute(
            """
            SELECT results.score, results.total, results.created_at, users.username
            FROM results
            JOIN users ON users.id = results.user_id
            ORDER BY results.created_at DESC
            """
        ).fetchall()
    return render_template("admin_results.html", results=results)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
