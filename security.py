"""Простая защита от prompt-injection и опасного Python-кода."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass


PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|system|developer) instructions",
    r"disregard (all )?(previous|system|developer) instructions",
    r"forget (all )?(previous|system|developer) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"show (the )?(system|developer) prompt",
    r"disable (safety|guardrails|rules|restrictions)",
    r"bypass (safety|guardrails|rules|restrictions)",
    r"забудь .*инструкц",
    r"игнорируй .*инструкц",
    r"покажи .*системн.*промпт",
    r"раскрой .*системн.*промпт",
    r"отключи .*огранич",
    r"обойди .*огранич",
    r"выполни .*опасн.*код",
    r"удали .*файл",
    r"скачай .*файл",
    r"отправь .*данн",
]

DANGEROUS_CODE_WORDS = {
    "__import__", "eval", "exec", "compile", "open", "input", "globals", "locals",
    "vars", "dir", "getattr", "setattr", "delattr", "help", "breakpoint",
    "os", "sys", "subprocess", "socket", "requests", "urllib", "pathlib", "shutil",
    "pickle", "marshal", "ctypes", "multiprocessing", "threading", "builtins",
    "importlib", "site", "pip", "conda", "venv", "http", "ftplib", "smtplib",
}

BANNED_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.While,
)


@dataclass(frozen=True)
class SecurityCheckResult:
    ok: bool
    message: str


def check_user_instruction(instruction: str) -> SecurityCheckResult:
    text = instruction.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return SecurityCheckResult(False, "Инструкция похожа на prompt-injection или опасную команду.")
    return SecurityCheckResult(True, "Инструкция прошла проверку.")


def check_python_code(code: str) -> SecurityCheckResult:
    lowered = code.lower()
    for word in DANGEROUS_CODE_WORDS:
        if re.search(rf"\b{re.escape(word.lower())}\b", lowered):
            return SecurityCheckResult(False, f"Код содержит запрещенное слово или модуль: {word}")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return SecurityCheckResult(False, f"Синтаксическая ошибка в Python-коде: {exc}")

    for node in ast.walk(tree):
        if isinstance(node, BANNED_AST_NODES):
            return SecurityCheckResult(False, f"Запрещенная конструкция Python: {type(node).__name__}")


        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return SecurityCheckResult(False, f"Запрещен доступ к приватному атрибуту: {node.attr}")

        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return SecurityCheckResult(False, f"Запрещено имя, начинающееся с _: {node.id}")

    return SecurityCheckResult(True, "Код прошел проверку.")