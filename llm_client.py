"""Клиент для GateLLM/OpenAI-compatible API."""
from __future__ import annotations

from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


DEFAULT_BASE_URL = "https://gatellm.ru/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: int = 90
    temperature: float = 0.1
    max_tokens: int = 2500


@dataclass(frozen=True)
class LLMTextResponse:
    content: str
    model: str


class LLMClientError(RuntimeError):
    """Понятная ошибка для Streamlit-интерфейса."""


def build_client(settings: LLMSettings) -> OpenAI:
    api_key = (settings.api_key or "").strip()
    base_url = (settings.base_url or "").strip().rstrip("/")

    if not api_key or api_key in {"PASTE_YOUR_KEY_HERE", "sk-...", "your-api-key"}:
        raise LLMClientError("Не указан API-ключ GateLLM. Добавь его в файл .env или в боковой панели Streamlit.")
    if not base_url:
        raise LLMClientError("Не указан Base URL GateLLM.")

    return OpenAI(api_key=api_key, base_url=base_url, timeout=settings.timeout)



def chat_text_with_model(settings: LLMSettings, messages: list[dict[str, str]]) -> LLMTextResponse:
    client = build_client(settings)
    model = (settings.model or DEFAULT_MODEL).strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    except APITimeoutError as exc:
        raise LLMClientError(f"Таймаут ответа LLM по модели {model}: {exc}") from exc
    except APIConnectionError as exc:
        raise LLMClientError(f"Не удалось подключиться к GateLLM API: {exc}") from exc
    except APIStatusError as exc:
        raise LLMClientError(f"Ошибка GateLLM API {exc.status_code} для модели {model}: {exc.message}") from exc
    except Exception as exc:
        raise LLMClientError(f"Неожиданная ошибка GateLLM API для модели {model}: {exc}") from exc

    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)

    if not content:
        raise LLMClientError(f"GateLLM вернул пустой ответ. Модель: {model}")

    return LLMTextResponse(content=str(content).strip(), model=model)



def chat_text(settings: LLMSettings, messages: list[dict[str, str]]) -> str:
    """Метод для агента: возвращает только текст ответа."""
    return chat_text_with_model(settings, messages).content



def check_llm(settings: LLMSettings) -> tuple[bool, str]:
    """Проверка API-ключа, Base URL и точной модели."""
    model = (settings.model or DEFAULT_MODEL).strip()
    try:
        response = chat_text_with_model(
            LLMSettings(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=model,
                timeout=min(settings.timeout, 20),
                temperature=0,
                max_tokens=50,
            ),
            [
                {"role": "system", "content": "Ответь одним словом: OK"},
                {"role": "user", "content": "Проверка связи"},
            ],
        )
        return True, f"API отвечает. Модель: {response.model}. Ответ: {response.content[:100]}"
    except Exception as exc:
        return False, str(exc)
