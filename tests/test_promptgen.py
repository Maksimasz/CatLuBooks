"""Проверки промпта агента: задание модели, чистка ответа, шаблон-запасной."""
import unittest

from config import PromptConfig
from core import promptgen


class ComposeMessagesTests(unittest.TestCase):
    def setUp(self):
        self.prompt = PromptConfig()

    def test_question_contains_title_and_topic(self):
        messages = promptgen.compose_messages(self.prompt, "Схемы", "даташиты разъёмов")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Схемы", messages[0]["content"])
        self.assertIn("даташиты разъёмов", messages[0]["content"])

    def test_missing_topic_is_named_in_the_question(self):
        text = promptgen.compose_messages(self.prompt, "Схемы", "  ")[0]["content"]

        self.assertIn("не указана", text)

    def test_template_prompt_takes_topic_or_title(self):
        with_topic = promptgen.fallback_prompt(self.prompt, "Схемы", "даташиты")
        without = promptgen.fallback_prompt(self.prompt, "Схемы", "")

        self.assertIn("даташиты", with_topic)
        self.assertIn("Схемы", without)


class CleanTests(unittest.TestCase):
    def test_markdown_fence_is_stripped(self):
        self.assertEqual(promptgen.clean("```\nТы — ассистент.\n```"), "Ты — ассистент.")
        self.assertEqual(promptgen.clean("```markdown\nТы — ассистент.\n```"), "Ты — ассистент.")
        self.assertEqual(promptgen.clean("```Ты — ассистент.```"), "Ты — ассистент.")

    def test_signature_before_prompt_is_stripped(self):
        self.assertEqual(promptgen.clean("Системный промпт: Ты — ассистент."),
                         "Ты — ассистент.")

    def test_too_long_answer_is_cut(self):
        cleaned = promptgen.clean("я" * 5000)

        self.assertEqual(len(cleaned), promptgen.MAX_PROMPT_CHARS)


class _FakeClient:
    """Модель с готовым ответом на внепоточный запрос."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def chat(self, messages, max_tokens=None):
        self.calls.append((messages, max_tokens))
        return self.answer


class ComposeTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_answer_is_cleaned_and_question_limited(self):
        client = _FakeClient("```Системный промпт: Ты — ассистент по схемам.```")

        prompt = await promptgen.compose(PromptConfig(), client, "Схемы", "даташиты")

        self.assertEqual(prompt, "Ты — ассистент по схемам.")
        self.assertEqual(client.calls[0][1], promptgen.COMPOSE_MAX_TOKENS)

    async def test_empty_answer_falls_back_to_template(self):
        prompt = await promptgen.compose(PromptConfig(), _FakeClient("   "),
                                         "Схемы", "даташиты")

        self.assertIn("даташиты", prompt)


if __name__ == '__main__':
    unittest.main()
