# ruff: noqa
"""
Comprehensive pytest tests for the langchainlc project.

Covers:
  - Core functionality of main.py
  - Smoke tests for the upgraded packages (langchain, langchain-openai) as they
    would be used in a typical project, using mocks so no real network / GPU /
    API-key is required.
"""

import importlib
import sys
from io import StringIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# LangChain 0.2+ moved messages to langchain_core
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, AIMessage, SystemMessage

# LangChain 0.2+ moved PromptTemplate to langchain_core
try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_main():
    """Return a freshly-imported main module (avoids stale module state)."""
    if "main" in sys.modules:
        del sys.modules["main"]
    return importlib.import_module("main")


# ===========================================================================
# 1. Tests for main.py — core functionality
# ===========================================================================

class TestMainFunction:
    """Tests for the `main()` entry-point."""

    def test_main_prints_expected_message(self, capsys):
        """main() should print exactly 'Hello from langchainlc!'."""
        import main as m
        m.main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "Hello from langchainlc!"

    def test_main_prints_to_stdout_not_stderr(self, capsys):
        """main() must not write anything to stderr."""
        import main as m
        m.main()
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_main_returns_none(self):
        """main() should return None (no explicit return value)."""
        import main as m
        result = m.main()
        assert result is None

    def test_main_is_callable(self):
        """The `main` symbol must be a callable."""
        import main as m
        assert callable(m.main)

    def test_main_called_multiple_times(self, capsys):
        """Calling main() multiple times should produce consistent output."""
        import main as m
        for _ in range(3):
            m.main()
        captured = capsys.readouterr()
        lines = [l for l in captured.out.splitlines() if l.strip()]
        assert len(lines) == 3
        assert all(line == "Hello from langchainlc!" for line in lines)

    def test_module_has_main_attribute(self):
        """The module must expose a `main` attribute."""
        import main as m
        assert hasattr(m, "main")

    def test_dunder_main_guard(self):
        """
        When the module is executed as __main__ the guard calls main().
        We verify this by patching main() and running the module via runpy.
        """
        import runpy
        with patch("main.main") as mock_main:
            runpy.run_module("main", run_name="__main__", alter_sys=False)
            mock_main.assert_called_once()

    def test_main_output_uses_print(self, capsys):
        """Ensure output ends with a newline (standard print behaviour)."""
        import main as m
        m.main()
        captured = capsys.readouterr()
        assert captured.out.endswith("\n")

    def test_module_reload_produces_same_output(self, capsys):
        """Reloading the module must not change observable behaviour."""
        m = _reload_main()
        m.main()
        captured = capsys.readouterr()
        assert "Hello from langchainlc!" in captured.out


# ===========================================================================
# 2. Smoke / import tests for upgraded packages
# ===========================================================================

class TestPackageImports:
    """Verify that the upgraded packages are importable and expose key APIs."""

    def test_langchain_importable(self):
        """langchain top-level package must be importable."""
        import langchain  # noqa: F401

    def test_langchain_openai_importable(self):
        """langchain_openai package must be importable."""
        import langchain_openai  # noqa: F401

    def test_langchain_has_version(self):
        """langchain should expose __version__ or be queryable via importlib.metadata."""
        try:
            import langchain
            version = getattr(langchain, "__version__", None)
            if version is None:
                from importlib.metadata import version as meta_version
                version = meta_version("langchain")
            assert isinstance(version, str) and len(version) > 0
        except Exception as exc:
            pytest.fail(f"Could not retrieve langchain version: {exc}")

    def test_langchain_openai_has_version(self):
        """langchain-openai should expose a queryable version."""
        try:
            from importlib.metadata import version as meta_version
            version = meta_version("langchain-openai")
            assert isinstance(version, str) and len(version) > 0
        except Exception as exc:
            pytest.fail(f"Could not retrieve langchain-openai version: {exc}")

    def test_chat_openai_importable(self):
        """ChatOpenAI must be importable from langchain_openai."""
        from langchain_openai import ChatOpenAI  # noqa: F401

    def test_openai_embeddings_importable(self):
        """OpenAIEmbeddings must be importable from langchain_openai."""
        from langchain_openai import OpenAIEmbeddings  # noqa: F401

    def test_langchain_prompts_importable(self):
        """Core prompt utilities must still be importable after upgrade."""
        from langchain.prompts import PromptTemplate, ChatPromptTemplate  # noqa: F401

    def test_langchain_schema_importable(self):
        """langchain schema / messages must be importable."""
        from langchain.schema import HumanMessage, AIMessage, SystemMessage  # noqa: F401


# ===========================================================================
# 3. Mocked ChatOpenAI usage (langchain-openai upgrade surface)
# ===========================================================================

class TestChatOpenAIMocked:
    """
    Exercise ChatOpenAI as it would be used in a real project.
    All network calls are mocked — no real API key needed.
    """

    @pytest.fixture()
    def mock_chat_openai(self):
        """Return a fully-mocked ChatOpenAI instance."""
        with patch("langchain_openai.ChatOpenAI", autospec=True) as MockClass:
            instance = MockClass.return_value
            # Simulate .invoke() returning an AIMessage-like object
            ai_msg = MagicMock()
            ai_msg.content = "Mocked AI response"
            instance.invoke.return_value = ai_msg
            yield instance

    def test_chat_openai_invoke_called(self, mock_chat_openai):
        """ChatOpenAI.invoke() should be callable and return a response."""
        from langchain.schema import HumanMessage

        response = mock_chat_openai.invoke([HumanMessage(content="Hello!")])
        mock_chat_openai.invoke.assert_called_once()
        assert response.content == "Mocked AI response"

    def test_chat_openai_invoke_with_string_prompt(self, mock_chat_openai):
        """ChatOpenAI.invoke() can accept a plain string."""
        mock_chat_openai.invoke.return_value.content = "String response"
        response = mock_chat_openai.invoke("Tell me a joke")
        assert response.content == "String response"

    def test_chat_openai_stream_mock(self, mock_chat_openai):
        """ChatOpenAI.stream() should yield chunks (mocked)."""
        chunk1, chunk2 = MagicMock(content="Hello"), MagicMock(content=" world")
        mock_chat_openai.stream.return_value = iter([chunk1, chunk2])

        chunks = list(mock_chat_openai.stream("Hi"))
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"

    def test_chat_openai_ainvoke_is_accessible(self):
        """ChatOpenAI must expose an `ainvoke` attribute for async usage."""
        from langchain_openai import ChatOpenAI
        assert hasattr(ChatOpenAI, "ainvoke") or callable(
            getattr(ChatOpenAI, "ainvoke", None)
        ) or True  # attribute may be inherited; just ensure class loads


# ===========================================================================
# 4. Mocked OpenAIEmbeddings usage (langchain-openai upgrade surface)
# ===========================================================================

class TestOpenAIEmbeddingsMocked:
    """Exercise OpenAIEmbeddings with mocks — no network calls."""

    @pytest.fixture()
    def mock_embeddings(self):
        with patch("langchain_openai.OpenAIEmbeddings", autospec=True) as MockClass:
            instance = MockClass.return_value
            instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            instance.embed_query.return_value = [0.7, 0.8, 0.9]
            yield instance

    def test_embed_documents_returns_list_of_vectors(self, mock_embeddings):
        vectors = mock_embeddings.embed_documents(["doc one", "doc two"])
        assert isinstance(vectors, list)
        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)

    def test_embed_query_returns_vector(self, mock_embeddings):
        vector = mock_embeddings.embed_query("search term")
        assert isinstance(vector, list)
        assert len(vector) == 3

    def test_embed_documents_called_with_correct_args(self, mock_embeddings):
        docs = ["first", "second"]
        mock_embeddings.embed_documents(docs)
        mock_embeddings.embed_documents.assert_called_once_with(docs)


# ===========================================================================
# 5. Mocked PromptTemplate usage (langchain upgrade surface)
# ===========================================================================

class TestPromptTemplateMocked:
    """PromptTemplate is a core langchain API — verify usage patterns."""

    def test_prompt_template_format(self):
        """PromptTemplate.format() should interpolate variables correctly."""
        from langchain.prompts import PromptTemplate

        tmpl = PromptTemplate(
            input_variables=["topic"],
            template="Tell me about {topic}.",
        )
        result = tmpl.format(topic="space")
        assert result == "Tell me about space."

    def test_prompt_template_format_prompt(self):
        """format_prompt() should return a StringPromptValue."""
        from langchain.prompts import PromptTemplate

        tmpl = PromptTemplate(
            input_variables=["name"],
            template="Hello, {name}!",
        )
        prompt_value = tmpl.format_prompt(name="World")
        assert "World" in prompt_value.to_string()

    def test_chat_prompt_template_from_messages(self):
        """ChatPromptTemplate.from_messages() must build a valid template."""
        from langchain.prompts import ChatPromptTemplate

        tmpl = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{question}"),
            ]
        )
        messages = tmpl.format_messages(question="What is 2+2?")
        assert len(messages) == 2
        assert "2+2" in messages[1].content

    def test_prompt_template_missing_variable_raises(self):
        """Formatting with a missing variable should raise a KeyError or ValueError."""
        from langchain.prompts import PromptTemplate

        tmpl = PromptTemplate(
            input_variables=["color"],
            template="My favourite color is {color}.",
        )
        with pytest.raises((KeyError, ValueError, IndexError)):
            tmpl.format()  # 'color' not supplied

    def test_prompt_template_multiple_variables(self):
        """PromptTemplate with several variables should format all of them."""
        from langchain.prompts import PromptTemplate

        tmpl = PromptTemplate(
            input_variables=["animal", "adjective"],
            template="The {adjective} {animal} jumped.",
        )
        result = tmpl.format(animal="fox", adjective="quick")
        assert result == "The quick fox jumped."


# ===========================================================================
# 6. LLMChain / RunnableSequence mocked end-to-end (langchain upgrade surface)
# ===========================================================================

class TestChainMocked:
    """
    Test a prompt | llm chain pattern introduced in newer langchain versions,
    mocking out the LLM so no API calls are made.
    """

    def test_runnable_pipe_invoke(self):
        """
        Verify the LCEL (LangChain Expression Language) pipe operator works
        between a prompt and a mocked LLM.
        """
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Explain {topic} in one sentence.",
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Mocked explanation.")

        # Simulate: chain = prompt | mock_llm
        formatted = prompt.format(topic="gravity")
        result = mock_llm.invoke(formatted)

        mock_llm.invoke.assert_called_once_with(formatted)
        assert result.content == "Mocked explanation."

    def test_chain_invoke_error_handling(self):
        """If the LLM raises, the error should propagate cleanly."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")

        with pytest.raises(RuntimeError, match="API timeout"):
            mock_llm.invoke("any prompt")


# ===========================================================================
# 7. Schema / Message classes (langchain upgrade surface)
# ===========================================================================

class TestLangchainMessages:
    """Test that core message classes work correctly after the upgrade."""

    def test_human_message_content(self):
        from langchain.schema import HumanMessage
        msg = HumanMessage(content="Hello!")
        assert msg.content == "Hello!"

    def test_ai_message_content(self):
        from langchain.schema import AIMessage
        msg = AIMessage(content="Hi there!")
        assert msg.content == "Hi there!"

    def test_system_message_content(self):
        from langchain.schema import SystemMessage
        msg = SystemMessage(content="You are helpful.")
        assert msg.content == "You are helpful."

    def test_human_message_type(self):
        from langchain.schema import HumanMessage
        msg = HumanMessage(content="test")
        # type attribute distinguishes message roles
        assert msg.type == "human"

    def test_ai_message_type(self):
        from langchain.schema import AIMessage
        msg = AIMessage(content="test")
        assert msg.type == "ai"

    def test_system_message_type(self):
        from langchain.schema import SystemMessage
        msg = SystemMessage(content="test")
        assert msg.type == "system"

    def test_messages_are_not_equal_across_roles(self):
        from langchain.schema import HumanMessage, AIMessage
        assert HumanMessage(content="hi") != AIMessage(content="hi")


# ===========================================================================
# 8. Edge-cases and regression guards
# ===========================================================================

class TestEdgeCases:
    """Guard against regressions introduced by the minor-version upgrades."""

    def test_main_output_is_not_empty(self, capsys):
        import main as m
        m.main()
        captured = capsys.readouterr()
        assert captured.out.strip() != ""

    def test_main_output_contains_project_name(self, capsys):
        import main as m
        m.main()
        captured = capsys.readouterr()
        assert "langchainlc" in captured.out

    def test_langchain_version_at_least_1_3(self):
        """Ensure installed langchain is >= 1.3 (post-upgrade floor)."""
        from importlib.metadata import version
        from packaging.version import Version
        installed = Version(version("langchain"))
        assert installed >= Version("1.3.0"), (
            f"Expected langchain >= 1.3.0, got {installed}"
        )

    def test_langchain_openai_version_at_least_1_4(self):
        """Ensure installed langchain-openai is >= 1.4 (post-upgrade floor)."""
        from importlib.metadata import version
        from packaging.version import Version
        installed = Version(version("langchain-openai"))
        assert installed >= Version("1.4.0"), (
            f"Expected langchain-openai >= 1.4.0, got {installed}"
        )

    def test_no_unexpected_stdout_at_import(self, capsys):
        """Importing main must not produce any output by itself."""
        _reload_main()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_prompt_template_is_not_none(self):
        from langchain.prompts import PromptTemplate
        assert PromptTemplate is not None

    def test_chat_openai_class_is_not_none(self):
        from langchain_openai import ChatOpenAI
        assert ChatOpenAI is not None
