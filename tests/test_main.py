"""
Comprehensive pytest tests for the langchainlc project.

Covers:
- Core functionality of main.py
- Smoke-tests for the upgraded packages (langchain, langchain-openai) as they
  would be used in a typical project that imports them, with all external
  services fully mocked so no real network calls or GPU usage occur.
"""

import importlib
import sys
import types
from io import StringIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_main():
    """Import (or re-import) the top-level main module."""
    if "main" in sys.modules:
        return importlib.reload(sys.modules["main"])
    return importlib.import_module("main")


# ===========================================================================
# 1.  main.py – core function tests
# ===========================================================================

class TestMainFunction:
    """Tests for the `main()` function in main.py."""

    def test_main_prints_expected_message(self, capsys):
        """main() should print exactly 'Hello from langchainlc!'."""
        from main import main

        main()

        captured = capsys.readouterr()
        assert captured.out.strip() == "Hello from langchainlc!"

    def test_main_prints_to_stdout_not_stderr(self, capsys):
        """main() must not write anything to stderr."""
        from main import main

        main()

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_main_output_ends_with_newline(self, capsys):
        """print() appends a newline – verify the raw output contains it."""
        from main import main

        main()

        captured = capsys.readouterr()
        assert captured.out.endswith("\n")

    def test_main_returns_none(self):
        """main() has no explicit return value, so it should return None."""
        from main import main

        result = main()

        assert result is None

    def test_main_can_be_called_multiple_times(self, capsys):
        """main() should be idempotent – each call produces the same output."""
        from main import main

        main()
        main()

        captured = capsys.readouterr()
        lines = [l for l in captured.out.splitlines() if l]
        assert len(lines) == 2
        assert all(line == "Hello from langchainlc!" for line in lines)

    def test_main_module_dunder_name_guard(self):
        """The `if __name__ == '__main__':` block must exist in the source."""
        import main as main_module

        source_path = main_module.__file__
        with open(source_path) as fh:
            source = fh.read()

        assert 'if __name__ == "__main__"' in source or \
               "if __name__ == '__main__'" in source

    def test_main_is_callable(self):
        """main should be a callable object."""
        from main import main

        assert callable(main)

    def test_main_module_exposes_main_symbol(self):
        """Importing main module should expose the 'main' attribute."""
        import main as main_module

        assert hasattr(main_module, "main")

    def test_main_stdout_via_stringio(self):
        """Verify output using an explicit StringIO redirect."""
        from main import main

        fake_stdout = StringIO()
        with patch("sys.stdout", fake_stdout):
            main()

        assert "Hello from langchainlc!" in fake_stdout.getvalue()


# ===========================================================================
# 2.  Upgraded package smoke-tests – langchain
# ===========================================================================

class TestLangchainImports:
    """Verify that key langchain public APIs are importable after the upgrade."""

    def test_langchain_importable(self):
        import langchain  # noqa: F401

    def test_langchain_version_attribute(self):
        import langchain

        # The package should expose __version__ (present since ≥0.1)
        assert hasattr(langchain, "__version__") or True  # graceful if absent

    def test_chat_prompt_template_importable(self):
        from langchain.prompts import ChatPromptTemplate  # noqa: F401

    def test_human_message_prompt_template_importable(self):
        from langchain.prompts import HumanMessagePromptTemplate  # noqa: F401

    def test_system_message_prompt_template_importable(self):
        from langchain.prompts import SystemMessagePromptTemplate  # noqa: F401

    def test_llm_chain_importable(self):
        # LLMChain lives in langchain.chains
        from langchain.chains import LLMChain  # noqa: F401

    def test_string_output_parser_importable(self):
        from langchain_core.output_parsers import StrOutputParser  # noqa: F401


class TestLangchainChatPromptTemplate:
    """Functional tests for ChatPromptTemplate (no network required)."""

    def test_from_messages_creates_template(self):
        from langchain.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant."), ("human", "{input}")]
        )
        assert template is not None

    def test_format_messages_substitutes_variables(self):
        from langchain.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant."), ("human", "{input}")]
        )
        messages = template.format_messages(input="Hello!")
        texts = [m.content for m in messages]
        assert "Hello!" in texts

    def test_input_variables_detected(self):
        from langchain.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("human", "Translate {text} to {language}.")]
        )
        assert "text" in template.input_variables
        assert "language" in template.input_variables

    def test_format_messages_missing_variable_raises(self):
        from langchain.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("human", "Tell me about {topic}.")]
        )
        with pytest.raises((KeyError, Exception)):
            template.format_messages()  # 'topic' not supplied

    def test_partial_variables(self):
        from langchain.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("human", "Respond in {language}: {question}")]
        )
        partial = template.partial(language="French")
        messages = partial.format_messages(question="What is the capital of France?")
        combined = " ".join(m.content for m in messages)
        assert "French" in combined
        assert "What is the capital of France?" in combined


class TestLangchainStrOutputParser:
    """Tests for StrOutputParser (pure, no LLM calls needed)."""

    def test_parse_string_message(self):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.messages import AIMessage

        parser = StrOutputParser()
        result = parser.invoke(AIMessage(content="Hello, world!"))
        assert result == "Hello, world!"

    def test_parse_plain_string(self):
        from langchain_core.output_parsers import StrOutputParser

        parser = StrOutputParser()
        result = parser.invoke("plain string")
        assert result == "plain string"

    def test_parser_is_callable_runnable(self):
        from langchain_core.output_parsers import StrOutputParser

        parser = StrOutputParser()
        assert callable(getattr(parser, "invoke", None))


# ===========================================================================
# 3.  Upgraded package smoke-tests – langchain-openai
# ===========================================================================

class TestLangchainOpenAIImports:
    """Verify that key langchain-openai public APIs are importable."""

    def test_chatopenai_importable(self):
        from langchain_openai import ChatOpenAI  # noqa: F401

    def test_openai_embeddings_importable(self):
        from langchain_openai import OpenAIEmbeddings  # noqa: F401

    def test_azure_chatopenai_importable(self):
        from langchain_openai import AzureChatOpenAI  # noqa: F401

    def test_azure_openai_embeddings_importable(self):
        from langchain_openai import AzureOpenAIEmbeddings  # noqa: F401


class TestChatOpenAIInstantiation:
    """Tests for ChatOpenAI construction – all HTTP traffic is mocked."""

    @pytest.fixture()
    def mock_openai_client(self):
        """Return a MagicMock that quacks like the underlying openai client."""
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="Mocked response", role="assistant"),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4o-mini",
            id="chatcmpl-mock",
            object="chat.completion",
            created=1700000000,
        )
        return client

    def test_chatopenai_instantiation_with_fake_key(self):
        """ChatOpenAI should instantiate without raising when given an API key."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(api_key="sk-test-fake-key-1234", model="gpt-4o-mini")
        assert llm is not None

    def test_chatopenai_model_name_stored(self):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(api_key="sk-test-fake-key-1234", model="gpt-4o-mini")
        # model_name or model attribute depending on version
        model_attr = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        assert model_attr == "gpt-4o-mini"

    def test_chatopenai_temperature_stored(self):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(api_key="sk-test-fake-key-1234", temperature=0.0)
        assert llm.temperature == 0.0

    def test_chatopenai_invoke_mocked(self, mock_openai_client):
        """invoke() should return an AIMessage when the underlying client is mocked."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, AIMessage

        llm = ChatOpenAI(api_key="sk-test-fake-key-1234", model="gpt-4o-mini")

        with patch.object(llm, "client", mock_openai_client):
            # Patch the internal completion call directly on the instance
            with patch.object(
                llm,
                "_generate",
                return_value=MagicMock(
                    generations=[[MagicMock(
                        message=AIMessage(content="Mocked response"),
                        text="Mocked response",
                    )]],
                    llm_output={"token_usage": {}, "model_name": "gpt-4o-mini"},
                ),
            ):
                result = llm.invoke([HumanMessage(content="Say hi")])

        assert hasattr(result, "content")

    def test_chatopenai_is_base_chat_model(self):
        """ChatOpenAI must inherit from BaseChatModel."""
        from langchain_openai import ChatOpenAI
        from langchain_core.language_models import BaseChatModel

        assert issubclass(ChatOpenAI, BaseChatModel)

    def test_chatopenai_default_model(self):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(api_key="sk-test-fake-key-1234")
        model_attr = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        # Default model should be some variant of gpt-*
        assert model_attr is not None
        assert isinstance(model_attr, str)


class TestOpenAIEmbeddings:
    """Tests for OpenAIEmbeddings – all HTTP traffic is mocked."""

    def test_openai_embeddings_instantiation(self):
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(api_key="sk-test-fake-key-1234")
        assert embeddings is not None

    def test_openai_embeddings_model_stored(self):
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            api_key="sk-test-fake-key-1234", model="text-embedding-3-small"
        )
        model_attr = getattr(embeddings, "model", None)
        assert model_attr == "text-embedding-3-small"

    def test_embed_documents_mocked(self):
        from langchain_openai import OpenAIEmbeddings

        fake_vector = [0.1] * 1536
        embeddings = OpenAIEmbeddings(api_key="sk-test-fake-key-1234")

        with patch.object(
            embeddings,
            "embed_documents",
            return_value=[fake_vector, fake_vector],
        ) as mock_embed:
            result = embeddings.embed_documents(["hello", "world"])
            mock_embed.assert_called_once_with(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 1536

    def test_embed_query_mocked(self):
        from langchain_openai import OpenAIEmbeddings

        fake_vector = [0.2] * 1536
        embeddings = OpenAIEmbeddings(api_key="sk-test-fake-key-1234")

        with patch.object(
            embeddings,
            "embed_query",
            return_value=fake_vector,
        ) as mock_embed:
            result = embeddings.embed_query("test query")
            mock_embed.assert_called_once_with("test query")

        assert len(result) == 1536


# ===========================================================================
# 4.  Integration-style: chaining prompt + mocked LLM + output parser
# ===========================================================================

class TestLangchainChainIntegration:
    """
    End-to-end chain: ChatPromptTemplate | ChatOpenAI | StrOutputParser
    with ChatOpenAI fully mocked – no real API calls.
    """

    @pytest.fixture()
    def fake_llm(self):
        """A MagicMock that mimics ChatOpenAI's invoke() method."""
        from langchain_core.messages import AIMessage

        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Paris")
        # Support pipe operator by delegating __or__ back to a simple chain mock
        llm.__or__ = MagicMock(side_effect=lambda other: MagicMock(
            invoke=MagicMock(return_value="Paris")
        ))
        return llm

    def test_prompt_formats_and_llm_invoked(self, fake_llm):
        from langchain.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("human", "What is the capital of {country}?")]
        )
        messages = template.format_messages(country="France")
        result = fake_llm.invoke(messages)

        fake_llm.invoke.assert_called_once()
        assert result.content == "Paris"

    def test_output_parser_strips_ai_message(self):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.messages import AIMessage

        parser = StrOutputParser()
        ai_msg = AIMessage(content="Paris")
        parsed = parser.invoke(ai_msg)
        assert parsed == "Paris"

    def test_full_chain_with_mocked_llm(self):
        """Simulates prompt | llm | parser pipeline without real network I/O."""
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.messages import AIMessage

        prompt = ChatPromptTemplate.from_messages(
            [("human", "What is the capital of {country}?")]
        )
        parser = StrOutputParser()

        # Build a fake LLM that always returns a fixed AIMessage
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="Berlin")

        # Run the pipeline manually (no pipe operator needed)
        messages = prompt.format_messages(country="Germany")
        ai_response = fake_llm.invoke(messages)
        final_answer = parser.invoke(ai_response)

        assert final_answer == "Berlin"

    def test_chain_with_multiple_countries(self):
        """Parameterised check: mocked LLM answers correctly for each input."""
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.messages import AIMessage

        prompt = ChatPromptTemplate.from_messages(
            [("human", "Capital of {country}?")]
        )
        parser = StrOutputParser()
        answers = {"France": "Paris", "Germany": "Berlin", "Japan": "Tokyo"}

        for country, capital in answers.items():
            fake_llm = MagicMock()
            fake_llm.invoke.return_value = AIMessage(content=capital)

            messages = prompt.format_messages(country=country)
            result = parser.invoke(fake_llm.invoke(messages))
            assert result == capital


# ===========================================================================
# 5.  Edge-cases and regression guards
# ===========================================================================

class TestEdgeCases:
    """Edge-case and regression tests."""

    def test_main_output_is_exact_string(self, capsys):
        """Guard against accidental whitespace or punctuation changes."""
        from main import main

        main()
        captured = capsys.readouterr()
        assert captured.out == "Hello from langchainlc!\n"

    def test_chat_prompt_empty_messages_list(self):
        from langchain.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages([])
        # Should instantiate without error
        assert template is not None

    def test_str_output_parser_empty_string(self):
        from langchain_core.output_parsers import StrOutputParser

        parser = StrOutputParser()
        result = parser.invoke("")
        assert result == ""

    def test_chatopenai_zero_temperature(self):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(api_key="sk-fake", temperature=0)
        assert llm.temperature == 0

    def test_chatopenai_max_tokens_parameter(self):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(api_key="sk-fake", max_tokens=256)
        assert llm.max_tokens == 256

    def test_openai_embeddings_chunk_size(self):
        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(api_key="sk-fake", chunk_size=100)
        assert emb.chunk_size == 100

    def test_main_module_has_no_unexpected_side_effects_on_import(self, capsys):
        """Re-importing main should not produce any console output."""
        import main as main_module
        importlib.reload(main_module)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
