import os
import re
from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_HF_MODEL = "Qwen/Qwen3-0.6B"
OPENAI_PROVIDER = "1"
HUGGINGFACE_PROVIDER = "2"
_qwen_tokenizer = None
_qwen_model = None


def _format_error(error: Exception, secret: str = "") -> str:
    message = str(error) or error.__class__.__name__
    if secret:
        message = message.replace(secret, "[redacted]")
    return message[:500]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "si"}


def _strip_thinking_content(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _load_qwen_model():
    global _qwen_tokenizer, _qwen_model

    if _qwen_tokenizer is not None and _qwen_model is not None:
        return _qwen_tokenizer, _qwen_model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.getenv("HUGGINGFACE_MODEL", DEFAULT_HF_MODEL)
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    token_kwargs = {"token": api_token} if api_token else {}

    _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
    _qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        **token_kwargs,
    )
    _qwen_model.eval()

    return _qwen_tokenizer, _qwen_model

""" Extrae la sección específica del prompt dada su etiqueta (e.g., "Context", "Question") ----------------------------------------------"""
def _extract_prompt_section(prompt: str, section_name: str) -> str:
    pattern = rf"{section_name}:\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]+:|\n\s*Answer\.|\Z)"
    match = re.search(pattern, prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""

""" Genera una respuesta local basada en el contexto recuperado, si la consulta al LLM falla ----------------------------------------------"""
def _fallback_answer(prompt: str, provider_name: str, reason: str) -> str:
    context = _extract_prompt_section(prompt, "Context")
    question = _extract_prompt_section(prompt, "Question")

    if not context:
        return f"No pude consultar el modelo externo ({reason}) y no hay contexto disponible para responder."

    question_terms = {
        term
        for term in re.findall(r"\b\w+\b", question.lower())
        if len(term) > 3
    }
    sentences = re.split(r"(?<=[.!?])\s+", context)

    scored_sentences = []
    for sentence in sentences:
        sentence_terms = set(re.findall(r"\b\w+\b", sentence.lower()))
        score = len(question_terms & sentence_terms)
        if score:
            scored_sentences.append((score, sentence.strip()))

    selected = [sentence for _, sentence in sorted(scored_sentences, reverse=True)[:4]]
    if not selected:
        selected = [context[:1200].strip()]

    answer = "\n".join(f"- {sentence}" for sentence in selected if sentence)
    return (
        f"No pude consultar {provider_name} ({reason}). Respuesta local basada solo en los documentos recuperados:\n"
        f"{answer}"
    )


""" Sends the prompt to OpenAI and returns the generated response ----------------------------------------------"""
def _generate_openai_answer(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_answer(prompt, "OpenAI", "OPENAI_API_KEY no configurada")

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            input=prompt
        )
    except RateLimitError as error:
        error_code = getattr(error, "code", None) or "quota/rate limit"
        return _fallback_answer(prompt, "OpenAI", error_code)
    except APIError as error:
        error_code = getattr(error, "code", None) or error.__class__.__name__
        return _fallback_answer(prompt, "OpenAI", error_code)

    return response.output_text.strip()


""" Sends the prompt to HuggingFace and returns the generated response ----------------------------------------------"""
def _generate_huggingface_answer(prompt: str) -> str:
    try:
        import torch

        tokenizer, model = _load_qwen_model()
        enable_thinking = _env_bool("QWEN_ENABLE_THINKING", False)
        messages = [{"role": "user", "content": prompt}]

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        temperature = float(
            os.getenv("HUGGINGFACE_TEMPERATURE", "0.6" if enable_thinking else "0.7")
        )
        top_p = float(os.getenv("HUGGINGFACE_TOP_P", "0.95" if enable_thinking else "0.8"))

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", "1024")),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=int(os.getenv("HUGGINGFACE_TOP_K", "20")),
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    except Exception as error:
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or ""
        return _fallback_answer(prompt, "HuggingFace Qwen3 local", _format_error(error, api_token))

    if not _env_bool("QWEN_SHOW_THINKING", False):
        response = _strip_thinking_content(response)

    return response


""" Routes the prompt to the selected LLM provider ----------------------------------------------"""
def generate_answer(prompt: str, provider: str = OPENAI_PROVIDER) -> str:
    if provider == HUGGINGFACE_PROVIDER:
        return _generate_huggingface_answer(prompt)

    return _generate_openai_answer(prompt)
