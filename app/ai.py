import json
import logging
import re
import urllib.error
import urllib.request
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from .config import settings

logger = logging.getLogger("legalai.ai")


LANGUAGE_NAMES = {
    "az": "Azərbaycan",
    "en": "English",
    "ru": "русском",
}


def _language_line(language: str) -> str:
    name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES["az"])
    return f"Cavabların (summary, risks, grammar_issues daxil) yalnız {name} dilində olmalıdır."


SYSTEM_PROMPT = """Sən hüquqi sənədləri analiz edən köməkçisən. {language_line}
Bu ilkin texniki yoxlamadır, hüquqi məsləhət deyil. Yalnız sənəddə açıq yazılan faktlardan istifadə et; heç nə uydurma.
client_name_found yalnız müştəri/tərəf kimi açıq göstərilən şəxsin həm adı, həm soyadı olduqda true olsun.
Sədr, hakim, katib, vəkil kimi vəzifə sahiblərini müştəri hesab etmə. Onları officials siyahısına "Sədr: Ad Soyad" və ya "Hakim: Ad Soyad" formatında əlavə et.
Müştərinin ad-soyadı yoxdursa client_name_found false və client_name boş sətir olsun.
summary qısa, faydalı ən azı bir cümləlik xülasə olsun.
Risk siyahısına yalnız konkret çatışmayan, ziddiyyətli, vaxtı keçmiş və ya qeyri-müəyyən məqamları yaz. Hər riskdə sənəddən qısa hissəni dırnaq içində ver, sonra səbəbini yaz: məsələn, "15.05.2022" — tarix keçmiş ola bilər. Yoxdursa boş siyahı qaytar.
grammar_issues siyahısında yalnız həqiqi yazı/qrammatika xətalarını ver. Buraya sözlər arasında boşluq unudulub bir neçə sözün bitişik yazıldığı hallar da daxildir (məs. uzun bir "söz" əslində bir neçə sözün birləşməsidirsə). Hər bənddə səhv hissəni dırnaq içində, sonra təklif edilən düzəlişi yaz: məsələn, "məhkəməqərarı" → "məhkəmə qərarı", ya da "tarixliqərarınsaxlanılması" → "tarixli qərarın saxlanılması". Yoxdursa boş siyahı qaytar.
Sənəd müqavilədirsə və bitmə/xitam/qüvvədəolma tarixi aydın yazılıbsa, contract_end_date sahəsinə YYYY-MM-DD formatında yaz. Tarix yoxdursa və ya sənəd müqavilə deyilsə, boş sətir qaytar.
risk_score MÜTLƏQ risks siyahısının uzunluğu və ciddiliyi ilə uyğun olmalıdır: risks siyahısı boşdursa risk_score 0 olsun; risks siyahısında 1 və ya daha çox bənd varsa, risk_score HEÇ VAXT 0 ola bilməz — hər bənd üçün azı 20 bal say, ən ciddi hallarda 100-ə qədər.
Yalnız sxemdə tələb olunan JSON obyektini qaytar."""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "client_name": {"type": "string"},
        "client_name_found": {"type": "boolean"},
        "officials": {"type": "array", "items": {"type": "string"}},
        "document_type": {"type": "string"},
        "summary": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "grammar_issues": {"type": "array", "items": {"type": "string"}},
        "extracted_dates": {"type": "array", "items": {"type": "string"}},
        "contract_end_date": {"type": "string"},
        "risk_score": {"type": "number"},
    },
    "required": ["client_name", "client_name_found", "officials", "document_type", "summary", "risks", "grammar_issues", "extracted_dates", "contract_end_date", "risk_score"],
}


def _fallback(text: str) -> dict:
    dates = re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text)
    name_match = re.search(r"(?:Müştəri|Ad[ıı]|Şəxs)\s*[:\-]\s*([^\n,]{3,80})", text, re.I)
    client_name = name_match.group(1).strip() if name_match else "Naməlum müştəri"
    lower = text.lower()
    risks = []
    for word, label in [("imza", "İmza bölməsini yoxlayın"), ("etibarnamə", "Etibarnamənin qüvvədəolma tarixini yoxlayın"), ("iddia", "İddia müddətini və əsaslandırmanı yoxlayın")]:
        if word in lower:
            risks.append(label)
    return {
        "client_name": client_name,
        "client_name_found": client_name != "Naməlum müştəri",
        "officials": [],
        "document_type": "Məhkəmə qərarı" if "məhkəmə" in lower else "Hüquqi sənəd",
        "summary": "Demo analizi: sənəd saxlanıldı. Ollama/Gemma işlədikdə daha ətraflı analiz veriləcək.",
        "risks": risks,
        "grammar_issues": [],
        "extracted_dates": dates,
        "contract_end_date": "",
        "risk_score": min(100, len(risks) * 25),
    }


def _call_model(prompt_value) -> str:
    contents = "\n".join(str(message.content) for message in prompt_value.messages)
    payload = json.dumps({
        "model": settings.ollama_model,
        "prompt": contents,
        "stream": False,
        "format": ANALYSIS_SCHEMA,
        "keep_alive": "0",
        # num_ctx: Ollama-nın defolt context uzunluğu (server konfiqurasiyasından asılı
        # olaraq 2048-4096 ola bilər) uzun hüquqi sənədləri kəsə bilər — açıq təyin edirik.
        "options": {"temperature": 0, "num_ctx": 4096},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = json.loads(response.read().decode("utf-8"))["response"]
            # Modelin xam cavabını loglayırıq ki, "grammar_issues niyə boşdur" kimi halları
            # docker compose logs api ilə diaqnostika etmək mümkün olsun.
            logger.info("Ollama xam cavabı (analiz): %s", raw[:3000])
            return raw
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
        return json.dumps(_fallback(prompt_value.messages[-1].content), ensure_ascii=False)


def _parse_json(output: str) -> dict:
    cleaned = output.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = _fallback(output)
    # Bəzi kiçik modellər cavabı {"analysis": {...}} kimi sarıyır.
    if isinstance(data.get("analysis"), dict):
        data = data["analysis"]
    officials = [str(person).strip() for person in (data.get("officials") or []) if str(person).strip()]
    raw_client_name = str(data.get("client_name") or "").strip()
    if data.get("client_name_found") is not True:
        data["client_name"] = "; ".join(officials) if officials else "Sənəd daxilində ad və soyad qeyd olunmayıb"
    elif not raw_client_name or len(raw_client_name) < 3:
        data["client_name"] = "Sənəd daxilində ad və soyad qeyd olunmayıb"
    elif len(raw_client_name) > 120:
        # Kiçik model bəzən bütöv abzası "müştəri adı" kimi hallüsinasiya edir —
        # bu, real ad-soyad deyil, ona görə bazaya yazmırıq.
        data["client_name"] = "; ".join(officials) if officials else "Sənəd daxilində ad və soyad qeyd olunmayıb"
    else:
        data["client_name"] = raw_client_name
    if not data.get("document_type"):
        data["document_type"] = "Hüquqi sənəd"
    if not data.get("summary") or len(str(data["summary"]).strip()) < 15:
        data["summary"] = "AI sənəddən strukturlaşdırılmış xülasə yarada bilmədi; ilkin yoxlama tələb olunur."
    for key in ("risks", "extracted_dates"):
        data[key] = [item for item in (data.get(key) or []) if _looks_valid(item)]
    # grammar_issues-a "boşluqsuz uzun mətn" filtrini tətbiq etmirik — sözlər bitişik yazılan
    # həqiqi yazı xətaları da məhz belə (uzun, boşluqsuz) görünür və filtr onları da silərdi.
    data["grammar_issues"] = [str(item).strip() for item in (data.get("grammar_issues") or []) if str(item).strip()]
    data["contract_end_date"] = str(data.get("contract_end_date") or "").strip()
    try:
        risk_score = max(0, min(100, float(data.get("risk_score", 0))))
    except (TypeError, ValueError):
        risk_score = 0
    # Kiçik modellər bəzən risks siyahısı dolu olsa belə risk_score-u 0 qaytarır.
    # Bu, real risklərin göstərilməməsinə səbəb olmasın deyə minimum bal tətbiq edirik.
    if data["risks"] and risk_score < 20:
        risk_score = min(100, max(20, len(data["risks"]) * 20))
    data["risk_score"] = risk_score
    return data


def _looks_valid(item) -> bool:
    """Kiçik modellər bəzən boşluqsuz, mənasız 'identifier' tipli mətn hallüsinasiya edir — bunları süzürük."""
    text = str(item).strip()
    if not text:
        return False
    if " " not in text and len(text) > 25:
        return False
    return True


def embed_text(text: str) -> list[float] | None:
    """Ollama-nın embedding modelindən (nomic-embed-text) mətnin vektor təmsilini alır.
    Semantik axtarış üçün istifadə olunur — Ollama əlçatan deyilsə None qaytarır (axtarış
    o zaman sadəcə hərfi ILIKE nəticələrinə keçir)."""
    payload = json.dumps({
        "model": settings.embedding_model,
        "prompt": text[:6000],
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
            embedding = data.get("embedding")
            return embedding if embedding else None
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Embedding alına bilmədi (Ollama/%s əlçatan deyilmi?): %s", settings.embedding_model, exc)
        return None


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Sənəd mətni:\n{document_text}"),
])
analysis_chain = prompt | RunnableLambda(_call_model) | StrOutputParser() | RunnableLambda(_parse_json)


def analyze_document(text: str, language: str = "az") -> dict:
    return analysis_chain.invoke({"document_text": text[:30000], "language_line": _language_line(language)})


# Təlimatları hədəf dildə yazırıq — kiçik modellər öz dilindəki təlimatı ingiliscə "answer in X"
# tapşırığından daha etibarlı izləyir.
ANSWER_INSTRUCTIONS = {
    "az": """Sən hüquqi sənəd köməkçisisən. Cavabını YALNIZ Azərbaycan dilində yaz — heç bir kəlməni başqa dildə yazma.
Yalnız aşağıdakı SƏNƏDLƏR hissəsindən istifadə et. Cavab sənədlərdə yoxdursa, bunu aydın bildir.
Hüquqi məsləhət vermə, fakt uydurma. Cavabı qısa və praktik saxla.
VACIB: sual "risk" sözünü əlavə detalsız ehtiva edirsə (risk, risklər, riskin nə olduğu), bu HƏMİŞƏ
"bu konkret sənədlərdə hansı risklər tapılıb" mənasını verir — hər sənəd üçün aşağıdakı "risk balı" və
"konkret risklər" sətirlərindən istifadə et. HEÇ VAXT "risk" sözünün lüğəvi mənasını izah etmə.""",
    "en": """You are a legal-document assistant. Answer ONLY in English — do not use any other language.
Use only the DOCUMENTS below. If the answer is not in the documents, say that clearly.
Do not provide legal advice or invent facts. Keep the answer short and practical.
IMPORTANT: if the question mentions "risk" without further detail, this ALWAYS means "what risks were
found in these specific documents" — use the risk score and specific-risks lines below for each document.
NEVER give a generic dictionary definition of the word "risk".""",
    "ru": """Ты юридический ассистент по документам. Отвечай ТОЛЬКО на русском языке — не используй другой язык.
Используй только раздел ДОКУМЕНТЫ ниже. Если ответа там нет, чётко скажи об этом.
Не давай юридических советов и не выдумывай факты. Ответ должен быть коротким и практичным.
ВАЖНО: если вопрос касается слова «риск» без уточнения, это ВСЕГДА означает «какие риски найдены в
этих документах» — используй строки «оценка риска» и «конкретные риски» ниже для каждого документа.
НИКОГДА не давай общее словарное определение слова «риск».""",
}

AZ_SPECIFIC_CHARS = set("əıöüşçƏİÖÜŞÇ")


def _looks_azerbaijani(text: str) -> bool:
    return any(ch in AZ_SPECIFIC_CHARS for ch in text)


def _translate_to_azerbaijani(text: str) -> str:
    """Model başqa dildə cavab verəndə son çarə kimi Azərbaycan dilinə tərcümə edir."""
    prompt_text = f"""Aşağıdakı mətni Azərbaycan dilinə tərcümə et. Yalnız tərcüməni yaz, izah əlavə etmə.

Mətn:
{text}"""
    payload = json.dumps({
        "model": settings.ollama_model,
        "prompt": prompt_text,
        "stream": False,
        # 5m: bu funksiya çox zaman rag.py/answer_question-dan dərhal sonra, eyni sorğu
        # içində çağırılır — modeli dərhal boşaltsaq (keep_alive=0) növbəti çağırış yenidən
        # yükləmə gecikməsinə düşür (zəif CPU-lu serverdə bu, timeout-a səbəb olurdu).
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_ctx": 4096},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            translated = json.loads(response.read().decode("utf-8")).get("response", "").strip()
            return translated or text
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return text


def answer_question(question: str, document_context: str, language: str = "az") -> str:
    """Yüklənmiş sənədlərə əsaslanan qısa chat cavabı."""
    instructions = ANSWER_INSTRUCTIONS.get(language, ANSWER_INSTRUCTIONS["az"])
    prompt_text = f"""{instructions}

SƏNƏDLƏR / DOCUMENTS:
{document_context[:24000]}

SUAL / QUESTION: {question}"""
    payload = json.dumps({
        "model": settings.ollama_model,
        "prompt": prompt_text,
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_ctx": 4096},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            answer = json.loads(response.read().decode("utf-8")).get("response", "").strip()
            if not answer:
                return "Sənədlərə əsaslanan cavab yaradıla bilmədi."
            if language == "az" and not _looks_azerbaijani(answer):
                logger.info("Cavab azərbaycanca görünmür, tərcümə edilir: %s", answer[:200])
                answer = _translate_to_azerbaijani(answer)
            return answer
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return "Ollama ilə əlaqə qurulmadı. Ollama-nın açıq olduğunu yoxlayın."
