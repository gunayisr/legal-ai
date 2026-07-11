"""LlamaIndex əsaslı RAG orkestrasiyası: chunking, retrieval-dan sonra rerank, sonra LLM.

Axın:
  1. Sənəd yükləndikdə (app/main.py) mətn LlamaIndex-in SentenceSplitter-i ilə kiçik
     parçalara (chunk) bölünür və hər parça ayrıca embed olunub pgvector-da saxlanılır
     (bax: models.DocumentChunk). Bütöv-sənəd embedding-i (models.Document.embedding)
     də qalır — o, /search-dakı sənəd səviyyəli axtarış üçündür.
  2. /ask sorğusunda sualın embedding-i ilə ən yaxın 10 chunk pgvector cosine-distance
     ilə tapılır (app/main.py).
  3. Bu modul həmin 10 namizədi LlamaIndex-in LLMRerank-ı ilə yenidən qiymətləndirib
     ən yaxşı 5-ə endirir (rerank uğursuz olsa sadə oxşarlıq sırası saxlanılır).
  4. Son 5 parça LlamaIndex-in response synthesizer-inə (Ollama LLM) verilib cavab
     yaradılır; hər parçanın mənbə sənədi və uyğunluq faizi də UI-da göstərmək üçün
     geri qaytarılır.

Nomic-embed-text asimmetrik modeldir — sənəd və sual üçün fərqli prefiks tələb edir
("search_document: " / "search_query: "), bunu OllamaEmbedding-in
text_instruction/query_instruction parametrləri ilə düzgün tətbiq edirik. Bu, əvvəlki
"təsadüfi/uyğunsuz cavab" problemini əsaslı azaltmalıdır.
"""
import logging

from llama_index.core import PromptTemplate, QueryBundle, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from .ai import ANSWER_INSTRUCTIONS, _looks_azerbaijani, _translate_to_azerbaijani
from .config import settings

logger = logging.getLogger("legalai.rag")

_llm = Ollama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    request_timeout=90.0,
    additional_kwargs={"temperature": 0},
)
_embed_model = OllamaEmbedding(
    model_name=settings.embedding_model,
    base_url=settings.ollama_base_url,
    text_instruction="search_document: ",
    query_instruction="search_query: ",
)
_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)


def chunk_text(text: str) -> list[str]:
    """Mətni LlamaIndex-in cümlə-hörmətli splitter-i ilə ~800 tokenlik parçalara bölür."""
    if not text or not text.strip():
        return []
    try:
        return [chunk for chunk in _splitter.split_text(text) if chunk.strip()]
    except Exception as exc:
        logger.warning("Chunking alınmadı, bütöv mətn tək parça kimi saxlanılır: %s", exc)
        return [text]


def embed_document_text(text: str) -> list[float] | None:
    """Sənəd/chunk mətninin embedding-i ("search_document: " prefiksi ilə)."""
    try:
        return _embed_model.get_text_embedding(text[:2000])
    except Exception as exc:
        logger.warning("Sənəd embedding-i alınmadı: %s", exc)
        return None


def embed_query(text: str) -> list[float] | None:
    """Sualın embedding-i ("search_query: " prefiksi ilə)."""
    try:
        return _embed_model.get_query_embedding(text[:2000])
    except Exception as exc:
        logger.warning("Sual embedding-i alınmadı: %s", exc)
        return None


def _qa_template(language: str) -> PromptTemplate:
    instructions = ANSWER_INSTRUCTIONS.get(language, ANSWER_INSTRUCTIONS["az"])
    return PromptTemplate(
        instructions + "\n\nSƏNƏDLƏR:\n{context_str}\n\nSUAL: {query_str}\nCAVAB:"
    )


def _rerank(question: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
    if len(nodes) <= top_n:
        return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
    try:
        reranker = LLMRerank(llm=_llm, top_n=top_n, choice_batch_size=len(nodes))
        return reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=question))
    except Exception as exc:
        logger.warning("LLMRerank alınmadı, sadə oxşarlıq sırası istifadə olunur: %s", exc)
        return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[:top_n]


def rerank_and_answer(
    question: str,
    candidates: list[tuple[str, float, dict]],
    language: str = "az",
    top_n: int = 5,
) -> tuple[str, list[dict]]:
    """candidates: [(mətn, oxşarlıq_balı_0_1, metadata_dict), ...].

    Namizədləri LlamaIndex node-larına çevirir, LLMRerank ilə TOP-N-ə endirir, sonra
    response synthesizer (Ollama LLM) ilə cavab yaradır. Geri: (cavab, mənbələr) —
    mənbələr UI-da faizlə göstərilən top-N cədvəl üçündür.
    """
    if not candidates:
        return "Sənədlərə əsaslanan cavab yaradıla bilmədi.", []

    nodes = [
        NodeWithScore(node=TextNode(text=text, id_=str(i), metadata=meta), score=score)
        for i, (text, score, meta) in enumerate(candidates)
    ]
    top_nodes = _rerank(question, nodes, top_n)

    # UI cədvəlində eyni sənəd bir neçə parça (chunk) ilə seçilsə belə TƏKRAR göstərilmir —
    # hər fayl üçün ən yüksək balı saxlayırıq, sıra yenə relevanslığa görədir.
    seen_filenames: set[str] = set()
    sources: list[dict] = []
    for n in top_nodes:
        filename = n.node.metadata.get("filename", "sənəd")
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)
        sources.append({
            "filename": filename,
            "score": round(max(0.0, min(1.0, n.score or 0.0)) * 100, 1),
        })

    synthesizer = get_response_synthesizer(
        llm=_llm,
        response_mode="compact",
        text_qa_template=_qa_template(language),
    )
    try:
        response = synthesizer.synthesize(query=question, nodes=top_nodes)
        answer = str(response).strip()
    except Exception as exc:
        logger.warning("LlamaIndex synthesis xətası: %s", exc)
        raise

    if not answer:
        return "Sənədlərə əsaslanan cavab yaradıla bilmədi.", sources
    if language == "az" and not _looks_azerbaijani(answer):
        logger.info("RAG cavabı azərbaycanca görünmür, tərcümə edilir: %s", answer[:200])
        answer = _translate_to_azerbaijani(answer)
    return answer, sources
