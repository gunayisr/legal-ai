# LegalAI MVP

Vəkillər üçün sənəd yaddaşı, AI risk analizi və müştəri üzrə axtarış API-si.

## İşə salmaq

1. Mac-də [Ollama](https://ollama.com) quraşdırın. Terminalda yüngül Gemma modelini endirib başladın:

```bash
ollama run gemma3:1b
```

Bu pəncərəni açıq saxlayın. Model yalnız sənəd analiz ediləndə işləyir.
API analizi bitən kimi modeli RAM-dan çıxarır (`keep_alive: 0`), buna görə kompüter boşda əlavə yük daşımır.
2. Başqa terminalda layihə qovluğunda işlədin:

```bash
docker compose up --build
```

3. API sənədləri: http://localhost:8000/docs
4. Sadə demo UI: http://localhost:8000/ui/
5. n8n: http://localhost:5678

## Demo

`/docs` səhifəsində `POST /documents/upload` endpointindən TXT, PDF və ya DOCX yükləyin. Sonra `GET /search?q=Müştəri adı` ilə həmin şəxsə bağlı sənədləri görün.

## Kurs mövzuları

- GenAI və prompt engineering: `app/ai.py` sistem promptu + Ollama/Gemma
- LangChain + LCEL: `analysis_chain`
- FastAPI/Pydantic/SQLAlchemy/PostgreSQL: API və məlumat modeli
- Docker: `docker-compose.yml`
- n8n: Telegram, Gemma və MCP workflow-ları üçün hazır servis

## n8n-də son konfiqurasiya

1. n8n-də Telegram credential yaradın (BotFather tokeni).
2. `Schedule Trigger → HTTP Request (GET http://api:8000/court-events/upcoming) → Telegram` workflow-u qurun.
3. n8n-də **Ollama Chat Model** node-u əlavə edin, Base URL olaraq `http://host.docker.internal:11434` və model olaraq `gemma3:1b` yazın. Sonra `Webhook → Ollama Chat Model → Telegram` workflow-u yaradın.
4. Həmin workflow-u n8n MCP Server Trigger ilə MCP tool kimi paylaşın.

Qeyd: Sistem hüquqi məsləhət və ya rəsmi sənəd həqiqiliyi təsdiqi vermir; yalnız ilkin AI indikatorları verir.
