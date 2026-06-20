import json
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HISTORY_DIR = Path(__file__).resolve().parent / "chat_history"
HISTORY_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Classificador de Genero - Maratona de Boston", page_icon="🏁")
st.title("🏁 Classificador de Gênero — Maratona de Boston")

with st.sidebar:
    st.header("Projeto")
    st.write("Este modelo de Machine Learning foi treinado com dados da Maratona de Boston 2019 para classificar o gênero do corredor a partir da idade e do tempo de prova. Um agente baseado no Gemini explica os resultados e responde perguntas sobre o tema.")
    st.caption(f"Backend: {BACKEND_URL}")

def _session_path(session_id: str) -> Path:
    return HISTORY_DIR / f"{session_id}.json"


def save_session(session_id: str, messages: list[dict]) -> None:
    """Salva o JSON da sessao com as mensagens atuais."""
    now = datetime.now().isoformat(timespec="seconds")
    started_at = now
    path = _session_path(session_id)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            started_at = existing.get("started_at", now)
        except (json.JSONDecodeError, OSError):
            pass

    data = {
        "session_id": session_id,
        "started_at": started_at,
        "updated_at": now,
        "messages": messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_sessions() -> list[dict]:
    """Le todos os JSONs da pasta de historico, mais recentes primeiro."""
    sessions = []
    for path in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        messages = data.get("messages", [])
        primeira_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        preview = primeira_msg[:60] + ("…" if len(primeira_msg) > 60 else "")
        sessions.append(
            {
                "session_id": data.get("session_id", path.stem),
                "started_at": data.get("started_at", ""),
                "preview": preview or "(sem mensagens)",
                "n_messages": len(messages),
            }
        )
    return sessions

def load_session(session_id: str) -> list[dict]:
    path = _session_path(session_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data.get("messages", [])


tab_sliders, tab_chat, tab_historico = st.tabs(
    ["🏁 Classificar", "💬 Chat com o Agente", "📜 Histórico de Conversas"]
)

# Aba 1 - Classificacao por sliders
with tab_sliders:
    age = st.slider("Idade", min_value=18, max_value=80, value=35)
    result_sec = st.slider(
        "Tempo de prova (segundos)",
        min_value=7200,
        max_value=21600,
        value=12000,
        step=60,
    )

    h = int(result_sec // 3600)
    m = int((result_sec % 3600) // 60)
    s = int(result_sec % 60)
    st.caption(f"Tempo selecionado: {h:02d}h{m:02d}m{s:02d}s")

    if st.button("Classificar", type="primary"):
        payload = {"age": age, "result_sec": result_sec}
        
        try:
            with st.spinner("Consultando o modelo..."):
                pred_resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
                pred_resp.raise_for_status()
                result = pred_resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Erro ao conectar com o backend")
            st.stop()
        except requests.exceptions.RequestException as exc:
            st.error(f"Erro ao consultar o backend: {exc}")
            st.stop()

        col1, col2, col3 = st.columns(3)
        col1.metric("Masculino", f"{result['prob_male']:.1%}")
        col2.metric("Feminino", f"{result['prob_female']:.1%}")
        with col3:
            st.success(f"Classe predita: {result['prediction']}")

        # Explicacao
        with st.expander("Explicação do Gemini", expanded=True):
            try:
                with st.spinner("Gerando explicação com o Gemini..."):
                    explain_resp = requests.post(f"{BACKEND_URL}/explain", json=payload, timeout=30)
                    explain_resp.raise_for_status()
                    explanation = explain_resp.json()["explanation"]
                st.write(explanation)
            except requests.exceptions.ConnectionError:
                st.warning("Não foi possível conectar ao backend para gerar a explicação.")
            except requests.exceptions.HTTPError:
                if explain_resp.status_code == 503:
                    st.warning("GEMINI_API_KEY não configurada no backend: Explicação Indisponível.")
                else:
                    st.warning(f"Erro ao gerar explicação (HTTP {explain_resp.status_code}).")
            except requests.exceptions.RequestException as exc:
                st.warning(f"Erro ao gerar explicação: {exc}")

# Aba 2 - Chat livre com o agente
with tab_chat:
    st.write("Fale sobre a Maratona de Boston 2019. Perguntas fora desse tema não serão respondidas pois o agente foi instruído a recusá-las.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] 
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None

    for msg in st.session_state.chat_history:
        display_role = "assistant" if msg["role"] == "model" else "user"
        with st.chat_message(display_role):
            st.write(msg["content"])

    user_input = st.chat_input("Pergunte sobre a maratona...")

    if user_input:
        if st.session_state.chat_session_id is None:
            st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        reply = None
        try:
            with st.spinner("Pensando..."):
                chat_resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"messages": st.session_state.chat_history},
                    timeout=30,
                )
                chat_resp.raise_for_status()
                reply = chat_resp.json()["reply"]
        except requests.exceptions.ConnectionError:
            reply = ("Erro ao conectar com o backend")
        except requests.exceptions.HTTPError:
            if chat_resp.status_code == 503:
                reply = "GEMINI_API_KEY não configurada no backend: chat indisponível."
            else:
                reply = f"Erro do backend (HTTP {chat_resp.status_code})."
        except requests.exceptions.RequestException as exc:
            reply = f"Erro ao consultar o backend: {exc}"

        st.session_state.chat_history.append({"role": "model", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

        # Persiste a conversa em um arquivo JSON
        save_session(st.session_state.chat_session_id, st.session_state.chat_history)

    if st.session_state.chat_history:
        if st.button("Iniciar nova conversa"):
            st.session_state.chat_history = []
            st.session_state.chat_session_id = None
            st.rerun()

# Aba 3 - Historico de conversas salvas
with tab_historico:
    st.write("Conversas salvas anteriormente na aba 💬 Chat com o Agente.")
    st.caption(f"Arquivos em: `{HISTORY_DIR}`")

    sessions = list_sessions()  # Recarrega os arquivos

    if not sessions:
        st.info("Nenhuma conversa salva ainda. Use a aba 💬 Chat com o Agente para começar uma.")
    else:
        rotulo_para_id = {
            f"{s['started_at']} — {s['preview']} ({s['n_messages']} msgs)": s["session_id"]
            for s in sessions
        }
        opcoes = ["Selecione uma conversa"] + list(rotulo_para_id.keys())
        escolha = st.selectbox("Conversas disponíveis", options=opcoes)

        if escolha != opcoes[0]:
            session_id = rotulo_para_id[escolha]
            mensagens = load_session(session_id)

            st.divider()
            if not mensagens:
                st.warning("Não foi possível ler essa conversa.")
            else:
                for msg in mensagens:
                    display_role = "assistant" if msg.get("role") == "model" else "user"
                    with st.chat_message(display_role):
                        st.write(msg.get("content", ""))
