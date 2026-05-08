import gradio as gr
import traceback
from datetime import date
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from block3_agent.agent import (
    create_agent,
    vector_memory, 
    list_files, 
    chat_with_session, 
    session_memory, 
    new_session, 
    switch_session, 
    delete_session, 
    get_all_sessions, 
    get_current_session_id,
    add_uploaded_file_to_session
)
from shared.config import DATA_DIR, SESSIONS_DIR

try:
    agent = create_agent()
    print("Агент успешно загружен!")
except Exception as e:
    print(f"Ошибка загрузки агента: {e}")
    traceback.print_exc()
    agent = None

def load_session_history(session_id: str) -> list:
    if not session_id:
        return []
    
    messages = session_memory.get_history(session_id)
    history = []
    for msg in messages:
        history.append({
            "role": msg['role'],
            "content": msg['content']
        })
    return history


def get_session_info():
    current_id = get_current_session_id()
    if not current_id:
        return "### Текущий диалог\n\nНет активной сессии"
    
    history = session_memory.get_history(current_id)
    msg_count = len(history)
    
    files_count = len(session_memory.get_uploaded_files(current_id))
    
    sessions = get_all_sessions()
    title = "Новый диалог"
    for s in sessions:
        if s['session_id'] == current_id:
            title = s['title']
            break
    
    return f"""### Текущий диалог

**Название:** {title}
**ID:** `{current_id}`
**Сообщений:** {msg_count}
**Файлов в сессии:** {files_count}
"""

def refresh_files():
    return list_files()


def respond(message, history):
    if not message or not message.strip():
        return "", history
    
    history.append({"role": "user", "content": message})
    answer, _ = chat_with_session(message)
    history.append({"role": "assistant", "content": answer})
    
    return "", history


def upload_file(file):
    if file is None:
        return "Файл не выбран", refresh_files()
    
    try:
        filename = Path(file.name).name
        dest_path = DATA_DIR / filename
        shutil.copy2(file, dest_path)
        
        result = vector_memory.add_document(str(dest_path))
        status = f"{result}"
        
        current_id = get_current_session_id()
        if current_id:
            add_uploaded_file_to_session(current_id, filename)
        
        return status, refresh_files()
        
    except Exception as e:
        return f"Ошибка: {str(e)}", refresh_files()


def on_new_session():
    new_session()
    sessions = get_all_sessions()
    session_choices = [(s['session_id'], s['title']) for s in sessions]
    current_id = get_current_session_id()
    
    new_history = load_session_history(current_id)
    info = get_session_info()
    
    return "Новый диалог создан", gr.update(choices=session_choices, value=current_id), info, new_history


def on_switch_session(session_input):
    if not session_input:
        return "Сессия не выбрана", gr.update(), gr.update(), []

    sessions = get_all_sessions()
    real_id = None
    for s in sessions:
        if s['session_id'] == session_input or s['title'] == session_input:
            real_id = s['session_id']
            break
    
    if real_id is None:
        return f"Сессия '{session_input}' не найдена", gr.update(), gr.update(), []
    
    result = switch_session(real_id)
    sessions = get_all_sessions()
    session_choices = [(s['session_id'], s['title']) for s in sessions]
    current_id = get_current_session_id()
    new_history = load_session_history(current_id)
    info = get_session_info()
    
    return result, gr.update(choices=session_choices, value=current_id), info, new_history


def on_delete_session(session_input):
    if not session_input:
        return "Сессия не выбрана", gr.update(), gr.update(), []

    sessions = get_all_sessions()
    real_id = None
    for s in sessions:
        if s['session_id'] == session_input or s['title'] == session_input:
            real_id = s['session_id']
            break
    
    if real_id is None:
        return f"Сессия '{session_input}' не найдена", gr.update(), gr.update(), []
    
    result = delete_session(real_id)
    sessions = get_all_sessions()
    session_choices = [(s['session_id'], s['title']) for s in sessions]
    current_id = get_current_session_id()
    new_history = load_session_history(current_id) if current_id else []
    info = get_session_info()
    
    return result, gr.update(choices=session_choices, value=current_id), info, new_history


def on_refresh_sessions():
    sessions = get_all_sessions()
    session_choices = [(s['session_id'], s['title']) for s in sessions]
    current_id = get_current_session_id()
    info = get_session_info()
    
    return gr.update(choices=session_choices, value=current_id), info


def create_interface():
    with gr.Blocks(title="RAG Ассистент") as demo:
        gr.Markdown("# RAG Ассистент")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Файлы")
                
                file_upload = gr.File(
                    label="Загрузить документ",
                    file_types=[".txt", ".pdf", ".docx"]
                )
                
                upload_status = gr.Textbox(
                    label="Статус",
                    interactive=False,
                    value="Готов к загрузке"
                )
                
                file_list_box = gr.Textbox(
                    label="Доступные файлы",
                    value=refresh_files(),
                    interactive=False,
                    lines=5
                )
                
                refresh_btn = gr.Button("Обновить список файлов")
                
                gr.Markdown("### Сессии")
                
                session_info = gr.Markdown(get_session_info())
                
                sessions = get_all_sessions()
                session_choices = [(s['session_id'], s['title']) for s in sessions]
                current_id = get_current_session_id()
                
                session_dropdown = gr.Dropdown(
                    label="Все диалоги",
                    choices=session_choices,
                    interactive=True,
                    value=current_id
                )
                
                with gr.Row():
                    new_session_btn = gr.Button("Новый диалог", size="sm", variant="primary")
                    switch_btn = gr.Button("Переключить", size="sm", variant="secondary")
                
                delete_btn = gr.Button("Удалить выбранный диалог", size="sm", variant="stop")
                
                refresh_sessions_btn = gr.Button("Обновить список диалогов", size="sm")
                
                session_status = gr.Textbox(
                    label="Статус",
                    interactive=False,
                    value="Готов",
                    visible=True
                )
                
                gr.Markdown("### Информация")
                gr.Markdown(f"Папка с файлами: agent_data")
                gr.Markdown(f"Папка с сессиями: sessions")
                gr.Markdown("Поддерживаемые форматы: TXT, PDF, DOCX")
            
            with gr.Column(scale=3):
                gr.Markdown("### Диалог")
                
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Сообщение",
                        placeholder="Введите вопрос...",
                        scale=4
                    )
                    send_btn = gr.Button("Отправить", scale=1, variant="primary")
                
                clear_btn = gr.Button("Очистить чат", variant="secondary", size="sm")
        
        file_upload.upload(
            upload_file, 
            inputs=[file_upload], 
            outputs=[upload_status, file_list_box]
            )
        
        refresh_btn.click(
            refresh_files, 
            outputs=[file_list_box]
            )
        
        send_btn.click(
            respond, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot]
            )
        
        msg.submit(
            respond, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot]
            )
        
        clear_btn.click(
            lambda: [], 
            outputs=[chatbot]
            )
        
        new_session_btn.click(
            on_new_session, 
            outputs=[session_status, session_dropdown, session_info, chatbot]
            )
        
        switch_btn.click(
            on_switch_session, 
            inputs=[session_dropdown], 
            outputs=[session_status, session_dropdown, session_info, chatbot]
            )
        
        delete_btn.click(
            on_delete_session, 
            inputs=[session_dropdown], 
            outputs=[session_status, session_dropdown, session_info, chatbot]
            )
        
        refresh_sessions_btn.click(
            on_refresh_sessions, 
            outputs=[session_dropdown, session_info]
            )

        demo.load(
            lambda: (
                refresh_files(),
                gr.update(choices=[(s['session_id'], s['title']) for s in get_all_sessions()], value=get_current_session_id()),
                get_session_info(),
                load_session_history(get_current_session_id())
            ),
            outputs=[file_list_box, session_dropdown, session_info, chatbot]
        )
    
    return demo


def run_server(share=False, server_name="0.0.0.0", server_port=8080):
    demo = create_interface()
    print(f"\nДоступ: http://localhost:{server_port}")
    demo.launch(
        server_name=server_name, 
        server_port=server_port, 
        share=share, 
        inbrowser=False
        )


if __name__ == "__main__":
    run_server()