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
    get_current_session_id
    )
from shared.config import DATA_DIR, SESSIONS_DIR
from shared.__init__ import SessionManager

print("\n")
print("Загрузка ReAct-агента...")
print("\n")


current_session_id=None
session_manager = SessionManager(SESSIONS_DIR)


try:
    agent = create_agent()
    print("Агент успешно загружен!")
except Exception as e:
    print(f"Ошибка загрузки агента: {e}")
    traceback.print_exc()
    agent = None
    
def upload_file(file) -> tuple:
            if file is None:
                return "Файл не выбран", list_files()
            
            try:
                filename = Path(file.name).name
                dest_path = DATA_DIR / filename
                shutil.copy2(file, dest_path)
            
                try:

                    result = vector_memory.add_document(str(dest_path))
                    status = f"{result}"
                    current_id = get_current_session_id()
                    if current_id and hasattr(session_manager, 'add_uploaded_file'):
                        session_manager.add_uploaded_file(current_id, filename)
        
                    sessions_list = session_manager.get_all_sessions()
                    sessions_choices = [(s['session_id'], f"{s['title']} ({s['message_count']} сообщ)") 
                                        for s in sessions_list]
                    return status, list_files(), gr.update(choices=sessions_choices), current_id
                
                except:
                    status = f"Файл сохранен: {filename}"
                
                return status, list_files()
                
            except Exception as e:
                return f"Ошибка: {str(e)}", list_files()
            
def chat_only(message: str) -> str:
    if not message or not message.strip():
        return ""
    
    answer, _ = chat_with_session(message)
    
    return answer


def chat(message, history):
    if message is None:
        return ""
    
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            user_text = "".join(text_parts)
        else:
            user_text = str(content)
    elif isinstance(message, str):
        user_text = message
    elif isinstance(message, tuple):
        user_text = str(message[0]) if message else ""
    elif isinstance(message, list):
        user_text= "\n".join(str(item) for item in message)
    else:
        user_text = str(message)
    
    user_text = user_text.strip()
    
    if not user_text:
        return ""
    
    print(f"\nСообщение: {user_text[:100]}...")
    
    try:
        answer = chat_only(user_text)
        print(f"Тип answer из агента: {type(answer)}")
        print(f"answer: {answer}")
    
        return answer
    
    except Exception as e:
        print(f"Ошибка: {e}")
        traceback.print_exc()
        return f"Ошибка: {str(e)}"

def refresh_files():
    return list_files()

def refresh_sessions():
    sessions = get_all_sessions()
    session_choices = [(s['session_id'], s['title']) for s in sessions]
    current_id = get_current_session_id()
    return gr.update(choices=session_choices, value=current_id)


def on_new_session():
    new_session()
    sessions = get_all_sessions()
    session_choices = [(s['session_id'], s['title']) for s in sessions]
    current_id = get_current_session_id()
    return "Новый диалог создан", gr.update(choices=session_choices, value=current_id)


def on_switch_session(session_id):
    if not session_id:
        return "Сессия не выбрана", gr.update(), gr.update(), []
    
    print(f"\nПереключение на сессию: {session_id}")

    sessions = get_all_sessions()
    existing_ids = [s['session_id'] for s in sessions]
    print(f"Существующие ID: {existing_ids}")
   
    real_id = None
    if session_id not in existing_ids:
        for s in sessions:
            if s['title'] == session_id or session_id in s['title']:
                real_id = s['session_id']
                print(f"Найден ID: {real_id}")
                break
    
    if real_id is None:
        return f"Сессия '{session_id}' не найдена", gr.update(), gr.update(), []
    
    result = switch_session(real_id)
    
    
    sessions = get_all_sessions()
    session_choices = [(s['session_id'], s['title']) for s in sessions]
    current_id = get_current_session_id()
    info = get_session_info()
    
    return result, gr.update(choices=session_choices, value=current_id), info

def on_delete_session(session_id): 
    if not session_id:
        return "Сессия не выбрана", gr.update(), gr.update(), []
    
    sessions = get_all_sessions()
    existing_ids = [s['session_id'] for s in sessions]
    
    print(f"Существующие ID: {existing_ids}")
   
    if session_id not in existing_ids:
        for s in sessions:
            if s['title'] == session_id or session_id in s['title']:
                session_id = s['session_id']
                print(f"Найден ID: {session_id}")
                break
        else:
            return f"Сессия '{session_id}' не найдена", gr.update(), gr.update(), []
    
    result = delete_session(session_id)
    print(f"Результат удаления: {result}")
    
    sessions = get_all_sessions()
    session_choices = [(s['session_id'], s['title']) for s in sessions]
    current_id = get_current_session_id()
    info = get_session_info()
    
    return result, gr.update(choices=session_choices, value=current_id), info


def get_session_info():
    current_id = get_current_session_id()
    if not current_id:
        return "### Текущий диалог\n\nНет активной сессии"
    
    history = session_memory.get_history(current_id)
    msg_count = len(history)
    
    sessions = get_all_sessions()
    title = "Новый диалог"
    for s in sessions:
        if s['session_id'] == current_id:
            title = s['title']
            break
    
    return f"""### Текущий диалог

Название: {title}
\nID: `{current_id}`
\nСообщений: {msg_count}
\nФайлов в сессии: {len(session_memory._sessions.get(current_id, []))}
"""



def create_interface():
    with gr.Blocks(title="RAG Ассистент") as demo:
        gr.Markdown("# RAG Ассистент")
        gr.Markdown("Ассистент для работы с документами. Загружайте файлы и задавайте вопросы!")
        
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
                    value=list_files(),
                    interactive=False,
                    lines=5
                )
                
                refresh_btn = gr.Button("Обновить список файлов")
                
                gr.Markdown("### Сессии диалогов")

                session_info = gr.Markdown(get_session_info())
                
                sessions = get_all_sessions()
                session_choices = [(s['session_id'], s['title']) for s in sessions]
                current_id = get_current_session_id()
                
                session_dropdown = gr.Dropdown(
                    label="Все диалоги",
                    choices=session_choices,
                    interactive=True,
                    value=current_id,
                    allow_custom_value=True
                )
                
                with gr.Row():
                    new_session_btn = gr.Button("Новый диалог", size="sm", variant="primary")
                    switch_btn = gr.Button("Переключить", size="sm", variant="secondary")
                
                delete_btn = gr.Button("Удалить выбранный диалог", size="sm", variant="stop")
                
                refresh_sessions_btn = gr.Button("Обновить список диалогов", size="sm")
                
                session_status = gr.Textbox(
                    label="Статус сессии",
                    interactive=False,
                    value="Готов",
                    visible=True
                )
                
                gr.Markdown("### Информация")
                gr.Markdown(f"Папка с файлами: agent_data")
                gr.Markdown(f"Папка с сессиями: sessions")
                gr.Markdown("Поддерживаемые форматы: TXT, PDF, DOCX")

            
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat,
                    title="Диалог с ассистентом",
                    examples=[
                        "Привет! Расскажи о себе.",
                        "Какие файлы у меня загружены?",
                        "Помоги найти информацию в документе",
                        "Что такое RAG?"
                    ],
                )

        file_upload.upload(
            upload_file,
            inputs=[file_upload],
            outputs=[upload_status, file_list_box]
        )
        
        refresh_btn.click(
            refresh_files,
            inputs=[],
            outputs=[file_list_box]
        )
        
        new_session_btn.click(
            on_new_session,
            inputs=[],
            outputs=[session_status, session_dropdown]
        ).then(
            get_session_info,
            inputs=[],
            outputs=[session_info]
        )

        switch_btn.click(
            on_switch_session,
            inputs=[session_dropdown],
            outputs=[session_status, session_dropdown, session_info]
        ).then(
            None,
            js="() => { setTimeout(() => location.reload(), 500); }"
        )
        
        delete_btn.click(
            on_delete_session,
            inputs=[session_dropdown],
            outputs=[session_status, session_dropdown]
        ).then(
            get_session_info,
            inputs=[],
            outputs=[session_info]
        )
        
        refresh_sessions_btn.click(
            refresh_sessions,
            inputs=[],
            outputs=[session_dropdown]
        ).then(
            get_session_info,
            inputs=[],
            outputs=[session_info]
        )
        demo.load(
            lambda: (gr.update(choices=[(s['session_id'], s['title']) for s in get_all_sessions()],
                               value=get_current_session_id()),
                     get_session_info()),
            inputs=[],
            outputs=[session_dropdown, session_info]
        )
    
    return demo


def run_server(share=False, server_port=8080):
    demo = create_interface()
    print(f"Доступ:")
    print(f"   • http://localhost:{server_port}")
    if share:
        print(f"   • Публично: будет сгенерирована ссылка")
        
    demo.launch(
        server_port=server_port,
        share=share,
        inbrowser=False
    )


if __name__ == "__main__":
    run_server()
