# Локальный ИИ-ассистент для аналитики документов и построения сводок
## Проект направлен на реализацию и построение локальной llm-модели, не требующей больших вычислительных ресурсов и сложных установочных процессов. Модель способна сохранять историю диалога, хранить в векторном виде необходимые тексты и docx-, pdf-документы для последующего анализа информации в них и вывода необходимых данных на основе запроса пользователя. Есть возможность вставять и тестировать несколько вариантов llm-, emdedding-моделей для возможного улучшения работы программы, в экспериментальных целях или в целях оптимизации процесса.  
## Setup:
- git clone https://github.com/d1seternal/local_AI.git<br>cd local_AI
- conda activate your-env
- pip install llama-cpp-python<br>Пример загрузки llm-модели через HuggingFace:<br> pip install huggingface-hub<br>hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q6_K.gguf --local-dir
- pip install -r requirements.txt
## Requirements.txt (текущие в проекте):
conda==26.1.1<br>cmake==3.29.5-msvc4<br>llama-cpp-python==0.3.16<br>torch==2.10.0<br>chromadb==1.5.2<br>sentence-transformers==5.2.3<br>PyPDF2==3.0.1<br>python-docx==1.2.0<br>python==3.11.4
## Структура проекта:
```mermaid
graph TD
    subgraph "local_AI"
        A[requirements.txt]
        
        subgraph LLM[llm/]
            subgraph B1[block1]
                B1F[block1_llm_setup.py]
            end
            
            subgraph B2[block2]
                B2F1[memory.py]
                B2F2[rag.py]
                B2F3[main_llm_rag.py]
            end
            
            subgraph B3[block3]
                
            end
            
            subgraph B4[block4]
                
            end
            
            subgraph FP[final_project]
                
            end
        end
    end
