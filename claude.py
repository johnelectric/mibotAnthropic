from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
import PyPDF2
import io
import anthropic
from dotenv import load_dotenv
from typing import List, Dict
import markdown2  # Nueva librería para Markdown

# Cargar variables de entorno
load_dotenv()

app = FastAPI()

# Configuración desde variables de entorno
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "7000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Verificar que la API key esté configurada
if not ANTHROPIC_API_KEY:
    raise ValueError("La variable de entorno ANTHROPIC_API_KEY no está configurada")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Variables globales para almacenar estado
pdf_text = ""
conversation_history: List[Dict[str, str]] = []

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot con PDF</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                height: 100vh;
                display: flex;
                flex-direction: column;
                background-color: #f5f5f5;
            }}
            .container {{
                display: flex;
                flex-direction: column;
                height: 100%;
                padding: 20px;
                box-sizing: border-box;
            }}
            .header {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .upload-section {{
                display: flex;
                flex-direction: column;
                margin-bottom: 20px;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .chat-container {{
                display: flex;
                flex: 1;
                gap: 20px;
            }}
            .chat-box {{
                flex: 1;
                display: flex;
                flex-direction: column;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .chat-messages {{
                flex: 1;
                padding: 15px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            .message {{
                max-width: 80%;
                padding: 10px 15px;
                border-radius: 18px;
                word-wrap: break-word;
            }}
            .user-message {{
                align-self: flex-end;
                background-color: #4CAF50;
                color: white;
            }}
            .bot-message {{
                align-self: flex-start;
                background-color: #e5e5ea;
                color: black;
            }}
            .bot-message-content {{
                line-height: 1.5;
            }}
            .bot-message-content h1, 
            .bot-message-content h2, 
            .bot-message-content h3 {{
                margin-top: 0.5em;
                margin-bottom: 0.5em;
            }}
            .bot-message-content p {{
                margin-bottom: 1em;
            }}
            .bot-message-content ul, 
            .bot-message-content ol {{
                margin-bottom: 1em;
                padding-left: 2em;
            }}
            .bot-message-content code {{
                background-color: #f0f0f0;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }}
            .bot-message-content pre {{
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            .bot-message-content blockquote {{
                border-left: 3px solid #ccc;
                padding-left: 10px;
                margin-left: 0;
                color: #555;
            }}
            .chat-input {{
                display: flex;
                padding: 15px;
                border-top: 1px solid #ddd;
            }}
            #message-input {{
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 20px;
                outline: none;
                font-size: 16px;
            }}
            #send-button {{
                margin-left: 10px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 20px;
                cursor: pointer;
                font-size: 16px;
            }}
            #send-button:hover {{
                background-color: #45a049;
            }}
            .upload-button {{
                padding: 10px 15px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }}
            .upload-button:hover {{
                background-color: #45a049;
            }}
            .status {{
                margin-top: 10px;
                font-style: italic;
                color: #666;
            }}
            .typing-indicator {{
                display: inline-block;
                padding-left: 5px;
            }}
            .typing-dot {{
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #888;
                margin-right: 3px;
                animation: typingAnimation 1.4s infinite ease-in-out;
            }}
            .typing-dot:nth-child(1) {{
                animation-delay: 0s;
            }}
            .typing-dot:nth-child(2) {{
                animation-delay: 0.2s;
            }}
            .typing-dot:nth-child(3) {{
                animation-delay: 0.4s;
            }}
            @keyframes typingAnimation {{
                0%, 60%, 100% {{ transform: translateY(0); }}
                30% {{ transform: translateY(-5px); }}
            }}
            @media (max-width: 768px) {{
                .chat-container {{
                    flex-direction: column;
                }}
                .message {{
                    max-width: 90%;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Chatbot Inteligente</h1>
                <p>Puedes chatear normalmente o subir un PDF para preguntar sobre su contenido</p>
            </div>
            
            <div class="upload-section">
                <label for="pdf-upload">Sube un PDF (opcional):</label>
                <input type="file" id="pdf-upload" accept=".pdf">
                <button class="upload-button" onclick="uploadPDF()">Cargar PDF</button>
                <div id="upload-status" class="status">No hay PDF cargado (puedes chatear igualmente)</div>
            </div>
            
            <div class="chat-container">
                <div class="chat-box">
                    <div class="chat-messages" id="chat-messages">
                        <!-- Mensajes aparecerán aquí -->
                    </div>
                    <div class="chat-input">
                        <input type="text" id="message-input" placeholder="Escribe tu mensaje..." onkeypress="handleKeyPress(event)">
                        <button id="send-button" onclick="sendMessage()">Enviar</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Función para mostrar mensajes en el chat
            function displayMessage(role, content) {{
                const chatMessages = document.getElementById('chat-messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${{role}}-message`;
                
                if (role === 'bot') {{
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'bot-message-content';
                    contentDiv.innerHTML = content;
                    messageDiv.appendChild(contentDiv);
                }} else {{
                    messageDiv.textContent = content;
                }}
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }}
            
            // Función para mostrar indicador de "escribiendo"
            function showTypingIndicator() {{
                const chatMessages = document.getElementById('chat-messages');
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message bot-message';
                typingDiv.id = 'typing-indicator';
                typingDiv.innerHTML = `
                    <span>Pensando</span>
                    <span class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </span>
                `;
                chatMessages.appendChild(typingDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }}
            
            // Función para ocultar indicador de "escribiendo"
            function hideTypingIndicator() {{
                const typingIndicator = document.getElementById('typing-indicator');
                if (typingIndicator) {{
                    typingIndicator.remove();
                }}
            }}
            
            // Función para manejar la tecla Enter
            function handleKeyPress(event) {{
                if (event.key === 'Enter') {{
                    sendMessage();
                }}
            }}
            
            // Función para enviar mensaje
            async function sendMessage() {{
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Mostrar mensaje del usuario
                displayMessage('user', message);
                input.value = '';
                
                // Mostrar indicador de "escribiendo"
                showTypingIndicator();
                
                try {{
                    const response = await fetch('/ask-question', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ question: message }})
                    }});
                    
                    const result = await response.json();
                    
                    // Ocultar indicador de "escribiendo"
                    hideTypingIndicator();
                    
                    if (response.ok) {{
                        displayMessage('bot', result.answer);
                    }} else {{
                        displayMessage('bot', `Error: ${{result.detail || 'Error desconocido'}}`);
                    }}
                }} catch (error) {{
                    hideTypingIndicator();
                    displayMessage('bot', "Error al conectar con el servidor");
                    console.error('Error:', error);
                }}
            }}
            
            async function uploadPDF() {{
                const fileInput = document.getElementById('pdf-upload');
                const statusDiv = document.getElementById('upload-status');
                
                if (fileInput.files.length === 0) {{
                    statusDiv.textContent = "No seleccionaste ningún archivo";
                    return;
                }}
                
                const file = fileInput.files[0];
                if (file.type !== "application/pdf") {{
                    statusDiv.textContent = "El archivo debe ser un PDF";
                    return;
                }}
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {{
                    const response = await fetch('/upload-pdf', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const result = await response.json();
                    if (response.ok) {{
                        statusDiv.textContent = `PDF cargado: ${{file.name}}`;
                        displayMessage('bot', `He cargado el PDF "${{file.name}}". Ahora puedo responder preguntas sobre su contenido.`);
                    }} else {{
                        statusDiv.textContent = `Error: ${{result.detail || 'Error desconocido'}}`;
                        displayMessage('bot', `Error al cargar el PDF: ${{result.detail || 'Error desconocido'}}`);
                    }}
                }} catch (error) {{
                    statusDiv.textContent = "Error al cargar el PDF";
                    displayMessage('bot', "Error al conectar con el servidor");
                    console.error('Error:', error);
                }}
            }}
            
            // Mensaje inicial del bot
            window.onload = function() {{
                displayMessage('bot', "¡Hola! Soy un chatbot inteligente. Puedes hablar conmigo normalmente o subir un PDF y preguntarme sobre su contenido.");
                
                // Configurar el scroll automático
                const chatMessages = document.getElementById('chat-messages');
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Observar cambios en el chat para mantener el scroll abajo
                const observer = new MutationObserver(function(mutations) {{
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }});
                
                observer.observe(chatMessages, {{ childList: true }});
            }};
        </script>
    </body>
    </html>
    """

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_text, conversation_history
    if file.content_type != "application/pdf":
        return JSONResponse(
            status_code=400,
            content={"detail": "El archivo debe ser un PDF"}
        )
    
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        pdf_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
        
        # Limpiar el historial de conversación
        conversation_history = []
        
        return {"detail": "PDF procesado correctamente"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error al procesar el PDF: {str(e)}"}
        )

@app.post("/ask-question")
async def ask_question(data: dict):
    global pdf_text, conversation_history
    question = data.get("question", "")
    
    if not question.strip():
        return JSONResponse(
            status_code=400,
            content={"detail": "La pregunta no puede estar vacía"}
        )
    
    try:
        # Construir el mensaje del sistema según si hay PDF cargado o no
        system_message = (
            f"Responde basándote en este PDF:\n\n{pdf_text}\n\nSi la pregunta no está relacionada con el PDF, responde normalmente. Usa formato Markdown para tus respuestas (encabezados, listas, código, etc.)."
            if pdf_text else
            "Eres un asistente útil. Responde las preguntas de manera clara y concisa usando formato Markdown (encabezados, listas, código, etc.) cuando sea apropiado."
        )
        
        # Agregar pregunta al historial
        conversation_history.append({"role": "user", "content": question})
        
        # Limitar el historial a las últimas 4 interacciones para no sobrecargar
        recent_history = conversation_history[-4:]
        
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_message,
            messages=recent_history
        )
        
        answer = response.content[0].text
        
        # Convertir Markdown a HTML
        html_answer = markdown2.markdown(answer)
        
        # Agregar respuesta al historial
        conversation_history.append({"role": "assistant", "content": answer})
        
        return {"answer": html_answer}
    except anthropic.APIConnectionError as e:
        return JSONResponse(
            status_code=503,
            content={"detail": f"Error de conexión con la API: {str(e)}"}
        )
    except anthropic.RateLimitError as e:
        return JSONResponse(
            status_code=429,
            content={"detail": "Estamos recibiendo muchas solicitudes. Por favor espera un momento."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error al procesar la pregunta: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")