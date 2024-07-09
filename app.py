from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS  # 导入 Flask-CORS
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.pdf import PyPDFLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage
import os
import uuid
import time
import json
from functools import wraps

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# Storage for models, knowledge bases, and API keys
models = {}
knowledge_bases = {}
api_keys = {}
embeddings = {}

def require_apikey(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if request.headers.get('Authorization'):
            api_key = request.headers['Authorization'].split(' ')[-1]
            if api_key in api_keys:
                return view_function(*args, **kwargs)
        return jsonify({"error": "Invalid or missing API key"}), 401
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/model', methods=['POST'])
def add_model():
    data = request.json
    model_type = data['type']
    model_id = str(uuid.uuid4())
    
    if model_type == 'openai':
        models[model_id] = ChatOpenAI(
            base_url=data['base_url'],
            api_key=data['api_key'],
            model_name=data['chat_model'],
            streaming=True
        )
    elif model_type == 'azure':
        models[model_id] = AzureChatOpenAI(
            deployment_name=data['deployment_name'],
            openai_api_version=data['api_version'],
            azure_endpoint=data['endpoint'],
            api_key=data['api_key'],
            streaming=True
        )
    elif model_type == 'ollama':
        models[model_id] = Ollama(
            base_url=data['base_url'],
            model=data['model_name']
        )
    else:
        return jsonify({"error": "Invalid model type"}), 400

    api_key = f"sk-{''.join([str(uuid.uuid4()).replace('-', '') for _ in range(2)])[:32]}"
    api_keys[api_key] = model_id

    return jsonify({"model_id": model_id, "api_key": api_key}), 201

@app.route('/api/model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    if model_id in models:
        del models[model_id]
        api_key_to_delete = None
        for api_key, model in api_keys.items():
            if model == model_id:
                api_key_to_delete = api_key
                break
        if api_key_to_delete:
            del api_keys[api_key_to_delete]
        return jsonify({"message": "API key deleted successfully"}), 200
    else:
        return jsonify({"error": "Model not found"}), 404

@app.route('/api/knowledge', methods=['POST'])
def add_knowledge():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    embedding_model = request.form.get('embedding_model')
    base_url = request.form.get('base_url')
    api_key = request.form.get('api_key')

    if not embedding_model or not base_url or not api_key:
        return jsonify(error="Missing embedding parameters"), 400
    
    kb_id = str(uuid.uuid4())
    kb_dir = f"knowledge_bases/{kb_id}"
    os.makedirs(kb_dir, exist_ok=True)
    file_path = os.path.join(kb_dir, file.filename)
    file.save(file_path)

    # Process and index the knowledge base
    if file.filename.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file.filename.endswith('.csv'):
        loader = CSVLoader(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    # Create OpenAIEmbeddings with the specified model, base URL, and API key
    embedding = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=base_url,
        openai_api_key=api_key
    )
    embeddings[kb_id] = embedding
    
    knowledge_bases[kb_id] = Chroma.from_documents(texts, embedding)

    return jsonify({
        "kb_id": kb_id,
        "embedding_model": embedding_model,
        "embedding_base_url": base_url
    }), 201

@app.route('/api/knowledge/<kb_id>', methods=['DELETE'])
def delete_knowledge_base(kb_id):
    if kb_id in knowledge_bases:
        del knowledge_bases[kb_id]
        if kb_id in embeddings:
            del embeddings[kb_id]
        # 删除对应的文件
        kb_dir = f"knowledge_bases/{kb_id}"
        if os.path.exists(kb_dir):
            for file in os.listdir(kb_dir):
                os.remove(os.path.join(kb_dir, file))
            os.rmdir(kb_dir)
        return jsonify({"message": "Knowledge base deleted successfully"}), 200
    else:
        return jsonify({"error": "Knowledge base not found"}), 404


@app.route('/v1/chat/completions', methods=['POST'])
@require_apikey
def chat_completions():
    data = request.json
    api_key = request.headers['Authorization'].split(' ')[-1]
    model_id = api_keys[api_key]
    model = models[model_id]
    
    messages = data.get('messages', [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    query = messages[-1]['content']
    kb_id = data.get('kb_id')
    stream = data.get('stream', False)
    
    if kb_id:
        if kb_id not in knowledge_bases:
            return jsonify({"error": "Knowledge base not found"}), 404
        kb = knowledge_bases[kb_id]

        # 创建 PromptTemplate
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="请根据以下信息回答问题：\n\n{context}\n\n问题：{question}"
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            model,
            kb.as_retriever(),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            prompt_template=prompt_template
        )
        
        if stream:
            return Response(stream_with_context(stream_response(qa_chain, query)), content_type='text/event-stream')
        else:
            response = qa_chain({"question": query})
            return create_chat_completion_response(model_id, response['answer'], "knowledgebot")
    else:
        if stream:
            return Response(stream_with_context(stream_response(model, query)), content_type='text/event-stream')
        else:
            response = model([HumanMessage(content=query)])
            return create_chat_completion_response(model_id, response.content)

def stream_response(model_or_chain, query):
    start_time = time.time()
    full_response = ""
    
    if isinstance(model_or_chain, ConversationalRetrievalChain):
        for chunk in model_or_chain.stream({"question": query}):
            if 'answer' in chunk:
                full_response += chunk['answer']
                yield f"data: {json.dumps(create_chat_completion_chunk(chunk['answer'], model='knowledgebot'))}\n\n"
    else:  # Assuming it's a LangChain chat model
        for chunk in model_or_chain.stream([HumanMessage(content=query)]):
            if chunk.content:
                full_response += chunk.content
                yield f"data: {json.dumps(create_chat_completion_chunk(chunk.content))}\n\n"
    
    yield f"data: {json.dumps(create_chat_completion_chunk('[DONE]', finish_reason='stop', model='knowledgebot' if isinstance(model_or_chain, ConversationalRetrievalChain) else None))}\n\n"

def create_chat_completion_response(model_id, content, bot_name=None):
    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": bot_name or model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(content),  # This is an approximation
            "completion_tokens": len(content),  # This is an approximation
            "total_tokens": len(content) * 2  # This is an approximation
        }
    }

def create_chat_completion_chunk(content, finish_reason=None, model=None):
    chunk = {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or "gpt-3.5-turbo-0613",  # Use the provided model name or default
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": content
                },
                "finish_reason": finish_reason
            }
        ]
    }
    return chunk

@app.route('/v1/models', methods=['GET'])
@require_apikey
def list_models():
    api_key = request.headers['Authorization'].split(' ')[-1]
    model_id = api_keys[api_key]
    
    response = {
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ],
        "object": "list"
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)