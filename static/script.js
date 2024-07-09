document.addEventListener('DOMContentLoaded', function() {
    updateModelParams();
    document.getElementById('model-type').addEventListener('change', updateModelParams);
    loadSavedData();
});

function updateModelParams() {
    const modelType = document.getElementById('model-type').value;
    let params = '';
    if (modelType === 'openai') {
        params = `
            <input type="text" id="base-url" placeholder="Base URL">
            <input type="text" id="api-key" placeholder="API Key">
            <select id="chat-model">
                <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                <option value="gpt-4">gpt-4</option>
                <option value="gpt-4o">gpt-4o</option>
            </select>
        `;
    } else if (modelType === 'azure') {
        params = `
            <input type="text" id="deployment-name" placeholder="Deployment Name">
            <input type="text" id="api-version" placeholder="API Version">
            <input type="text" id="endpoint" placeholder="Endpoint">
            <input type="text" id="api-key" placeholder="API Key">
        `;
    } else if (modelType === 'ollama') {
        params = `
            <input type="text" id="base-url" placeholder="Base URL">
            <input type="text" id="model-name" placeholder="Model Name">
        `;
    }
    document.getElementById('model-params').innerHTML = params;
}

async function addModelAndGenerateKey() {
    const modelType = document.getElementById('model-type').value;
    let modelData = { type: modelType };

    if (modelType === 'openai') {
        modelData.base_url = document.getElementById('base-url').value;
        modelData.api_key = document.getElementById('api-key').value;
        modelData.chat_model = document.getElementById('chat-model').value;
    } else if (modelType === 'azure') {
        modelData.deployment_name = document.getElementById('deployment-name').value;
        modelData.api_version = document.getElementById('api-version').value;
        modelData.endpoint = document.getElementById('endpoint').value;
        modelData.api_key = document.getElementById('api-key').value;
    } else if (modelType === 'ollama') {
        modelData.base_url = document.getElementById('base-url').value;
        modelData.model_name = document.getElementById('model-name').value;
    }

    try {
        const response = await fetch('/api/model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(modelData),
        });
        const data = await response.json();
        if (response.ok) {
            displayApiKey(data.api_key, modelType, modelData, data.model_id);
            saveApiKey(data.model_id, modelType, modelData, data.api_key);
        } else {
            throw new Error(data.error || 'Failed to generate API key');
        }
    } catch (error) {
        alert('Error generating API key: ' + error.message);
    }
}

function displayApiKey(apiKey, modelType, modelData, modelId) {
    const apiKeysContainer = document.getElementById('api-keys');
    const apiKeyElement = document.createElement('div');
    apiKeyElement.className = 'api-key';
    apiKeyElement.dataset.modelId = modelId;
    
    let modelDetails = '';
    for (const [key, value] of Object.entries(modelData)) {
        if (key !== 'type' && key !== 'api_key') {
            modelDetails += `<strong>${key}:</strong> ${value}<br>`;
        }
    }

    apiKeyElement.innerHTML = `
        <h3>${modelType} Model</h3>
        <strong>API Key:</strong> ${apiKey}<br>
        ${modelDetails}
        <strong>Knowledge Base:</strong> <span class="kb-status">None</span><br>
        <button onclick="updateKnowledgeBase('${modelId}')">Update Knowledge Base</button>
        <button class="delete-api-key" onclick="deleteApiKey('${modelId}')">Delete API Key</button>
    `;
    apiKeysContainer.appendChild(apiKeyElement);
}

async function uploadKnowledgeBase() {
    const fileInput = document.getElementById('kb-file');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload');
        return;
    }

    const embeddingModel = document.getElementById('embedding-model').value;
    const baseUrl = document.getElementById('base-url').value;
    const apiKey = document.getElementById('api-key').value;

    if (!baseUrl || !apiKey) {
        alert('Please add an OpenAI model and generate an API key first');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('embedding_model', embeddingModel);
    formData.append('base_url', baseUrl);
    formData.append('api_key', apiKey);

    try {
        const response = await fetch('/api/knowledge', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (response.ok) {
            displayKnowledgeBase(data.kb_id, file.name, file.type, file.size, data.embedding_model);
            updateAllApiKeys(data.kb_id);
            saveKnowledgeBase(data.kb_id, file.name, file.type, file.size, data.embedding_model);
        } else {
            throw new Error(data.error || 'Failed to upload knowledge base');
        }
    } catch (error) {
        alert('Error uploading knowledge base: ' + error.message);
    }
}

function displayKnowledgeBase(kbId, fileName, fileType, fileSize, embeddingModel) {
    const kbContainer = document.getElementById('knowledge-bases');
    const kbElement = document.createElement('div');
    kbElement.className = 'kb-item';
    kbElement.dataset.kbId = kbId;
    kbElement.innerHTML = `
        <h3>Knowledge Base</h3>
        <strong>ID:</strong> ${kbId}<br>
        <strong>File Name:</strong> ${fileName}<br>
        <strong>File Type:</strong> ${fileType}<br>
        <strong>File Size:</strong> ${formatFileSize(fileSize)}<br>
        <strong>Embedding Model:</strong> ${embeddingModel}<br>
        <button class="delete-btn" onclick="deleteKnowledgeBase('${kbId}')">Delete</button>
    `;
    kbContainer.appendChild(kbElement);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' bytes';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
    else if (bytes < 1073741824) return (bytes / 1048576).toFixed(2) + ' MB';
    else return (bytes / 1073741824).toFixed(2) + ' GB';
}

async function deleteKnowledgeBase(kbId) {
    try {
        const response = await fetch(`/api/knowledge/${kbId}`, {
            method: 'DELETE',
        });
        if (response.ok) {
            const kbElement = document.querySelector(`.kb-item[data-kb-id="${kbId}"]`);
            kbElement.remove();
            updateAllApiKeys('None');
            removeKnowledgeBaseFromStorage(kbId);
            alert('Knowledge base deleted successfully');
        } else {
            const data = await response.json();
            throw new Error(data.error || 'Failed to delete knowledge base');
        }
    } catch (error) {
        alert('Error deleting knowledge base: ' + error.message);
    }
}

function updateAllApiKeys(kbId) {
    const apiKeyElements = document.querySelectorAll('.api-key');
    apiKeyElements.forEach(apiKeyElement => {
        apiKeyElement.querySelector('.kb-status').textContent = kbId;
    });
}

function loadSavedData() {
    // Load saved API keys and knowledge bases from local storage and display them
    const savedApiKeys = JSON.parse(localStorage.getItem('apiKeys') || '[]');
    savedApiKeys.forEach(apiKeyData => {
        displayApiKey(apiKeyData.apiKey, apiKeyData.modelType, apiKeyData.modelData, apiKeyData.modelId);
    });

    const savedKnowledgeBases = JSON.parse(localStorage.getItem('knowledgeBases') || '[]');
    savedKnowledgeBases.forEach(kbData => {
        displayKnowledgeBase(kbData.kbId, kbData.fileName, kbData.fileType, kbData.fileSize, kbData.embeddingModel);
    });
}

function saveApiKey(modelId, modelType, modelData, apiKey) {
    const savedApiKeys = JSON.parse(localStorage.getItem('apiKeys') || '[]');
    savedApiKeys.push({ modelId, modelType, modelData, apiKey });
    localStorage.setItem('apiKeys', JSON.stringify(savedApiKeys));
}

function saveKnowledgeBase(kbId, fileName, fileType, fileSize, embeddingModel) {
    const savedKnowledgeBases = JSON.parse(localStorage.getItem('knowledgeBases') || '[]');
    savedKnowledgeBases.push({ kbId, fileName, fileType, fileSize, embeddingModel });
    localStorage.setItem('knowledgeBases', JSON.stringify(savedKnowledgeBases));
}

function removeKnowledgeBaseFromStorage(kbId) {
    const savedKnowledgeBases = JSON.parse(localStorage.getItem('knowledgeBases') || '[]');
    const updatedKnowledgeBases = savedKnowledgeBases.filter(kb => kb.kbId !== kbId);
    localStorage.setItem('knowledgeBases', JSON.stringify(updatedKnowledgeBases));
}

// 添加deleteApiKey函数
async function deleteApiKey(modelId) {
    try {
        const response = await fetch(`/api/model/${modelId}`, {
            method: 'DELETE',
        });
        if (response.ok) {
            const apiKeyElement = document.querySelector(`.api-key[data-model-id="${modelId}"]`);
            apiKeyElement.remove();
            removeApiKeyFromStorage(modelId);
            alert('API key deleted successfully');
        } else {
            const data = await response.json();
            throw new Error(data.error || 'Failed to delete API key');
        }
    } catch (error) {
        alert('Error deleting API key: ' + error.message);
    }
}

// 删除存储中的API key
function removeApiKeyFromStorage(modelId) {
    const savedApiKeys = JSON.parse(localStorage.getItem('apiKeys') || '[]');
    const updatedApiKeys = savedApiKeys.filter(apiKeyData => apiKeyData.modelId !== modelId);
    localStorage.setItem('apiKeys', JSON.stringify(updatedApiKeys));
}