class MedicalRAGApp {
    constructor() {
        this.currentScreen = 'workspaceSelection';
        this.workspaceType = null;
        this.selectedDocuments = [];
        this.availableDocuments = [];
        this.currentPDF = null;
        this.currentPage = 1;
        this.totalPages = 0;
        this.zoomLevel = 1.0;
        this.chatHistory = [];
        this.apiAvailable = false;
        this.isProcessing = false;
        this.markdownAvailable = false;
        this.init();
    }

    async init() {
        this.bindEvents();
        await this.checkAPIHealth();
        await this.loadAvailableDocuments();
        this.configurePDFJS();
        this.checkMarkdownAvailability();
        console.log('Medical RAG App initialized');
    }

    checkMarkdownAvailability() {
        this.markdownAvailable = typeof marked !== 'undefined';
        if (this.markdownAvailable) {
            console.log('✅ Markdown rendering available');
            // Configure marked options for medical content
            marked.setOptions({
                breaks: true,
                gfm: true,
                sanitize: false // Be careful with this in production
            });
        } else {
            console.warn('⚠️ Marked.js not available - using basic text rendering');
        }
    }

    configurePDFJS() {
        if (typeof pdfjsLib !== 'undefined') {
            try {
                pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
            } catch (error) {
                console.warn('PDF.js worker configuration failed:', error);
            }
        } else {
            console.warn('PDF.js library not loaded');
        }
    }

    bindEvents() {
        // Workspace selection
        document.querySelectorAll('.workspace-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const type = card.dataset.type;
                this.selectWorkspaceType(type);
            });
        });

        // Navigation
        const backBtn = document.querySelector('.back-btn');
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                this.showScreen('workspaceSelection');
            });
        }

        const switchWorkspace = document.getElementById('switchWorkspace');
        if (switchWorkspace) {
            switchWorkspace.addEventListener('click', () => {
                this.showScreen('workspaceSelection');
            });
        }

        // Document selection
        const clearSelection = document.getElementById('clearSelection');
        if (clearSelection) {
            clearSelection.addEventListener('click', () => {
                this.clearDocumentSelection();
            });
        }

        const startWorkspace = document.getElementById('startWorkspace');
        if (startWorkspace) {
            startWorkspace.addEventListener('click', () => {
                this.startWorkspace();
            });
        }

        // File upload
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        if (uploadZone && fileInput) {
            uploadZone.addEventListener('click', () => fileInput.click());
            uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadZone.addEventListener('drop', this.handleDrop.bind(this));
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Chat functionality
        const questionInput = document.getElementById('questionInput');
        const sendButton = document.getElementById('sendQuestion');
        if (questionInput && sendButton) {
            questionInput.addEventListener('input', () => {
                this.autoResize(questionInput);
                sendButton.disabled = !questionInput.value.trim() || this.isProcessing;
            });

            questionInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (questionInput.value.trim() && !this.isProcessing) {
                        this.sendQuestion();
                    }
                }
            });

            sendButton.addEventListener('click', () => {
                if (questionInput.value.trim() && !this.isProcessing) {
                    this.sendQuestion();
                }
            });
        }

        // Document selector
        const documentSelector = document.getElementById('documentSelector');
        if (documentSelector) {
            documentSelector.addEventListener('change', (e) => {
                if (e.target.value) {
                    this.loadPDFDocument(e.target.value);
                } else {
                    this.showPDFPlaceholder();
                }
            });
        }

        // PDF controls
        const prevPage = document.getElementById('prevPage');
        const nextPage = document.getElementById('nextPage');
        const zoomOut = document.getElementById('zoomOut');
        const zoomIn = document.getElementById('zoomIn');

        if (prevPage) {
            prevPage.addEventListener('click', () => {
                if (this.currentPage > 1) {
                    this.currentPage--;
                    this.renderPDFPage();
                }
            });
        }

        if (nextPage) {
            nextPage.addEventListener('click', () => {
                if (this.currentPage < this.totalPages) {
                    this.currentPage++;
                    this.renderPDFPage();
                }
            });
        }

        if (zoomOut) {
            zoomOut.addEventListener('click', () => {
                this.zoomLevel = Math.max(0.5, this.zoomLevel - 0.25);
                this.renderPDFPage();
            });
        }

        if (zoomIn) {
            zoomIn.addEventListener('click', () => {
                this.zoomLevel = Math.min(3.0, this.zoomLevel + 0.25);
                this.renderPDFPage();
            });
        }

        // Modal events
        this.bindModalEvents();
    }

    bindModalEvents() {
        const errorModal = document.getElementById('errorModal');
        if (!errorModal) return;

        const closeButtons = errorModal.querySelectorAll('.modal-close');
        closeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.hideModal();
            });
        });

        errorModal.addEventListener('click', (e) => {
            if (e.target === errorModal) {
                this.hideModal();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideModal();
            }
        });
    }

    async checkAPIHealth() {
        try {
            const response = await fetch('/api/health');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            this.apiAvailable = data.backend_ready === true;

            if (this.apiAvailable) {
                console.log('✅ API connection successful');
            } else {
                console.warn('⚠️ API available but backend not ready');
            }
        } catch (error) {
            console.warn('❌ API not available - running in demo mode:', error.message);
            this.apiAvailable = false;
        }
    }

    async loadAvailableDocuments() {
        if (!this.apiAvailable) {
            this.availableDocuments = this.getDemoDocuments();
            return;
        }

        try {
            const response = await fetch('/api/documents');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            this.availableDocuments = [...(data.permanent || []), ...(data.temporary || [])];
        } catch (error) {
            console.error('Error loading documents:', error);
            this.availableDocuments = this.getDemoDocuments();
        }
    }

    getDemoDocuments() {
        return [
            {
                id: 'demo-1',
                name: 'Medical Terminology Guide.pdf',
                type: 'permanent',
                size: 2458912,
                uploaded_at: '2024-01-15T10:30:00Z'
            },
            {
                id: 'demo-2',
                name: 'Patient Care Protocols.pdf',
                type: 'permanent',
                size: 1875456,
                uploaded_at: '2024-01-10T14:20:00Z'
            },
            {
                id: 'demo-3',
                name: 'Emergency Procedures Manual.pdf',
                type: 'temporary',
                size: 3245678,
                uploaded_at: '2024-01-20T09:15:00Z'
            }
        ];
    }

    selectWorkspaceType(type) {
        this.workspaceType = type;
        this.showScreen('documentSelection');
        this.setupDocumentSelection();
    }

    setupDocumentSelection() {
        const title = document.getElementById('documentSelectionTitle');
        const desc = document.getElementById('documentSelectionDesc');
        const uploadSection = document.getElementById('uploadSection');

        if (this.workspaceType === 'specific') {
            if (title) title.textContent = 'Select Permanent Documents';
            if (desc) desc.textContent = 'Choose from your existing document library';
            if (uploadSection) uploadSection.classList.add('hidden');
        } else {
            if (title) title.textContent = 'Create Temporary Workspace';
            if (desc) desc.textContent = 'Select existing documents or upload new ones';
            if (uploadSection) uploadSection.classList.remove('hidden');
        }

        this.populateDocumentGrid();
    }

    populateDocumentGrid() {
        const grid = document.getElementById('documentGrid');
        if (!grid) return;

        grid.innerHTML = '';

        const filteredDocs = this.workspaceType === 'specific' 
            ? this.availableDocuments.filter(doc => doc.type === 'permanent')
            : this.availableDocuments;

        if (filteredDocs.length === 0) {
            grid.innerHTML = `
                <div class="col-span-full text-center py-8">
                    <p class="text-gray-500">No documents available. ${this.workspaceType === 'temporary' ? 'Upload some documents to get started.' : 'Please add some permanent documents first.'}</p>
                </div>
            `;
            return;
        }

        filteredDocs.forEach(doc => {
            const item = this.createDocumentItem(doc);
            grid.appendChild(item);
        });

        this.updateSelectionActions();
    }

    createDocumentItem(doc) {
        const item = document.createElement('div');
        item.className = 'document-item';
        item.dataset.docId = doc.id;

        const sizeText = doc.size ? (doc.size / 1024 / 1024).toFixed(1) + ' MB' : 'Unknown';
        const dateText = doc.uploaded_at ? new Date(doc.uploaded_at).toLocaleDateString() : 'Unknown';

        item.innerHTML = `
            <div class="document-preview">
                <svg width="48" height="48" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
                </svg>
            </div>
            <div class="document-info">
                <h4>${doc.name}</h4>
                <div class="document-meta">
                    <span>${sizeText}</span>
                    <span class="document-type-badge ${doc.type}">${doc.type}</span>
                </div>
                <div class="document-meta">
                    <span>${dateText}</span>
                </div>
            </div>
        `;

        item.addEventListener('click', () => {
            this.toggleDocumentSelection(doc, item);
        });

        return item;
    }

    toggleDocumentSelection(doc, element) {
        const isSelected = element.classList.contains('selected');
        
        if (isSelected) {
            element.classList.remove('selected');
            this.selectedDocuments = this.selectedDocuments.filter(d => d.id !== doc.id);
        } else {
            element.classList.add('selected');
            this.selectedDocuments.push(doc);
        }

        this.updateSelectionActions();
    }

    updateSelectionActions() {
        const actions = document.querySelector('.selection-actions');
        const countElement = document.querySelector('.selected-count');
        const startButton = document.getElementById('startWorkspace');

        if (countElement) {
            countElement.textContent = `${this.selectedDocuments.length} document(s) selected`;
        }

        if (startButton) {
            startButton.disabled = this.selectedDocuments.length === 0;
        }

        if (actions) {
            actions.style.display = this.selectedDocuments.length > 0 ? 'flex' : 'none';
        }
    }

    clearDocumentSelection() {
        this.selectedDocuments = [];
        document.querySelectorAll('.document-item.selected').forEach(item => {
            item.classList.remove('selected');
        });
        this.updateSelectionActions();
    }

    async startWorkspace() {
        if (this.selectedDocuments.length === 0) {
            this.showError('Please select at least one document.');
            return;
        }

        try {
            this.showLoading('Setting up workspace...');

            // If API is available, load selected documents
            if (this.apiAvailable) {
                await this.loadSelectedDocuments();
            } else {
                // Simulate workspace setup in demo mode
                await new Promise(resolve => setTimeout(resolve, 1500));
            }

            this.hideLoading();
            this.showScreen('mainInterface');
            this.populateDocumentSelector();  // CORRECTION: Ajouter cette ligne
            this.initializeChat();
        } catch (error) {
            this.hideLoading();
            this.showError('Failed to start workspace: ' + error.message);
        }
    }

    // CORRECTION: Fonction populateDocumentSelector corrigée
    populateDocumentSelector() {
        const selector = document.getElementById('documentSelector');
        if (!selector) return;

        selector.innerHTML = '<option value="">Select a document to view</option>';

        this.selectedDocuments.forEach(doc => {
            const option = document.createElement('option');
            option.value = doc.name;
            option.textContent = doc.name;
            selector.appendChild(option);
        });

        console.log('📋 Document selector populated with', this.selectedDocuments.length, 'documents');
    }

    async loadSelectedDocuments() {
        // This would typically send the selected documents to the API
        // For now, we'll just log them
        console.log('Loading selected documents:', this.selectedDocuments);
        
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    initializeChat() {
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
            this.addMessage('assistant', 'Hello! I\'m ready to help you with your medical questions. Please ask me anything about your selected documents.', [], true);
        }
    }

    async sendQuestion() {
        const input = document.getElementById('questionInput');
        const question = input.value.trim();
        
        if (!question || this.isProcessing) return;

        this.isProcessing = true;
        this.updateSendButton();

        // Add user message
        this.addMessage('user', question);
        
        // Clear input
        input.value = '';
        this.autoResize(input);

        try {
            this.showLoading('Processing your question...');
            
            let response;
            if (this.apiAvailable) {
                response = await this.sendQuestionToAPI(question);
            } else {
                response = await this.getDemoResponse(question);
            }

            this.hideLoading();
            
            // Add assistant response
            this.addMessage('assistant', response.answer, response.sources || []);
            
        } catch (error) {
            this.hideLoading();
            console.error('Error sending question:', error);
            this.addMessage('assistant', 'Sorry, I encountered an error while processing your question. Please try again.');
        } finally {
            this.isProcessing = false;
            this.updateSendButton();
        }
    }

    // CORRECTION: Fonction sendQuestionToAPI avec filtres
    async sendQuestionToAPI(question) {
        try {
            // Récupérer le document sélectionné
            const documentSelector = document.getElementById('documentSelector');
            const selectedDocument = documentSelector ? documentSelector.value : '';
            
            // Construire les filtres
            const filters = {};
            if (selectedDocument && selectedDocument !== '') {
                filters.document = selectedDocument;
                console.log('🎯 Recherche limitée au document:', selectedDocument);
            } else {
                console.log('🌐 Recherche dans tous les documents');
            }

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    debug: false,
                    filters: filters  // CORRECTION: Ajouter les filtres
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const processedResponse = this.processAPIResponse(data);
            
            console.log('✅ Réponse API traitée:', processedResponse);
            return processedResponse;
            
        } catch (error) {
            console.error('❌ Erreur API:', error);
            throw new Error(`Erreur lors de la communication avec l'API: ${error.message}`);
        }
    }

    processAPIResponse(data) {
        return {
            answer: data.answer || 'No answer provided',
            sources: (data.sources || []).map(source => ({
                index: source.index,
                score: source.score,
                document: source.document,
                page: source.page,
                content: source.content
            }))
        };
    }

    async getDemoResponse(question) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        return {
            answer: "This is a demo response. In real usage, this would be generated by the medical RAG system based on your selected documents.",
            sources: [
                {
                    index: 1,
                    score: 0.95,
                    document: "Medical Terminology Guide.pdf",
                    page: "15",
                    content: "Sample content from the medical guide..."
                }
            ]
        };
    }

    addMessage(type, content, sources = [], isWelcome = false) {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type} ${isWelcome ? 'welcome-message' : ''}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (this.markdownAvailable && type === 'assistant') {
            contentDiv.innerHTML = marked.parse(content);
        } else {
            contentDiv.textContent = content;
        }

        messageDiv.appendChild(contentDiv);

        // Add sources if provided
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message-sources';
            
            const sourcesTitle = document.createElement('div');
            sourcesTitle.textContent = 'Sources:';
            sourcesTitle.style.fontWeight = 'bold';
            sourcesTitle.style.marginBottom = '8px';
            sourcesDiv.appendChild(sourcesTitle);

            sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-citation';
                sourceItem.onclick = () => this.navigateToSource(source);  // CORRECTION: Fonction maintenant définie
                
                sourceItem.innerHTML = `
                    <div class="source-number">[${index + 1}]</div>
                    <div class="source-info">
                        <div class="source-document">${source.document}</div>
                        <div class="source-page">Page ${source.page}</div>
                        <div class="source-score">Score: ${source.score?.toFixed(3)}</div>
                    </div>
                `;
                
                sourcesDiv.appendChild(sourceItem);
            });

            messageDiv.appendChild(sourcesDiv);
        }

        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();
        messageDiv.appendChild(timestamp);

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        this.chatHistory.push({ type, content, sources, timestamp: Date.now() });
    }

    // CORRECTION: Fonction navigateToSource définie
    navigateToSource(source) {
        console.log('🎯 Navigation vers la source:', source);
        
        // Extraire le nom du document et le numéro de page
        const documentName = source.document;
        const pageNumber = source.page;
        
        // Sélectionner le document dans le sélecteur
        const documentSelector = document.getElementById('documentSelector');
        if (documentSelector && documentName) {
            // Trouver et sélectionner le bon document
            for (let option of documentSelector.options) {
                if (option.text.toLowerCase().includes(documentName.toLowerCase())) {
                    documentSelector.value = option.value;
                    documentSelector.dispatchEvent(new Event('change'));
                    break;
                }
            }
            
            // Naviguer vers la page si possible
            if (pageNumber && pageNumber !== 'N/A') {
                setTimeout(() => {
                    this.navigateToPage(parseInt(pageNumber));
                }, 500);
            }
        }
    }

    // CORRECTION: Nouvelle fonction navigateToPage
    navigateToPage(pageNumber) {
        if (!this.currentPDF || !pageNumber) return;
        
        // Valider le numéro de page
        const targetPage = Math.max(1, Math.min(pageNumber, this.totalPages));
        
        if (targetPage !== this.currentPage) {
            this.currentPage = targetPage;
            this.renderPDFPage();
            
            // Effet visuel de navigation
            this.highlightPageNavigation();
            
            console.log(`📄 Navigation vers la page ${targetPage}`);
        }
    }

    // CORRECTION: Nouvelle fonction pour l'effet visuel
    highlightPageNavigation() {
        const canvas = document.getElementById('pdfCanvas');
        if (canvas) {
            canvas.style.transition = 'all 0.3s ease';
            canvas.style.boxShadow = '0 0 20px rgba(33, 128, 141, 0.6)';
            
            setTimeout(() => {
                canvas.style.boxShadow = '';
            }, 1500);
        }
    }

    updateSendButton() {
        const sendButton = document.getElementById('sendQuestion');
        const questionInput = document.getElementById('questionInput');
        
        if (sendButton && questionInput) {
            sendButton.disabled = !questionInput.value.trim() || this.isProcessing;
        }
    }

    autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    showScreen(screenName) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        const targetScreen = document.getElementById(screenName);
        if (targetScreen) {
            targetScreen.classList.add('active');
            this.currentScreen = screenName;
        }
    }

    showLoading(message = 'Loading...') {
        let overlay = document.getElementById('loadingOverlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="loading-content">
                    <div class="spinner"></div>
                    <p id="loadingMessage">${message}</p>
                </div>
            `;
            document.body.appendChild(overlay);
        }
        
        const messageElement = document.getElementById('loadingMessage');
        if (messageElement) {
            messageElement.textContent = message;
        }
        
        overlay.style.display = 'flex';
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    showError(message) {
        const modal = document.getElementById('errorModal');
        if (modal) {
            const messageElement = modal.querySelector('.modal-body p');
            if (messageElement) {
                messageElement.textContent = message;
            }
            modal.style.display = 'flex';
        } else {
            alert(message);
        }
    }

    hideModal() {
        const modal = document.getElementById('errorModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        this.handleFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.handleFiles(files);
    }

    async handleFiles(files) {
        if (!files.length) return;

        const allowedTypes = ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        const validFiles = files.filter(file => allowedTypes.includes(file.type));

        if (validFiles.length !== files.length) {
            this.showError('Some files were skipped. Only PDF, TXT, DOC, and DOCX files are supported.');
        }

        if (validFiles.length === 0) return;

        try {
            this.showLoading('Uploading files...');
            
            if (this.apiAvailable) {
                await this.uploadFiles(validFiles);
            } else {
                await new Promise(resolve => setTimeout(resolve, 2000));
                validFiles.forEach(file => {
                    this.availableDocuments.push({
                        id: `uploaded-${Date.now()}-${file.name}`,
                        name: file.name,
                        type: 'temporary',
                        size: file.size,
                        uploaded_at: new Date().toISOString()
                    });
                });
            }

            this.hideLoading();
            this.populateDocumentGrid();
        } catch (error) {
            this.hideLoading();
            this.showError('Upload failed: ' + error.message);
        }
    }

    async uploadFiles(files) {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        formData.append('type', 'temporary');

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();
        if (result.uploaded_files) {
            result.uploaded_files.forEach(file => {
                this.availableDocuments.push({
                    id: file.name,
                    name: file.name,
                    type: file.type,
                    size: file.size_bytes,
                    uploaded_at: file.uploaded_at
                });
            });
        }
    }

    async loadPDFDocument(documentName) {
        if (!documentName) return;

        try {
            const url = `/api/documents/pdf/${encodeURIComponent(documentName)}`;
            if (typeof pdfjsLib !== 'undefined') {
                const pdf = await pdfjsLib.getDocument(url).promise;
                this.currentPDF = pdf;
                this.totalPages = pdf.numPages;
                this.currentPage = 1;
                this.renderPDFPage();
            } else {
                this.showPDFUnavailable();
            }
        } catch (error) {
            console.error('Error loading PDF:', error);
            this.showPDFError();
        }
    }

    async renderPDFPage() {
        if (!this.currentPDF) return;

        try {
            const page = await this.currentPDF.getPage(this.currentPage);
            const canvas = document.getElementById('pdfCanvas');
            if (!canvas) return;

            const context = canvas.getContext('2d');
            const viewport = page.getViewport({ scale: this.zoomLevel });

            canvas.height = viewport.height;
            canvas.width = viewport.width;

            await page.render({
                canvasContext: context,
                viewport: viewport
            }).promise;

            this.updatePDFControls();
        } catch (error) {
            console.error('Error rendering PDF page:', error);
        }
    }

    updatePDFControls() {
        const pageInfo = document.getElementById('pageInfo');
        const zoomLevel = document.getElementById('zoomLevel');
        const prevBtn = document.getElementById('prevPage');
        const nextBtn = document.getElementById('nextPage');

        if (pageInfo) {
            pageInfo.textContent = `Page ${this.currentPage} of ${this.totalPages}`;
        }

        if (zoomLevel) {
            zoomLevel.textContent = `${Math.round(this.zoomLevel * 100)}%`;
        }

        if (prevBtn) {
            prevBtn.disabled = this.currentPage <= 1;
        }

        if (nextBtn) {
            nextBtn.disabled = this.currentPage >= this.totalPages;
        }
    }

    showPDFPlaceholder() {
        const container = document.querySelector('.pdf-container');
        if (container) {
            container.innerHTML = `
                <div class="viewer-placeholder">
                    <svg width="64" height="64" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
                    </svg>
                    <p>Choose a document from the dropdown above to view its contents</p>
                </div>
            `;
        }
    }

    showPDFUnavailable() {
        const container = document.querySelector('.pdf-container');
        if (container) {
            container.innerHTML = `
                <div class="viewer-placeholder">
                    <svg width="64" height="64" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M11,16.5L6.5,12L8.5,10L11,12.5L15.5,8L17.5,10L11,16.5Z" />
                    </svg>
                    <p>PDF.js library not loaded. PDF viewing is not available.</p>
                </div>
            `;
        }
    }

    showPDFError() {
        const container = document.querySelector('.pdf-container');
        if (container) {
            container.innerHTML = `
                <div class="viewer-placeholder">
                    <svg width="64" height="64" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M13,13H11V7H13M13,17H11V15H13M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" />
                    </svg>
                    <p>Failed to load the selected document. Please try again.</p>
                </div>
            `;
        }
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.medicalRAGApp = new MedicalRAGApp();
});
