// Base API URL
const API_URL = '/api';

// DOM elements
const elements = {
    // Navigation
    dashboardLink: document.getElementById('dashboardLink'),
    classesLink: document.getElementById('classesLink'),
    inspectionsLink: document.getElementById('inspectionsLink'),
    
    // Views
    dashboardView: document.getElementById('dashboardView'),
    classDetailView: document.getElementById('classDetailView'),
    inspectionsView: document.getElementById('inspectionsView'),
    
    // Dashboard elements
    serverStatus: document.getElementById('serverStatus'),
    serverVersion: document.getElementById('serverVersion'),
    serverHost: document.getElementById('serverHost'),
    modulesInfo: document.getElementById('modulesInfo'),
    serverInfo: document.getElementById('serverInfo'),
    dockerInfo: document.getElementById('dockerInfo'),
    
    totalClasses: document.getElementById('totalClasses'),
    totalObjects: document.getElementById('totalObjects'),
    
    classesTableBody: document.getElementById('classesTableBody'),
    classesLoading: document.getElementById('classesLoading'),
    classesTableContainer: document.getElementById('classesTableContainer'),
    refreshBtn: document.getElementById('refreshBtn'),
    inspectBtn: document.getElementById('inspectBtn'),
    createClassBtn: document.getElementById('createClassBtn'),
    exportClassesBtn: document.getElementById('exportClassesBtn'),
    
    // Class detail elements
    classDetailTitle: document.getElementById('classDetailTitle'),
    classDetailLoading: document.getElementById('classDetailLoading'),
    classDetailInfo: document.getElementById('classDetailInfo'),
    backToClassesBtn: document.getElementById('backToClassesBtn'),
    
    classObjectCount: document.getElementById('classObjectCount'),
    classPropertyCount: document.getElementById('classPropertyCount'),
    classVectorType: document.getElementById('classVectorType'),
    classVectorDimension: document.getElementById('classVectorDimension'),
    classStorageEstimate: document.getElementById('classStorageEstimate'),
    classQueryTime: document.getElementById('classQueryTime'),
    classProperties: document.getElementById('classProperties'),
    sampleObjectsContainer: document.getElementById('sampleObjectsContainer'),
    viewAllObjectsBtn: document.getElementById('viewAllObjectsBtn'),
    
    // Query elements
    queryForm: document.getElementById('queryForm'),
    queryText: document.getElementById('queryText'),
    queryType: document.getElementById('queryType'),
    queryLimit: document.getElementById('queryLimit'),
    runQueryBtn: document.getElementById('runQueryBtn'),
    queryResults: document.getElementById('queryResults'),
    queryInfo: document.getElementById('queryInfo'),
    queryResultsContainer: document.getElementById('queryResultsContainer'),
    queryLoading: document.getElementById('queryLoading'),
    
    // Inspections elements
    inspectionsLoading: document.getElementById('inspectionsLoading'),
    inspectionsTableContainer: document.getElementById('inspectionsTableContainer'),
    inspectionsTableBody: document.getElementById('inspectionsTableBody'),
    newInspectionBtn: document.getElementById('newInspectionBtn'),
    
    // Modals
    deleteClassModal: new bootstrap.Modal(document.getElementById('deleteClassModal')),
    deleteClassName: document.getElementById('deleteClassName'),
    confirmDeleteBtn: document.getElementById('confirmDeleteBtn'),
    
    newInspectionModal: new bootstrap.Modal(document.getElementById('newInspectionModal')),
    includeSamplesCheck: document.getElementById('includeSamplesCheck'),
    runBenchmarksCheck: document.getElementById('runBenchmarksCheck'),
    startInspectionBtn: document.getElementById('startInspectionBtn'),
    
    // New Class Modal
    newClassModal: new bootstrap.Modal(document.getElementById('newClassModal')),
    newClassName: document.getElementById('newClassName'),
    classDescription: document.getElementById('classDescription'),
    vectorizer: document.getElementById('vectorizer'),
    vectorDimension: document.getElementById('vectorDimension'),
    propertiesContainer: document.getElementById('propertiesContainer'),
    addPropertyBtn: document.getElementById('addPropertyBtn'),
    createClassError: document.getElementById('createClassError'),
    createClassSubmitBtn: document.getElementById('createClassSubmitBtn'),
    
    // Update Class Modal
    updateClassModal: new bootstrap.Modal(document.getElementById('updateClassModal')),
    updateClassName: document.getElementById('updateClassName'),
    updateVectorizer: document.getElementById('updateVectorizer'),
    updateVectorDimension: document.getElementById('updateVectorDimension'),
    updateVectorizeClassName: document.getElementById('updateVectorizeClassName'),
    updateMaxConnections: document.getElementById('updateMaxConnections'),
    updateEfConstruction: document.getElementById('updateEfConstruction'),
    updateDistanceMetric: document.getElementById('updateDistanceMetric'),
    updateClassError: document.getElementById('updateClassError'),
    updateClassSubmitBtn: document.getElementById('updateClassSubmitBtn'),
    updatePropContainer: document.getElementById('updatePropContainer'),
    updateClassSuccess: document.getElementById('updateClassSuccess'),
    
    // Docker Login Modal
    dockerLoginModal: new bootstrap.Modal(document.getElementById('dockerLoginModal')),
    dockerUsername: document.getElementById('username'),
    dockerPassword: document.getElementById('password'),
    loginError: document.getElementById('loginError'),
    submitLoginBtn: document.getElementById('submitLoginBtn'),
    geminiModel: document.getElementById('geminiModel')
};

// Current state
let state = {
    currentView: 'dashboard',
    currentClass: null,
    classes: [],
    inspections: [],
    meta: {},
    classToDelete: null,
    classToUpdate: null,
    isAuthenticated: false
};

// API request helper
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_URL}${endpoint}`, options);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error(`API Error for ${endpoint}:`, error);
        // You could add a toast notification here
        throw error;
    }
}

// View navigation
function showView(viewName) {
    // Hide all views
    elements.dashboardView.style.display = 'none';
    elements.classDetailView.style.display = 'none';
    elements.inspectionsView.style.display = 'none';
    
    // Reset active links
    elements.dashboardLink.classList.remove('active');
    elements.classesLink.classList.remove('active');
    elements.inspectionsLink.classList.remove('active');
    
    // Show selected view
    switch (viewName) {
        case 'dashboard':
            elements.dashboardView.style.display = 'block';
            elements.dashboardLink.classList.add('active');
            loadDashboard();
            break;
        case 'classDetail':
            elements.classDetailView.style.display = 'block';
            elements.classesLink.classList.add('active');
            break;
        case 'inspections':
            elements.inspectionsView.style.display = 'block';
            elements.inspectionsLink.classList.add('active');
            loadInspections();
            break;
    }
    
    state.currentView = viewName;
}

// Load dashboard data
async function loadDashboard() {
    try {
        // Show loading indicators
        elements.classesTableContainer.style.display = 'none';
        elements.classesLoading.style.display = 'flex';
        
        // Fetch server metadata
        const meta = await fetchAPI('/meta');
        state.meta = meta;
        
        // Update server info
        if (meta && meta.health) {
            elements.serverStatus.textContent = meta.health.status === 'OK' ? 'Ready' : 'Not Ready';
            elements.serverStatus.className = meta.health.status === 'OK' ? 'badge bg-success' : 'badge bg-danger';
        }
        
        if (meta && meta.meta) {
            elements.serverVersion.textContent = meta.meta.version || 'Unknown';
            elements.serverHost.textContent = meta.meta.hostname || 'Unknown';
            
            // Display modules
            if (meta.meta.modules) {
                const modulesList = Object.entries(meta.meta.modules).map(([name, info]) => {
                    return `<span class="badge bg-info me-2">${name}: ${info.version || 'Unknown'}</span>`;
                }).join(' ');
                
                elements.modulesInfo.innerHTML = modulesList || 'No modules found';
            }
        }

        // Fetch Docker container information
        try {
            const dockerInfo = await fetchAPI('/docker-info');
            if (dockerInfo && dockerInfo.status === 'success' && dockerInfo.containers && dockerInfo.containers.length > 0) {
                // Display Docker container information
                let dockerHtml = `
                    <div class="card mt-4">
                        <div class="card-header bg-info text-white d-flex justify-content-between align-items-center">
                            <div><i class="bi bi-hdd-rack"></i> Docker Container Information</div>
                            <button class="btn btn-sm btn-light refresh-docker-btn">
                                <i class="bi bi-arrow-clockwise"></i> Refresh
                            </button>
                        </div>
                        <div class="card-body">
                            <div class="mb-4">
                                <div class="row mb-3">
                                    <div class="col-md-6">
                                        <div id="dockerCpuGauge" class="gauge-container"></div>
                                    </div>
                                    <div class="col-md-6">
                                        <div id="dockerMemoryGauge" class="gauge-container"></div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-12">
                                        <div id="dockerSystemInfo" class="system-info p-3 bg-light rounded">
                                            <h6><i class="bi bi-info-circle"></i> System Information</h6>
                                            <div class="table-responsive">
                                                <table class="table table-sm">
                                                    <tbody>
                                                        <tr>
                                                            <th width="30%">Total Containers:</th>
                                                            <td>${dockerInfo.containers.length}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Weaviate Version:</th>
                                                            <td>${dockerInfo.containers[0].image.split(':')[1] || 'Unknown'}</td>
                                                        </tr>
                                                        ${dockerInfo.system ? `
                                                        <tr>
                                                            <th>Total System Containers:</th>
                                                            <td>${dockerInfo.system.total_containers || 'Unknown'} (${dockerInfo.system.running_containers || 'Unknown'} running)</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Total Images:</th>
                                                            <td>${dockerInfo.system.total_images || 'Unknown'}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Volumes:</th>
                                                            <td>${dockerInfo.system.volumes || 'Unknown'}</td>
                                                        </tr>
                                                        ` : ''}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <nav>
                                <div class="nav nav-tabs" id="docker-nav-tab" role="tablist">
                                    <button class="nav-link active" id="docker-overview-tab" data-bs-toggle="tab" data-bs-target="#docker-overview" 
                                            type="button" role="tab" aria-controls="docker-overview" aria-selected="true">
                                        Overview
                                    </button>
                                    <button class="nav-link" id="docker-resources-tab" data-bs-toggle="tab" data-bs-target="#docker-resources" 
                                            type="button" role="tab" aria-controls="docker-resources" aria-selected="false">
                                        Resource Usage
                                    </button>
                                    <button class="nav-link" id="docker-network-tab" data-bs-toggle="tab" data-bs-target="#docker-network" 
                                            type="button" role="tab" aria-controls="docker-network" aria-selected="false">
                                        Network & I/O
                                    </button>
                                    <button class="nav-link" id="docker-details-tab" data-bs-toggle="tab" data-bs-target="#docker-details" 
                                            type="button" role="tab" aria-controls="docker-details" aria-selected="false">
                                        Container Details
                                    </button>
                                    <button class="nav-link" id="docker-logs-tab" data-bs-toggle="tab" data-bs-target="#docker-logs" 
                                            type="button" role="tab" aria-controls="docker-logs" aria-selected="false">
                                        Logs
                                    </button>
                                </div>
                            </nav>
                            <div class="tab-content p-3 border border-top-0 rounded-bottom" id="docker-nav-tabContent">
                                <div class="tab-pane fade show active" id="docker-overview" role="tabpanel" aria-labelledby="docker-overview-tab">
                                    <div class="table-responsive">
                                        <table class="table table-striped table-hover">
                                            <thead class="table-primary">
                                                <tr>
                                                    <th>Container</th>
                                                    <th>Image</th>
                                                    <th>Status</th>
                                                    <th>CPU Usage</th>
                                                    <th>Memory Usage</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                `;
                
                // Add container rows
                dockerInfo.containers.forEach(container => {
                    const cpuPercentage = parseFloat(container.cpu_usage) || 0;
                    const memPercentage = parseFloat(container.memory_percent) || 0;
                    
                    let statusBadgeClass = 'bg-warning';
                    if (container.status.includes('Up')) {
                        statusBadgeClass = 'bg-success';
                    } else if (container.status.includes('Exit')) {
                        statusBadgeClass = 'bg-danger';
                    }
                    
                    dockerHtml += `
                        <tr>
                            <td>
                                <div class="d-flex align-items-center">
                                    <span class="container-status-indicator ${container.status.includes('Up') ? 'online' : 'offline'}"></span>
                                    <strong>${container.name}</strong>
                                </div>
                                <div class="small text-muted">${container.id.substring(0, 12)}</div>
                            </td>
                            <td>
                                <div>${container.image}</div>
                            </td>
                            <td><span class="badge ${statusBadgeClass}">${container.status}</span></td>
                            <td>
                                <div class="progress" role="progressbar" aria-label="CPU Usage" aria-valuenow="${cpuPercentage}" aria-valuemin="0" aria-valuemax="100">
                                    <div class="progress-bar ${cpuPercentage > 80 ? 'bg-danger' : cpuPercentage > 50 ? 'bg-warning' : 'bg-primary'}" 
                                         style="width: ${cpuPercentage}%">${container.cpu_usage}</div>
                                </div>
                            </td>
                            <td>
                                <div class="progress" role="progressbar" aria-label="Memory Usage" aria-valuenow="${memPercentage}" aria-valuemin="0" aria-valuemax="100">
                                    <div class="progress-bar ${memPercentage > 80 ? 'bg-danger' : memPercentage > 50 ? 'bg-warning' : 'bg-info'}" 
                                         style="width: ${memPercentage}%">${container.memory_usage}</div>
                                </div>
                            </td>
                        </tr>
                    `;
                });
                
                dockerHtml += `
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                                <div class="tab-pane fade" id="docker-resources" role="tabpanel" aria-labelledby="docker-resources-tab">
                                    <div class="row">
                                        <div class="col-md-12 mb-4">
                                            <div id="dockerResourceTimeline" style="height: 300px;"></div>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="col-md-6">
                                            <div class="card mb-3">
                                                <div class="card-header bg-primary text-white">
                                                    <i class="bi bi-cpu"></i> CPU Utilization
                                                </div>
                                                <div class="card-body">
                                                    <div id="dockerCpuChart"></div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="card mb-3">
                                                <div class="card-header bg-info text-white">
                                                    <i class="bi bi-memory"></i> Memory Utilization
                                                </div>
                                                <div class="card-body">
                                                    <div id="dockerMemoryChart"></div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="table-responsive mt-3">
                                        <table class="table table-bordered">
                                            <thead class="table-dark">
                                                <tr>
                                                    <th>Container</th>
                                                    <th>CPU Cores</th>
                                                    <th>CPU Usage</th>
                                                    <th>Memory Used</th>
                                                    <th>Memory Total</th>
                                                    <th>Memory %</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                `;
                
                // Add detailed resource rows
                dockerInfo.containers.forEach(container => {
                    const memParts = container.memory_usage.split(' / ');
                    const memUsed = memParts[0] || 'N/A';
                    const memTotal = memParts.length > 1 ? memParts[1] : 'N/A';
                    const memPercent = container.memory_percent || 'N/A';
                    
                    dockerHtml += `
                        <tr>
                            <td><strong>${container.name}</strong></td>
                            <td>${parseFloat(container.cpu_usage) / 100 || 'N/A'}</td>
                            <td>${container.cpu_usage}</td>
                            <td>${memUsed}</td>
                            <td>${memTotal}</td>
                            <td>${memPercent}</td>
                        </tr>
                    `;
                });
                
                dockerHtml += `
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                                <div class="tab-pane fade" id="docker-network" role="tabpanel" aria-labelledby="docker-network-tab">
                                    <div class="row">
                                        <div class="col-md-12 mb-4">
                                            <div id="dockerNetworkGraph" style="height: 300px;"></div>
                                        </div>
                                    </div>
                                    <div class="table-responsive">
                                        <table class="table table-striped">
                                            <thead class="table-dark">
                                                <tr>
                                                    <th>Container</th>
                                                    <th>Network I/O</th>
                                                    <th>Block I/O</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                `;
                
                // Add network rows
                dockerInfo.containers.forEach(container => {
                    dockerHtml += `
                        <tr>
                            <td><strong>${container.name}</strong></td>
                            <td>
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-arrow-down text-success me-1"></i>
                                    <i class="bi bi-arrow-up text-danger me-2"></i>
                                    ${container.network_io}
                                </div>
                            </td>
                            <td>
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-hdd me-2"></i>
                                    ${container.block_io}
                                </div>
                            </td>
                        </tr>
                    `;
                });
                
                dockerHtml += `
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                                <div class="tab-pane fade" id="docker-details" role="tabpanel" aria-labelledby="docker-details-tab">
                                    <div class="row">
                                        <div class="col-12">
                                            <h5 class="mb-3">Container Configuration Details</h5>
                                            <div class="accordion" id="containerDetailsAccordion">
                                                ${dockerInfo.containers.map((container, index) => {
                                                    const details = container.details || {};
                                                    const ports = details.ports || [];
                                                    const volumes = details.volumes || [];
                                                    const envVars = details.env_vars || [];
                                                    const resourceLimits = details.resource_limits || {};
                                                    const healthStatus = details.health_status || 'N/A';
                                                    
                                                    // Format creation date
                                                    let creationDate = 'Unknown';
                                                    if (details.created) {
                                                        try {
                                                            creationDate = new Date(details.created).toLocaleString();
                                                        } catch (e) {
                                                            creationDate = details.created;
                                                        }
                                                    }
                                                    
                                                    return `
                                                        <div class="accordion-item">
                                                            <h2 class="accordion-header" id="heading${index}">
                                                                <button class="accordion-button ${index === 0 ? '' : 'collapsed'}" type="button" data-bs-toggle="collapse" 
                                                                        data-bs-target="#collapse${index}" aria-expanded="${index === 0 ? 'true' : 'false'}" 
                                                                        aria-controls="collapse${index}">
                                                                    <div class="d-flex align-items-center w-100">
                                                                        <span class="container-status-indicator ${container.status.includes('Up') ? 'online' : 'offline'}"></span>
                                                                        <strong class="me-2">${container.name}</strong>
                                                                        <span class="badge ${healthStatus === 'healthy' ? 'bg-success' : 
                                                                                            healthStatus === 'unhealthy' ? 'bg-danger' : 
                                                                                            'bg-secondary'} ms-auto">
                                                                            ${healthStatus}
                                                                        </span>
                                                                    </div>
                                                                </button>
                                                            </h2>
                                                            <div id="collapse${index}" class="accordion-collapse collapse ${index === 0 ? 'show' : ''}" 
                                                                aria-labelledby="heading${index}" data-bs-parent="#containerDetailsAccordion">
                                                                <div class="accordion-body">
                                                                    <div class="row">
                                                                        <div class="col-md-6">
                                                                            <h6><i class="bi bi-info-circle"></i> Basic Information</h6>
                                                                            <table class="table table-sm">
                                                                                <tbody>
                                                                                    <tr>
                                                                                        <th width="30%">Container ID:</th>
                                                                                        <td><code>${container.id}</code></td>
                                                                                    </tr>
                                                                                    <tr>
                                                                                        <th>Image:</th>
                                                                                        <td>${container.image}</td>
                                                                                    </tr>
                                                                                    <tr>
                                                                                        <th>Status:</th>
                                                                                        <td>${container.status}</td>
                                                                                    </tr>
                                                                                    <tr>
                                                                                        <th>Created:</th>
                                                                                        <td>${creationDate}</td>
                                                                                    </tr>
                                                                                    <tr>
                                                                                        <th>Restarts:</th>
                                                                                        <td>${details.restarts || 0}</td>
                                                                                    </tr>
                                                                                </tbody>
                                                                            </table>
                                                                        </div>
                                                                        <div class="col-md-6">
                                                                            <h6><i class="bi bi-gear"></i> Resource Limits</h6>
                                                                            <table class="table table-sm">
                                                                                <tbody>
                                                                                    <tr>
                                                                                        <th width="30%">Memory Limit:</th>
                                                                                        <td>${resourceLimits.memory || 'No limit'}</td>
                                                                                    </tr>
                                                                                    <tr>
                                                                                        <th>CPU Limit:</th>
                                                                                        <td>${resourceLimits.cpu || 'No limit'}</td>
                                                                                    </tr>
                                                                                    <tr>
                                                                                        <th>Current CPU:</th>
                                                                                        <td>${container.cpu_usage}</td>
                                                                                    </tr>
                                                                                    <tr>
                                                                                        <th>Current Memory:</th>
                                                                                        <td>${container.memory_usage}</td>
                                                                                    </tr>
                                                                                </tbody>
                                                                            </table>
                                                                        </div>
                                                                    </div>
                                                                    
                                                                    <div class="row mt-3">
                                                                        <div class="col-md-6">
                                                                            <h6><i class="bi bi-hdd-network"></i> Network Configuration</h6>
                                                                            ${ports.length > 0 ? `
                                                                                <table class="table table-sm table-bordered">
                                                                                    <thead class="table-light">
                                                                                        <tr>
                                                                                            <th>Port Mappings</th>
                                                                                        </tr>
                                                                                    </thead>
                                                                                    <tbody>
                                                                                        ${ports.map(port => `
                                                                                            <tr>
                                                                                                <td><code>${port}</code></td>
                                                                                            </tr>
                                                                                        `).join('')}
                                                                                    </tbody>
                                                                                </table>
                                                                            ` : `<p class="text-muted">No ports exposed</p>`}
                                                                        </div>
                                                                        <div class="col-md-6">
                                                                            <h6><i class="bi bi-hdd"></i> Storage Volumes</h6>
                                                                            ${volumes.length > 0 ? `
                                                                                <table class="table table-sm table-bordered">
                                                                                    <thead class="table-light">
                                                                                        <tr>
                                                                                            <th>Volume Mounts</th>
                                                                                        </tr>
                                                                                    </thead>
                                                                                    <tbody>
                                                                                        ${volumes.map(volume => `
                                                                                            <tr>
                                                                                                <td><code>${volume}</code></td>
                                                                                            </tr>
                                                                                        `).join('')}
                                                                                    </tbody>
                                                                                </table>
                                                                            ` : `<p class="text-muted">No volumes mounted</p>`}
                                                                        </div>
                                                                    </div>
                                                                    
                                                                    <div class="row mt-3">
                                                                        <div class="col-12">
                                                                            <h6><i class="bi bi-list-ul"></i> Environment Variables</h6>
                                                                            ${envVars.length > 0 ? `
                                                                                <div class="env-vars-container p-2 bg-light rounded" style="max-height: 150px; overflow-y: auto;">
                                                                                    <pre class="mb-0"><code>${envVars.join('\n')}</code></pre>
                                                                                </div>
                                                                            ` : `<p class="text-muted">No environment variables available or all are filtered for security</p>`}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    `;
                                                }).join('')}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="tab-pane fade" id="docker-logs" role="tabpanel" aria-labelledby="docker-logs-tab">
                                    <div class="row">
                                        <div class="col-12">
                                            <h5 class="mb-3">Container Logs</h5>
                                            <div class="accordion" id="containerLogsAccordion">
                                                ${dockerInfo.containers.map((container, index) => {
                                                    const logs = container.logs || [];
                                                    
                                                    return `
                                                        <div class="accordion-item">
                                                            <h2 class="accordion-header" id="logHeading${index}">
                                                                <button class="accordion-button ${index === 0 ? '' : 'collapsed'}" type="button" data-bs-toggle="collapse" 
                                                                        data-bs-target="#logCollapse${index}" aria-expanded="${index === 0 ? 'true' : 'false'}" 
                                                                        aria-controls="logCollapse${index}">
                                                                    <div class="d-flex align-items-center">
                                                                        <span class="container-status-indicator ${container.status.includes('Up') ? 'online' : 'offline'}"></span>
                                                                        <strong>${container.name}</strong>
                                                                    </div>
                                                                </button>
                                                            </h2>
                                                            <div id="logCollapse${index}" class="accordion-collapse collapse ${index === 0 ? 'show' : ''}" 
                                                                aria-labelledby="logHeading${index}" data-bs-parent="#containerLogsAccordion">
                                                                <div class="accordion-body">
                                                                    <div class="logs-container bg-dark text-light p-3 rounded" style="max-height: 300px; overflow-y: auto;">
                                                                        ${logs.length > 0 ? `
                                                                            <pre class="mb-0"><code>${logs.join('\n')}</code></pre>
                                                                        ` : `<p class="text-muted">No logs available</p>`}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    `;
                                                }).join('')}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="alert alert-info mt-3">
                                <i class="bi bi-info-circle"></i> Weaviate Docker container statistics are updated on refresh.
                                For real-time monitoring, consider using a dedicated tool like Prometheus with Grafana.
                            </div>
                        </div>
                    </div>
                `;
                
                // Create a container for Docker info
                if (!elements.dockerInfo) {
                    elements.dockerInfo = document.createElement('div');
                    elements.dockerInfo.id = 'dockerInfo';
                    elements.serverInfo.parentNode.insertBefore(elements.dockerInfo, elements.serverInfo.nextSibling);
                }
                
                elements.dockerInfo.innerHTML = dockerHtml;
                
                // Add event listener for refresh button
                const refreshDockerBtn = document.querySelector('.refresh-docker-btn');
                if (refreshDockerBtn) {
                    refreshDockerBtn.addEventListener('click', async () => {
                        try {
                            const updatedDockerInfo = await fetchAPI('/docker-info');
                            if (updatedDockerInfo && updatedDockerInfo.status === 'success') {
                                // Re-render the Docker information
                                const currentView = state.currentView;
                                showView('dashboard');
                            }
                        } catch (error) {
                            console.error('Error refreshing Docker info:', error);
                        }
                    });
                }
                
                // Create enhanced visualizations using Plotly
                if (typeof Plotly !== 'undefined') {
                    // Create CPU Gauge
                    const avgCpu = dockerInfo.containers.reduce((sum, container) => sum + (parseFloat(container.cpu_usage) || 0), 0) / dockerInfo.containers.length;
                    
                    const cpuGaugeData = [
                        {
                            type: "indicator",
                            mode: "gauge+number",
                            value: avgCpu,
                            title: { text: "Average CPU Usage (%)", font: { size: 16 } },
                            gauge: {
                                axis: { range: [null, 100], tickwidth: 1 },
                                bar: { color: avgCpu > 80 ? "#f44336" : avgCpu > 50 ? "#ff9800" : "#2196f3" },
                                bgcolor: "white",
                                borderwidth: 2,
                                bordercolor: "gray",
                                steps: [
                                    { range: [0, 30], color: "rgba(33, 150, 243, 0.3)" },
                                    { range: [30, 70], color: "rgba(255, 152, 0, 0.3)" },
                                    { range: [70, 100], color: "rgba(244, 67, 54, 0.3)" }
                                ],
                                threshold: {
                                    line: { color: "red", width: 4 },
                                    thickness: 0.75,
                                    value: 90
                                }
                            }
                        }
                    ];
                    
                    const gaugeLayout = { 
                        width: 350, 
                        height: 250, 
                        margin: { t: 25, r: 25, l: 25, b: 25 },
                        paper_bgcolor: "white",
                        font: { color: "darkblue", family: "Arial" }
                    };
                    
                    Plotly.newPlot('dockerCpuGauge', cpuGaugeData, gaugeLayout);
                    
                    // Create Memory Gauge
                    const avgMem = dockerInfo.containers.reduce((sum, container) => sum + (parseFloat(container.memory_percent) || 0), 0) / dockerInfo.containers.length;
                    
                    const memGaugeData = [
                        {
                            type: "indicator",
                            mode: "gauge+number",
                            value: avgMem,
                            title: { text: "Average Memory Usage (%)", font: { size: 16 } },
                            gauge: {
                                axis: { range: [null, 100], tickwidth: 1 },
                                bar: { color: avgMem > 80 ? "#f44336" : avgMem > 50 ? "#ff9800" : "#4caf50" },
                                bgcolor: "white",
                                borderwidth: 2,
                                bordercolor: "gray",
                                steps: [
                                    { range: [0, 30], color: "rgba(76, 175, 80, 0.3)" },
                                    { range: [30, 70], color: "rgba(255, 152, 0, 0.3)" },
                                    { range: [70, 100], color: "rgba(244, 67, 54, 0.3)" }
                                ],
                                threshold: {
                                    line: { color: "red", width: 4 },
                                    thickness: 0.75,
                                    value: 90
                                }
                            }
                        }
                    ];
                    
                    Plotly.newPlot('dockerMemoryGauge', memGaugeData, gaugeLayout);
                    
                    // Create CPU Bar Chart with more details
                    const cpuData = [{
                        x: dockerInfo.containers.map(c => c.name),
                        y: dockerInfo.containers.map(c => parseFloat(c.cpu_usage) || 0),
                        type: 'bar',
                        marker: {
                            color: dockerInfo.containers.map(c => {
                                const val = parseFloat(c.cpu_usage) || 0;
                                return val > 80 ? 'rgba(244, 67, 54, 0.8)' : 
                                       val > 50 ? 'rgba(255, 152, 0, 0.8)' : 
                                       'rgba(33, 150, 243, 0.8)';
                            }),
                            line: {
                                color: dockerInfo.containers.map(c => {
                                    const val = parseFloat(c.cpu_usage) || 0;
                                    return val > 80 ? 'rgb(244, 67, 54)' : 
                                           val > 50 ? 'rgb(255, 152, 0)' : 
                                           'rgb(33, 150, 243)';
                                }),
                                width: 1.5
                            }
                        },
                        hovertemplate: '<b>%{x}</b><br>CPU Usage: %{y}%<extra></extra>'
                    }];
                    
                    const cpuLayout = {
                        title: 'Container CPU Usage (%)',
                        height: 250,
                        margin: { l: 40, r: 10, t: 30, b: 80 },
                        yaxis: {
                            title: 'CPU Usage (%)',
                            range: [0, 100]
                        },
                        xaxis: {
                            tickangle: -45
                        },
                        bargap: 0.15,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };
                    
                    const cpuConfig = {
                        responsive: true,
                        displayModeBar: false
                    };
                    
                    Plotly.newPlot('dockerCpuChart', cpuData, cpuLayout, cpuConfig);
                    
                    // Create Memory Bar Chart with more details
                    const memData = [{
                        x: dockerInfo.containers.map(c => c.name),
                        y: dockerInfo.containers.map(c => parseFloat(c.memory_percent) || 0),
                        type: 'bar',
                        marker: {
                            color: dockerInfo.containers.map(c => {
                                const val = parseFloat(c.memory_percent) || 0;
                                return val > 80 ? 'rgba(244, 67, 54, 0.8)' : 
                                       val > 50 ? 'rgba(255, 152, 0, 0.8)' : 
                                       'rgba(76, 175, 80, 0.8)';
                            }),
                            line: {
                                color: dockerInfo.containers.map(c => {
                                    const val = parseFloat(c.memory_percent) || 0;
                                    return val > 80 ? 'rgb(244, 67, 54)' : 
                                           val > 50 ? 'rgb(255, 152, 0)' : 
                                           'rgb(76, 175, 80)';
                                }),
                                width: 1.5
                            }
                        },
                        hovertemplate: '<b>%{x}</b><br>Memory Usage: %{y}%<br>%{text}<extra></extra>',
                        text: dockerInfo.containers.map(c => c.memory_usage)
                    }];
                    
                    const memLayout = {
                        title: 'Container Memory Usage (%)',
                        height: 250,
                        margin: { l: 40, r: 10, t: 30, b: 80 },
                        yaxis: {
                            title: 'Memory Usage (%)',
                            range: [0, 100]
                        },
                        xaxis: {
                            tickangle: -45
                        },
                        bargap: 0.15,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };
                    
                    const memConfig = {
                        responsive: true,
                        displayModeBar: false
                    };
                    
                    Plotly.newPlot('dockerMemoryChart', memData, memLayout, memConfig);
                    
                    // Create Dummy Resource Timeline (simulated values)
                    const timePoints = [];
                    const cpuValues = [];
                    const memValues = [];
                    
                    // Create sample data for the last 24 hours
                    const now = new Date();
                    for (let i = 24; i >= 0; i--) {
                        const time = new Date(now.getTime() - i * 3600000);
                        timePoints.push(time);
                        
                        // Create realistic-looking fluctuations
                        const baseValue = 30 + Math.random() * 10;
                        const cpuVal = baseValue + Math.sin(i / 4) * 15 + Math.random() * 10;
                        const memVal = baseValue + Math.cos(i / 6) * 10 + Math.random() * 15;
                        
                        cpuValues.push(cpuVal);
                        memValues.push(memVal);
                    }
                    
                    const timelineData = [
                        {
                            x: timePoints,
                            y: cpuValues,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'CPU (%)',
                            line: {
                                color: 'rgb(33, 150, 243)',
                                width: 2
                            },
                            marker: {
                                color: 'rgb(33, 150, 243)',
                                size: 5
                            }
                        },
                        {
                            x: timePoints,
                            y: memValues,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Memory (%)',
                            line: {
                                color: 'rgb(76, 175, 80)',
                                width: 2
                            },
                            marker: {
                                color: 'rgb(76, 175, 80)',
                                size: 5
                            }
                        }
                    ];
                    
                    const timelineLayout = {
                        title: 'Resource Usage Over Time (Simulated)',
                        xaxis: {
                            title: 'Time',
                            showgrid: true,
                            zeroline: false
                        },
                        yaxis: {
                            title: 'Usage (%)',
                            range: [0, 100],
                            showgrid: true,
                            zeroline: false
                        },
                        legend: {
                            x: 0.1,
                            y: 1.1,
                            orientation: 'h'
                        },
                        margin: { l: 40, r: 10, t: 30, b: 40 },
                        hovermode: 'closest',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(248,249,250,1)'
                    };
                    
                    Plotly.newPlot('dockerResourceTimeline', timelineData, timelineLayout);
                    
                    // Create Network I/O Graph
                    // This is a placeholder visualization showing network connections
                    const nodes = [
                        { id: 'host', label: 'Host System', shape: 'dot', size: 20, color: { background: '#4CAF50', border: '#388E3C' } }
                    ];
                    
                    const edges = [];
                    
                    dockerInfo.containers.forEach((container, index) => {
                        nodes.push({
                            id: container.name,
                            label: container.name,
                            shape: 'dot',
                            size: 15,
                            color: { background: '#2196F3', border: '#1565C0' }
                        });
                        
                        edges.push({
                            from: 'host',
                            to: container.name,
                            arrows: {
                                to: { enabled: true, scaleFactor: 0.5 },
                                from: { enabled: true, scaleFactor: 0.5 }
                            },
                            color: { color: '#90CAF9' },
                            label: container.network_io.split(' / ')[0]
                        });
                    });
                    
                    // Create dummy data for network graph using Plotly
                    const networkData = [{
                        type: 'sankey',
                        orientation: 'h',
                        node: {
                            pad: 15,
                            thickness: 20,
                            line: { color: 'black', width: 0.5 },
                            label: ['Host'].concat(dockerInfo.containers.map(c => c.name)),
                            color: ['#4CAF50'].concat(Array(dockerInfo.containers.length).fill('#2196F3'))
                        },
                        link: {
                            source: Array(dockerInfo.containers.length).fill(0),
                            target: Array.from({ length: dockerInfo.containers.length }, (_, i) => i + 1),
                            value: dockerInfo.containers.map(c => 1 + Math.random() * 2),
                            label: dockerInfo.containers.map(c => c.network_io.split(' / ')[0] || 'N/A')
                        }
                    }];
                    
                    const networkLayout = {
                        title: 'Network Connections',
                        font: { size: 12 },
                        margin: { l: 0, r: 0, t: 40, b: 0 }
                    };
                    
                    Plotly.newPlot('dockerNetworkGraph', networkData, networkLayout);
                }
            } else if (dockerInfo && dockerInfo.status === 'not_found') {
                // No Docker container found
                if (!elements.dockerInfo) {
                    elements.dockerInfo = document.createElement('div');
                    elements.dockerInfo.id = 'dockerInfo';
                    elements.serverInfo.parentNode.insertBefore(elements.dockerInfo, elements.serverInfo.nextSibling);
                }
                
                elements.dockerInfo.innerHTML = `
                    <div class="alert alert-warning mt-4">
                        <i class="bi bi-exclamation-triangle"></i> No Weaviate Docker containers found.
                    </div>
                `;
            } else if (dockerInfo && dockerInfo.status === 'error') {
                // Error retrieving Docker info
                if (!elements.dockerInfo) {
                    elements.dockerInfo = document.createElement('div');
                    elements.dockerInfo.id = 'dockerInfo';
                    elements.serverInfo.parentNode.insertBefore(elements.dockerInfo, elements.serverInfo.nextSibling);
                }
                
                // Show login button if authentication is required
                if (dockerInfo.message.includes("requires permissions") || dockerInfo.message.includes("login")) {
                    elements.dockerInfo.innerHTML = `
                        <div class="alert alert-danger mt-4">
                            <i class="bi bi-x-circle"></i> Error retrieving Docker information: ${dockerInfo.message}
                            <div class="mt-2">
                                <button class="btn btn-primary btn-sm" id="showDockerLoginBtn">
                                    <i class="bi bi-key"></i> Login for Docker Access
                                </button>
                            </div>
                        </div>
                    `;
                    
                    // Add event listener to the login button
                    document.getElementById('showDockerLoginBtn').addEventListener('click', () => {
                        elements.dockerLoginModal.show();
                    });
                } else {
                    elements.dockerInfo.innerHTML = `
                        <div class="alert alert-danger mt-4">
                            <i class="bi bi-x-circle"></i> Error retrieving Docker information: ${dockerInfo.message}
                        </div>
                    `;
                }
            }
        } catch (dockerError) {
            console.error('Error fetching Docker info:', dockerError);
        }
        
        elements.serverInfo.style.display = 'block';
        
        // Fetch classes data
        await loadClasses();
        
    } catch (error) {
        console.error('Error loading dashboard:', error);
        elements.serverInfo.innerHTML = `<div class="alert alert-danger">Error loading server info: ${error.message}</div>`;
        elements.serverInfo.style.display = 'block';
        elements.classesLoading.style.display = 'none';
    }
}

// Load classes data
async function loadClasses() {
    try {
        // Fetch classes
        const classes = await fetchAPI('/classes');
        state.classes = classes;
        
        // Calculate totals
        const totalObjects = classes.reduce((sum, cls) => sum + cls.object_count, 0);
        
        // Update stats
        elements.totalClasses.textContent = classes.length;
        elements.totalObjects.textContent = totalObjects;
        
        // Render classes table
        renderClassesTable(classes);
        
    } catch (error) {
        console.error('Error loading classes:', error);
        elements.classesTableContainer.innerHTML = `<div class="alert alert-danger">Error loading classes: ${error.message}</div>`;
        elements.classesTableContainer.style.display = 'block';
    } finally {
        elements.classesLoading.style.display = 'none';
    }
}

// Render classes table
function renderClassesTable(classes) {
    if (!classes || classes.length === 0) {
        elements.classesTableBody.innerHTML = `<tr><td colspan="6" class="text-center">No classes found</td></tr>`;
        elements.classesTableContainer.style.display = 'block';
        return;
    }
    
    const rows = classes.map(cls => {
        return `
            <tr class="class-row" data-class="${cls.class_name}">
                <td><span class="badge badge-class">${cls.class_name}</span></td>
                <td><span class="badge badge-count">${cls.object_count}</span></td>
                <td>${cls.property_count}</td>
                <td>${cls.vector_type}</td>
                <td>${cls.vector_dimension}</td>
                <td>
                    <div class="btn-group">
                        <button class="btn btn-sm btn-outline-primary view-class-btn" data-class="${cls.class_name}" title="View Class">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-warning update-class-btn" data-class="${cls.class_name}" title="Edit Class">
                            <i class="bi bi-pencil"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger delete-class-btn" data-class="${cls.class_name}" title="Delete Class">
                            <i class="bi bi-trash"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-success export-class-btn" data-class="${cls.class_name}" title="Export Class">
                            <i class="bi bi-file-earmark-excel"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
    
    elements.classesTableBody.innerHTML = rows;
    elements.classesTableContainer.style.display = 'block';
    
    // Add event listeners for class actions
    document.querySelectorAll('.view-class-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const className = btn.getAttribute('data-class');
            loadClassDetail(className);
        });
    });
    
    document.querySelectorAll('.update-class-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const className = btn.getAttribute('data-class');
            showUpdateClassModal(className);
        });
    });
    
    document.querySelectorAll('.delete-class-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const className = btn.getAttribute('data-class');
            showDeleteClassModal(className);
        });
    });
    
    document.querySelectorAll('.export-class-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const className = btn.getAttribute('data-class');
            exportClass(className);
        });
    });
    
    document.querySelectorAll('.class-row').forEach(row => {
        row.addEventListener('click', () => {
            const className = row.getAttribute('data-class');
            loadClassDetail(className);
        });
    });
}

// Load class detail
async function loadClassDetail(className) {
    try {
        state.currentClass = className;
        
        // Update title
        elements.classDetailTitle.textContent = className;
        
        // Show loading
        elements.classDetailInfo.style.display = 'none';
        elements.classDetailLoading.style.display = 'flex';
        
        // Switch to class detail view
        showView('classDetail');
        
        // Fetch class details
        const classDetail = await fetchAPI(`/classes/${className}`);
        
        // Update class info
        elements.classObjectCount.textContent = classDetail.object_count;
        elements.classPropertyCount.textContent = classDetail.properties.length;
        elements.classVectorType.textContent = classDetail.vector_config.type;
        elements.classVectorDimension.textContent = classDetail.vector_config.config.dimension || 'N/A';
        
        if (classDetail.storage_estimate) {
            const sizeInMB = (classDetail.storage_estimate.estimated_size_bytes / (1024 * 1024)).toFixed(2);
            elements.classStorageEstimate.textContent = `${sizeInMB} MB`;
        } else {
            elements.classStorageEstimate.textContent = 'N/A';
        }
        
        elements.classQueryTime.textContent = (classDetail.count_query_time * 1000).toFixed(2);
        
        // Render properties
        const propertiesHtml = classDetail.properties.map(prop => {
            return `<div class="property-pill">
                <strong>${prop.name}</strong> (${prop.dataType[0]})
            </div>`;
        }).join('');
        
        elements.classProperties.innerHTML = propertiesHtml || 'No properties found';
        
        // Render sample objects
        renderSampleObjects(classDetail.sample_objects);
        
        // Setup view all objects button
        elements.viewAllObjectsBtn.onclick = () => {
            window.open(`/objects/${className}`, '_blank');
        };
        
        // Setup query form 
        elements.queryForm.onsubmit = (e) => {
            e.preventDefault();
            runQuery(className);
        };
        
        // Reset query form and results
        elements.queryText.value = '';
        elements.queryResults.style.display = 'none';
        
        // Show class info
        elements.classDetailInfo.style.display = 'block';
        
    } catch (error) {
        console.error('Error loading class detail:', error);
        elements.classDetailInfo.innerHTML = `<div class="alert alert-danger">Error loading class details: ${error.message}</div>`;
        elements.classDetailInfo.style.display = 'block';
    } finally {
        elements.classDetailLoading.style.display = 'none';
    }
}

// Render sample objects
function renderSampleObjects(objects) {
    if (!objects || objects.length === 0) {
        elements.sampleObjectsContainer.innerHTML = '<p class="text-muted">No sample objects available</p>';
        return;
    }
    
    const objectsHtml = objects.map((obj, index) => {
        // Get object ID and metadata
        const id = obj._additional?.id || 'Unknown';
        const creationTime = obj._additional?.creationTimeUnix ? 
            new Date(obj._additional.creationTimeUnix).toLocaleString() : 'Unknown';
        
        // Check if vector data is available
        const hasVector = !!obj._additional?.vector;
        const vectorLength = hasVector ? obj._additional.vector.length : 0;
        
        // Prepare properties display
        const propertiesHtml = Object.entries(obj)
            .filter(([key]) => key !== '_additional')
            .map(([key, value]) => {
                let displayValue = value;
                
                // Format display value based on type
                if (typeof value === 'object') {
                    displayValue = JSON.stringify(value).substring(0, 100) + (JSON.stringify(value).length > 100 ? '...' : '');
                } else if (typeof value === 'string' && value.length > 100) {
                    displayValue = value.substring(0, 100) + '...';
                }
                
                return `<tr>
                    <td width="30%"><strong>${key}</strong></td>
                    <td>${displayValue}</td>
                </tr>`;
            }).join('');
        
        // Add metadata section
        const metadataHtml = `
            <tr class="table-secondary">
                <td colspan="2"><strong>Metadata</strong></td>
            </tr>
            <tr>
                <td width="30%"><strong>ID</strong></td>
                <td>${id}</td>
            </tr>
            <tr>
                <td width="30%"><strong>Created</strong></td>
                <td>${creationTime}</td>
            </tr>
        `;

        // Add vector section separately
        const vectorHtml = hasVector ? `
            <tr class="table-info">
                <td colspan="2"><strong>Vector Information</strong></td>
            </tr>
            <tr>
                <td width="30%"><strong>Vector Dimensions</strong></td>
                <td>${vectorLength}</td>
            </tr>
            <tr>
                <td width="30%"><strong>Vector</strong></td>
                <td>
                    <button class="btn btn-sm btn-outline-secondary toggle-vector-btn" data-index="${index}">
                        <i class="bi bi-eye"></i> Show/Hide Vector
                    </button>
                    <div class="vector-preview" id="vector-preview-${index}" style="display: none; margin-top: 10px; max-height: 200px; overflow-y: auto;">
                        <pre class="mb-0"><code>${hasVector ? JSON.stringify(obj._additional.vector, null, 2) : 'No vector data'}</code></pre>
                    </div>
                </td>
            </tr>
        ` : '';
        
        return `
            <div class="card mb-3">
                <div class="card-header bg-light">
                    Object ${index + 1}
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <tbody>
                                ${propertiesHtml}
                                ${metadataHtml}
                                ${vectorHtml}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    elements.sampleObjectsContainer.innerHTML = objectsHtml;
    
    // Add event listeners to toggle vector display
    document.querySelectorAll('.toggle-vector-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const index = btn.getAttribute('data-index');
            const preview = document.getElementById(`vector-preview-${index}`);
            if (preview) {
                const isVisible = preview.style.display !== 'none';
                preview.style.display = isVisible ? 'none' : 'block';
                btn.innerHTML = isVisible ? 
                    '<i class="bi bi-eye"></i> Show Vector' : 
                    '<i class="bi bi-eye-slash"></i> Hide Vector';
            }
        });
    });
}

// Show delete class confirmation modal
function showDeleteClassModal(className) {
    state.classToDelete = className;
    elements.deleteClassName.textContent = className;
    elements.deleteClassModal.show();
}

// Delete class
async function deleteClass(className) {
    try {
        await fetchAPI(`/classes/${className}`, { method: 'DELETE' });
        
        // Close modal
        elements.deleteClassModal.hide();
        
        // Reload classes data
        await loadClasses();
        
        // Show success message (you could add a toast notification here)
        console.log(`Class ${className} deleted successfully`);
    } catch (error) {
        console.error('Error deleting class:', error);
        // Show error message
        alert(`Error deleting class: ${error.message}`);
    }
}

// Load inspections
async function loadInspections() {
    try {
        // Show loading
        elements.inspectionsTableContainer.style.display = 'none';
        elements.inspectionsLoading.style.display = 'flex';
        
        // Fetch inspections
        const response = await fetchAPI('/inspect');
        state.inspections = response.reports || [];
        
        // Render inspections table
        renderInspectionsTable();
        
    } catch (error) {
        console.error('Error loading inspections:', error);
        elements.inspectionsTableContainer.innerHTML = `<div class="alert alert-danger">Error loading inspections: ${error.message}</div>`;
        elements.inspectionsTableContainer.style.display = 'block';
    } finally {
        elements.inspectionsLoading.style.display = 'none';
    }
}

// Render inspections table
function renderInspectionsTable() {
    if (!state.inspections || state.inspections.length === 0) {
        elements.inspectionsTableBody.innerHTML = `<tr><td colspan="5" class="text-center">No inspection reports found</td></tr>`;
        elements.inspectionsTableContainer.style.display = 'block';
        return;
    }
    
    const rows = state.inspections.map(inspection => {
        // Format timestamp
        const timestamp = new Date(inspection.timestamp).toLocaleString();
        
        // Status badge class
        let statusBadgeClass = 'badge-secondary';
        if (inspection.status === 'completed') statusBadgeClass = 'badge-completed';
        if (inspection.status === 'running') statusBadgeClass = 'badge-running';
        if (inspection.status === 'error') statusBadgeClass = 'badge-error';
        
        return `
            <tr>
                <td>${inspection.id}</td>
                <td>${timestamp}</td>
                <td><span class="badge inspection-badge ${statusBadgeClass}">${inspection.status}</span></td>
                <td>${inspection.collections_count}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary view-inspection-btn" data-id="${inspection.id}">
                        <i class="bi bi-eye"></i> View
                    </button>
                </td>
            </tr>
        `;
    }).join('');
    
    elements.inspectionsTableBody.innerHTML = rows;
    elements.inspectionsTableContainer.style.display = 'block';
    
    // Add event listeners for inspection actions
    document.querySelectorAll('.view-inspection-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const inspectionId = btn.getAttribute('data-id');
            window.open(`/inspect/${inspectionId}`, '_blank');
        });
    });
}

// Start a new inspection
async function startInspection() {
    try {
        const requestBody = {
            include_samples: elements.includeSamplesCheck.checked,
            run_benchmarks: elements.runBenchmarksCheck.checked
        };
        
        // Close modal
        elements.newInspectionModal.hide();
        
        // Start inspection
        const response = await fetchAPI('/inspect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        // Show success message
        alert(`Inspection started with ID: ${response.report_id}`);
        
        // Switch to inspections view and reload
        showView('inspections');
    } catch (error) {
        console.error('Error starting inspection:', error);
        alert(`Error starting inspection: ${error.message}`);
    }
}

// Run a query against a class
async function runQuery(className) {
    try {
        const queryText = elements.queryText.value.trim();
        if (!queryText) {
            alert('Please enter a query');
            return;
        }
        
        const queryType = elements.queryType.value;
        const limit = parseInt(elements.queryLimit.value) || 5;
        
        // Get Gemini API key if needed
        let geminiApiKey = null;
        let geminiModel = null;
        if (queryType === 'gemini') {
            // Check if we have a stored API key
            geminiApiKey = localStorage.getItem('gemini_api_key');
            
            // If not, prompt the user for it
            if (!geminiApiKey) {
                geminiApiKey = prompt('Please enter your Gemini API key (it will be stored locally for future use):', '');
                if (!geminiApiKey) {
                    alert('Gemini API key is required for Gemini search');
                    return;
                }
                // Save the API key in local storage for future use
                localStorage.setItem('gemini_api_key', geminiApiKey);
            }
            
            // Get the selected Gemini model
            geminiModel = elements.geminiModel ? elements.geminiModel.value : 'gemini-1.5-flash';
        }
        
        // Show loading
        elements.queryResults.style.display = 'none';
        elements.queryLoading.style.display = 'flex';
        
        // Send the query
        const response = await fetchAPI('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                class_name: className,
                query_text: queryText,
                limit: limit,
                search_type: queryType,
                gemini_api_key: geminiApiKey,
                gemini_model: geminiModel
            })
        });
        
        // Display results
        elements.queryInfo.innerHTML = `
            <strong>${queryType.charAt(0).toUpperCase() + queryType.slice(1)} search</strong> for: "${queryText}"
            <br>
            Found ${response.results ? response.results.length : 0} results
            <div class="mt-2">
                <button class="btn btn-sm btn-outline-primary open-in-new-tab-btn">
                    <i class="bi bi-box-arrow-up-right"></i> Open in New Tab
                </button>
                ${queryType === 'gemini' ? `
                <button class="btn btn-sm btn-outline-danger ms-2" id="resetGeminiApiBtn">
                    <i class="bi bi-key"></i> Reset Gemini API Key
                </button>
                <span class="ms-2">Model: ${geminiModel}</span>
                ` : ''}
            </div>
        `;
        
        // Add event listener to the "Open in New Tab" button
        document.querySelector('.open-in-new-tab-btn').addEventListener('click', () => {
            const resultsUrl = `/results?query=${encodeURIComponent(queryText)}&class=${encodeURIComponent(className)}&type=${encodeURIComponent(queryType)}&limit=${limit}`;
            window.open(resultsUrl, '_blank');
        });
        
        // Add event listener to the "Reset Gemini API Key" button if it exists
        const resetGeminiApiBtn = document.getElementById('resetGeminiApiBtn');
        if (resetGeminiApiBtn) {
            resetGeminiApiBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to reset your Gemini API key?')) {
                    localStorage.removeItem('gemini_api_key');
                    alert('Gemini API key has been reset. You will be prompted for a new key on your next Gemini search.');
                }
            });
        }
        
        if (response.status === 'success' && response.results && response.results.length > 0) {
            // Display Gemini analysis if available
            if (queryType === 'gemini' && response.gemini_analysis) {
                const geminiHtml = `
                    <div class="card mb-4">
                        <div class="card-header bg-primary text-white">
                            <i class="bi bi-stars"></i> Gemini AI Analysis
                        </div>
                        <div class="card-body">
                            <div class="gemini-analysis">
                                ${marked.parse(response.gemini_analysis)}
                            </div>
                        </div>
                    </div>
                `;
                elements.queryResultsContainer.innerHTML = geminiHtml;
            }
            
            // Render results
            const resultsHtml = response.results.map((result, index) => {
                // Basic object info
                const id = result.id || result._additional?.id || 'Unknown';
                
                // Format the creation time correctly
                let creationTime = 'Unknown';
                if (result._additional?.creationTimeUnix) {
                    // Convert unix timestamp (milliseconds) to Date object
                    const timestamp = parseInt(result._additional.creationTimeUnix);
                    if (!isNaN(timestamp)) {
                        creationTime = new Date(timestamp).toLocaleString();
                    }
                }
                
                // Check if this result has search_type (for combined search)
                const searchType = result._additional?.search_type;
                const searchTypeBadge = searchType ? 
                    `<span class="badge ${searchType === 'vector' ? 'bg-primary' : 'bg-success'} ms-2">${searchType}</span>` : '';
                
                // Extract key document properties
                const text = result.text || '';
                const source = result.source || '';
                const filename = result.filename || '';
                const page = result.page !== undefined ? `Page ${result.page}` : '';
                
                // Display document source information
                const sourceInfo = filename ? 
                    `<div class="alert alert-secondary">
                        <strong>Source:</strong> ${filename} ${page ? `(${page})` : ''}
                     </div>` : '';
                
                // Display text excerpt
                const textExcerpt = text ? 
                    `<div class="card mb-3">
                        <div class="card-header">Text excerpt</div>
                        <div class="card-body">
                            <p>${text.length > 500 ? text.substring(0, 500) + '...' : text}</p>
                        </div>
                     </div>` : '';
                
                // Format properties (excluding text and source which we display separately)
                const propertiesHtml = Object.entries(result)
                    .filter(([key]) => key !== '_additional' && key !== 'text' && key !== 'source' && key !== 'filename' && key !== 'page')
                    .map(([key, value]) => {
                        let displayValue = value;
                        
                        // Format display value based on type
                        if (typeof value === 'object') {
                            displayValue = JSON.stringify(value).substring(0, 100) + (JSON.stringify(value).length > 100 ? '...' : '');
                        } else if (typeof value === 'string' && value.length > 100) {
                            displayValue = value.substring(0, 100) + '...';
                        }
                        
                        return `<tr>
                            <td width="30%"><strong>${key}</strong></td>
                            <td>${displayValue}</td>
                        </tr>`;
                    }).join('');
                
                // Format certainty if available (for vector search)
                const certaintyHtml = result._additional?.certainty ? `
                    <tr>
                        <td width="30%"><strong>Certainty</strong></td>
                        <td>${(result._additional.certainty * 100).toFixed(2)}%</td>
                    </tr>
                ` : '';
                
                // Check if vector data is available
                const hasVector = !!result._additional?.vector;
                const vectorLength = hasVector ? result._additional.vector.length : 0;
                
                // Add vector section separately
                const vectorHtml = hasVector ? `
                    <tr class="table-info">
                        <td colspan="2"><strong>Vector Information</strong></td>
                    </tr>
                    <tr>
                        <td width="30%"><strong>Vector Dimensions</strong></td>
                        <td>${vectorLength}</td>
                    </tr>
                    <tr>
                        <td width="30%"><strong>Vector</strong></td>
                        <td>
                            <button class="btn btn-sm btn-outline-secondary toggle-vector-btn" data-index="result-${index}">
                                <i class="bi bi-eye"></i> Show/Hide Vector
                            </button>
                            <div class="vector-preview" id="vector-preview-result-${index}" style="display: none; margin-top: 10px; max-height: 200px; overflow-y: auto;">
                                <pre class="mb-0"><code>${hasVector ? JSON.stringify(result._additional.vector, null, 2) : 'No vector data'}</code></pre>
                            </div>
                        </td>
                    </tr>
                ` : '';
                
                return `
                    <div class="card mb-4">
                        <div class="card-header bg-light">
                            Result ${index + 1} - ID: ${id.substring(0, 8)}... ${searchTypeBadge}
                        </div>
                        <div class="card-body">
                            ${sourceInfo}
                            ${textExcerpt}
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <tbody>
                                        ${propertiesHtml ? `
                                            <tr class="table-secondary">
                                                <td colspan="2"><strong>Properties</strong></td>
                                            </tr>
                                            ${propertiesHtml}
                                        ` : ''}
                                        <tr class="table-secondary">
                                            <td colspan="2"><strong>Metadata</strong></td>
                                        </tr>
                                        <tr>
                                            <td width="30%"><strong>Created</strong></td>
                                            <td>${creationTime}</td>
                                        </tr>
                                        ${certaintyHtml}
                                        ${vectorHtml}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Append the regular results after the Gemini analysis if it exists
            if (queryType === 'gemini' && response.gemini_analysis) {
                elements.queryResultsContainer.innerHTML += `
                    <h5 class="mt-4 mb-3">Original Vector Search Results</h5>
                    ${resultsHtml}
                `;
            } else {
                elements.queryResultsContainer.innerHTML = resultsHtml;
            }
            
            // Add event listeners to toggle vector display
            document.querySelectorAll('.toggle-vector-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const index = btn.getAttribute('data-index');
                    const preview = document.getElementById(`vector-preview-${index}`);
                    if (preview) {
                        const isVisible = preview.style.display !== 'none';
                        preview.style.display = isVisible ? 'none' : 'block';
                        btn.innerHTML = isVisible ? 
                            '<i class="bi bi-eye"></i> Show Vector' : 
                            '<i class="bi bi-eye-slash"></i> Hide Vector';
                    }
                });
            });
        } else if (response.status === 'error') {
            elements.queryResultsContainer.innerHTML = `<div class="alert alert-danger">${response.message}</div>`;
        } else {
            elements.queryResultsContainer.innerHTML = `<div class="alert alert-warning">No results found</div>`;
        }
        
        // Show results
        elements.queryResults.style.display = 'block';
    } catch (error) {
        console.error('Error running query:', error);
        elements.queryResultsContainer.innerHTML = `<div class="alert alert-danger">Error running query: ${error.message}</div>`;
        elements.queryResults.style.display = 'block';
    } finally {
        elements.queryLoading.style.display = 'none';
    }
}

// Create a new class
async function createClass() {
    try {
        elements.createClassError.style.display = 'none';
        
        // Validate class name (required)
        const className = elements.newClassName.value.trim();
        if (!className) {
            throw new Error('Class name is required');
        }
        
        // Validate class name format (should be CamelCase)
        if (!/^[A-Z][a-zA-Z0-9]*$/.test(className)) {
            throw new Error('Class name should start with uppercase letter and only contain letters and numbers (CamelCase)');
        }
        
        // Get description (optional)
        const description = elements.classDescription.value.trim();
        
        // Get vectorizer
        const vectorizer = elements.vectorizer.value;
        
        // Get vector dimension
        const vectorDimension = parseInt(elements.vectorDimension.value);
        if (isNaN(vectorDimension) || vectorDimension < 2) {
            throw new Error('Vector dimension must be at least 2');
        }
        
        // Collect properties
        const properties = [];
        const propertyRows = elements.propertiesContainer.querySelectorAll('.property-row');
        
        if (propertyRows.length === 0) {
            throw new Error('At least one property is required');
        }
        
        propertyRows.forEach(row => {
            const name = row.querySelector('.property-name').value.trim();
            const dataType = row.querySelector('.property-datatype').value;
            const indexInverted = row.querySelector('.property-index').value === 'true';
            const description = row.querySelector('.property-description').value.trim();
            
            if (!name) {
                throw new Error('All properties must have a name');
            }
            
            // Create property object
            const property = {
                name: name,
                dataType: [dataType],
                indexInverted: indexInverted
            };
            
            if (description) {
                property.description = description;
            }
            
            properties.push(property);
        });
        
        // Create class request
        const requestData = {
            class_name: className,
            vectorizer: vectorizer,
            vector_dimension: vectorDimension,
            properties: properties
        };
        
        if (description) {
            requestData.description = description;
        }
        
        // Send request to create class
        const response = await fetchAPI('/classes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        // Hide modal
        elements.newClassModal.hide();
        
        // Reload classes to show the new class
        await loadClasses();
        
        // Show success alert
        alert(`Class "${className}" successfully created`);
        
    } catch (error) {
        console.error('Error creating class:', error);
        elements.createClassError.textContent = `Error: ${error.message}`;
        elements.createClassError.style.display = 'block';
    }
}

// Add a new property row
function addPropertyRow() {
    const propertyTemplate = `
        <div class="property-row mb-3 border p-3 rounded">
            <div class="row g-2">
                <div class="col-md-4">
                    <label class="form-label">Name</label>
                    <input type="text" class="form-control property-name" placeholder="e.g., text, title, content" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Data Type</label>
                    <select class="form-select property-datatype">
                        <option value="text">text</option>
                        <option value="string">string</option>
                        <option value="int">int</option>
                        <option value="number">number</option>
                        <option value="boolean">boolean</option>
                        <option value="date">date</option>
                    </select>
                </div>
                <div class="col-md-3">
                    <label class="form-label">Index</label>
                    <select class="form-select property-index">
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                    </select>
                </div>
                <div class="col-md-1 d-flex align-items-end">
                    <button type="button" class="btn btn-sm btn-outline-danger remove-property-btn"><i class="bi bi-trash"></i></button>
                </div>
            </div>
            <div class="mt-2">
                <label class="form-label">Description (optional)</label>
                <input type="text" class="form-control property-description" placeholder="Property description">
            </div>
        </div>
    `;
    
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = propertyTemplate.trim();
    const propertyRow = tempDiv.firstChild;
    
    // Add event listener to remove button
    propertyRow.querySelector('.remove-property-btn').addEventListener('click', function() {
        propertyRow.remove();
    });
    
    elements.propertiesContainer.appendChild(propertyRow);
}

// Function to show the update class modal
function showUpdateClassModal(className) {
    state.classToUpdate = className;
    elements.updateClassName.value = className;
    elements.updateClassError.style.display = 'none';
    elements.updateClassSuccess.style.display = 'none';
    elements.updatePropContainer.innerHTML = ''; // Clear existing property fields
    
    // Fetch the current class configuration
    fetchAPI(`/classes/${className}`)
        .then(classData => {
            // Populate the form with the current values
            if (classData.vector_config) {
                // Set vectorizer
                elements.updateVectorizer.value = classData.vector_config.type || 'none';
                
                // Set vector dimension
                if (classData.vector_config.dimension) {
                    elements.updateVectorDimension.value = classData.vector_config.dimension;
                } else {
                    elements.updateVectorDimension.value = '';
                }
                
                // Set distance metric if available
                if (classData.vector_index_config && classData.vector_index_config.distance) {
                    elements.updateDistanceMetric.value = classData.vector_index_config.distance;
                } else {
                    elements.updateDistanceMetric.value = 'cosine';
                }
                
                // Set vector index config values if available
                if (classData.vector_index_config) {
                    elements.updateMaxConnections.value = classData.vector_index_config.maxConnections || '';
                    elements.updateEfConstruction.value = classData.vector_index_config.efConstruction || '';
                }
            }
            
            // Set vectorize class name if available
            if (classData.module_config && 
                classData.vector_config && 
                classData.vector_config.type !== 'none' &&
                classData.module_config[classData.vector_config.type] &&
                'vectorizeClassName' in classData.module_config[classData.vector_config.type]) {
                
                elements.updateVectorizeClassName.checked = 
                    classData.module_config[classData.vector_config.type].vectorizeClassName;
            } else {
                elements.updateVectorizeClassName.checked = false;
            }
            
            // Display current properties
            if (classData.properties && classData.properties.length > 0) {
                const propertiesDiv = document.createElement('div');
                propertiesDiv.className = 'mb-4';
                propertiesDiv.innerHTML = `
                    <h5>Current Properties</h5>
                    <table class="table table-sm table-striped">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Data Type</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${classData.properties.map(prop => `
                                <tr>
                                    <td>${prop.name}</td>
                                    <td><span class="badge badge-secondary">${Array.isArray(prop.dataType) ? prop.dataType.join(', ') : prop.dataType}</span></td>
                                    <td>${prop.description || ''}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
                elements.updatePropContainer.appendChild(propertiesDiv);
            }
            
            // Create section for adding new properties
            const newPropsDiv = document.createElement('div');
            newPropsDiv.className = 'mb-4';
            newPropsDiv.innerHTML = `
                <h5>Add New Property</h5>
                <p class="text-info small">
                    <i class="bi bi-info-circle"></i> 
                    Adding properties to existing collections has limitations. The new property will only be indexed 
                    for new objects, not existing ones.
                </p>
                <div id="newPropertiesContainer">
                    <div class="property-input mb-3 p-3 border rounded">
                        <div class="row mb-2">
                            <div class="col-md-6">
                                <label class="form-label">Property Name</label>
                                <input type="text" class="form-control property-name" placeholder="Property name">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Data Type</label>
                                <select class="form-select property-datatype">
                                    <option value="text">Text</option>
                                    <option value="int">Integer</option>
                                    <option value="number">Number</option>
                                    <option value="boolean">Boolean</option>
                                    <option value="date">Date</option>
                                    <option value="text[]">Text Array</option>
                                    <option value="int[]">Integer Array</option>
                                    <option value="number[]">Number Array</option>
                                    <option value="boolean[]">Boolean Array</option>
                                    <option value="date[]">Date Array</option>
                                </select>
                            </div>
                        </div>
                        <div class="row mb-2">
                            <div class="col-md-12">
                                <label class="form-label">Description (optional)</label>
                                <input type="text" class="form-control property-description" placeholder="Property description">
                            </div>
                        </div>
                        <div class="row mb-2">
                            <div class="col-md-6">
                                <label class="form-label">Tokenization (optional)</label>
                                <select class="form-select property-tokenization">
                                    <option value="">Default</option>
                                    <option value="word">Word</option>
                                    <option value="lowercase">Lowercase</option>
                                    <option value="whitespace">Whitespace</option>
                                    <option value="field">Field</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <div class="form-check mt-4">
                                    <input class="form-check-input property-index" type="checkbox" checked>
                                    <label class="form-check-label">
                                        Index this property
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <button type="button" class="btn btn-outline-primary btn-sm" id="updateAddPropertyBtn">
                    <i class="bi bi-plus-circle"></i> Add Another Property
                </button>
            `;
            elements.updatePropContainer.appendChild(newPropsDiv);
            
            // Add handler for "Add Another Property" button
            const addPropertyBtn = document.getElementById('updateAddPropertyBtn');
            if (addPropertyBtn) {
                // Remove any existing event listeners
                addPropertyBtn.replaceWith(addPropertyBtn.cloneNode(true));
                
                // Add fresh event listener
                document.getElementById('updateAddPropertyBtn').addEventListener('click', function() {
                    console.log('Add property button clicked');
                    addPropertyField();
                });
            } else {
                console.error('Could not find Add Another Property button');
            }
            
            // Add an inverted index config section
            const invertedIndexDiv = document.createElement('div');
            invertedIndexDiv.className = 'mb-4';
            invertedIndexDiv.innerHTML = `
                <h5>Inverted Index Configuration</h5>
                <p class="text-info small">
                    <i class="bi bi-info-circle"></i> 
                    You may need to delete and recreate the collection to change some of these settings if you have existing data.
                </p>
                <div class="row mb-3">
                    <div class="col-md-6">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="updateIndexNullState" 
                                ${classData.inverted_index_config && classData.inverted_index_config.indexNullState ? 'checked' : ''}>
                            <label class="form-check-label" for="updateIndexNullState">
                                Index Null State
                            </label>
                            <div class="form-text small">Enables filtering on null/not-null property values</div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="updateIndexPropertyLength" 
                                ${classData.inverted_index_config && classData.inverted_index_config.indexPropertyLength ? 'checked' : ''}>
                            <label class="form-check-label" for="updateIndexPropertyLength">
                                Index Property Length
                            </label>
                            <div class="form-text small">Enables filtering on property length</div>
                        </div>
                    </div>
                </div>
                <div class="row mb-3">
                    <div class="col-md-6">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="updateIndexTimestamps" 
                                ${classData.inverted_index_config && classData.inverted_index_config.indexTimestamps ? 'checked' : ''}>
                            <label class="form-check-label" for="updateIndexTimestamps">
                                Index Timestamps
                            </label>
                            <div class="form-text small">Enables filtering on creation/update times</div>
                        </div>
                    </div>
                </div>
                <div class="row mb-3">
                    <div class="col-md-6">
                        <label class="form-label">BM25 k1 Parameter</label>
                        <input type="number" class="form-control" id="updateBM25k1" step="0.1" min="0" 
                            value="${classData.inverted_index_config && classData.inverted_index_config.bm25 ? classData.inverted_index_config.bm25.k1 : 1.2}">
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">BM25 b Parameter</label>
                        <input type="number" class="form-control" id="updateBM25b" step="0.1" min="0" max="1" 
                            value="${classData.inverted_index_config && classData.inverted_index_config.bm25 ? classData.inverted_index_config.bm25.b : 0.75}">
                    </div>
                </div>
            `;
            elements.updatePropContainer.appendChild(invertedIndexDiv);
            
            // Show the modal
            elements.updateClassModal.show();
        })
        .catch(error => {
            console.error('Error fetching class data:', error);
            elements.updateClassError.innerText = 'Error loading class data: ' + error.message;
            elements.updateClassError.style.display = 'block';
        });
}

// Function to add a new property input field
function addPropertyField() {
    // Find the container for new properties
    const container = document.getElementById('newPropertiesContainer');
    
    if (!container) {
        console.error('Property container not found');
        return;
    }
    
    const newField = document.createElement('div');
    newField.className = 'property-input mb-3 p-3 border rounded';
    newField.innerHTML = `
        <div class="row mb-2">
            <div class="col-md-6">
                <label class="form-label">Property Name</label>
                <input type="text" class="form-control property-name" placeholder="Property name">
            </div>
            <div class="col-md-6">
                <label class="form-label">Data Type</label>
                <select class="form-select property-datatype">
                    <option value="text">Text</option>
                    <option value="int">Integer</option>
                    <option value="number">Number</option>
                    <option value="boolean">Boolean</option>
                    <option value="date">Date</option>
                    <option value="text[]">Text Array</option>
                    <option value="int[]">Integer Array</option>
                    <option value="number[]">Number Array</option>
                    <option value="boolean[]">Boolean Array</option>
                    <option value="date[]">Date Array</option>
                </select>
            </div>
        </div>
        <div class="row mb-2">
            <div class="col-md-12">
                <label class="form-label">Description (optional)</label>
                <input type="text" class="form-control property-description" placeholder="Property description">
            </div>
        </div>
        <div class="row">
            <div class="col-md-6">
                <label class="form-label">Tokenization (optional)</label>
                <select class="form-select property-tokenization">
                    <option value="">Default</option>
                    <option value="word">Word</option>
                    <option value="lowercase">Lowercase</option>
                    <option value="whitespace">Whitespace</option>
                    <option value="field">Field</option>
                </select>
            </div>
            <div class="col-md-6">
                <div class="form-check mt-4">
                    <input class="form-check-input property-index" type="checkbox" checked>
                    <label class="form-check-label">
                        Index this property
                    </label>
                </div>
            </div>
        </div>
        <button type="button" class="btn btn-sm btn-outline-danger mt-2 remove-property-btn">
            <i class="bi bi-trash"></i> Remove
        </button>
    `;
    
    container.appendChild(newField);
    
    // Add event listener for the remove button
    const removeBtn = newField.querySelector('.remove-property-btn');
    if (removeBtn) {
        removeBtn.addEventListener('click', function() {
            container.removeChild(newField);
        });
    }
    
    console.log('Added new property field');
}

// Function to update a class with new configuration
function updateClass() {
    const className = state.classToUpdate;
    elements.updateClassError.style.display = 'none';
    elements.updateClassSuccess.style.display = 'none';
    elements.updateClassSubmitBtn.disabled = true;
    elements.updateClassSubmitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Updating...';
    
    // Gather the base configuration from form fields
    const config = {
        vectorizer: elements.updateVectorizer.value,
        distance: elements.updateDistanceMetric.value,
        max_connections: elements.updateMaxConnections.value ? parseInt(elements.updateMaxConnections.value) : null,
        ef_construction: elements.updateEfConstruction.value ? parseInt(elements.updateEfConstruction.value) : null,
        vector_dimension: elements.updateVectorDimension.value ? parseInt(elements.updateVectorDimension.value) : null,
        vectorize_class_name: elements.updateVectorizeClassName.checked
    };
    
    // Gather new properties from the form
    const newProperties = [];
    document.querySelectorAll('.property-input').forEach(propInput => {
        const name = propInput.querySelector('.property-name').value.trim();
        if (name) {
            // Get the selected data type
            const dataType = propInput.querySelector('.property-datatype').value;
            
            const property = {
                name: name,
                // Important: Weaviate expects dataType to be an array of strings
                dataType: [dataType],
                indexInverted: propInput.querySelector('.property-index').checked
            };
            
            const description = propInput.querySelector('.property-description').value.trim();
            if (description) {
                property.description = description;
            }
            
            const tokenization = propInput.querySelector('.property-tokenization').value;
            if (tokenization) {
                property.tokenization = tokenization;
            }
            
            newProperties.push(property);
            console.log('Adding property:', property);
        }
    });
    
    // Add properties to config if we have any
    if (newProperties.length > 0) {
        config.add_properties = newProperties;
    }
    
    // Get inverted index configuration
    const invertedIndexConfig = {
        indexNullState: document.getElementById('updateIndexNullState').checked,
        indexPropertyLength: document.getElementById('updateIndexPropertyLength').checked,
        indexTimestamps: document.getElementById('updateIndexTimestamps').checked,
        bm25: {
            k1: parseFloat(document.getElementById('updateBM25k1').value),
            b: parseFloat(document.getElementById('updateBM25b').value)
        }
    };
    
    // Add inverted index config to the main config
    config.inverted_index_config = invertedIndexConfig;
    
    console.log('Updating class with config:', config);
    
    // Update the class with the new configuration
    fetchAPI(`/classes/${className}/config`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
    })
        .then(response => {
            console.log('Update response:', response);
            
            // Re-enable form and show appropriate messages
            elements.updateClassSubmitBtn.disabled = false;
            elements.updateClassSubmitBtn.innerHTML = 'Update Collection';
            
            // Process the response
            if (response.status === 'success') {
                // Show success message
                elements.updateClassSuccess.innerText = 'Collection updated successfully!';
                elements.updateClassSuccess.style.display = 'block';
                
                // Process property results if any
                if (response.property_updates) {
                    const successful = response.property_updates.filter(p => p.status === 'success').length;
                    const failed = response.property_updates.filter(p => p.status === 'error').length;
                    
                    if (successful > 0) {
                        elements.updateClassSuccess.innerText += ` Added ${successful} new properties.`;
                    }
                    
                    if (failed > 0) {
                        elements.updateClassError.innerText = `Failed to add ${failed} properties.`;
                        
                        // Show detailed error messages for each failed property
                        const failedDetails = response.property_updates
                            .filter(p => p.status === 'error')
                            .map(p => `Property "${p.name}": ${p.message || 'Unknown error'}`)
                            .join('. ');
                            
                        if (failedDetails) {
                            elements.updateClassError.innerText += ` Details: ${failedDetails}`;
                        }
                        
                        elements.updateClassError.style.display = 'block';
                    }
                }
                
                // Hide modal after successful update
                setTimeout(() => {
                    elements.updateClassModal.hide();
                    
                    // Refresh the classes list
                    loadClasses();
                    
                    // Then refresh the class detail view if we're currently on it
                    if (state.currentView === 'classDetail' && state.currentClass === className) {
                        loadClassDetail(className);
                    }
                }, 1000);
            } else {
                // Show error message for unsuccessful update
                let errorMessage = 'Update failed. ';
                
                if (response.message) {
                    errorMessage += response.message;
                }
                
                if (response.config_update === 'error' && response.config_message) {
                    errorMessage += ' ' + response.config_message;
                }
                
                // Format property errors if any
                if (response.property_updates && response.property_updates.some(p => p.status === 'error')) {
                    const propertyErrors = response.property_updates
                        .filter(p => p.status === 'error')
                        .map(p => `Property "${p.name}": ${p.message || 'Unknown error'}`)
                        .join('. ');
                        
                    if (propertyErrors) {
                        errorMessage += ' Property errors: ' + propertyErrors;
                    }
                }
                
                elements.updateClassError.innerText = errorMessage;
                elements.updateClassError.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error updating class:', error);
            elements.updateClassSubmitBtn.disabled = false;
            elements.updateClassSubmitBtn.innerHTML = 'Update Collection';
            
            // Create a more readable error message
            let errorMessage = 'Error updating collection: ';
            
            if (error.message) {
                errorMessage += error.message;
            } else if (typeof error === 'object') {
                try {
                    // Try to stringify the error object in a readable way
                    errorMessage += JSON.stringify(error, null, 2);
                } catch (e) {
                    errorMessage += 'Unknown error occurred';
                }
            } else {
                errorMessage += String(error);
            }
            
            elements.updateClassError.innerText = errorMessage;
            elements.updateClassError.style.display = 'block';
        });
}

// Export all classes to Excel
function exportAllClasses() {
    window.open('/api/classes/export', '_blank');
}

// Export a specific class (currently this just triggers the full export)
function exportClass(className) {
    window.open('/api/classes/export', '_blank');
}

// Login function for Docker access
async function loginForDockerAccess() {
    try {
        // Get form values
        const username = elements.dockerUsername.value.trim();
        const password = elements.dockerPassword.value.trim();
        
        if (!username || !password) {
            elements.loginError.textContent = 'Username and password are required';
            elements.loginError.style.display = 'block';
            return;
        }
        
        // Send login request
        const response = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                password: password
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Login failed');
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Update authentication state
            state.isAuthenticated = true;
            
            // Hide modal and error
            elements.loginError.style.display = 'none';
            elements.dockerLoginModal.hide();
            
            // Reload dashboard to update Docker info
            await loadDashboard();
        } else {
            // Show error message
            elements.loginError.textContent = result.message || 'Login failed';
            elements.loginError.style.display = 'block';
        }
    } catch (error) {
        console.error('Error during login:', error);
        elements.loginError.textContent = error.message || 'Login error occurred';
        elements.loginError.style.display = 'block';
    }
}

// Initialize the dashboard
function initDashboard() {
    // Navigation listeners
    elements.dashboardLink.addEventListener('click', (e) => {
        e.preventDefault();
        showView('dashboard');
    });
    
    elements.classesLink.addEventListener('click', (e) => {
        e.preventDefault();
        showView('dashboard');
    });
    
    elements.inspectionsLink.addEventListener('click', (e) => {
        e.preventDefault();
        showView('inspections');
    });
    
    // Back button from class detail
    elements.backToClassesBtn.addEventListener('click', () => {
        showView('dashboard');
    });
    
    // Refresh button
    elements.refreshBtn.addEventListener('click', () => {
        loadDashboard();
    });
    
    // Inspect button
    elements.inspectBtn.addEventListener('click', () => {
        elements.newInspectionModal.show();
    });

    // Create Class button
    elements.createClassBtn.addEventListener('click', () => {
        // Reset the form
        elements.newClassName.value = '';
        elements.classDescription.value = '';
        elements.vectorizer.value = 'none';
        elements.vectorDimension.value = '768';
        
        // Clear properties container except for the first property
        elements.propertiesContainer.innerHTML = '';
        addPropertyRow();
        
        // Clear errors
        elements.createClassError.style.display = 'none';
        
        // Show modal
        elements.newClassModal.show();
    });
    
    // Export Classes button
    if (elements.exportClassesBtn) {
        elements.exportClassesBtn.addEventListener('click', () => {
            exportAllClasses();
        });
    }
    
    // Add Property button
    elements.addPropertyBtn.addEventListener('click', addPropertyRow);
    
    // Create Class Submit button
    elements.createClassSubmitBtn.addEventListener('click', createClass);
    
    // New inspection button
    elements.newInspectionBtn.addEventListener('click', () => {
        elements.newInspectionModal.show();
    });
    
    // Start inspection button
    elements.startInspectionBtn.addEventListener('click', () => {
        startInspection();
    });
    
    // Delete class confirmation
    elements.confirmDeleteBtn.addEventListener('click', () => {
        if (state.classToDelete) {
            deleteClass(state.classToDelete);
        }
    });
    
    // Initialize first property row remove button
    const firstPropertyRow = elements.propertiesContainer.querySelector('.property-row');
    if (firstPropertyRow) {
        firstPropertyRow.querySelector('.remove-property-btn').addEventListener('click', function() {
            if (elements.propertiesContainer.querySelectorAll('.property-row').length > 1) {
                firstPropertyRow.remove();
            } else {
                alert('At least one property is required');
            }
        });
    }
    
    // Update class submit button - properly set up the event listener
    if (elements.updateClassSubmitBtn) {
        elements.updateClassSubmitBtn.addEventListener('click', updateClass);
        console.log('Added event listener to update class button');
    } else {
        console.error('Update class submit button not found');
    }
    
    // Docker login submit button
    if (elements.submitLoginBtn) {
        elements.submitLoginBtn.addEventListener('click', loginForDockerAccess);
    }
    
    // Initial view
    showView('dashboard');
    
    // Add event listener for the query type change to show/hide Gemini model selection
    if (elements.queryType) {
        elements.queryType.addEventListener('change', function() {
            const queryType = this.value;
            const geminiModelContainer = document.getElementById('geminiModelContainer');
            
            if (queryType === 'gemini') {
                // Show the Gemini model selection if it exists
                if (geminiModelContainer) {
                    geminiModelContainer.style.display = 'block';
                } else {
                    // Create the Gemini model selection if it doesn't exist
                    const container = document.createElement('div');
                    container.id = 'geminiModelContainer';
                    container.className = 'row mb-3';
                    container.innerHTML = `
                        <div class="col-md-4">
                            <div class="input-group">
                                <span class="input-group-text">Gemini Model</span>
                                <select class="form-select" id="geminiModel">
                                    <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
                                    <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                                    <option value="gemini-1.0-pro">Gemini 1.0 Pro</option>
                                    <option value="gemini-pro">Gemini Pro</option>
                                    <option value="gemini-pro-vision">Gemini Pro Vision</option>
                                </select>
                            </div>
                        </div>
                    `;
                    
                    // Insert it after the limit input
                    const limitRow = document.querySelector('#queryForm .row.mb-3');
                    limitRow.parentNode.insertBefore(container, limitRow.nextSibling);
                    
                    // Add it to the elements object
                    elements.geminiModel = document.getElementById('geminiModel');
                }
            } else if (geminiModelContainer) {
                // Hide the Gemini model selection
                geminiModelContainer.style.display = 'none';
            }
        });
    }
}

// Start the dashboard when page is loaded
document.addEventListener('DOMContentLoaded', initDashboard); 