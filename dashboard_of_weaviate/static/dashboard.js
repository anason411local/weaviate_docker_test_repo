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
    
    totalClasses: document.getElementById('totalClasses'),
    totalObjects: document.getElementById('totalObjects'),
    avgQueryTime: document.getElementById('avgQueryTime'),
    
    classesTableBody: document.getElementById('classesTableBody'),
    classesLoading: document.getElementById('classesLoading'),
    classesTableContainer: document.getElementById('classesTableContainer'),
    refreshBtn: document.getElementById('refreshBtn'),
    inspectBtn: document.getElementById('inspectBtn'),
    
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
    startInspectionBtn: document.getElementById('startInspectionBtn')
};

// Current state
let state = {
    currentView: 'dashboard',
    currentClass: null,
    classes: [],
    inspections: [],
    meta: {},
    classToDelete: null
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
        elements.serverInfo.style.display = 'none';
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
        const avgQueryTime = 0; // We'd need to run benchmarks to get this
        
        // Update stats
        elements.totalClasses.textContent = classes.length;
        elements.totalObjects.textContent = totalObjects;
        elements.avgQueryTime.textContent = avgQueryTime || 'N/A';
        
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
                    <button class="btn btn-sm btn-outline-primary view-class-btn" data-class="${cls.class_name}">
                        <i class="bi bi-eye"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger delete-class-btn" data-class="${cls.class_name}">
                        <i class="bi bi-trash"></i>
                    </button>
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
    
    document.querySelectorAll('.delete-class-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const className = btn.getAttribute('data-class');
            showDeleteClassModal(className);
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
        // Get object ID
        const id = obj._additional?.id || 'Unknown';
        
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
        
        return `
            <div class="card mb-3">
                <div class="card-header bg-light">
                    Object ${index + 1} - ID: ${id}
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <tbody>
                                ${propertiesHtml}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    elements.sampleObjectsContainer.innerHTML = objectsHtml;
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
    
    // Initial view
    showView('dashboard');
}

// Start the dashboard when page is loaded
document.addEventListener('DOMContentLoaded', initDashboard); 