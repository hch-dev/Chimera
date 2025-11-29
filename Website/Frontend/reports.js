// Reports Dashboard JavaScript

// Mock data for demonstration
const mockScans = [
    {
        id: 1,
        date: new Date('2025-01-15T14:30:00'),
        type: 'URL',
        target: 'https://example.com',
        riskLevel: 'Low',
        riskClass: 'safe',
        score: 85,
        details: 'SSL valid, domain age 5+ years'
    },
    {
        id: 2,
        date: new Date('2025-01-15T13:45:00'),
        type: 'Email',
        target: 'suspicious@fake.com',
        riskLevel: 'High',
        riskClass: 'danger',
        score: 25,
        details: 'Spoofed domain, urgent language detected'
    },
    {
        id: 3,
        date: new Date('2025-01-15T12:20:00'),
        type: 'URL',
        target: 'https://suspicious-site123.com',
        riskLevel: 'Medium',
        riskClass: 'warning',
        score: 60,
        details: 'Numeric domain, self-signed SSL'
    },
    {
        id: 4,
        date: new Date('2025-01-15T11:15:00'),
        type: 'Email',
        target: 'noreply@legitimate-company.com',
        riskLevel: 'Low',
        riskClass: 'safe',
        score: 90,
        details: 'Valid headers, no suspicious patterns'
    },
    {
        id: 5,
        date: new Date('2025-01-15T10:30:00'),
        type: 'URL',
        target: 'https://phishing-bank-login.com',
        riskLevel: 'High',
        riskClass: 'danger',
        score: 15,
        details: 'Fake banking site, suspicious redirects'
    },
    {
        id: 6,
        date: new Date('2025-01-15T09:45:00'),
        type: 'Email',
        target: 'winner@lottery-scam.com',
        riskLevel: 'High',
        riskClass: 'danger',
        score: 20,
        details: 'Prize scam, malicious attachments'
    },
    {
        id: 7,
        date: new Date('2025-01-15T08:20:00'),
        type: 'URL',
        target: 'https://microsoft.com',
        riskLevel: 'Low',
        riskClass: 'safe',
        score: 95,
        details: 'Legitimate site, valid certificate'
    },
    {
        id: 8,
        date: new Date('2025-01-15T07:30:00'),
        type: 'Email',
        target: 'support@tech-company.com',
        riskLevel: 'Medium',
        riskClass: 'warning',
        score: 55,
        details: 'Urgent language, suspicious links'
    },
    {
        id: 9,
        date: new Date('2025-01-14T18:45:00'),
        type: 'URL',
        target: 'https://google.com',
        riskLevel: 'Low',
        riskClass: 'safe',
        score: 98,
        details: 'Trusted domain, perfect security'
    },
    {
        id: 10,
        date: new Date('2025-01-14T17:20:00'),
        type: 'Email',
        target: 'admin@company-internal.com',
        riskLevel: 'Low',
        riskClass: 'safe',
        score: 88,
        details: 'Internal email, verified sender'
    },
    {
        id: 11,
        date: new Date('2025-01-14T16:15:00'),
        type: 'URL',
        target: 'https://fake-paypal-login.com',
        riskLevel: 'High',
        riskClass: 'danger',
        score: 10,
        details: 'Paypal spoof, malicious intent'
    },
    {
        id: 12,
        date: new Date('2025-01-14T15:30:00'),
        type: 'Email',
        target: 'newsletter@legit-retailer.com',
        riskLevel: 'Low',
        riskClass: 'safe',
        score: 92,
        details: 'Marketing email, safe content'
    }
];

// Pagination variables
let currentPage = 1;
const itemsPerPage = 10;
let filteredScans = [...mockScans];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    // Load data from localStorage or use mock data
    loadDashboardData();
    
    // Initialize charts
    initializeCharts();
    
    // Setup event listeners
    setupEventListeners();
    
    // Render table
    renderTable();
    
    // Update statistics
    updateStatistics();
});

// Load dashboard data
function loadDashboardData() {
    // Try to load saved scans from localStorage
    const savedScans = localStorage.getItem('chimera_scans');
    if (savedScans) {
        try {
            const parsed = JSON.parse(savedScans);
            if (Array.isArray(parsed) && parsed.length > 0) {
                // Convert date strings back to Date objects
                parsed.forEach(scan => {
                    scan.date = new Date(scan.date);
                });
                mockScans.push(...parsed);
            }
        } catch (e) {
            console.error('Error loading saved scans:', e);
        }
    }
    
    filteredScans = [...mockScans];
}

// Initialize charts
function initializeCharts() {
    // Simple chart implementation using CSS bars
    createTrendChart();
    createDistributionChart();
}

// Create trend chart
function createTrendChart() {
    const canvas = document.getElementById('trendChart');
    const ctx = canvas.getContext('2d');
    
    // Generate mock trend data
    const days = 30;
    const data = generateTrendData(days);
    
    // Clear canvas
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Draw simple line chart
    drawLineChart(ctx, data, canvas.width, canvas.height);
}

// Generate trend data
function generateTrendData(days) {
    const data = [];
    const now = new Date();
    
    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        
        // Generate random scan counts with some pattern
        const baseCount = 5 + Math.random() * 10;
        const weekendBoost = (date.getDay() === 0 || date.getDay() === 6) ? 1.5 : 1;
        const scanCount = Math.floor(baseCount * weekendBoost);
        
        data.push({
            date: date,
            scans: scanCount,
            threats: Math.floor(scanCount * (0.1 + Math.random() * 0.3))
        });
    }
    
    return data;
}

// Draw line chart
function drawLineChart(ctx, data, width, height) {
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;
    
    // Find max value for scaling
    const maxScans = Math.max(...data.map(d => d.scans));
    
    // Draw axes
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-color');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
    // Draw data points and lines
    const xStep = chartWidth / (data.length - 1);
    
    // Draw scans line
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--primary-color');
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((point, index) => {
        const x = padding + index * xStep;
        const y = height - padding - (point.scans / maxScans) * chartHeight;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
    
    // Draw data points
    data.forEach((point, index) => {
        const x = padding + index * xStep;
        const y = height - padding - (point.scans / maxScans) * chartHeight;
        
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--primary-color');
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    });
}

// Create distribution chart
function createDistributionChart() {
    const canvas = document.getElementById('distributionChart');
    const ctx = canvas.getContext('2d');
    
    // Calculate distribution
    const distribution = calculateRiskDistribution();
    
    // Clear canvas
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Draw pie chart
    drawPieChart(ctx, distribution, canvas.width, canvas.height);
}

// Calculate risk distribution
function calculateRiskDistribution() {
    const distribution = {
        safe: 0,
        warning: 0,
        danger: 0
    };
    
    mockScans.forEach(scan => {
        distribution[scan.riskClass]++;
    });
    
    return distribution;
}

// Draw pie chart
function drawPieChart(ctx, data, width, height) {
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 20;
    
    const total = data.safe + data.warning + data.danger;
    const colors = {
        safe: getComputedStyle(document.documentElement).getPropertyValue('--success-color'),
        warning: getComputedStyle(document.documentElement).getPropertyValue('--warning-color'),
        danger: getComputedStyle(document.documentElement).getPropertyValue('--danger-color')
    };
    
    let currentAngle = -Math.PI / 2;
    
    Object.entries(data).forEach(([key, value]) => {
        if (value === 0) return;
        
        const sliceAngle = (value / total) * Math.PI * 2;
        
        // Draw slice
        ctx.fillStyle = colors[key];
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
        ctx.closePath();
        ctx.fill();
        
        // Draw percentage text
        const textAngle = currentAngle + sliceAngle / 2;
        const textX = centerX + Math.cos(textAngle) * (radius * 0.7);
        const textY = centerY + Math.sin(textAngle) * (radius * 0.7);
        
        ctx.fillStyle = 'white';
        ctx.font = 'bold 14px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(Math.round((value / total) * 100) + '%', textX, textY);
        
        currentAngle += sliceAngle;
    });
}

// Setup event listeners
function setupEventListeners() {
    // Search functionality
    document.getElementById('searchScans').addEventListener('input', (e) => {
        filterScans(e.target.value);
    });
    
    // Period selector
    document.getElementById('trendPeriod').addEventListener('change', () => {
        createTrendChart();
    });
}

// Filter scans
function filterScans(searchTerm) {
    const term = searchTerm.toLowerCase();
    
    if (term === '') {
        filteredScans = [...mockScans];
    } else {
        filteredScans = mockScans.filter(scan => 
            scan.target.toLowerCase().includes(term) ||
            scan.type.toLowerCase().includes(term) ||
            scan.riskLevel.toLowerCase().includes(term)
        );
    }
    
    currentPage = 1;
    renderTable();
    updatePagination();
}

// Render table
function renderTable() {
    const tbody = document.getElementById('scansTableBody');
    const start = (currentPage - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const pageData = filteredScans.slice(start, end);
    
    tbody.innerHTML = pageData.map(scan => `
        <tr>
            <td>${formatDate(scan.date)}</td>
            <td>
                <span class="type-badge type-${scan.type.toLowerCase()}">
                    <i class="fas fa-${scan.type === 'URL' ? 'link' : 'envelope'}"></i>
                    ${scan.type}
                </span>
            </td>
            <td>
                <div class="target-cell">
                    <span class="target-text">${truncateText(scan.target, 30)}</span>
                    <button class="copy-btn" onclick="copyToClipboard('${scan.target}')" title="Copy">
                        <i class="fas fa-copy"></i>
                    </button>
                </div>
            </td>
            <td>
                <span class="risk-badge ${scan.riskClass}">
                    ${scan.riskLevel}
                </span>
            </td>
            <td>
                <div class="score-cell">
                    <div class="score-bar">
                        <div class="score-fill ${scan.riskClass}" style="width: ${scan.score}%"></div>
                    </div>
                    <span class="score-text">${scan.score}</span>
                </div>
            </td>
            <td>
                <div class="action-buttons">
                    <button class="action-btn" onclick="viewDetails(${scan.id})" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="action-btn" onclick="reportScan(${scan.id})" title="Report">
                        <i class="fas fa-flag"></i>
                    </button>
                    <button class="action-btn" onclick="deleteScan(${scan.id})" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
    
    updatePagination();
}

// Update pagination
function updatePagination() {
    const totalPages = Math.ceil(filteredScans.length / itemsPerPage);
    const start = (currentPage - 1) * itemsPerPage + 1;
    const end = Math.min(currentPage * itemsPerPage, filteredScans.length);
    
    // Update pagination info
    document.getElementById('showingStart').textContent = filteredScans.length > 0 ? start : 0;
    document.getElementById('showingEnd').textContent = end;
    document.getElementById('totalRecords').textContent = filteredScans.length;
    
    // Update button states
    document.getElementById('prevBtn').disabled = currentPage === 1;
    document.getElementById('nextBtn').disabled = currentPage === totalPages;
    
    // Update page numbers
    const pageNumbers = document.getElementById('pageNumbers');
    pageNumbers.innerHTML = '';
    
    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || (i >= currentPage - 1 && i <= currentPage + 1)) {
            const pageBtn = document.createElement('button');
            pageBtn.className = `page-number ${i === currentPage ? 'active' : ''}`;
            pageBtn.textContent = i;
            pageBtn.onclick = () => goToPage(i);
            pageNumbers.appendChild(pageBtn);
        } else if (i === currentPage - 2 || i === currentPage + 2) {
            const dots = document.createElement('span');
            dots.className = 'page-dots';
            dots.textContent = '...';
            pageNumbers.appendChild(dots);
        }
    }
}

// Navigation functions
function previousPage() {
    if (currentPage > 1) {
        currentPage--;
        renderTable();
    }
}

function nextPage() {
    const totalPages = Math.ceil(filteredScans.length / itemsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        renderTable();
    }
}

function goToPage(page) {
    currentPage = page;
    renderTable();
}

// Update statistics
function updateStatistics() {
    // Calculate stats
    const totalScans = mockScans.length;
    const threatsDetected = mockScans.filter(s => s.riskClass === 'danger').length;
    const threatRate = totalScans > 0 ? Math.round((threatsDetected / totalScans) * 100) : 0;
    
    // Update overview stats
    animateNumber('totalScans', totalScans);
    animateNumber('threatsDetected', threatsDetected);
    document.getElementById('threatRate').textContent = threatRate + '%';
    
    // Update last scan
    if (mockScans.length > 0) {
        const lastScan = mockScans[mockScans.length - 1];
        const now = new Date();
        const diffMs = now - lastScan.date;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        
        if (diffHours < 1) {
            document.getElementById('lastScan').textContent = 'Just now';
        } else if (diffHours < 24) {
            document.getElementById('lastScan').textContent = `${diffHours}h ago`;
        } else {
            const diffDays = Math.floor(diffHours / 24);
            document.getElementById('lastScan').textContent = `${diffDays}d ago`;
        }
    }
    
    // Update detailed stats
    const urlScans = mockScans.filter(s => s.type === 'URL').length;
    const emailChecks = mockScans.filter(s => s.type === 'Email').length;
    const fileAnalysis = mockScans.filter(s => s.type === 'File').length;
    
    animateNumber('urlScans', urlScans);
    animateNumber('emailChecks', emailChecks);
    animateNumber('fileAnalysis', fileAnalysis);
}

// Animate number counting
function animateNumber(elementId, target) {
    const element = document.getElementById(elementId);
    const duration = 1000;
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current);
    }, 16);
}

// Utility functions
function formatDate(date) {
    return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Action functions
function viewDetails(scanId) {
    const scan = mockScans.find(s => s.id === scanId);
    if (scan) {
        showNotification(`Details for ${scan.target}: ${scan.details}`, 'info');
    }
}

function reportScan(scanId) {
    const scan = mockScans.find(s => s.id === scanId);
    if (scan) {
        sessionStorage.setItem('reportUrl', scan.target);
        window.location.href = 'report.html';
    }
}

function deleteScan(scanId) {
    if (confirm('Are you sure you want to delete this scan record?')) {
        const index = mockScans.findIndex(s => s.id === scanId);
        if (index > -1) {
            mockScans.splice(index, 1);
            filteredScans = [...mockScans];
            renderTable();
            updateStatistics();
            showNotification('Scan record deleted successfully', 'success');
        }
    }
}

// Export data
function exportData() {
    const data = {
        exportDate: new Date().toISOString(),
        totalScans: mockScans.length,
        scans: mockScans.map(scan => ({
            date: scan.date.toISOString(),
            type: scan.type,
            target: scan.target,
            riskLevel: scan.riskLevel,
            score: scan.score,
            details: scan.details
        }))
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `chimera-reports-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    showNotification('Reports data exported successfully', 'success');
}

// Add dashboard specific CSS
const dashboardCSS = `
.dashboard-main {
    padding: 2rem 0 4rem;
    min-height: calc(100vh - 200px);
}

.dashboard-header {
    text-align: center;
    margin-bottom: 3rem;
}

.dashboard-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
        color: var(--text-primary);
}

.dashboard-header p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
}

.overview-stats {
    margin-bottom: 3rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
}

.stat-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2rem;
    border: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.stat-icon {
    width: 60px;
    height: 60px;
    background: var(--gradient-primary);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-size: 1.5rem;
    flex-shrink: 0;
}

.stat-content {
    flex: 1;
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.stat-label {
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.stat-change {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    font-weight: 500;
}

.stat-change.positive {
    color: var(--success-color);
}

.stat-change.negative {
    color: var(--danger-color);
}

.stat-change.neutral {
    color: var(--text-secondary);
}

.charts-section {
    margin-bottom: 3rem;
}

.charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 2rem;
}

.chart-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2rem;
    border: 1px solid var(--border-color);
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
}

.chart-header h3 {
    font-size: 1.25rem;
    color: var(--text-primary);
}

.period-select {
    padding: 0.5rem 1rem;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.9rem;
}

.chart-legend {
    display: flex;
    gap: 1rem;
    font-size: 0.85rem;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.legend-color {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.legend-color.safe {
    background: var(--success-color);
}

.legend-color.warning {
    background: var(--warning-color);
}

.legend-color.danger {
    background: var(--danger-color);
}

.chart-container {
    position: relative;
    height: 200px;
}

.chart-container canvas {
    width: 100%;
    height: 100%;
}

.recent-scans {
    margin-bottom: 3rem;
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
}

.section-header h2 {
    font-size: 1.75rem;
    color: var(--text-primary);
}

.table-controls {
    display: flex;
    gap: 1rem;
    align-items: center;
}

.search-input {
    padding: 0.75rem 1rem;
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.9rem;
    min-width: 250px;
}

.search-input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
}

.table-container {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    border: 1px solid var(--border-color);
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.scans-table {
    width: 100%;
    border-collapse: collapse;
}

.scans-table th {
    background: var(--bg-tertiary);
    padding: 1rem;
    text-align: left;
    font-weight: 600;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-color);
}

.scans-table td {
    padding: 1rem;
    border-bottom: 1px solid var(--border-color);
}

.scans-table tr:hover {
    background: var(--bg-tertiary);
}

.type-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

.type-badge.type-url {
    background: rgba(0, 102, 204, 0.1);
    color: var(--primary-color);
}

.type-badge.type-email {
    background: rgba(0, 200, 150, 0.1);
    color: var(--secondary-color);
}

.target-cell {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.target-text {
    font-family: monospace;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.copy-btn {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0.25rem;
    border-radius: 4px;
    transition: var(--transition);
}

.copy-btn:hover {
    color: var(--primary-color);
    background: rgba(0, 102, 204, 0.1);
}

.risk-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

.risk-badge.safe {
    background: rgba(40, 167, 69, 0.1);
    color: var(--success-color);
}

.risk-badge.warning {
    background: rgba(255, 193, 7, 0.1);
    color: var(--warning-color);
}

.risk-badge.danger {
    background: rgba(220, 53, 69, 0.1);
    color: var(--danger-color);
}

.score-cell {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.score-bar {
    width: 60px;
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
}

.score-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}

.score-fill.safe {
    background: var(--success-color);
}

.score-fill.warning {
    background: var(--warning-color);
}

.score-fill.danger {
    background: var(--danger-color);
}

.score-text {
    font-weight: 600;
    color: var(--text-primary);
    min-width: 30px;
}

.action-buttons {
    display: flex;
    gap: 0.5rem;
}

.action-btn {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0.5rem;
    border-radius: 4px;
    transition: var(--transition);
}

.action-btn:hover {
    color: var(--primary-color);
    background: rgba(0, 102, 204, 0.1);
}

.table-pagination {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
}

.pagination-info {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.pagination-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.page-numbers {
    display: flex;
    gap: 0.5rem;
}

.page-number {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    transition: var(--transition);
    font-size: 0.9rem;
}

.page-number:hover {
    background: var(--bg-tertiary);
}

.page-number.active {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.page-dots {
    padding: 0.5rem 0.75rem;
    color: var(--text-muted);
}

.stats-summary {
    margin-bottom: 3rem;
}

.stats-summary h2 {
    font-size: 1.75rem;
    margin-bottom: 2rem;
    color: var(--text-primary);
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.summary-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2rem;
    border: 1px solid var(--border-color);
}

.summary-card h3 {
    font-size: 1.25rem;
    margin-bottom: 1.5rem;
    color: var(--text-primary);
}

.summary-content {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.summary-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: var(--bg-tertiary);
    border-radius: 6px;
}

.summary-label {
    color: var(--text-secondary);
    font-weight: 500;
}

.summary-value {
    font-weight: 700;
    color: var(--primary-color);
}

.threat-item {
    margin-bottom: 1rem;
}

.threat-name {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: var(--text-primary);
}

.threat-bar {
    width: 100%;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.threat-fill {
    height: 100%;
    background: var(--gradient-secondary);
    border-radius: 4px;
    transition: width 0.3s ease;
}

.threat-count {
    position: absolute;
    right: 0.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
}

.time-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: var(--bg-tertiary);
    border-radius: 6px;
}

.time-label {
    color: var(--text-secondary);
    font-weight: 500;
}

.time-value {
    font-weight: 600;
    color: var(--text-primary);
}

@media (max-width: 768px) {
    .dashboard-header h1 {
        font-size: 2rem;
    }
    
    .stats-grid {
        grid-template-columns: 1fr;
    }
    
    .charts-grid {
        grid-template-columns: 1fr;
    }
    
    .section-header {
        flex-direction: column;
        gap: 1rem;
        align-items: flex-start;
    }
    
    .table-controls {
        flex-direction: column;
        width: 100%;
    }
    
    .search-input {
        width: 100%;
        min-width: auto;
    }
    
    .table-container {
        overflow-x: auto;
    }
    
    .scans-table {
        min-width: 600px;
    }
    
    .table-pagination {
        flex-direction: column;
        gap: 1rem;
        align-items: center;
    }
    
    .summary-grid {
        grid-template-columns: 1fr;
    }
    
    .stat-card {
        flex-direction: column;
        text-align: center;
    }
}
`;

// Inject dashboard CSS
const dashboardStyleSheet = document.createElement('style');
dashboardStyleSheet.textContent = dashboardCSS;
document.head.appendChild(dashboardStyleSheet);
