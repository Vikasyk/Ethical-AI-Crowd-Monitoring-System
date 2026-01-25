document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const routeSelect = document.getElementById('routeSelect');
    const originVal = document.getElementById('originVal');
    const destVal = document.getElementById('destVal');
    const densitySlider = document.getElementById('densitySlider');
    const densityVal = document.getElementById('densityVal');
    const toggleBtn = document.getElementById('toggleBtn');
    const videoFeed = document.getElementById('videoFeed');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const passengerCount = document.getElementById('passengerCount');
    const densityMetric = document.getElementById('densityMetric');
    const alertCard = document.getElementById('alertCard');
    const alertMessage = document.getElementById('alertMessage');
    const alertIcon = document.getElementById('alertIcon');
    const systemStatus = document.getElementById('systemStatus');
    const graphModal = document.getElementById('graphModal');

    // New Elements for Source
    const videoSourceRadios = document.getElementsByName('videoSource');
    const uploadContainer = document.getElementById('uploadContainer');
    const videoUpload = document.getElementById('videoUpload');
    const uploadStatus = document.getElementById('uploadStatus');

    let densityChart = null; // Chart instance

    let routesData = [];
    let isMonitoring = false;
    let pollInterval = null;
    let chartInterval = null;

    // State for source
    let currentSource = 'upload'; // default
    let uploadedFilename = null;

    // 1. Fetch Routes
    fetch('/api/routes')
        .then(res => res.json())
        .then(data => {
            routesData = data;
            routeSelect.innerHTML = '';
            if (data.length === 0) {
                const opt = document.createElement('option');
                opt.text = "No routes found (Manual)";
                opt.value = "500-D";
                routeSelect.add(opt);
            } else {
                data.forEach(r => {
                    const opt = document.createElement('option');
                    opt.value = r.route_no;
                    opt.text = `${r.route_no} (${r.origin} -> ${r.destination})`;
                    routeSelect.add(opt);
                });
                updateRouteInfo(data[0].route_no);
            }
        });

    // 2. Event Listeners
    routeSelect.addEventListener('change', (e) => updateRouteInfo(e.target.value));

    densitySlider.addEventListener('input', (e) => {
        densityVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    toggleBtn.addEventListener('click', () => {
        if (!isMonitoring) {
            startMonitoring();
        } else {
            stopMonitoring();
        }
    });

    // Source Selection Logic
    videoSourceRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            currentSource = e.target.value;
            if (currentSource === 'upload') {
                uploadContainer.style.display = 'block';
            } else {
                uploadContainer.style.display = 'none';
            }
        });
    });

    // File Upload Logic
    videoUpload.addEventListener('change', () => {
        const file = videoUpload.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        uploadStatus.textContent = "Uploading...";

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    uploadedFilename = data.filename;
                    uploadStatus.textContent = "Upload Complete";
                    uploadStatus.style.color = "#4ade80"; // success green
                } else {
                    uploadStatus.textContent = "Upload Failed";
                    uploadStatus.style.color = "#ef4444"; // error red
                }
            })
            .catch(err => {
                console.error(err);
                uploadStatus.textContent = "Error uploading";
            });
    });


    // --- MODAL FUNCTIONS ---
    window.showGraphModal = function () {
        graphModal.style.display = 'flex';
        initChart();
        chartInterval = setInterval(updateChartData, 1000);
    };

    window.closeGraphModal = function () {
        graphModal.style.display = 'none';
        if (chartInterval) clearInterval(chartInterval);
    };

    function initChart() {
        if (densityChart) return;
        const ctx = document.getElementById('densityChart').getContext('2d');
        densityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Crowd Density (P/m²)',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 2
                },
                {
                    label: 'Safety Threshold',
                    data: [],
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 6,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                }
            }
        });
    }

    function updateChartData() {
        if (!densityChart) return;
        fetch('/api/history')
            .then(res => res.json())
            .then(data => {
                const labels = data.map(d => d.time);
                const densities = data.map(d => d.density);
                const thresholds = data.map(d => d.threshold);

                densityChart.data.labels = labels;
                densityChart.data.datasets[0].data = densities;
                densityChart.data.datasets[1].data = thresholds;
                densityChart.update('none');
            });
    }


    function updateRouteInfo(routeId) {
        const route = routesData.find(r => r.route_no === routeId);
        if (route) {
            originVal.textContent = route.origin;
            destVal.textContent = route.destination;
        } else {
            // Fallback
            originVal.textContent = "Source";
            destVal.textContent = "Destination";
        }
    }

    function startMonitoring() {
        // Validation for Upload
        if (currentSource === 'upload' && !uploadedFilename) {
            alert("Please upload a video first or select Webcam.");
            return;
        }

        fetch('/api/start', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                isMonitoring = true;

                // Update UI to running state
                toggleBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Monitoring';
                toggleBtn.classList.remove('btn-primary');
                toggleBtn.classList.add('btn-danger');

                systemStatus.innerHTML = '<span class="dot"></span> Monitoring Active';
                systemStatus.style.color = 'var(--success)';

                videoPlaceholder.style.display = 'none';
                videoFeed.style.display = 'block';

                // Start Video Stream
                const route = routeSelect.value;
                const density = densitySlider.value;

                // Construct URL with source
                let src = `/video_feed?route=${route}&density=${density}&source_type=${currentSource}`;
                if (currentSource === 'upload' && uploadedFilename) {
                    src += `&filename=${uploadedFilename}`;
                }

                videoFeed.src = src;

                // Start Polling Metrics
                pollInterval = setInterval(fetchMetrics, 1000);
            });
    }

    function stopMonitoring() {
        fetch('/api/stop', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                isMonitoring = false;

                // Update UI to stopped state
                toggleBtn.innerHTML = '<i class="fa-solid fa-play"></i> Start Monitoring';
                toggleBtn.classList.remove('btn-danger');
                toggleBtn.classList.add('btn-primary');

                systemStatus.innerHTML = '<span class="dot" style="animation:none; background:gray;"></span> System Standby';
                systemStatus.style.color = 'gray';

                videoFeed.src = "";
                videoFeed.style.display = 'none';
                videoPlaceholder.style.display = 'block';

                if (pollInterval) clearInterval(pollInterval);
            });
    }

    function fetchMetrics() {
        fetch('/api/metrics')
            .then(res => res.json())
            .then(data => {
                // Update text
                passengerCount.textContent = data.passenger_count;
                densityMetric.textContent = data.current_density.toFixed(2);
                alertMessage.textContent = data.alert_msg;

                // Handle Alert Styling
                if (data.alert_type === 'error') {
                    alertCard.classList.add('danger');
                    alertIcon.classList.remove('fa-circle-check');
                    alertIcon.classList.add('fa-triangle-exclamation');
                } else {
                    alertCard.classList.remove('danger');
                    alertIcon.classList.add('fa-circle-check');
                    alertIcon.classList.remove('fa-triangle-exclamation');
                }
            });
    }
});
function openAlertsWindow() {
    window.open(
        "/alerts",
        "AlertDashboard",
        "width=1200,height=650,menubar=no,toolbar=no"
    );
}
