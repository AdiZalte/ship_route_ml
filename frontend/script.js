
const API_URL = "http://127.0.0.1:8000";

const statusElement = document.getElementById("api-status");
const trackInput = document.getElementById("track-data");
const sampleButton = document.getElementById("sample-btn");
const predictButton = document.getElementById("predict-btn");
const messageElement = document.getElementById("message");

let map;
let routeLayer;

// Initialize the map
function initializeMap() {
    map = L.map("map").setView([33.75, -118.25], 7);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19
    }).addTo(map);

    routeLayer = L.layerGroup().addTo(map);
}

// Check whether the backend is running
async function checkAPI() {
    try {
        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("API health check failed");
        }

        statusElement.textContent = "● API Connected";
        statusElement.style.background = "#166534";
    } catch (error) {
        statusElement.textContent = "● API Offline";
        statusElement.style.background = "#991b1b";
    }
}

// Generate a sample track with 20 observations.
// These are synthetic demonstration coordinates near Long Beach.
function loadSampleTrack() {
    const observations = [];
    const now = Date.now();

    for (let i = 19; i >= 0; i--) {
        observations.push({
            timestamp: new Date(now - i * 5 * 60 * 1000)
    .toISOString()
    .replace("T", " ")
    .replace(/\.\d{3}Z$/, ""),
            latitude: 33.55 + (19 - i) * 0.005,
            longitude: -118.45 + (19 - i) * 0.005,
            sog: 10,
            cog: 45,
            wave_height: 1.0,
            wind_speed: 8
        });
    }

    trackInput.value = JSON.stringify(observations, null, 2);
    messageElement.textContent =
        "Sample track loaded. Click Predict Route to continue.";
    messageElement.className = "message-success";
}


function displayValue(id, value, fallback = "Not available") {
    document.getElementById(id).textContent =
        value !== undefined && value !== null ? value : fallback;
}

function drawRoute(observations, result) {
    routeLayer.clearLayers();

    const current = result.last_position || observations[observations.length - 1];

    const points = [
        [Number(current.latitude), Number(current.longitude)]
    ];

    // Your API returns future positions inside "trajectory".
    if (Array.isArray(result.trajectory)) {
        result.trajectory.forEach(point => {
            points.push([
                Number(point.latitude),
                Number(point.longitude)
            ]);
        });
    }

    L.marker(points[0])
        .addTo(routeLayer)
        .bindPopup("Latest vessel position");

    if (points.length > 1) {
        L.polyline(points, {
            color: "#0891b2",
            weight: 4
        }).addTo(routeLayer);

        L.circleMarker(points[points.length - 1], {
            radius: 7,
            color: "#047857",
            fillColor: "#10b981",
            fillOpacity: 1
        })
            .addTo(routeLayer)
            .bindPopup("Predicted position");

        map.fitBounds(L.latLngBounds(points).pad(0.2));
    }
}

async function predictRoute() {
    messageElement.textContent = "";
    messageElement.className = "";

    let observations;

    try {
        observations = JSON.parse(trackInput.value);

        if (!Array.isArray(observations) || observations.length < 20) {
            throw new Error("Please provide at least 20 AIS observations.");
        }
    } catch (error) {
        messageElement.textContent =
            "Invalid input: " + error.message;
        messageElement.className = "message-error";
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
    track: observations,
    wave_height: observations[observations.length - 1].wave_height ?? 1.0,
    wind_speed: observations[observations.length - 1].wind_speed ?? 8.0
})
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof result.detail === "string"
                    ? result.detail
                    : JSON.stringify(result.detail || result)
            );
        }

        console.log("Prediction API response:", result);

        // Nearest port and ETA
        displayValue("nearest-port", result.eta?.port);

        if (result.eta?.applicable && result.eta.eta_minutes != null) {
            displayValue(
                "eta",
                `${Number(result.eta.eta_minutes).toFixed(1)} minutes`
            );
        } else {
            displayValue(
                "eta",
                result.eta?.reason || "ETA unavailable"
            );
        }

        // Distance to port
        displayValue(
            "distance",
            result.eta?.distance_km != null
                ? `${Number(result.eta.distance_km).toFixed(2)} km`
                : undefined
        );

        // Shipping lane
        if (result.lane) {
            const laneText = result.lane.off_lane
                ? `Off lane (${Number(result.lane.distance_to_lane_km).toFixed(2)} km away)`
                : `On lane (${Number(result.lane.distance_to_lane_km).toFixed(2)} km away)`;

            displayValue("lane-status", laneText);
        } else {
            displayValue("lane-status", undefined);
        }

        // Risk model
        if (result.risk?.available && result.risk.probability != null) {
            const probability = Number(result.risk.probability);
            const riskLabel = result.risk.high_risk
                ? "High"
                : "Low";

            displayValue(
                "risk",
                `${riskLabel} (${(probability * 100).toFixed(1)}%)`
            );
        } else {
            displayValue(
                "risk",
                result.risk?.reason || "Risk unavailable"
            );
        }

        drawRoute(observations, result);

        messageElement.textContent =
            "Prediction completed successfully.";
        messageElement.className = "message-success";

    } catch (error) {
        console.error("Prediction error:", error);

        messageElement.textContent =
            "Prediction failed: " + error.message;
        messageElement.className = "message-error";
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict Route";
    }
}


// Connect buttons
sampleButton.addEventListener("click", loadSampleTrack);
predictButton.addEventListener("click", predictRoute);

// Start the page
initializeMap();
checkAPI();
