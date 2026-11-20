/**
 * FreshTrack Central API Client with Smart Host Detection & Glassmorphism UI Bridge
 */

const getInitialBaseUrl = () => {
    if (window.__API_BASE_URL__) {
        return window.__API_BASE_URL__;
    }
    const hostname = window.location.hostname || "localhost";
    const port = "8000";
    if (window.location.protocol === "file:") {
        return "http://localhost:8000/api";
    }
    return `http://${hostname}:${port}/api`;
};

const API_CONFIG = {
    BASE_URL: getInitialBaseUrl(),
    FALLBACK_URLS: [
        "http://localhost:8000/api",
        "http://127.0.0.1:8000/api"
    ],
    TIMEOUT_MS: 10000
};

const apiClient = {
    getBaseUrl() {
        return API_CONFIG.BASE_URL;
    },

    setBaseUrl(newUrl) {
        if (newUrl) {
            API_CONFIG.BASE_URL = newUrl.replace(/\/+$/, "");
        }
    },

    async request(endpoint, options = {}) {
        const cleanEndpoint = endpoint.startsWith("/") ? endpoint : "/" + endpoint;
        const defaultHeaders = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        };

        const executeFetch = async (baseUrl) => {
            const url = `${baseUrl}${cleanEndpoint}`;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.TIMEOUT_MS);

            try {
                const response = await fetch(url, {
                    ...options,
                    signal: controller.signal,
                    headers: {
                        ...defaultHeaders,
                        ...(options.headers || {})
                    }
                });
                clearTimeout(timeoutId);

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    const errorMsg = data.detail || (data.details && data.details.join("; ")) || data.message || `HTTP ${response.status}: ${response.statusText}`;
                    const err = new Error(errorMsg);
                    err.status = response.status;
                    err.data = data;
                    throw err;
                }
                return data;
            } catch (err) {
                clearTimeout(timeoutId);
                throw err;
            }
        };

        // Try primary BASE_URL first
        try {
            return await executeFetch(API_CONFIG.BASE_URL);
        } catch (primaryErr) {
            // If primary failed due to network/host difference, try fallbacks
            for (const fallback of API_CONFIG.FALLBACK_URLS) {
                if (fallback !== API_CONFIG.BASE_URL) {
                    try {
                        const fallbackData = await executeFetch(fallback);
                        // Save working fallback as primary
                        API_CONFIG.BASE_URL = fallback;
                        return fallbackData;
                    } catch (e) {
                        // continue to next fallback
                    }
                }
            }
            throw primaryErr;
        }
    },

    get(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: "GET" });
    },

    post(endpoint, body, options = {}) {
        return this.request(endpoint, {
            ...options,
            method: "POST",
            body: JSON.stringify(body)
        });
    },

    async checkHealth() {
        return this.get("/health");
    },

    async getDatasetSummary() {
        return this.get("/analysis/summary");
    },

    async getFruitAnalysis() {
        return this.get("/analysis/fruits");
    },

    async getEnvironmentalAnalysis() {
        return this.get("/analysis/environment");
    },

    async getSupportedFruits() {
        return this.get("/predictions/fruits");
    },

    async getScatterData(fruit = "All", limit = 600) {
        return this.get(`/analysis/scatter-data?fruit=${encodeURIComponent(fruit)}&limit=${limit}`);
    },

    async getSampleRecords(fruit = "Tomato", limit = 25, offset = 0, quality = null) {
        let ep = `/predictions/sample-records?fruit_type=${encodeURIComponent(fruit)}&limit=${limit}&offset=${offset}`;
        if (quality && quality !== "All") {
            ep += `&quality=${encodeURIComponent(quality)}`;
        }
        return this.get(ep);
    },

    async predictQuality(telemetry) {
        const payload = {
            fruit_type: telemetry.fruit_type || telemetry.fruit || "Tomato",
            temperature: telemetry.temperature !== undefined ? telemetry.temperature : (telemetry.temperature_c !== undefined ? telemetry.temperature_c : 23.5),
            humidity: telemetry.humidity !== undefined ? telemetry.humidity : (telemetry.humidity_percent !== undefined ? telemetry.humidity_percent : 85.0),
            light_intensity: telemetry.light_intensity !== undefined ? telemetry.light_intensity : (telemetry.light_lux !== undefined ? telemetry.light_lux : 15.0),
            co2: telemetry.co2 !== undefined ? telemetry.co2 : (telemetry.co2_ppm !== undefined ? telemetry.co2_ppm : 340.0),
            // backward compat keys
            temperature_c: telemetry.temperature !== undefined ? telemetry.temperature : telemetry.temperature_c,
            humidity_percent: telemetry.humidity !== undefined ? telemetry.humidity : telemetry.humidity_percent,
            light_lux: telemetry.light_intensity !== undefined ? telemetry.light_intensity : telemetry.light_lux,
            co2_ppm: telemetry.co2 !== undefined ? telemetry.co2 : telemetry.co2_ppm
        };

        const res = await this.post("/predictions/predict", payload);
        const isGood = (res.predicted_quality === "Good" || res.quality_prediction === "Good");
        const conf = res.confidence_score !== undefined 
            ? res.confidence_score 
            : (isGood ? (res.good_probability || 0.95) : (res.bad_probability || 0.95));

        return {
            ...res,
            predicted_quality: res.predicted_quality || (isGood ? "Good" : "Bad"),
            quality_prediction: res.predicted_quality || (isGood ? "Good" : "Bad"),
            confidence_score: conf,
            confidence: conf,
            is_good: isGood
        };
    }
};

window.API_CONFIG = API_CONFIG;
window.apiClient = apiClient;
