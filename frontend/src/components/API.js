import axios from "axios";

// Pick base URLs from environment variables
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000";
const ML_API_URL  = import.meta.env.VITE_ML_API_URL  || "http://127.0.0.1:8000";

// Create axios instance for backend (sensors, logs, etc.)
const API = axios.create({
  baseURL: `${BACKEND_URL}/api`,
  timeout: 10000,
});

// Optional: second instance for ML API if endpoints differ
const ML_API = axios.create({
  baseURL: "http://127.0.0.1:8000",   // no /api
  timeout: 10000,
});

// ----------------- API calls -----------------
export const fetchSensors     = () => API.get("/sensors").then(r => r.data);
export const fetchLogs        = () => API.get("/logs").then(r => r.data);
export const postSensor       = (payload) => API.post("/sensors", payload).then(r => r.data);

// Predictions might come from ML_API instead of backend
export const fetchPredictions = (payload = {}) =>
  ML_API.post("/predict", payload).then(r => r.data);


export default API;
