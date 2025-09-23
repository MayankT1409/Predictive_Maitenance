// import axios from "axios";

// // Pick base URLs from environment variables
// const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000";
// const ML_API_URL  = import.meta.env.VITE_ML_API_URL  || "http://127.0.0.1:8000";

// // Create axios instance for backend (sensors, logs, etc.)
// const API = axios.create({
//   baseURL: `${BACKEND_URL}/api`,
//   timeout: 10000,
// });

// // Optional: second instance for ML API if endpoints differ
// const ML_API = axios.create({
//   baseURL: "http://127.0.0.1:8000",   // no /api
//   timeout: 10000,
// });

// // ----------------- API calls -----------------
// export const fetchSensors     = () => API.get("/sensors").then(r => r.data);
// export const fetchLogs        = () => API.get("/logs").then(r => r.data);
// export const postSensor       = (payload) => API.post("/sensors", payload).then(r => r.data);

// // Predictions might come from ML_API instead of backend
// export const fetchPredictions = (payload = {}) =>
//   ML_API.post("/predict", payload).then(r => r.data);


// export default API;


// Sample data - move this to a separate data service
export const sensorData = [
  { time: '00:00', pressure: 45, temperature: 75, vibration: 2.1 },
  { time: '04:00', pressure: 48, temperature: 78, vibration: 2.3 },
  { time: '08:00', pressure: 52, temperature: 82, vibration: 2.8 },
  { time: '12:00', pressure: 49, temperature: 79, vibration: 2.4 },
  { time: '16:00', pressure: 46, temperature: 76, vibration: 2.2 },
  { time: '20:00', pressure: 44, temperature: 74, vibration: 2.0 },
];

export const equipmentData = [
  { id: 'EQ001', name: 'Motor A1', status: 'Good', health: 85, daysLeft: 45, risk: 'Low' },
  { id: 'EQ002', name: 'Pump B2', status: 'Warning', health: 60, daysLeft: 15, risk: 'Medium' },
  { id: 'EQ003', name: 'Compressor C3', status: 'Critical', health: 25, daysLeft: 3, risk: 'High' },
  { id: 'EQ004', name: 'Generator D4', status: 'Good', health: 90, daysLeft: 60, risk: 'Low' },
];

export const healthDistribution = [
  { name: 'Good', value: 65, color: '#10B981' },
  { name: 'Warning', value: 25, color: '#F59E0B' },
  { name: 'Critical', value: 10, color: '#EF4444' },
];