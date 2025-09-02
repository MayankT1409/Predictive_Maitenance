import { Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Sensors from "./pages/Sensors.jsx";
import Predictions from "./pages/Predictions.jsx";
import Logs from "./pages/Logs.jsx";
// import NotFound from "./pages/NotFound.jsx";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/sensors" element={<Sensors />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/logs" element={<Logs />} />
          {/* <Route path="*" element={<NotFound />} /> */}
        </Routes>
      </main>
    </div>
  );
}
