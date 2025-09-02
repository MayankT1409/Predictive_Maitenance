import { useEffect, useState } from "react";
import LoadingSpinner from "../components/LoadingSpinner.jsx";
import SensorTable from "../components/SensorTable.jsx";
import { fetchSensors } from "../components/API.js";

export default function Sensors() {
  const [loading, setLoading] = useState(true);
  const [rows, setRows] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const data = await fetchSensors();
        const formatted = data.map(d => ({
          timestamp: new Date(d.timestamp).toLocaleString(),
          temperature: d.temperature,
          vibration: d.vibration,
          pressure: d.pressure,
        }));
        setRows(formatted);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <LoadingSpinner />;

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Sensors</h2>
      <SensorTable rows={rows} />
    </div>
  );
}
