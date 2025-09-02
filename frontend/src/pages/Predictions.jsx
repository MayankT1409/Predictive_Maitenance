import { useEffect, useState } from "react";
import LoadingSpinner from "../components/LoadingSpinner.jsx";
import PredictionTable from "../components/PredictionTable.jsx";
import { fetchPredictions } from "../components/API.js";

export default function Predictions() {
  const [loading, setLoading] = useState(true);
  const [rows, setRows] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const data = await fetchPredictions();
        const formatted = data.map(d => ({
          timestamp: new Date(d.timestamp).toLocaleString(),
          status: d.status,
          probability: d.probability,
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
      <h2 className="text-2xl font-semibold">Predictions</h2>
      <PredictionTable rows={rows} />
    </div>
  );
}
