import { useEffect, useState } from "react";
import LoadingSpinner from "../components/LoadingSpinner.jsx";
import MaintenanceLog from "../components/MaintenanceLog.jsx";
import { fetchLogs } from "../components/API.js";

export default function Logs() {
  const [loading, setLoading] = useState(true);
  const [rows, setRows] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const data = await fetchLogs();
        const formatted = data.map(d => ({
          date: new Date(d.date).toLocaleString(),
          issue: d.issue,
          actionTaken: d.actionTaken,
          performedBy: d.performedBy,
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
      <h2 className="text-2xl font-semibold">Maintenance Logs</h2>
      <MaintenanceLog rows={rows} />
    </div>
  );
}
