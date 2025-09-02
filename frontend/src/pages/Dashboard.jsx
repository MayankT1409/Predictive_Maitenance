import { useEffect, useMemo, useState } from "react";
import StatCard from "../components/StateCard.jsx";
import SensorChart from "../components/SensorChart.jsx";
import RiskGauge from "../components/RiskGauge.jsx";
import LoadingSpinner from "../components/LoadingSpinner.jsx";
import { fetchSensors, fetchPredictions } from "../components/API.js";

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [sensorRows, setSensorRows] = useState([]);
  const [predRows, setPredRows] = useState([]);

  useEffect(() => {
    const runPrediction = async () => {
      try {
        const samplePayload = {
          sensor1: 10,
          sensor2: 20,
          // ...other features required by your model
        };

        const result = await fetchPredictions(samplePayload);
        console.log("Prediction:", result);
      } catch (err) {
        console.error("Prediction error:", err);
      }
    };

    runPrediction();
  }, []);


  // Build lightweight live-like chart from last items
  const chartData = useMemo(() => {
    return sensorRows.slice(0, 30).reverse().map(r => ({
      time: new Date(r.timestamp).toLocaleTimeString(),
      temperature: r.temperature,
      vibration: r.vibration,
      pressure: r.pressure,
    }));
  }, [sensorRows]);

  const latestProb = predRows?.[0]?.probability ?? 0;

  if (loading) return <LoadingSpinner />;

  return (
    <div className="space-y-6">
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Sensors Tracked" value={sensorRows.length} subtitle="Recent window" />
        <StatCard title="Predictions Recorded" value={predRows.length} />
        <StatCard title="Latest Status" value={predRows?.[0]?.status || "—"} subtitle={predRows?.[0]?.timestamp && new Date(predRows[0].timestamp).toLocaleString()} />
        <StatCard title="Latest Risk" value={`${Math.round(latestProb * 100)}%`} />
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <SensorChart data={chartData} />
        </div>
        <RiskGauge probability={latestProb} />
      </div>
    </div>
  );
}
