import { LineChart, Line, Tooltip, CartesianGrid, XAxis, YAxis, Legend, ResponsiveContainer } from "recharts";

export default function SensorChart({ data }) {
  return (
    <div className="bg-white rounded-2xl shadow-sm p-5">
      <h3 className="font-semibold mb-3">Live Sensor Trends</h3>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="temperature" name="Temperature" stroke="#ef4444" dot={false} />
            <Line type="monotone" dataKey="vibration"   name="Vibration"   stroke="#10b981" dot={false} />
            <Line type="monotone" dataKey="pressure"    name="Pressure"    stroke="#3b82f6" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
