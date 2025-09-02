import { RadialBarChart, RadialBar, ResponsiveContainer, PolarAngleAxis } from "recharts";

export default function RiskGauge({ probability = 0 }) {
  const pct = Math.round(probability * 100);
  const data = [{ name: "risk", value: pct, fill: pct > 60 ? "#ef4444" : pct > 30 ? "#f59e0b" : "#10b981" }];

  return (
    <div className="bg-white rounded-2xl shadow-sm p-5 flex flex-col items-center">
      <h3 className="font-semibold mb-2">Failure Risk</h3>
      <div className="w-full h-56">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart innerRadius="70%" outerRadius="100%" data={data} startAngle={180} endAngle={0}>
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar minAngle={15} dataKey="value" cornerRadius={20} clockWise />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 text-3xl font-semibold">{pct}%</div>
      <div className="text-xs text-gray-500">Probability of maintenance required</div>
    </div>
  );
}
