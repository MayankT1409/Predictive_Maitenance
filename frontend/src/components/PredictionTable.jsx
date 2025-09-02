export default function PredictionTable({ rows = [] }) {
  const badge = (p) =>
    p >= 0.6 ? "bg-red-100 text-red-700" : p >= 0.3 ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700";

  return (
    <div className="bg-white rounded-2xl shadow-sm p-5 overflow-x-auto">
      <h3 className="font-semibold mb-3">Prediction History</h3>
      <table className="min-w-full text-sm">
        <thead className="text-gray-500">
          <tr>
            <th className="py-2 pr-4 text-left">Timestamp</th>
            <th className="py-2 pr-4 text-left">Status</th>
            <th className="py-2 pr-4 text-left">Probability</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t">
              <td className="py-2 pr-4">{r.timestamp}</td>
              <td className="py-2 pr-4">{r.status}</td>
              <td className="py-2 pr-4">
                <span className={`px-2 py-1 rounded-lg text-xs ${badge(r.probability)}`}>
                  {(r.probability * 100).toFixed(1)}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <p className="text-gray-400 text-sm mt-2">No predictions yet.</p>}
    </div>
  );
}
