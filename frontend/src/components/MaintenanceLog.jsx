export default function MaintenanceLog({ rows = [] }) {
  return (
    <div className="bg-white rounded-2xl shadow-sm p-5 overflow-x-auto">
      <h3 className="font-semibold mb-3">Maintenance Logs</h3>
      <table className="min-w-full text-sm">
        <thead className="text-gray-500">
          <tr>
            <th className="py-2 pr-4 text-left">Date</th>
            <th className="py-2 pr-4 text-left">Issue</th>
            <th className="py-2 pr-4 text-left">Action Taken</th>
            <th className="py-2 pr-4 text-left">Performed By</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t">
              <td className="py-2 pr-4">{r.date}</td>
              <td className="py-2 pr-4">{r.issue}</td>
              <td className="py-2 pr-4">{r.actionTaken}</td>
              <td className="py-2 pr-4">{r.performedBy}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <p className="text-gray-400 text-sm mt-2">No logs yet.</p>}
    </div>
  );
}
