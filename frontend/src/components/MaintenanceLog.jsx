import React from 'react';
import { Upload } from 'lucide-react';

const MaintenanceLog = () => {
  const logs = [
    { date: '2024-09-23', equipment: 'Motor A1', activity: 'Routine Inspection', status: 'Completed', technician: 'John Doe' },
    { date: '2024-09-22', equipment: 'Pump B2', activity: 'Oil Change', status: 'Completed', technician: 'Jane Smith' },
    { date: '2024-09-21', equipment: 'Compressor C3', activity: 'Emergency Repair', status: 'In Progress', technician: 'Mike Johnson' },
    { date: '2024-09-20', equipment: 'Generator D4', activity: 'Filter Replacement', status: 'Scheduled', technician: 'Sarah Wilson' },
  ];

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Recent Activities</h3>
          <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors flex items-center">
            <Upload className="mr-2 h-4 w-4" />
            Import Logs
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Equipment</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Activity</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Technician</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log, index) => (
              <tr key={index} className="border-b border-gray-200">
                <td className="px-6 py-4 text-gray-900">{log.date}</td>
                <td className="px-6 py-4">
                  <div className="font-medium text-gray-900">{log.equipment}</div>
                </td>
                <td className="px-6 py-4 text-gray-900">{log.activity}</td>
                <td className="px-6 py-4">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    log.status === 'Completed' ? 'bg-green-100 text-green-800' :
                    log.status === 'In Progress' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-blue-100 text-blue-800'
                  }`}>
                    {log.status}
                  </span>
                </td>
                <td className="px-6 py-4 text-gray-900">{log.technician}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MaintenanceLog;