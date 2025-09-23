import React from 'react';
import MaintenanceLog from '../components/MaintenanceLog';

const Logs = () => (
  <div className="p-6 bg-gray-50 min-h-screen">
    <div className="mb-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Maintenance Logs</h2>
      <p className="text-gray-600">View maintenance history and equipment logs</p>
    </div>
    <MaintenanceLog />
  </div>
);

export default Logs;