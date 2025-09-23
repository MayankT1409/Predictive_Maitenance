import React from 'react';
import { equipmentData } from './API';

const SensorTable = () => (
  <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
    <div className="px-6 py-4 border-b border-gray-200">
      <h3 className="text-lg font-semibold">Equipment Status</h3>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Equipment</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Health Score</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Days Remaining</th>
          </tr>
        </thead>
        <tbody>
          {equipmentData.map((equipment) => (
            <tr key={equipment.id} className="border-b border-gray-200">
              <td className="px-6 py-4">
                <div className="font-medium text-gray-900">{equipment.name}</div>
                <div className="text-sm text-gray-500">{equipment.id}</div>
              </td>
              <td className="px-6 py-4">
                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                  equipment.status === 'Good' ? 'bg-green-100 text-green-800' :
                  equipment.status === 'Warning' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {equipment.status}
                </span>
              </td>
              <td className="px-6 py-4">
                <div className="flex items-center">
                  <span className="mr-2">{equipment.health}%</span>
                  <div className="w-16 bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${equipment.health >= 80 ? 'bg-green-500' : equipment.health >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
                      style={{ width: `${equipment.health}%` }}
                    />
                  </div>
                </div>
              </td>
              <td className="px-6 py-4 text-gray-900">{equipment.daysLeft} days</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

export default SensorTable;