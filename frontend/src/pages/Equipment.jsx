import React from 'react';
import { equipmentData } from '../components/API';

const Equipment = () => (
  <div className="p-6 bg-gray-50 min-h-screen">
    <div className="mb-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Equipment</h2>
      <p className="text-gray-600">Manage and monitor all industrial equipment</p>
    </div>

    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {equipmentData.map((equipment) => (
        <div key={equipment.id} className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">{equipment.name}</h3>
            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
              equipment.status === 'Good' ? 'bg-green-100 text-green-800' :
              equipment.status === 'Warning' ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'
            }`}>
              {equipment.status}
            </span>
          </div>
          <div className="space-y-3">
            <div>
              <p className="text-sm text-gray-600">Equipment ID</p>
              <p className="font-medium">{equipment.id}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Health Score</p>
              <div className="flex items-center">
                <span className="mr-2 font-medium">{equipment.health}%</span>
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${equipment.health >= 80 ? 'bg-green-500' : equipment.health >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
                    style={{ width: `${equipment.health}%` }}
                  />
                </div>
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-600">Estimated Remaining Life</p>
              <p className="font-medium">{equipment.daysLeft} days</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Risk Level</p>
              <p className={`font-medium ${
                equipment.risk === 'Low' ? 'text-green-600' :
                equipment.risk === 'Medium' ? 'text-yellow-600' :
                'text-red-600'
              }`}>{equipment.risk}</p>
            </div>
          </div>
          <button className="mt-4 w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors">
            View Details
          </button>
        </div>
      ))}
    </div>
  </div>
);

export default Equipment;