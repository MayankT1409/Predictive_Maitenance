import React from 'react';
import PredictionTable from '../components/PredictionTable';

const Predictions = () => (
  <div className="p-6 bg-gray-50 min-h-screen">
    <div className="mb-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Predictions</h2>
      <p className="text-gray-600">Enter equipment logs to get predictive maintenance insights</p>
    </div>
    <PredictionTable />
  </div>
);

export default Predictions;