import React, { useState } from 'react';
import { TrendingUp, CheckCircle } from 'lucide-react';
import RiskGauge from './RiskGauge';

const PredictionTable = () => {
  const [logInput, setLogInput] = useState('');
  const [prediction, setPrediction] = useState(null);

  const handlePrediction = () => {
    if (logInput.trim()) {
      const mockPrediction = {
        equipmentId: 'EQ005',
        healthScore: Math.floor(Math.random() * 100),
        remainingDays: Math.floor(Math.random() * 90),
        riskLevel: ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)],
        recommendations: ['Check oil levels', 'Inspect belts', 'Monitor temperature']
      };
      setPrediction(mockPrediction);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold mb-4">Equipment Log Input</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Equipment Log Data
            </label>
            <textarea
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              rows="8"
              placeholder="Enter equipment log data (temperature, pressure, vibration, etc.)...&#10;Example:&#10;Temperature: 78°C&#10;Pressure: 45 PSI&#10;Vibration: 2.3 mm/s&#10;Operating Hours: 1200&#10;Last Maintenance: 30 days ago"
              value={logInput}
              onChange={(e) => setLogInput(e.target.value)}
            />
          </div>
          <button
            onClick={handlePrediction}
            className="w-full bg-blue-500 text-white py-3 px-4 rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center"
          >
            <TrendingUp className="mr-2 h-5 w-5" />
            Generate Prediction
          </button>
        </div>
      </div>

      {/* Results Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
        {prediction ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <RiskGauge value={prediction.healthScore} label="Health Score" />
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{prediction.remainingDays}</div>
                  <div className="text-gray-600 text-sm">Days Remaining</div>
                </div>
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">Risk Level</p>
              <span className={`inline-flex px-3 py-1 text-sm font-semibold rounded-full mt-1 ${
                prediction.riskLevel === 'Low' ? 'bg-green-100 text-green-800' :
                prediction.riskLevel === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {prediction.riskLevel} Risk
              </span>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">Recommendations</p>
              <ul className="space-y-1">
                {prediction.recommendations.map((rec, index) => (
                  <li key={index} className="text-sm text-gray-600 flex items-center">
                    <CheckCircle className="mr-2 h-4 w-4 text-green-500" />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <TrendingUp className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <p>Enter equipment log data and click "Generate Prediction" to see results</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionTable;