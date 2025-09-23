import React from 'react';

const RiskGauge = ({ value, label }) => {
  const getColor = (val) => {
    if (val >= 80) return 'text-green-500';
    if (val >= 50) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200">
      <div className="text-center">
        <div className={`text-3xl font-bold ${getColor(value)}`}>{value}%</div>
        <div className="text-gray-600 text-sm">{label}</div>
        <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
          <div 
            className={`h-2 rounded-full ${value >= 80 ? 'bg-green-500' : value >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
            style={{ width: `${value}%` }}
          />
        </div>
      </div>
    </div>
  );
};

export default RiskGauge;