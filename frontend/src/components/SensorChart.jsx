import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { sensorData } from './API';

const SensorChart = () => (
  <div className="bg-white p-6 rounded-lg border border-gray-200">
    <h3 className="text-lg font-semibold mb-4">Live Sensor Trends</h3>
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={sensorData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis />
        <Line type="monotone" dataKey="pressure" stroke="#3B82F6" name="Pressure" />
        <Line type="monotone" dataKey="temperature" stroke="#EF4444" name="Temperature" />
        <Line type="monotone" dataKey="vibration" stroke="#10B981" name="Vibration" />
      </LineChart>
    </ResponsiveContainer>
    <div className="flex justify-center space-x-6 mt-2 text-sm">
      <span className="flex items-center"><div className="w-3 h-3 bg-blue-500 rounded mr-2"></div>Pressure</span>
      <span className="flex items-center"><div className="w-3 h-3 bg-red-500 rounded mr-2"></div>Temperature</span>
      <span className="flex items-center"><div className="w-3 h-3 bg-green-500 rounded mr-2"></div>Vibration</span>
    </div>
  </div>
);

export default SensorChart;