import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Activity, AlertTriangle, CheckCircle, Cpu } from 'lucide-react';
import StatusCard from '../components/StateCard';
import SensorChart from '../components/SensorChart';
import SensorTable from '../components/SensorTable';
import { healthDistribution } from '../components/API';

const Dashboard = () => (
  <div className="p-6 bg-gray-50 min-h-screen">
    <div className="mb-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Dashboard</h2>
      <p className="text-gray-600">Overview of equipment health and maintenance status</p>
    </div>

    {/* Status Cards */}
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      <StatusCard title="Total Equipment" value="24" icon={Cpu} color="blue" />
      <StatusCard title="Healthy" value="18" icon={CheckCircle} color="green" />
      <StatusCard title="Warnings" value="4" icon={AlertTriangle} color="yellow" />
      <StatusCard title="Critical" value="2" icon={AlertTriangle} color="red" />
    </div>

    {/* Charts */}
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Equipment Health Distribution</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={healthDistribution}
              cx="50%"
              cy="50%"
              outerRadius={80}
              dataKey="value"
            >
              {healthDistribution.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>

      <SensorChart />
    </div>

    {/* Equipment Overview */}
    <SensorTable />
  </div>
);

export default Dashboard;