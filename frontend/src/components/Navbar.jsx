import { NavLink } from "react-router-dom";

const link =
  "px-3 py-2 rounded-xl text-sm font-medium hover:bg-white/60 hover:text-blue-700 transition";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 bg-white/80 backdrop-blur border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="w-9 h-9 rounded-2xl bg-blue-600 inline-flex items-center justify-center text-white font-bold">PM</span>
          <h1 className="font-semibold text-lg">Predictive Maintenance</h1>
        </div>
        <nav className="flex items-center gap-1">
          <NavLink to="/dashboard" className={({isActive})=>`${link} ${isActive?'bg-blue-50 text-blue-700':'text-gray-700'}`}>Dashboard</NavLink>
          <NavLink to="/sensors" className={({isActive})=>`${link} ${isActive?'bg-blue-50 text-blue-700':'text-gray-700'}`}>Sensors</NavLink>
          <NavLink to="/predictions" className={({isActive})=>`${link} ${isActive?'bg-blue-50 text-blue-700':'text-gray-700'}`}>Predictions</NavLink>
          <NavLink to="/logs" className={({isActive})=>`${link} ${isActive?'bg-blue-50 text-blue-700':'text-gray-700'}`}>Logs</NavLink>
        </nav>
      </div>
    </header>
  );
}
