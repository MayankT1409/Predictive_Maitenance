import React from "react";
import { NavLink } from "react-router-dom";

const Navbar = () => {
  const tabs = [
    { name: "Dashboard", path: "/dashboard" },
    { name: "Equipment", path: "/equipment" },
    { name: "Predictions", path: "/predictions" },
    { name: "Logs", path: "/logs" },
  ];

  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Logo + Title */}
        <div className="flex items-center space-x-2">
          <div className="bg-blue-500 rounded-full p-2">
            <span className="text-white font-bold text-sm">PM</span>
          </div>
          <h1 className="text-xl font-semibold text-gray-900">
            Predictive Maintenance
          </h1>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-8">
          {tabs.map((tab) => (
            <NavLink
              key={tab.name}
              to={tab.path}
              className={({ isActive }) =>
                `px-3 py-2 text-sm font-medium ${
                  isActive
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-600 hover:text-gray-900"
                }`
              }
            >
              {tab.name}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
