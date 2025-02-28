import React from "react";

export const WeatherOverview = () => {
  return (
    <div className="card">
      <h2 className="text-lg text-blue-300">Today's Weather</h2>
      <div className="flex items-center justify-between mt-3">
        <div>
          <p className="text-4xl font-bold">26°C</p>
          <p className="text-sm text-gray-400">Partly Cloudy</p>
        </div>
        <div>
          <p className="text-sm">Humidity: 65%</p>
          <p className="text-sm">Wind: 12 km/h</p>
        </div>
      </div>
    </div>
  );
};
