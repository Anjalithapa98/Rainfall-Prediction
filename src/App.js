import React from "react";
import { WeatherOverview } from "./components/WeatherOverview";
import { RainfallPrediction } from "./components/RainfallPrediction";
import { Forecast } from "./components/SevenDayForecast";
import "./index.css";

function App() {
  return (
    <div className="p-5">
      {/* Header with Logo & Project Title */}
    <header className="flex justify-between items-center p-4 bg-gray-900 rounded-lg">
  {/* Logo & Title Together */}
  <div className="flex items-center space-x-3">
    <img src="/logo.jpg" alt="Logo" className="weather-logo h-12 w-12" />
    <h1 className="text-xl text-blue-400">Rainfall Prediction System</h1>
  </div>

  {/* Last Updated Time */}
  <span className="text-sm text-gray-400">Updated: Just now</span>
</header>

      {/* Greeting */}
      <p className="text-center greeting my-4">Welcome, User! Here's the latest weather update.</p>

      {/* Dashboard Layout */}
      <div className="dashboard">
        <WeatherOverview />
        <RainfallPrediction />
        <Forecast />
      </div>
    </div>
  );
}

export default App;
