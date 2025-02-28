import React from "react";

const forecastData = [
  { day: "Mon", temp: 29, icon: "🌤" },
  { day: "Tue", temp: 30, icon: "☀️" },
  { day: "Wed", temp: 28, icon: "⛅" },
  { day: "Thu", temp: 27, icon: "🌧" },
  { day: "Fri", temp: 26, icon: "🌦" },
  { day: "Sat", temp: 25, icon: "⛈" },
  { day: "Sun", temp: 27, icon: "☀️" },
];

export const Forecast = () => {
  return (
    <div className="card">
      <h2 className="text-lg text-blue-300">7-Day Forecast</h2>
      <div className="grid grid-cols-4 gap-2 mt-3">
        {forecastData.map((day) => (
          <div key={day.day} className="text-center">
            <p className="text-xl">{day.icon}</p>
            <p className="text-sm text-gray-300">{day.day}</p>
            <p className="text-lg">{day.temp}°C</p>
          </div>
        ))}
      </div>
    </div>
  );
};
