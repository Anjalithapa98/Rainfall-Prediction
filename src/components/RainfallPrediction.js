import React from "react";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

export const RainfallPrediction = () => {
  const rainfallPercentage = 72; // Example value

  return (
    <div className="card text-center">
      <h2 className="text-lg text-blue-300">Rainfall Prediction</h2>
      <div className="circular-progress mx-auto my-4">
        <CircularProgressbar
          value={rainfallPercentage}
          text={`${rainfallPercentage}%`}
          styles={buildStyles({
            textColor: "#fff",
            pathColor: "#38bdf8",
            trailColor: "rgba(255, 255, 255, 0.1)",
          })}
        />
      </div>
      <p className="text-sm text-gray-400">Chance of Rain</p>
    </div>
  );
};
