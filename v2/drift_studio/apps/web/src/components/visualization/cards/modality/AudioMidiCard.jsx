import React from "react";
import CardContainer from "../common/CardContainer";
import MetricGrid from "../../widgets/MetricGrid";
import BarChart from "../../widgets/BarChart";
import { toChartData, pickNumeric } from "../../utils";

export default function AudioMidiCard({ data }) {
  if (!data) return null;
  const summary = data.summary || data.stats || data;
  const numeric = pickNumeric(summary);
  const numericDrift = data.numericDrift || data.numeric_drift;
  const categoricalDrift = data.categoricalDrift || data.categorical_drift;
  const distributions = summary?.label_distributions;

  return (
    <CardContainer title="Audio MIDI">
      {Object.keys(numeric).length > 0 && (
        <div className="mb-4">
          <MetricGrid data={numeric} />
        </div>
      )}
      {numericDrift && (
        <div className="mb-4">
          <div className="text-xs text-gray-600 mb-2">numeric_drift</div>
          <BarChart data={toChartData(numericDrift)} />
        </div>
      )}
      {categoricalDrift && (
        <div className="mb-4">
          <div className="text-xs text-gray-600 mb-2">categorical_drift</div>
          <BarChart data={toChartData(categoricalDrift)} />
        </div>
      )}
      {distributions && (
        <div className="space-y-4">
          {Object.entries(distributions).map(([name, dist]) => {
            const chartData = toChartData(dist);
            if (!chartData.length) return null;
            return (
              <div key={name}>
                <div className="text-xs text-gray-600 mb-2">{name}</div>
                <BarChart data={chartData} height={240} />
              </div>
            );
          })}
        </div>
      )}
    </CardContainer>
  );
}
