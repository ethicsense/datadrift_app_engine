import { memo } from "react";

import type { WidgetConfig } from "../types";
import { renderWidget } from "./widgetRegistry";

type WidgetRendererProps = {
  widgets: WidgetConfig[];
};

function WidgetRendererComponent({ widgets }: WidgetRendererProps) {
  return (
    <div className="widget-grid">
      {widgets.map((widget) => (
        <div key={widget.id}>{renderWidget(widget)}</div>
      ))}
    </div>
  );
}

export const WidgetRenderer = memo(WidgetRendererComponent);
