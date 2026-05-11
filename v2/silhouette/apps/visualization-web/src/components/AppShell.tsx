import { NavLink, Outlet } from "react-router-dom";

import { navigationItems } from "../config/navigation";
import { useDashboardFilters } from "../lib/filters";
import { FilterPanel } from "./FilterPanel";

export function AppShell() {
  const { filters, update } = useDashboardFilters();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar__brand">
          <span>Silhouette</span>
          <strong>시각화 엔진</strong>
        </div>
        <nav className="sidebar__nav">
          {navigationItems.map((item) => (
            <NavLink
              key={item.key}
              to={item.path}
              className={({ isActive }) => (isActive ? "nav-link is-active" : "nav-link")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <FilterPanel filters={filters} onChange={update} />
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
