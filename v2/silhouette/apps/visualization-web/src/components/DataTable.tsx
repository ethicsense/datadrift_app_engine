import { useMemo, useState } from "react";

import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table";

import { formatCellValue, formatColumnLabel } from "../lib/formatters";

type DataTableProps = {
  rows: Record<string, unknown>[];
  highlightKey?: string;
  highlightValue?: string | null;
  rowClassName?: (row: Record<string, unknown>) => string | undefined;
  initialSorting?: SortingState;
  includeColumns?: string[];
  onRowSelect?: (row: Record<string, unknown>) => void;
};

export function DataTable({
  rows,
  highlightKey,
  highlightValue,
  rowClassName,
  initialSorting,
  includeColumns,
  onRowSelect,
}: DataTableProps) {
  const [sorting, setSorting] = useState<SortingState>(initialSorting ?? []);
  const columnHelper = useMemo(() => createColumnHelper<Record<string, unknown>>(), []);
  const sample = rows[0];
  const columns = useMemo(
    () => {
      if (!sample) {
        return [];
      }
      const keys = includeColumns?.length ? includeColumns.filter((key) => key in sample) : Object.keys(sample);
      return keys.map((key) =>
        columnHelper.accessor((row) => row[key], {
          id: key,
          header: () => formatColumnLabel(key),
          cell: (info) => {
            const value = info.getValue();
            return formatCellValue(value, key);
          },
        }),
      );
    },
    [columnHelper, sample, includeColumns],
  );

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (!sample) {
    return <div className="empty-state">표시할 데이터가 없습니다.</div>;
  }

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th key={header.id}>
                  {header.isPlaceholder ? null : (
                    <button
                      type="button"
                      className="data-table__sort-button"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <span>{flexRender(header.column.columnDef.header, header.getContext())}</span>
                      <small>
                        {header.column.getIsSorted() === "asc"
                          ? "▲"
                          : header.column.getIsSorted() === "desc"
                            ? "▼"
                            : "↕"}
                      </small>
                    </button>
                  )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => {
            const dimmed =
              highlightKey && highlightValue && String(row.original[highlightKey] ?? "") !== highlightValue
                ? "data-table__row--dimmed"
                : undefined;
            const selected =
              highlightKey && highlightValue && String(row.original[highlightKey] ?? "") === highlightValue
                ? "data-table__row--selected"
                : undefined;
            const clickable = onRowSelect ? "data-table__row--clickable" : undefined;
            const custom = rowClassName?.(row.original);
            const trClass = [dimmed, selected, clickable, custom].filter(Boolean).join(" ") || undefined;
            return (
              <tr
                key={row.id}
                className={trClass}
                onClick={onRowSelect ? () => onRowSelect(row.original) : undefined}
                role={onRowSelect ? "button" : undefined}
                tabIndex={onRowSelect ? 0 : undefined}
                onKeyDown={
                  onRowSelect
                    ? (event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          onRowSelect(row.original);
                        }
                      }
                    : undefined
                }
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
