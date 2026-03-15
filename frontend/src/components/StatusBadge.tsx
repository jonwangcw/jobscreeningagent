import type { Job } from "../api";

const colors: Record<string, string> = {
  new: "#3b82f6",
  reviewed: "#8b5cf6",
  applied: "#10b981",
  rejected: "#ef4444",
  parse_error: "#f59e0b",
};

export default function StatusBadge({ status }: { status: Job["status"] }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: 12,
        fontSize: 11,
        fontWeight: 600,
        background: (colors[status] ?? "#64748b") + "33",
        color: colors[status] ?? "#94a3b8",
        border: `1px solid ${colors[status] ?? "#64748b"}66`,
        textTransform: "uppercase",
        letterSpacing: "0.05em",
      }}
    >
      {status}
    </span>
  );
}
