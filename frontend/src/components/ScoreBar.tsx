interface ScoreBarProps {
  label: string;
  value: number | null;
  color?: string;
}

const colors: Record<string, string> = {
  role: "#6366f1",
  location: "#10b981",
  stack: "#f59e0b",
  composite: "#3b82f6",
};

export default function ScoreBar({ label, value, color }: ScoreBarProps) {
  const pct = value != null ? Math.round(value * 100) : 0;
  const bg = color ?? colors[label.toLowerCase()] ?? "#6366f1";

  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8", marginBottom: 2 }}>
        <span>{label}</span>
        <span>{value != null ? pct : "—"}</span>
      </div>
      <div style={{ height: 5, background: "#1e293b", borderRadius: 3 }}>
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: bg,
            borderRadius: 3,
            transition: "width 0.3s",
          }}
        />
      </div>
    </div>
  );
}
