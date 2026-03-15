import { useState } from "react";
import type { Job } from "../api";
import { generateMaterials, updateStatus } from "../api";
import ScoreBar from "./ScoreBar";
import StatusBadge from "./StatusBadge";

interface Props {
  job: Job;
  onUpdate: (updated: Job) => void;
}

export default function JobCard({ job, onUpdate }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [coverLetter, setCoverLetter] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);

  const composite = job.composite_score != null ? Math.round(job.composite_score * 100) : null;
  const scoreColor =
    composite == null ? "#94a3b8" : composite >= 70 ? "#10b981" : composite >= 45 ? "#f59e0b" : "#ef4444";

  async function handleStatusChange(newStatus: string) {
    try {
      const updated = await updateStatus(job.id, newStatus);
      onUpdate(updated);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleGenerate() {
    setGenerating(true);
    setError(null);
    try {
      const res = await generateMaterials(job.id, coverLetter);
      setOutputPath(res.output_path);
    } catch (e) {
      setError(String(e));
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div
      style={{
        background: "#1e293b",
        border: "1px solid #334155",
        borderRadius: 10,
        padding: "16px 20px",
        marginBottom: 12,
        cursor: "pointer",
      }}
      onClick={() => setExpanded((e) => !e)}
    >
      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {job.title}
          </div>
          <div style={{ color: "#94a3b8", fontSize: 13 }}>
            {job.company}
            {job.location ? ` · ${job.location}` : ""}
            {job.remote ? " · Remote" : ""}
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6, marginLeft: 16 }}>
          <div style={{ fontSize: 24, fontWeight: 800, color: scoreColor }}>
            {composite != null ? composite : "—"}
          </div>
          <StatusBadge status={job.status} />
        </div>
      </div>

      {/* Mini score bars */}
      <div style={{ marginTop: 12 }}>
        <ScoreBar label="Role" value={job.role_score} color="#6366f1" />
        <ScoreBar label="Location" value={job.location_score} color="#10b981" />
        <ScoreBar label="Stack" value={job.stack_score} color="#f59e0b" />
      </div>

      <div style={{ fontSize: 11, color: "#475569", marginTop: 8 }}>
        {job.source} · Found {new Date(job.created_at).toLocaleDateString()}
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div
          style={{ marginTop: 16, borderTop: "1px solid #334155", paddingTop: 16 }}
          onClick={(e) => e.stopPropagation()}
        >
          {job.rationale && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: "#94a3b8", marginBottom: 4 }}>RATIONALE</div>
              <div style={{ fontSize: 14, lineHeight: 1.6 }}>{job.rationale}</div>
            </div>
          )}

          {job.skill_gaps.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: "#94a3b8", marginBottom: 4 }}>SKILL GAPS</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {job.skill_gaps.map((gap) => (
                  <span key={gap} style={{ background: "#ef444422", color: "#fca5a5", border: "1px solid #ef444444", borderRadius: 6, padding: "2px 8px", fontSize: 12 }}>
                    {gap}
                  </span>
                ))}
              </div>
            </div>
          )}

          {job.url && (
            <a href={job.url} target="_blank" rel="noreferrer" style={{ fontSize: 13, color: "#6366f1" }}>
              View posting ↗
            </a>
          )}

          {/* Action buttons */}
          <div style={{ marginTop: 16, display: "flex", flexWrap: "wrap", gap: 8 }}>
            {job.status === "new" && (
              <button onClick={() => handleStatusChange("reviewed")} style={btnStyle("#6366f1")}>
                Mark reviewed
              </button>
            )}
            {job.status === "reviewed" && (
              <>
                <button onClick={() => handleStatusChange("applied")} style={btnStyle("#10b981")}>
                  Mark applied
                </button>
                <button onClick={() => handleStatusChange("rejected")} style={btnStyle("#ef4444")}>
                  Reject
                </button>
                <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, cursor: "pointer" }}>
                  <input type="checkbox" checked={coverLetter} onChange={(e) => setCoverLetter(e.target.checked)} />
                  Cover letter
                </label>
                <button onClick={handleGenerate} disabled={generating} style={btnStyle("#3b82f6")}>
                  {generating ? "Generating…" : "Generate materials"}
                </button>
              </>
            )}
            {job.status === "new" || job.status === "reviewed" ? null : null}
          </div>

          {outputPath && (
            <div style={{ marginTop: 10, fontSize: 13, color: "#10b981" }}>
              ✓ Saved to: <code style={{ fontSize: 12 }}>{outputPath}</code>
            </div>
          )}

          {error && (
            <div style={{ marginTop: 10, fontSize: 13, color: "#ef4444" }}>{error}</div>
          )}
        </div>
      )}
    </div>
  );
}

function btnStyle(color: string): React.CSSProperties {
  return {
    background: color + "22",
    color: color,
    border: `1px solid ${color}66`,
    borderRadius: 6,
    padding: "6px 14px",
    fontSize: 13,
    cursor: "pointer",
    fontWeight: 600,
  };
}
