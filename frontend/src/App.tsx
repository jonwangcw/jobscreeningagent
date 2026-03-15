import { useEffect, useState } from "react";
import type { Job } from "./api";
import { listJobs, triggerRun } from "./api";
import JobCard from "./components/JobCard";

const STATUS_TABS = ["all", "new", "reviewed", "applied", "rejected"] as const;
type Tab = (typeof STATUS_TABS)[number];

const styles: Record<string, React.CSSProperties> = {
  container: { maxWidth: 860, margin: "0 auto", padding: "24px 16px" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 },
  title: { fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" },
  tabs: { display: "flex", gap: 4, marginBottom: 20, flexWrap: "wrap" },
  runBtn: {
    background: "#6366f122",
    color: "#818cf8",
    border: "1px solid #6366f144",
    borderRadius: 8,
    padding: "8px 18px",
    cursor: "pointer",
    fontWeight: 700,
    fontSize: 14,
  },
  empty: { color: "#475569", textAlign: "center", padding: "60px 0", fontSize: 15 },
};

export default function App() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>("all");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [runMsg, setRunMsg] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listJobs(activeTab === "all" ? undefined : activeTab)
      .then(setJobs)
      .catch((e) => console.error(e))
      .finally(() => setLoading(false));
  }, [activeTab]);

  function handleJobUpdate(updated: Job) {
    setJobs((prev) => prev.map((j) => (j.id === updated.id ? updated : j)));
  }

  async function handleRun() {
    setRunning(true);
    setRunMsg(null);
    try {
      const res = await triggerRun();
      setRunMsg(res.message);
      // Refresh jobs after run
      const fresh = await listJobs(activeTab === "all" ? undefined : activeTab);
      setJobs(fresh);
    } catch (e) {
      setRunMsg(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.title}>Job Agent Dashboard</div>
        <button onClick={handleRun} disabled={running} style={styles.runBtn}>
          {running ? "Running…" : "▶ Run now"}
        </button>
      </div>

      {runMsg && (
        <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, padding: "10px 16px", marginBottom: 16, fontSize: 13, color: "#94a3b8" }}>
          {runMsg}
        </div>
      )}

      <div style={styles.tabs}>
        {STATUS_TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              background: activeTab === tab ? "#6366f133" : "transparent",
              color: activeTab === tab ? "#818cf8" : "#64748b",
              border: activeTab === tab ? "1px solid #6366f155" : "1px solid #1e293b",
              borderRadius: 8,
              padding: "6px 14px",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: activeTab === tab ? 700 : 400,
              textTransform: "capitalize",
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {loading ? (
        <div style={styles.empty}>Loading…</div>
      ) : jobs.length === 0 ? (
        <div style={styles.empty}>No jobs found in this category.</div>
      ) : (
        jobs.map((job) => (
          <JobCard key={job.id} job={job} onUpdate={handleJobUpdate} />
        ))
      )}
    </div>
  );
}
