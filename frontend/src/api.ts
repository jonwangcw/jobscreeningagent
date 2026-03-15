/** Typed fetch wrappers for all backend routes. */

const BASE = "";

export interface Job {
  id: number;
  posting_id: string;
  source: string;
  company: string;
  title: string;
  location: string | null;
  remote: boolean | null;
  url: string | null;
  role_score: number | null;
  location_score: number | null;
  stack_score: number | null;
  composite_score: number | null;
  rationale: string | null;
  skill_gaps: string[];
  status: "new" | "reviewed" | "applied" | "rejected" | "parse_error";
  created_at: string;
  updated_at: string;
}

export interface GenerateResponse {
  output_path: string;
}

export async function listJobs(status?: string, limit = 200, offset = 0): Promise<Job[]> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) params.set("status", status);
  const res = await fetch(`${BASE}/jobs?${params}`);
  if (!res.ok) throw new Error(`listJobs: ${res.status}`);
  return res.json();
}

export async function getJob(id: number): Promise<Job> {
  const res = await fetch(`${BASE}/jobs/${id}`);
  if (!res.ok) throw new Error(`getJob: ${res.status}`);
  return res.json();
}

export async function updateStatus(id: number, status: string): Promise<Job> {
  const res = await fetch(`${BASE}/jobs/${id}/status`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (!res.ok) throw new Error(`updateStatus: ${res.status}`);
  return res.json();
}

export async function generateMaterials(
  id: number,
  cover_letter: boolean
): Promise<GenerateResponse> {
  const res = await fetch(`${BASE}/jobs/${id}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cover_letter }),
  });
  if (!res.ok) throw new Error(`generate: ${res.status}`);
  return res.json();
}

export async function triggerRun(): Promise<{ message: string; stats?: object }> {
  const res = await fetch(`${BASE}/run`, { method: "POST" });
  if (!res.ok) throw new Error(`triggerRun: ${res.status}`);
  return res.json();
}
