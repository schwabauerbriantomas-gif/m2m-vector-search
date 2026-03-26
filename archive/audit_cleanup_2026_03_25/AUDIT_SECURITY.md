# 🔒 Security Audit Report

**Projects:** M2M Vector Search + EBM Language Model  
**Auditor:** Security Auditor (MASFactory Validation Committee)  
**Date:** 2026-03-20  
**Scope:** All Python source files in both projects  

---

## Executive Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH | 3 |
| MEDIUM | 7 |
| LOW | 5 |
| INFO | 3 |

**Overall Assessment:** The codebases are research-grade with several production-readiness gaps. The most urgent issues are unsafe deserialization (pickle) on potentially untrusted data and `torch.load` calls without `weights_only=True`. The M2M project has implemented several security fixes (noted as "H-XX FIX" and "P-XX FIX" in comments), but gaps remain. The EBM project has fewer mitigations in place.

---

## Findings

### [C-01] CRITICAL — Unsafe `pickle.loads()` on Index Data (M2M)

**File:** `m2m/storage/persistence.py`, line ~225 (`load_index`)  
**Also:** `m2m/storage/persistence.py`, line ~215 (`save_index` uses `pickle.dumps`)

**Description:** The `load_index()` method deserializes data with `pickle.loads(data)`. While HMAC verification is performed before deserialization, the HMAC secret has a hardcoded default:
```python
secret = os.environ.get("M2M_HMAC_SECRET", "m2m-default-hmac-secret-change-in-production")
```
If the environment variable is not set in production, the default secret is trivially guessable, making the HMAC check useless. An attacker who can modify the index file (e.g., via another vulnerability, shared storage, or supply chain) can achieve **remote code execution** via pickle deserialization.

**Impact:** RCE if an attacker can write to the index file.

**Recommended Fix:**
1. **Remove the hardcoded default** — raise an error if `M2M_HMAC_SECRET` is not set:
   ```python
   secret = os.environ.get("M2M_HMAC_SECRET")
   if secret is None:
       raise RuntimeError("M2M_HMAC_SECRET environment variable must be set")
   ```
2. **Replace pickle with `safetensors` or `json` + numpy** for index serialization.
3. If pickle must be used, restrict to `pickle.HIGHEST_PROTOCOL` with a custom `Unpickler` that restricts allowed classes.

**PoC:** Create a malicious `.idx` file with `pickle.dumps(__import__('os').system, protocol=4)`, compute HMAC with the default secret, prepend signature. When `load_index()` is called, arbitrary code executes.

---

### [H-01] HIGH — `torch.load()` Without `weights_only=True` (EBM)

**Files:**
- `ebm/diagnose.py`: `torch.load(checkpoint_path)` (line ~17)
- `ebm/diagnose.py`: `torch.load(os.path.join(checkpoint_dir, checkpoint_file))` (line ~80)
- `ebm/evaluate.py`: `torch.load(checkpoint_path)` (line ~113)
- `ebm/train.py`: `torch.load(checkpoint_path)` (line ~219)

**Description:** Multiple calls to `torch.load()` without `weights_only=True`. While `generate_samples.py` and `train_tinystories.py` correctly use `weights_only=True`, the majority of files do not. `torch.load()` with `weights_only=False` (the default) uses `pickle` under the hood, allowing arbitrary code execution if the checkpoint file is crafted maliciously.

**Impact:** RCE if attacker controls checkpoint files.

**Recommended Fix:** Add `weights_only=True` to all `torch.load()` calls:
```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
```
Note: This requires checkpoints to be saved using `model.state_dict()` (not full model objects).

---

### [H-02] HIGH — `pickle.load()` on Dataset Cache (M2M)

**File:** `m2m/dataset_transformer.py` (line ~35)

**Description:** The dataset transformer uses `pickle.load(f)` to load cached datasets:
```python
return pickle.load(f)
```
There is no integrity verification (unlike the index loading which has HMAC). If the cache file is poisoned, this leads to RCE.

**Impact:** RCE if attacker can write to the dataset cache directory.

**Recommended Fix:**
1. Replace pickle with `safetensors` or `numpy.savez`/`numpy.loadz`.
2. If pickle is necessary, add HMAC verification identical to `save_index`/`load_index`.

---

### [H-03] HIGH — Incomplete Path Traversal Prevention (M2M)

**Files:**
- `m2m/storage/persistence.py`, `__init__()` and `backup()` methods

**Description:** The path traversal check only looks for `..` in parts:
```python
if ".." in Path(storage_path).parts:
    raise ValueError(...)
```
This can be bypassed with:
- URL-encoded paths: `%2e%2e`
- Null bytes: `foo\x00../etc/passwd`
- Windows alternate data streams: `file::$DATA`
- Symlinks pointing outside the allowed directory
- `backup_path` in the `restore()` method has **no traversal check at all**:
  ```python
  def restore(self, backup_path: str):
      shutil.rmtree(str(self.storage_path), ignore_errors=True)  # Destructive!
      shutil.copytree(backup_path, str(self.storage_path))  # No validation!
  ```

**Impact:** Arbitrary file read/write/deletion via path traversal.

**Recommended Fix:**
1. After `Path.resolve()`, verify the result is under an allowed base directory.
2. Add validation to `restore()`:
   ```python
   def restore(self, backup_path: str):
       resolved = Path(backup_path).resolve()
       if not str(resolved).startswith(str(self.storage_path.parent)):
           raise ValueError("Backup path outside allowed directory")
   ```
3. Use `os.path.realpath()` to resolve symlinks before checking.

---

### [M-01] MEDIUM — Default HMAC Secret is Hardcoded (M2M)

**File:** `m2m/storage/persistence.py`, `save_index()` and `load_index()`

**Description:** Already detailed in C-01. The default secret `"m2m-default-hmac-secret-change-in-production"` is committed to the source code.

**Recommended Fix:** See C-01. Fail fast if environment variable is missing.

---

### [M-02] MEDIUM — Unbounded Memory Allocation in `HistoryBuffer` (EBM)

**File:** `ebm/soc.py`, `HistoryBuffer.__init__()`

**Description:** The `HistoryBuffer` pre-allocates `torch.zeros(capacity, latent_dim)` tensors in CPU memory. With default `capacity=10000` and `latent_dim=640`, this allocates ~25MB. However, there is no validation that `capacity` is reasonable:
```python
def __init__(self, capacity: int = 10000, latent_dim: int = 640):
    self.states = torch.zeros(capacity, latent_dim)  # No bounds check
```
If called with a maliciously large capacity (e.g., from an API parameter), this could exhaust memory.

**Recommended Fix:** Add a maximum capacity limit:
```python
MAX_BUFFER_CAPACITY = 100_000
def __init__(self, capacity=10000, latent_dim=640):
    if capacity > MAX_BUFFER_CAPACITY:
        raise ValueError(f"capacity {capacity} exceeds maximum {MAX_BUFFER_CAPACITY}")
```

---

### [M-03] MEDIUM — SQL Injection via `filter_by_metadata` (M2M)

**File:** `m2m/storage/persistence.py`, `filter_by_metadata()`

**Description:** While the current implementation fetches all rows and filters in Python (safe from SQL injection), the method accepts arbitrary `filter_dict` with operator keys like `$eq`, `$gt`, etc. This is not SQL injection per se, but the pattern of accepting user-provided filter operators could lead to issues if the implementation is later optimized to use SQL WHERE clauses without proper parameterization.

**Current Risk:** Low (filters in Python).  
**Future Risk:** Medium if someone "optimizes" this to raw SQL.

**Recommended Fix:** Add a comment warning against converting to raw SQL, or implement with parameterized queries from the start.

---

### [M-04] MEDIUM — `shell=True` in `final_diagnostic.py` (EBM)

**File:** `ebm/final_diagnostic.py`, `run_command()`

**Description:** The `run_command` function uses `shell=True`:
```python
result = subprocess.run(cmd, shell=True, ...)
```
While `cmd` is currently only called with hardcoded lists like `["python", "--version"]`, the function signature accepts arbitrary `cmd`. If this function is ever exposed to user input, it becomes a command injection vector.

**Recommended Fix:** Use `shell=False` (already done in `diagnostic_launcher.py` — inconsistency).

---

### [M-05] MEDIUM — Splat Count Growth Without Upper Bound Validation (EBM)

**File:** `ebm/splats.py`, `add_splat()`

**Description:** While `add_splat()` checks `n_active >= max_splats`, the `n_active` counter is a plain integer that could be manipulated if `add_splat()` is called concurrently without proper locking. Additionally, the `normalize()` method operates on `self.mu.data[:self.n_active]` — if `n_active` exceeds the actual parameter size, this could lead to out-of-bounds access.

**Recommended Fix:** Add bounds checking:
```python
def normalize(self):
    with torch.no_grad():
        n = min(self.n_active, self.max_splats)
        self.mu.data[:n] = normalize_sphere(self.mu.data[:n])
```

---

### [M-06] MEDIUM — Error Messages Leak Internal Paths (Both Projects)

**Files:** Multiple files across both projects

**Description:** Several error messages include full file system paths:
```python
raise ValueError(f"HMAC verification failed for index {name}. Possible tampering.")
# and
f"Path traversal detected in storage_path: {storage_path}"
```
If these errors propagate to API responses, they reveal server directory structure.

**Recommended Fix:** Use generic error messages in user-facing contexts; log detailed paths server-side only.

---

### [M-07] MEDIUM — Race Condition in WAL Write (M2M)

**File:** `m2m/storage/wal.py`

**Description:** The WAL uses `threading.Lock()` for single-process safety, but the file operations use Python's `open()` in append mode. If the process crashes between `self._file.write()` and `self._file.flush()`, data could be lost. The `sync_interval` mechanism batches flushes, meaning up to `sync_interval` entries could be lost on crash.

**Recommended Fix:** Ensure `fsync()` is called on critical operations, or document the data loss window in the API contract.

---

### [L-01] LOW — Hardcoded File Paths in Training Scripts (EBM)

**File:** `ebm/train_tinystories_fast.py`

```python
TINYSTORIES_TRAIN = "D:/datasets/ebm/tinystories_train.txt"
TINYSTORIES_VAL = "D:/datasets/ebm/tinystories_val.txt"
```

**Description:** Hardcoded Windows paths. Not a security vulnerability per se, but reveals internal directory structure and won't work on other machines.

**Recommended Fix:** Use `pathlib.Path` with relative paths or environment variables.

---

### [L-02] LOW — No Input Validation on API Query Parameters (M2M)

**File:** `m2m/api/coordinator_api.py`, `m2m/api/edge_api.py`

**Description:** While the coordinator has API key validation, individual query parameters (k, vector dimensions, etc.) lack explicit bounds checking. Extremely large `k` values could cause performance degradation.

**Recommended Fix:** Add `@field_validator` constraints on query models.

---

### [L-03] LOW — `divmod` / Division by Zero in Energy Functions (EBM)

**File:** `ebm/energy.py`, `ebm/energy_cuda.py`

**Description:** Several divisions use `.clamp(min=1e-8)` to prevent division by zero, but there are edge cases where clamped values could lead to numerical instability rather than outright crashes:
```python
energy = -torch.logsumexp(weighted_exponent, dim=-1)
```
If all exponents are `-inf` (e.g., all alpha values are zero), `logsumexp` returns `-inf`, making energy `+inf`.

**Recommended Fix:** Add explicit NaN/Inf checks after energy computation.

---

### [L-04] LOW — FAISS Import Fallback to Brute Force (M2M)

**File:** `m2m/splats.py` (M2M version), `find_neighbors()`

**Description:** When FAISS is not available, falls back to `torch.cdist()` which computes all pairwise distances. With large datasets, this is O(n*m) memory and time. Not a security issue, but could be exploited for DoS by inserting many vectors.

**Recommended Fix:** Implement approximate search fallback (e.g., random projection LSH).

---

### [L-05] LOW — Missing `weights_only=True` in `evaluate_embeddings.py` (M2M)

**File:** `m2m/evaluate_embeddings.py`

```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

**Description:** Explicitly uses `weights_only=False`. Should use `True` unless full model objects are stored.

**Recommended Fix:** Change to `weights_only=True` and update checkpoint format.

---

### [I-01] INFO — Good: API Key Authentication (M2M)

**File:** `m2m/api/coordinator_api.py`

**Description:** The coordinator API implements bearer token authentication with environment variable configuration. SSRF prevention for edge URLs is also implemented with scheme and IP validation.

**Note:** This is a positive finding — security measures already in place.

---

### [I-02] INFO — Good: Path Traversal Prevention Attempted (M2M)

**File:** `m2m/storage/persistence.py`

**Description:** The code includes explicit comments about path traversal fixes (H-01, P-04). While the implementation has gaps (see H-03), the awareness and attempt at mitigation is positive.

---

### [I-03] INFO — Good: `weights_only=True` in Some EBM Files (EBM)

**Files:** `ebm/generate_samples.py`, `ebm/train_tinystories.py`

**Description:** These files correctly use `weights_only=True` for `torch.load()`. The pattern exists and should be applied consistently.

---

## Priority Remediation Plan

### Immediate (Before Any Deployment)
1. **C-01:** Remove hardcoded HMAC default, fail on missing env var
2. **H-01:** Add `weights_only=True` to all `torch.load()` calls in EBM
3. **H-02:** Add HMAC verification or replace pickle in dataset transformer

### Short-Term (Within 1 Sprint)
4. **H-03:** Harden path traversal checks, especially in `restore()`
5. **M-04:** Remove `shell=True` from `final_diagnostic.py`
6. **M-06:** Sanitize error messages

### Medium-Term (Within 1 Month)
7. **M-02:** Add buffer capacity limits
8. **M-05:** Add bounds checking in splat storage
9. **M-07:** Document WAL data loss window
10. All LOW findings

---

## Methodology

- **Static Analysis:** Manual review of all Python source files
- **Pattern Matching:** Regex search for dangerous patterns (`eval`, `exec`, `pickle`, `torch.load`, `subprocess`, `os.system`, `open()`, path operations)
- **Dependency Review:** No `requirements.txt` was audited (recommend running `pip-audit` or `safety` separately)
- **Dynamic Testing:** Not performed (recommend fuzzing API endpoints)

---

*This audit covers source code only. Infrastructure security (network, OS, access control) was not in scope.*
