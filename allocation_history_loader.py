# -*- coding: utf-8 -*-
"""
allocation_history_loader.py  –  v4
────────────────────────────────────
Fixes in v4 (on top of v3):
  - Smart header-row detection: scans first 20 rows to find the row
    that contains date/type keywords, not blindly assumes header=0.
    Handles sheets where row 0 is blank or contains a title/subtitle.
  - Wider date-column keyword list (חודשים, תקופה, חודש דיווח, etc.)
  - Clear separation of "auth error" vs "structure/parse error"
  - Debug info shown in Streamlit warnings (sheet name, detected header,
    columns found, first 3 rows) so mismatches are easy to diagnose.
  - All v3 fixes retained (RTL invisible-char stripping).
"""

from __future__ import annotations

import re
import io
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ─── Strip invisible Unicode directional / zero-width chars ──────────────────
_INVIS_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f'
    r'\u202a\u202b\u202c\u202d\u202e'
    r'\u2066\u2067\u2068\u2069'
    r'\ufeff\u00a0\u00ad]'
)

def _clean(s: object) -> str:
    """Strip invisible Unicode chars and whitespace from any string."""
    return _INVIS_RE.sub('', str(s)).strip()


# ─── Hebrew month → number ────────────────────────────────────────────────────
_HEB_MONTHS = {
    "ינואר": 1, "פברואר": 2, "מרץ": 3, "מרס": 3,
    "אפריל": 4, "מאי": 5, "יוני": 6,
    "יולי": 7, "אוגוסט": 8, "ספטמבר": 9,
    "אוקטובר": 10, "נובמבר": 11, "דצמבר": 12,
}

# ─── Sheet-name → (manager, track) ───────────────────────────────────────────
_SHEET_META: dict[str, dict] = {
    "הראל כללי":   {"manager": "הראל", "track": "כללי"},
    "הראל מנייתי": {"manager": "הראל", "track": "מנייתי"},
    # ← add entries here as new sheets arrive
}
_MANAGER_PATTERNS = [
    "הראל", "מגדל", "כלל", "מנורה", "הפניקס", "אנליסט", "מיטב",
    "ילין", "פסגות", "אלטשולר", "ברקת", "אלומות",
]
_TRACK_PATTERNS = {
    "כלל": "כללי", "כללי": "כללי",
    "מנייתי": "מנייתי", "מניות": "מנייתי",
}


def _infer_meta(sheet_name: str) -> dict:
    s = _clean(sheet_name)
    for key, meta in _SHEET_META.items():
        if _clean(key) in s:
            return meta
    manager = next((m for m in _MANAGER_PATTERNS if m in s), s)
    track = "כללי"
    for pat, val in _TRACK_PATTERNS.items():
        if pat in s:
            track = val
            break
    return {"manager": manager, "track": track}


def _extract_sheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"לא ניתן לחלץ Sheet ID מהכתובת: {url}")
    return m.group(1)


def _csv_export_url(sheet_id: str, gid: int = 0) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={gid}"
    )


# ─── Sheet tab discovery ──────────────────────────────────────────────────────

def _discover_sheet_gids(sheet_id: str, max_probe: int = 12) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []

    # Attempt 1: parse HTML edit page
    try:
        r = requests.get(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit",
            timeout=15,
        )
        html = r.text
        for gid_str, title in re.findall(
            r'"sheetId":(\d+)[^}]{0,200}?"title":"([^"]+)"', html
        ):
            found.append((_clean(title), int(gid_str)))
        if not found:
            for title, gid_str in re.findall(
                r'"title":"([^"]+)"[^}]{0,200}?"sheetId":(\d+)', html
            ):
                found.append((_clean(title), int(gid_str)))
        if not found:
            for m in re.finditer(r'\["([^"]{1,80})",null,(\d+)', html):
                found.append((_clean(m.group(1)), int(m.group(2))))
    except Exception as e:
        logger.warning(f"HTML discovery failed: {e}")

    if found:
        return found

    # Attempt 2: probe gids 0..N
    for gid in range(max_probe):
        try:
            rr = requests.get(_csv_export_url(sheet_id, gid), timeout=15)
            ct = rr.headers.get("Content-Type", "")
            if rr.status_code != 200:
                break
            if "html" in ct.lower():
                break
            if len(rr.text.strip()) > 20:
                found.append((f"גליון_{gid}", gid))
        except Exception:
            break

    return found if found else [("גליון_0", 0)]


# ─── Smart header-row detection ───────────────────────────────────────────────

# All keywords that indicate a date/period column (broad list)
_DATE_KEYWORDS = {
    "תאריך", "חודש", "חודשים", "תקופה", "חודש דיווח", "תאריך דיווח",
    "date", "month", "months", "period", "report month", "month date", "time",
}
# Keywords that indicate a row-type / period-type column (not the date itself)
_TYPE_KEYWORDS = {"סוג", "type", "kind", "תקופה_סוג"}
# Values in the type column meaning "monthly row"
_MONTH_TYPE_VALUES = {"month", "חודשי", "חודש", "monthly"}


def _row_looks_like_header(row: pd.Series) -> bool:
    """
    Return True if this row looks like a header row.
    Heuristic: at least one cell matches a date keyword AND
               at least one other cell looks like an allocation column name
               (non-numeric, non-empty, not a type keyword).
    """
    cells = [_clean(str(v)).lower() for v in row.values]
    has_date = any(
        any(kw in c for kw in _DATE_KEYWORDS)
        for c in cells
    )
    # non-numeric, non-empty cells (potential column names)
    text_cells = [c for c in cells if c and c not in ("nan", "") and not _is_numeric(c)]
    has_enough_text = len(text_cells) >= 2
    return has_date and has_enough_text


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace("%", "").replace(",", "."))
        return True
    except ValueError:
        return False


def _find_header_row(raw_no_header: pd.DataFrame, max_scan: int = 20) -> int:
    """
    Scan the first max_scan rows of a header-less DataFrame to find
    the row index that looks like the actual column-header row.
    Returns 0 if nothing is found (safe fallback).
    """
    for i, (_, row) in enumerate(raw_no_header.head(max_scan).iterrows()):
        if _row_looks_like_header(row):
            return i
    return 0


def _read_csv_smart(csv_text: str) -> pd.DataFrame:
    """
    Read a CSV string using smart header detection.
    1. Count max columns to handle ragged rows (Google Sheets title rows
       often have fewer commas than data rows).
    2. Read header=None with explicit column count.
    3. Find the actual header row by scanning content.
    4. Rebuild DataFrame with correct columns, dropping rows above header.
    """
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()

    lines = [l for l in csv_text.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()

    max_cols = max(len(l.split(",")) for l in lines)

    try:
        raw = pd.read_csv(
            io.StringIO(csv_text),
            header=None,
            dtype=str,
            names=range(max_cols),
            engine="python",
        )
    except Exception:
        try:
            raw = pd.read_csv(io.StringIO(csv_text), header=0, dtype=str)
            raw.columns = [_clean(str(c)) for c in raw.columns]
            return raw.dropna(how="all").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame()

    header_idx = _find_header_row(raw)
    new_cols = [_clean(str(v)) for v in raw.iloc[header_idx].values]
    data = raw.iloc[header_idx + 1:].copy()
    data.columns = new_cols
    data = data.dropna(how="all").reset_index(drop=True)
    return data


# ─── Column role detection ────────────────────────────────────────────────────

def _find_date_col(columns: list) -> Optional[str]:
    """
    Find the date column. Priority: exact > ends-with > contains.
    Skips columns whose name contains type/period-kind keywords.
    """
    cleaned = {c: _clean(str(c)).lower() for c in columns}

    def is_type_col(cl: str) -> bool:
        return any(tk in cl for tk in _TYPE_KEYWORDS)

    # 1. Exact match
    for c, cl in cleaned.items():
        if cl in _DATE_KEYWORDS and not is_type_col(cl):
            return c

    # 2. Ends-with
    for c, cl in cleaned.items():
        if is_type_col(cl):
            continue
        if any(cl.endswith(kw) for kw in _DATE_KEYWORDS):
            return c

    # 3. Contains
    for c, cl in cleaned.items():
        if is_type_col(cl):
            continue
        if any(kw in cl for kw in _DATE_KEYWORDS):
            return c

    return None


def _find_type_col(columns: list) -> Optional[str]:
    for c in columns:
        cl = _clean(str(c)).lower()
        if any(tk in cl for tk in _TYPE_KEYWORDS) or cl in {"סוג התאריך", "סוג_תאריך"}:
            return c
    return None


# ─── Date value parsing ───────────────────────────────────────────────────────

def _parse_date_value(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.Timestamp(val).replace(day=1).to_pydatetime()

    s = _clean(str(val))
    if not s or s.lower() in ("nan", "none", ""):
        return None

    for heb, mn in _HEB_MONTHS.items():
        if heb in s:
            y = re.search(r"(\d{4})", s)
            if y:
                return datetime(int(y.group(1)), mn, 1)

    for fmt in (
        "%Y-%m-%d", "%d/%m/%Y", "%m/%Y", "%Y-%m",
        "%b-%Y", "%B %Y", "%b %Y", "%Y/%m/%d", "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(s, fmt).replace(day=1)
        except ValueError:
            pass

    try:
        return pd.to_datetime(s, dayfirst=True).replace(day=1).to_pydatetime()
    except Exception:
        return None


# ─── Percent value parsing ────────────────────────────────────────────────────

def _parse_percent(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and np.isnan(val):
            return None
        return round(float(val) * 100 if abs(val) <= 1.5 else float(val), 4)
    s = _clean(str(val)).replace(",", ".").replace("%", "").strip()
    if not s:
        return None
    try:
        f = float(s)
        return round(f * 100 if abs(f) <= 1.5 else f, 4)
    except ValueError:
        return None


# ─── Core normaliser ──────────────────────────────────────────────────────────

def _normalise_sheet_df(
    raw: pd.DataFrame,
    sheet_name: str,
    debug_warnings: list[str],
) -> pd.DataFrame:
    """
    Convert a smart-parsed wide DataFrame to normalised long format.
    Appends human-readable debug info to debug_warnings on failure.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    meta = _infer_meta(sheet_name)

    # Column names are already _clean()'d by _read_csv_smart
    date_col = _find_date_col(list(raw.columns))
    type_col = _find_type_col(list(raw.columns))

    if date_col is None:
        debug_warnings.append(
            f"⚠️ גליון **{sheet_name}**: לא נמצאה עמודת תאריך.\n"
            f"עמודות שנמצאו: `{list(raw.columns)[:10]}`\n"
            f"5 שורות ראשונות:\n```\n{raw.head(3).to_string()}\n```"
        )
        return pd.DataFrame()

    # Filter to Month-type rows only (if type column exists)
    if type_col is not None:
        month_mask = (
            raw[type_col].astype(str).apply(_clean).str.lower()
            .isin(_MONTH_TYPE_VALUES)
        )
        if month_mask.any():
            raw = raw[month_mask].copy()

    # Allocation columns = everything except date_col, type_col, Unnamed/empty
    skip = {date_col}
    if type_col:
        skip.add(type_col)
    alloc_cols = [
        c for c in raw.columns
        if c not in skip
        and not c.startswith("Unnamed")
        and c not in ("", "nan")
    ]

    if not alloc_cols:
        debug_warnings.append(
            f"⚠️ גליון **{sheet_name}**: נמצאה עמודת תאריך (`{date_col}`) "
            f"אך לא נמצאו עמודות אלוקציה. "
            f"עמודות: `{list(raw.columns)[:10]}`"
        )
        return pd.DataFrame()

    rows = []
    for _, row in raw.iterrows():
        dt = _parse_date_value(row[date_col])
        if dt is None:
            continue
        for col in alloc_cols:
            val = _parse_percent(row[col])
            if val is None:
                continue
            rows.append({
                "manager":          meta["manager"],
                "track":            meta["track"],
                "date":             pd.Timestamp(dt),
                "year":             dt.year,
                "month":            dt.month,
                "allocation_name":  col,
                "allocation_value": val,
                "source_sheet":     sheet_name,
            })

    if not rows:
        debug_warnings.append(
            f"⚠️ גליון **{sheet_name}**: עמודות זוהו אך כל השורות נכשלו בפרסור. "
            f"עמודת תאריך: `{date_col}` | "
            f"5 ערכי תאריך: `{list(raw[date_col].head())}`"
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ─── CSV transport ────────────────────────────────────────────────────────────

def _load_sheet_via_csv(
    sheet_id: str,
    gid: int,
    sheet_name: str,
    debug_warnings: list[str],
) -> pd.DataFrame:
    url = _csv_export_url(sheet_id, gid)
    try:
        r = requests.get(url, timeout=25, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")

        # Detect auth redirect (returns HTML login page)
        if r.status_code == 401 or r.status_code == 403:
            debug_warnings.append(
                f"🔒 גליון **{sheet_name}** (gid={gid}): שגיאת הרשאה (HTTP {r.status_code}). "
                "ודא שהגיליון משותף לכולם לפחות בצפייה."
            )
            return pd.DataFrame()

        if "html" in ct.lower() or r.text.strip().lower().startswith("<!doctype"):
            debug_warnings.append(
                f"🔒 גליון **{sheet_name}** (gid={gid}): ה-CSV חזר כדף HTML — "
                "ייתכן שדרוש אישור. נסה File → Share → Publish to web → CSV."
            )
            return pd.DataFrame()

        if r.status_code != 200:
            debug_warnings.append(f"⚠️ גליון **{sheet_name}**: HTTP {r.status_code}")
            return pd.DataFrame()

        # Smart parse
        df_raw = _read_csv_smart(r.text)
        return _normalise_sheet_df(df_raw, sheet_name, debug_warnings)

    except Exception as e:
        debug_warnings.append(f"⚠️ גליון **{sheet_name}** (gid={gid}): {e}")
        return pd.DataFrame()


# ─── gspread transport ────────────────────────────────────────────────────────

def _load_via_gspread(
    sheet_url: str,
    debug_warnings: list[str],
) -> pd.DataFrame:
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)

        frames = []
        for ws in sh.worksheets():
            try:
                data = ws.get_all_values()
                if not data or len(data) < 2:
                    continue
                raw_df = pd.DataFrame(data, dtype=str)
                smart = _read_csv_smart(raw_df.to_csv(index=False, header=False))
                norm = _normalise_sheet_df(smart, ws.title, debug_warnings)
                if not norm.empty:
                    frames.append(norm)
            except Exception as e:
                debug_warnings.append(f"gspread: גליון '{ws.title}' — {e}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as e:
        debug_warnings.append(f"gspread נכשל: {e}")
        return pd.DataFrame()


# ─── Main public API ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_allocation_history(sheet_url: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and normalise all sheets from a Google Sheets URL.
    Returns (df, debug_warnings).  df is empty on full failure.
    """
    debug_warnings: list[str] = []

    if not sheet_url or not sheet_url.strip():
        return pd.DataFrame(), ["לא הוגדר קישור ל-Google Sheets"]

    # gspread first (if service account configured)
    has_sa = hasattr(st, "secrets") and "gcp_service_account" in st.secrets
    if has_sa:
        df = _load_via_gspread(sheet_url, debug_warnings)
        if not df.empty:
            return df, debug_warnings
        debug_warnings.append("gspread נכשל — עובר ל-CSV ציבורי")

    try:
        sheet_id = _extract_sheet_id(sheet_url)
    except ValueError as e:
        return pd.DataFrame(), [str(e)]

    sheets = _discover_sheet_gids(sheet_id)
    frames: list[pd.DataFrame] = []

    for name, gid in sheets:
        df_sheet = _load_sheet_via_csv(sheet_id, gid, name, debug_warnings)
        if not df_sheet.empty:
            frames.append(df_sheet)

    if not frames:
        return pd.DataFrame(), debug_warnings + ["לא נטענו נתונים מאף גליון"]

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(
        ["manager", "track", "allocation_name", "date"]
    ).reset_index(drop=True)
    return df, debug_warnings
