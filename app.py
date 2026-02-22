import streamlit as st
import json
import os
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple

DATA_DIR = "data"
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
PLANS_PATH = os.path.join(DATA_DIR, "plans.json")


# -------------------------
# Storage
# -------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def ensure_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SETTINGS_PATH):
        save_json(SETTINGS_PATH, {
            "last_open_date": None,
            "parallel_high": 0.7,
            "parallel_mid": 0.4,
            "spread_days": 3,
            "subjects": ["영어", "수학", "국어"]
        })
    if not os.path.exists(PLANS_PATH):
        save_json(PLANS_PATH, {"plans_by_date": {}})

def get_plan(plans: dict, d: str) -> List[dict]:
    return plans.get("plans_by_date", {}).get(d, {"tasks": []}).get("tasks", [])

def set_plan(plans: dict, d: str, tasks: List[dict]):
    if "plans_by_date" not in plans:
        plans["plans_by_date"] = {}
    plans["plans_by_date"][d] = {"tasks": tasks}


# -------------------------
# Utils
# -------------------------
def completion_rate(tasks: List[dict]) -> float:
    if not tasks:
        return 1.0
    done = sum(1 for t in tasks if t.get("done") is True)
    return done / len(tasks)

def sort_tasks(tasks: List[dict]) -> List[dict]:
    def key(t):
        origin = 0 if t.get("origin", "today") == "backlog" else 1
        pri = int(t.get("priority", 3))
        est = int(t.get("est_min", 0))
        return (origin, -pri, est, t.get("subject", ""), t.get("title", ""))
    return sorted(tasks, key=key)

def pick_by_minutes(tasks: List[dict], budget_min: int) -> Tuple[List[dict], List[dict]]:
    picked, left = [], []
    total = 0
    for t in tasks:
        m = int(t.get("est_min", 0))
        if total + m <= budget_min:
            picked.append(t)
            total += m
        else:
            left.append(t)
    return picked, left

def spread_into_future(plans: dict, start_date: str, tasks: List[dict], days: int, logs: List[str]):
    if not tasks:
        return
    days = max(1, int(days))
    d0 = datetime.strptime(start_date, "%Y-%m-%d").date()

    tasks_sorted = sorted(tasks, key=lambda t: (-int(t.get("priority", 3)), int(t.get("est_min", 0))))
    buckets = [[] for _ in range(days)]
    for i, t in enumerate(tasks_sorted):
        buckets[i % days].append(t)

    for i in range(days):
        day = str(d0 + timedelta(days=i))
        existing = get_plan(plans, day)
        merged = existing + buckets[i]
        set_plan(plans, day, sort_tasks(merged))
        if buckets[i]:
            logs.append(f"{day}: 이동 {len(buckets[i])}개")


# -------------------------
# Stats (last 7 days)
# -------------------------
def last_7days_minutes_by_subject(plans: dict, today_str: str) -> dict:
    out = {}
    d0 = datetime.strptime(today_str, "%Y-%m-%d").date()
    for i in range(7):
        ds = str(d0 - timedelta(days=i))
        tasks = get_plan(plans, ds)
        for t in tasks:
            if t.get("done") is True:
                sub = (t.get("subject") or "").strip()
                if not sub:
                    continue
                out[sub] = out.get(sub, 0) + int(t.get("est_min", 0) or 0)
    return out

def apply_understudied_boost(user_priority_1to5: dict, mins7: dict, subjects: list, boost: int = 1) -> tuple:
    mins = {s: int(mins7.get(s, 0)) for s in subjects}
    if not mins:
        return user_priority_1to5, []
    min_val = min(mins.values())
    boosted = [s for s, v in mins.items() if v == min_val]
    eff = {}
    for s in subjects:
        p = int(user_priority_1to5.get(s, 3))
        if s in boosted:
            p = min(5, p + boost)
        eff[s] = p
    return eff, boosted


# -------------------------
# Auto generator
# -------------------------
def generate_tasks_for_today(
    total_minutes: int,
    subjects: List[str],
    priorities_1to5: Dict[str, int],
    avg_minutes: Dict[str, int],
    min_tasks_each: Dict[str, int],
    logs: List[str]
) -> List[dict]:
    total_minutes = max(0, int(total_minutes))
    tasks: List[dict] = []
    used = 0

    # 최소 개수부터
    for s in subjects:
        k = int(min_tasks_each.get(s, 0))
        m = max(5, int(avg_minutes.get(s, 30)))
        for j in range(k):
            if used + m > total_minutes:
                logs.append(f"{s}: 시간 부족(최소 개수 일부 생략)")
                break
            tasks.append({
                "subject": s,
                "title": f"{s} 할 일 {j+1}",
                "est_min": m,
                "done": False,
                "priority": 3,
                "origin": "today"
            })
            used += m

    remain = total_minutes - used
    if remain <= 0:
        logs.append("남은 시간 없음")
        return tasks

    # 남은 시간 가중치 배분
    weights = {s: max(1, int(priorities_1to5.get(s, 3))) for s in subjects}
    total_w = sum(weights.values())
    alloc = {s: int(remain * (weights[s] / total_w)) for s in subjects}

    leftover = remain - sum(alloc.values())
    if leftover > 0:
        sorted_sub = sorted(subjects, key=lambda s: -weights[s])
        idx = 0
        while leftover > 0 and sorted_sub:
            alloc[sorted_sub[idx % len(sorted_sub)]] += 1
            leftover -= 1
            idx += 1

    for s in subjects:
        m = max(5, int(avg_minutes.get(s, 30)))
        cnt = alloc[s] // m
        for j in range(cnt):
            tasks.append({
                "subject": s,
                "title": f"{s} 추가 {j+1}",
                "est_min": m,
                "done": False,
                "priority": 3,
                "origin": "today"
            })

    logs.append(f"총 {len(tasks)}개 생성")
    return tasks


# -------------------------
# Replan
# -------------------------
def replan(
    today_tasks: List[dict],
    yesterday_tasks: List[dict],
    settings: dict,
    done_yesterday: str,
    which_first: str,
    manual_rate: float,
    logs: List[str]
) -> Tuple[List[dict], List[dict]]:
    backlog = [t.copy() for t in yesterday_tasks if not t.get("done")]
    for t in backlog:
        t["origin"] = "backlog"

    today_copy = [t.copy() for t in today_tasks]
    for t in today_copy:
        t["origin"] = t.get("origin", "today")

    if done_yesterday == "완료":
        logs.append("전날 완료 → 오늘 유지")
        return sort_tasks(today_copy), []

    if which_first == "오늘":
        logs.append("오늘부터 → 오늘 유지")
        return sort_tasks(today_copy), backlog

    # 전날부터
    rate = float(manual_rate)
    th_high = float(settings.get("parallel_high", 0.7))
    th_mid = float(settings.get("parallel_mid", 0.4))

    backlog_min = sum(int(t.get("est_min", 0)) for t in backlog)
    today_min = sum(int(t.get("est_min", 0)) for t in today_copy)
    total_min = backlog_min + today_min

    if rate < th_mid:
        logs.append(f"전날 진행률 {int(rate*100)}% → 전날 먼저")
        logs.append("오늘은 미래로 이동")
        return sort_tasks(backlog), today_copy

    if rate >= th_high:
        logs.append(f"전날 진행률 {int(rate*100)}% → 같이 진행(균형)")
        keep_ratio = 0.6
    else:
        logs.append(f"전날 진행률 {int(rate*100)}% → 같이 진행(전날 중심)")
        keep_ratio = 0.3

    keep_today_budget = int(total_min * keep_ratio)
    today_sorted = sorted(today_copy, key=lambda t: (-int(t.get("priority", 3)), int(t.get("est_min", 0))))
    keep_today, move_today = pick_by_minutes(today_sorted, keep_today_budget)

    logs.append(f"오늘 유지 {len(keep_today)} / 이동 {len(move_today)}")
    final_today = sort_tasks(backlog + keep_today)
    return final_today, move_today


# =========================
# UI (Minimal, clean background + canary yellow accent)
# =========================
st.set_page_config(page_title="공부 도우미", layout="wide")

# 개나리색(노랑 포인트)
CANARY = "#FFD400"          # 개나리 느낌
CANARY_SOFT = "rgba(255,212,0,0.20)"
CANARY_BORDER = "rgba(255,212,0,0.45)"

st.markdown(
    f"""
    <style>
      /* 전체 배경: 파스텔 그라데이션/원형광 없음 (깔끔하게) */
      html, body, [data-testid="stAppViewContainer"] {{
        background: #ffffff;
        color: #111827;
      }}

      .block-container {{
        padding-top: 2.6rem;
        padding-bottom: 2.2rem;
        max-width: 1100px;
      }}

      h1 {{
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
      }}

      .small {{
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 0.9rem;
      }}

      /* 카드 */
      .card {{
        background: #ffffff;
        border: 1px solid rgba(17,24,39,0.08);
        border-radius: 16px;
        box-shadow: 0 10px 28px rgba(17,24,39,0.06);
        padding: 16px;
      }}

      /* 사이드바도 깔끔하게(배경만 약간 톤 다운) */
      [data-testid="stSidebar"] {{
        background: #fafafa;
        border-right: 1px solid rgba(17,24,39,0.08);
      }}

      /* 입력/선택 */
      .stTextInput input, .stNumberInput input {{
        border-radius: 12px !important;
      }}
      .stSelectbox div[data-baseweb="select"] {{
        border-radius: 12px !important;
      }}

      /* 버튼: 주황 대신 개나리 노랑 포인트 */
      .stButton > button {{
        border-radius: 12px !important;
        border: 1px solid {CANARY_BORDER} !important;
        background: {CANARY} !important;
        color: #111827 !important;
        font-weight: 700 !important;
        padding: 0.56rem 0.9rem !important;
        box-shadow: 0 10px 18px rgba(255,212,0,0.18) !important;
      }}
      .stButton > button:hover {{
        filter: brightness(0.98);
        transform: translateY(-1px);
        transition: 0.12s ease;
      }}

      /* 다운로드 버튼도 노랑 테두리로 통일 */
      [data-testid="stDownloadButton"] > button {{
        border-radius: 12px !important;
        border: 1px solid {CANARY_BORDER} !important;
        background: #ffffff !important;
        color: #111827 !important;
        font-weight: 700 !important;
      }}

      /* 슬라이더/토글/포커스 하이라이트가 주황으로 보이는 걸 노랑으로 강제 */
      [data-baseweb="slider"] div[role="slider"] {{
        background: {CANARY} !important;
        border-color: {CANARY} !important;
      }}
      /* 슬라이더 트랙(버전별 대응) */
      [data-baseweb="slider"] div {{
        accent-color: {CANARY};
      }}

      /* 체크박스 체크 색(일부 테마에서 주황 느낌 제거) */
      input[type="checkbox"] {{
        accent-color: {CANARY};
      }}

      /* 포커스 테두리 */
      *:focus {{
        outline-color: {CANARY_BORDER} !important;
      }}

      /* Expander도 둥글게 */
      details {{
        border-radius: 12px !important;
        border: 1px solid rgba(17,24,39,0.08) !important;
        background: #ffffff !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

ensure_files()
settings = load_json(SETTINGS_PATH)
plans = load_json(PLANS_PATH)

today_str = str(date.today())
yesterday_str = str(datetime.strptime(today_str, "%Y-%m-%d").date() - timedelta(days=1))
tomorrow_str = str(datetime.strptime(today_str, "%Y-%m-%d").date() + timedelta(days=1))

if "subjects" not in settings or not isinstance(settings["subjects"], list) or len(settings["subjects"]) == 0:
    settings["subjects"] = ["영어", "수학", "국어"]
    save_json(SETTINGS_PATH, settings)

st.title("공부 도우미")
st.markdown(f'<div class="small">오늘 {today_str}</div>', unsafe_allow_html=True)

# Sidebar (minimal)
st.sidebar.markdown("### 자동 생성")
total_min = st.sidebar.number_input("공부 시간(분)", 30, 720, 240, 10)

with st.sidebar.expander("과목 관리", expanded=False):
    new_subject = st.text_input("추가", value="", placeholder="예: 생명과학2")
    a, b = st.columns(2)
    if a.button("추가"):
        s = new_subject.strip()
        if s and s not in settings["subjects"]:
            settings["subjects"].append(s)
            save_json(SETTINGS_PATH, settings)
            st.success("추가됨")

    del_subject = st.selectbox("삭제", options=["(선택)"] + settings["subjects"])
    if b.button("삭제"):
        if del_subject != "(선택)":
            settings["subjects"] = [x for x in settings["subjects"] if x != del_subject]
            save_json(SETTINGS_PATH, settings)
            st.success("삭제됨")

subjects = st.sidebar.multiselect("과목", options=settings["subjects"], default=settings["subjects"])

mins7 = last_7days_minutes_by_subject(plans, today_str)
auto_balance = st.sidebar.toggle("최근 7일 적게 한 과목 +1", value=True)

with st.sidebar.expander("최근 7일(완료 기준)", expanded=False):
    if mins7:
        for k in sorted(mins7.keys()):
            st.write(f"{k}: {mins7[k]}분")
    else:
        st.write("아직 기록 없음")

with st.sidebar.expander("과목별 설정", expanded=True):
    priorities = {}
    avgmins = {}
    mintasks = {}
    for s in subjects:
        priorities[s] = st.slider(f"{s} 우선순위", 1, 5, 3)
        avgmins[s] = st.sidebar.number_input(f"{s} 1개당 분", 5, 180, 30, 5)
        mintasks[s] = st.sidebar.number_input(f"{s} 최소 개수", 0, 10, 0, 1)

if auto_balance:
    eff_priorities, boosted_subjects = apply_understudied_boost(priorities, mins7, subjects, boost=1)
else:
    eff_priorities = priorities
    boosted_subjects = []

auto_logs: List[str] = []
if st.sidebar.button("오늘 계획 만들기"):
    generated = generate_tasks_for_today(
        total_minutes=total_min,
        subjects=subjects,
        priorities_1to5=eff_priorities,
        avg_minutes=avgmins,
        min_tasks_each=mintasks,
        logs=auto_logs
    )
    set_plan(plans, today_str, generated)
    save_json(PLANS_PATH, plans)
    st.sidebar.success("생성 완료")

if boosted_subjects:
    st.sidebar.caption("자동 +1: " + ", ".join(boosted_subjects))

if auto_logs:
    with st.sidebar.expander("생성 메모", expanded=False):
        for l in auto_logs:
            st.write(l)

# Main
today_tasks = get_plan(plans, today_str)
yesterday_tasks = get_plan(plans, yesterday_str)
rate_auto = completion_rate(yesterday_tasks)

left_col, right_col = st.columns([1.25, 0.75], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("오늘 할 일")

    if not today_tasks:
        st.caption("왼쪽에서 '오늘 계획 만들기'를 누르면 자동으로 생성돼요.")
    else:
        for i, t in enumerate(today_tasks):
            r1, r2, r3 = st.columns([0.08, 0.2, 0.72])
            t["done"] = r1.checkbox("", value=bool(t.get("done", False)), key=f"td_{i}")
            t["subject"] = r2.text_input("과목", value=t.get("subject", ""), key=f"ts_{i}", label_visibility="collapsed")
            t["title"] = r3.text_input("할 일", value=t.get("title", ""), key=f"tt_{i}", label_visibility="collapsed")

            r4, r5 = st.columns([0.5, 0.5])
            t["est_min"] = r4.number_input("분", 0, 600, int(t.get("est_min", 0)), key=f"tm_{i}")
            t["priority"] = r5.selectbox("중요도", [1, 2, 3], index=int(t.get("priority", 3)) - 1, key=f"tp_{i}")
            t["origin"] = t.get("origin", "today")

        b1, b2 = st.columns([0.5, 0.5])
        if b1.button("저장"):
            set_plan(plans, today_str, today_tasks)
            save_json(PLANS_PATH, plans)
            st.success("저장됨")

        exported = json.dumps(plans, ensure_ascii=False, indent=2).encode("utf-8")
        b2.download_button("내보내기(plans.json)", data=exported, file_name="plans.json", mime="application/json")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top: 12px;">', unsafe_allow_html=True)
    st.subheader("전날 남은 것")
    if not yesterday_tasks:
        st.caption("전날 기록이 없어요.")
    else:
        st.caption(f"전날 진행률: {int(rate_auto * 100)}%")
        left_tasks = [t for t in yesterday_tasks if not t.get("done")]
        if left_tasks:
            for t in left_tasks:
                st.write(f"- {t.get('subject','')}: {t.get('title','')} ({t.get('est_min',0)}분)")
        else:
            st.write("남은 일 없음")
    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("오늘 정리")

    done_yesterday = st.radio("전날은?", ["완료", "미완료"], horizontal=True)

    if done_yesterday == "미완료":
        which_first_ui = st.radio("순서", ["전날 먼저", "오늘 먼저"], horizontal=True)
        which_first = "전날" if which_first_ui == "전날 먼저" else "오늘"
        manual_rate = st.slider("전날 진행률", 0.0, 1.0, float(rate_auto), 0.05)

        place_backlog = None
        if which_first == "오늘":
            place_backlog = st.selectbox("전날 남은 것 처리", ["내일에 넣기", "며칠에 나눠 넣기", "자동으로 안 넣기"])
    else:
        which_first = "전날"
        manual_rate = float(rate_auto)
        place_backlog = None

    with st.expander("조정 기준(선택)", expanded=False):
        settings["parallel_high"] = st.slider("같이 진행(균형) 기준", 0.5, 0.9, float(settings.get("parallel_high", 0.7)), 0.05)
        settings["parallel_mid"] = st.slider("전날 중심 기준", 0.2, 0.8, float(settings.get("parallel_mid", 0.4)), 0.05)
        settings["spread_days"] = st.number_input("나눠 넣는 기간(일)", 1, 7, int(settings.get("spread_days", 3)))

    logs: List[str] = []
    final_today: List[dict] = []
    move_future: List[dict] = []

    if st.button("미리보기"):
        final_today, move_future = replan(
            today_tasks=today_tasks,
            yesterday_tasks=yesterday_tasks,
            settings=settings,
            done_yesterday=done_yesterday,
            which_first=which_first,
            manual_rate=manual_rate,
            logs=logs
        )

        st.markdown("정리 결과")
        if not final_today:
            st.caption("비어있음")
        else:
            for i, t in enumerate(final_today, start=1):
                tag = "전날" if t.get("origin") == "backlog" else "오늘"
                st.write(f"{i}. {tag} / {t.get('subject','')}: {t.get('title','')} ({t.get('est_min',0)}분)")

        if done_yesterday == "미완료":
            if which_first == "전날":
                if move_future:
                    spread_into_future(plans, tomorrow_str, move_future, int(settings["spread_days"]), logs)
            else:
                backlog_to_move = [t.copy() for t in move_future]
                if place_backlog == "내일에 넣기":
                    spread_into_future(plans, tomorrow_str, backlog_to_move, 1, logs)
                elif place_backlog == "며칠에 나눠 넣기":
                    spread_into_future(plans, tomorrow_str, backlog_to_move, int(settings["spread_days"]), logs)
                else:
                    logs.append("전날 남은 건 자동으로 안 넣음")

        with st.expander("메모", expanded=False):
            for l in logs:
                st.write("- " + l)

        st.divider()
        if st.button("저장하기"):
            set_plan(plans, today_str, sort_tasks(final_today))
            settings["last_open_date"] = today_str
            save_json(SETTINGS_PATH, settings)
            save_json(PLANS_PATH, plans)
            st.success("저장됨")

    st.markdown('</div>', unsafe_allow_html=True)
