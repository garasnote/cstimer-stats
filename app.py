import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="cstimer Stats",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.penalty-plus2  { color: #f0c060; font-weight: bold; }
.penalty-dnf    { color: #e06060; font-weight: bold; }
.pb-badge       { color: #60c060; font-weight: bold; }
.metric-card    { background:#1e1e2e; border-radius:10px; padding:14px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ms_to_str(ms, penalty=0):
    """Convert milliseconds + penalty flag to display string."""
    if penalty == -1:
        return "DNF"
    total = ms + (2000 if penalty == 2000 else 0)
    s = total / 1000
    if s >= 60:
        m = int(s // 60)
        sec = s - m * 60
        return f"{m}:{sec:06.3f}"
    return f"{s:.3f}"

def effective_ms(ms, penalty):
    """Return None for DNF, else adjusted ms."""
    if penalty == -1:
        return None
    return ms + (2000 if penalty == 2000 else 0)

def rolling_average(times_ms, n):
    """Ao-N: drop best & worst in window, average the rest. Returns None if any DNF in window."""
    out = [None] * len(times_ms)
    for i in range(n - 1, len(times_ms)):
        window = times_ms[i - n + 1 : i + 1]
        if None in window:
            out[i] = None
            continue
        trimmed = sorted(window)[1:-1]
        out[i] = sum(trimmed) / len(trimmed)
    return out

def parse_sessions(data):
    """Parse cstimer JSON into a dict of DataFrames keyed by session name."""
    session_props = {}
    try:
        sp = json.loads(data.get("properties", {}).get("sessionData", "{}"))
        for k, v in sp.items():
            session_props[k] = v.get("name", k)
    except Exception:
        pass

    frames = {}
    for key, solves in data.items():
        if not key.startswith("session") or not isinstance(solves, list) or len(solves) == 0:
            continue
        sid = key.replace("session", "")
        raw_name = session_props.get(sid, key)
        # sessionData stores numeric names as integers (e.g. name:1 means "Session 1")
        if isinstance(raw_name, int):
            name = f"Session {raw_name}"
        else:
            name = str(raw_name)
        rows = []
        for idx, s in enumerate(solves):
            try:
                # cstimer has two export formats:
                # old: [penalty, time_ms, scramble, comment, timestamp]
                # new: [[penalty, time_ms], scramble, comment, timestamp]
                if isinstance(s[0], list):
                    penalty, time_ms = s[0][0], s[0][1]
                    scramble = s[1]
                    comment  = s[2] if len(s) > 2 else ""
                    ts       = s[3] if len(s) > 3 else None
                else:
                    penalty  = s[0]
                    time_ms  = s[1]
                    scramble = s[2]
                    comment  = s[3] if len(s) > 3 else ""
                    ts       = s[4] if len(s) > 4 else None
                eff      = effective_ms(time_ms, penalty)
                rows.append({
                    "solve_num" : idx + 1,
                    "time_ms"   : time_ms,
                    "penalty"   : penalty,
                    "eff_ms"    : eff,
                    "time_str"  : ms_to_str(time_ms, penalty),
                    "scramble"  : scramble,
                    "comment"   : comment,
                    "timestamp" : datetime.fromtimestamp(ts) if ts else None,
                    "is_dnf"    : penalty == -1,
                    "is_plus2"  : penalty == 2000,
                })
            except Exception:
                continue
        if rows:
            frames[name] = pd.DataFrame(rows)
    return frames

# ── Scramble visualizer ───────────────────────────────────────────────────────

FACE_COLORS = {
    'U': '#ffffff',  # white
    'D': '#ffff00',  # yellow
    'F': '#00cc00',  # green
    'B': '#0000cc',  # blue
    'L': '#ff8800',  # orange
    'R': '#cc0000',  # red
}

def init_cube():
    """Returns cube state as dict face -> 3x3 list of colors."""
    return {f: [[FACE_COLORS[f]]*3 for _ in range(3)] for f in 'UDLRFB'}

def rotate_face_cw(face):
    return [[face[2-j][i] for j in range(3)] for i in range(3)]

def rotate_face_ccw(face):
    return [[face[j][2-i] for j in range(3)] for i in range(3)]

def rotate_face_180(face):
    return [row[::-1] for row in face[::-1]]

def apply_move(cube, move):
    """Apply a single WCA move to the cube state."""
    base = move.rstrip("'2")
    suffix = move[len(base):]
    times = 2 if suffix == '2' else (3 if suffix == "'" else 1)

    for _ in range(times):
        c = {f: [row[:] for row in cube[f]] for f in cube}
        if base == 'U':
            cube['U'] = rotate_face_cw(c['U'])
            cube['F'][0], cube['R'][0], cube['B'][0], cube['L'][0] = \
                c['R'][0][:], c['B'][0][:], c['L'][0][:], c['F'][0][:]
        elif base == 'D':
            cube['D'] = rotate_face_cw(c['D'])
            cube['F'][2], cube['L'][2], cube['B'][2], cube['R'][2] = \
                c['L'][2][:], c['B'][2][:], c['R'][2][:], c['F'][2][:]
        elif base == 'R':
            cube['R'] = rotate_face_cw(c['R'])
            for r in range(3):
                cube['U'][r][2]   = c['F'][r][2]
                cube['F'][r][2]   = c['D'][r][2]
                cube['D'][r][2]   = c['B'][2-r][0]
                cube['B'][2-r][0] = c['U'][r][2]
        elif base == 'L':
            cube['L'] = rotate_face_cw(c['L'])
            for r in range(3):
                cube['U'][r][0]   = c['B'][2-r][2]
                cube['B'][2-r][2] = c['D'][r][0]
                cube['D'][r][0]   = c['F'][r][0]
                cube['F'][r][0]   = c['U'][r][0]
        elif base == 'F':
            cube['F'] = rotate_face_cw(c['F'])
            for i in range(3):
                cube['U'][2][i] = c['L'][2-i][2]
                cube['R'][i][0] = c['U'][2][i]
                cube['D'][0][2-i] = c['R'][i][0]
                cube['L'][2-i][2] = c['D'][0][2-i]
            for i in range(3):
                cube['U'][2][i] = c['L'][2-i][2]
                cube['L'][2-i][2] = c['D'][0][2-i]
                cube['D'][0][2-i] = c['R'][i][0]
                cube['R'][i][0] = c['U'][2][i]
        elif base == 'B':
            cube['B'] = rotate_face_cw(c['B'])
            for i in range(3):
                cube['U'][0][2-i] = c['R'][i][2]
                cube['L'][i][0] = c['U'][0][2-i]
                cube['D'][2][i] = c['L'][i][0]  
                cube['R'][2-i][2] = c['D'][2][i]
            for i in range(3):
                cube['U'][0][2-i] = c['R'][i][2]
                cube['R'][i][2] = c['D'][2][2-i]
                cube['D'][2][2-i] = c['L'][2-i][0]
                cube['L'][2-i][0] = c['U'][0][i]

def scramble_cube(scramble_str):
    """Apply scramble string and return cube state."""
    cube = init_cube()
    if not scramble_str or scramble_str.strip() == "":
        return cube
    moves = scramble_str.strip().split()
    for m in moves:
        # skip wide moves (Uw, Fw, Rw) for 3x3 visualization
        if any(c.islower() for c in m) or m.startswith(('Uw','Fw','Rw','Lw','Bw','Dw')):
            continue
        try:
            apply_move(cube, m)
        except Exception:
            continue
    return cube

def draw_cube_net(cube):
    """Draw the cube net as a Plotly figure."""
    # Layout: 4 rows, 3 cols of faces
    #         col0  col1  col2  col3
    # row0:         U
    # row1:   L     F     R     B
    # row2:         D
    face_positions = {
        'U': (0, 1), 'L': (1, 0), 'F': (1, 1),
        'R': (1, 2), 'B': (1, 3), 'D': (2, 1),
    }
    cell = 40  # px per sticker
    gap  = 4   # gap between faces
    size = 3

    shapes = []
    annotations = []

    for face, (row, col) in face_positions.items():
        ox = col * (size * cell + gap)
        oy = row * (size * cell + gap)
        for r in range(size):
            for c in range(size):
                color = cube[face][r][c]
                x0 = ox + c * cell + 2
                y0 = oy + r * cell + 2
                shapes.append(dict(
                    type='rect',
                    x0=x0, y0=y0, x1=x0+cell-4, y1=y0+cell-4,
                    fillcolor=color,
                    line=dict(color='#333', width=1.5),
                ))

    total_w = 4 * (size * cell + gap)
    total_h = 3 * (size * cell + gap)

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        xaxis=dict(range=[0, total_w], visible=False, scaleanchor='y'),
        yaxis=dict(range=[total_h, 0], visible=False),
        width=total_w + 20,
        height=total_h + 20,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# ── Statistics helpers ────────────────────────────────────────────────────────

def compute_stats(df):
    valid = df[~df['is_dnf']]['eff_ms'].dropna()
    if len(valid) == 0:
        return {}

    ao5  = rolling_average(df['eff_ms'].tolist(), 5)
    ao12 = rolling_average(df['eff_ms'].tolist(), 12)
    ao100= rolling_average(df['eff_ms'].tolist(), 100)

    # PB single (best non-DNF)
    pb_single = valid.min()
    # Running PB
    df = df.copy()
    df['running_pb'] = df['eff_ms'].expanding().min()

    # 95% CI over time (rolling window of 12)
    cis_lo, cis_hi = [], []
    for i in range(len(df)):
        window = df['eff_ms'].iloc[max(0,i-11):i+1].dropna().values
        if len(window) >= 2:
            m, se = np.mean(window), stats.sem(window)
            lo, hi = stats.t.interval(0.95, len(window)-1, loc=m, scale=se)
            cis_lo.append(lo); cis_hi.append(hi)
        else:
            cis_lo.append(None); cis_hi.append(None)
    df['ci_lo'] = cis_lo
    df['ci_hi'] = cis_hi
    df['ao5']   = ao5
    df['ao12']  = ao12
    df['ao100'] = ao100

    return df, pb_single

# ── Main app ──────────────────────────────────────────────────────────────────

st.title("🧩 cstimer Stats Dashboard")
st.caption("Upload your cstimer export (JSON) to explore your solves.")

# ── Sidebar: file upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Load Data")
    uploaded = st.file_uploader("Drop your cstimer JSON export", type=["json","txt"],
                               help="Export from cstimer: Options → Export. File is usually < 100KB.")
    
    # Fall back to embedded demo data
    use_demo = st.checkbox("Use embedded demo data", value=(uploaded is None))

    st.markdown("---")
    st.header("⚙️ Options")
    show_dnf = st.checkbox("Show DNF solves in table", value=True)
    ao_lines = st.multiselect("Rolling averages to show", ["Ao5","Ao12","Ao100"], default=["Ao5","Ao12"])
    show_ci  = st.checkbox("Show 95% Confidence Interval", value=True)
    show_pb  = st.checkbox("Show PB progression", value=True)

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded:
    raw = json.load(uploaded)
elif use_demo:
    # Embedded demo — the user's own data
    raw = {"session1":[[[0,16065],"L2 B' L' U2 F2 L D2 R' B2 L2 R' F2 U' R B2 U2 L2 F D'","",1772811889],[[0,17255],"D L2 R2 D B2 U B2 L2 F2 R' F' R2 U B L' D' R' D' B","",1772812936],[[0,16125],"R F2 L2 D' R2 B2 F2 D U2 L2 D L2 F' L' D U' F R' B' D F2","",1772822020],[[0,14675],"L2 D2 L2 U2 B' R2 B' D2 B2 D2 F' D' L2 F' U R D F L' D2 R","",1772830654],[[0,14721],"B L' B2 L2 F' L2 B' D2 R2 D2 L2 B' F' L D F' L' D B2 F2","",1772830724],[[0,16611],"B U' R2 U F2 D' F2 U2 B2 L2 U' L2 D2 B F' L U L' U R2 D'","",1772830760],[[0,15991],"B U F2 L' F2 B2 L' U' R' U2 B2 L2 D2 L2 D2 F L2 F R2","",1772830798],[[0,15116],"D2 L2 F L2 F' U2 F L2 B' D2 F' U2 R' F D' U' L D2 F2 L F'","",1772830978],[[0,15745],"R' U' F U2 B2 R' B' L U F' D2 F' U2 L2 F' R2 U2 R2 L2 B'","",1772831076],[[0,13824],"D' B2 D' L2 U B2 L2 F2 D2 F2 D B L B' R2 U B' F D' L2 F'","",1772831131],[[0,16433],"U' B2 R2 F2 U' F2 U R2 D2 F D' B U L F2 L D2 L","",1772831165],[[0,18479],"U2 F' L2 U2 F2 L2 F2 R2 D2 F2 D' F2 R2 U L' U2 F' D' F R U","",1772831207],[[0,14980],"D2 L2 B2 L2 R2 F D2 R2 B' D2 B U' R' F L' F2 D F2 R2 F L2","",1772831370],[[2000,15990],"L D' F' L' D2 B2 R2 D2 R2 F2 L F2 R D2 U' L2 R2 F' D B D2","",1772831464],[[0,17229],"L D' F' L' D2 B2 R2 D2 R2 F2 L F2 R D2 U' L2 R2 F' D B D2","",1772831971],[[0,17696],"D L' D2 B2 L' F' D2 L U2 B2 L2 U R2 D' R2 U B2 D L2 D' R","",1772832014],[[0,21016],"B2 L2 B2 F2 R U2 F2 L' U2 L' B2 L2 U' F L F L B F L' U","",1772832088],[[0,22252],"R2 D' U2 B2 D' R2 U L2 U' L2 F2 L D F2 L' B' D2 F R F L","",1772832138],[[0,16694],"L2 B' R2 U2 L2 R2 F' L2 F L R U B U R' B2 U L' R2","",1772832187],[[0,17528],"D2 F' D' B U2 B' U' R B' U2 R2 B2 D2 L D2 F2 D2 R2 U2 L","",1772839632],[[0,16076],"L2 B2 R2 B' U' F R' L2 D2 F' R2 B' D2 B' U2 F U2 F L' B'","",1772839676],[[0,18348],"U2 F L2 D2 U2 F2 R2 D2 R' B2 R' D2 B2 R2 D' R' U2 B' R U2 L","",1772839719],[[0,14033],"B' L2 B' D2 B U2 F D2 F2 R2 F' U2 L B2 D R' D' U2 R D2 F'","",1773138421],[[0,13152],"U2 R U2 L2 R B2 R' B2 D2 L2 U2 F L2 D' B' R B2 D2 R F","",1773138456],[[0,18496],"U' F2 R U2 L' B2 D2 R' D2 R' D2 F2 R B L2 D F' R' B F L'","",1773138490],[[0,15676],"D R2 D' F2 D' R2 U B2 U R2 F2 L' U B' F2 L' B R' F' L2 R2","",1773138546],[[0,15747],"U2 F' R' B D2 L2 F U R' U2 F2 L' F2 R F2 L' D2 B2 R' F2 U","",1773140652],[[0,17015],"R' B' L D R2 F2 R2 L F B2 U R2 D' F2 D2 R2 D L2 D R2 U","",1773214372],[[0,19277],"R2 D' U2 B2 D F2 R2 B2 R2 F2 U' F' U' L' U' B' U' R D' F","",1773214560],[[0,15877],"D2 B U2 B R2 B' L2 B' D2 L2 U2 L2 R' F' L2 D F' U' B2 R2 F","",1773214616],[[0,16973],"F' L2 D2 B2 L2 B2 L2 U L2 D' B2 R2 U' L B D' U2 B2 U2 L B","",1773214976],[[0,13985],"L2 F' U' B R' D2 L D' L' B R2 U2 B' D2 B' U2 D2 R2 B2 R2","",1773216362],[[0,16011],"F2 D B2 U' L2 U' B2 F2 U2 F2 U' B' D U2 F' U' B D L' D2 R","",1773216400],[[0,17208],"R2 D R' B2 U2 R D2 L2 U2 B2 R U2 R' U2 B U R D2 R F2 R","",1773216714],[[0,14369],"B2 F2 D' U' L2 D R2 B2 U' B2 F' L B2 D2 F U B L2 R'","",1773216753],[[0,19297],"L F2 U2 B2 U2 L2 R2 F' D2 R2 B F D R' B' D2 F2 R' F2 U'","",1773356647],[[0,15011],"B2 D2 B2 L2 U B2 D' F2 R2 F' D' R' B' D U L2 B' R2 F2","",1773356692],[[0,11198],"F' R D L2 F R' B D R2 F2 L U2 L F2 B2 U2 F2 L' B2 R","",1773356793],[[0,14456],"B D F2 U' D2 B' L U2 F2 B2 U B2 D L2 U B2 U' F2 L B' U2","",1773357098],[[0,14024],"U' F B2 U2 B2 D' B2 U F2 L2 U R2 U' F2 R' F L B2 U' B2 F'","",1773357151],[[0,20574],"B2 L2 U' L2 R2 U' F2 D R2 U B2 U B' F' L F D2 R D F","",1773357195],[[0,21318],"L' R' D2 L' F2 R' D2 U2 B2 F2 L2 R' B' U' B2 L' D R U2 R2 F","",1773357319],[[0,17072],"L2 D2 F2 R2 U' L2 D U' B2 L R' U' L F' L2 U' B D' U","",1773357367],[[0,19557],"F2 U2 B2 R2 F2 R2 U F2 U2 R2 U' F2 L F' U' B U2 B' L2 D2 F'","",1773357412],[[0,17779],"F R' L2 B2 D' F2 D2 L2 F2 L2 R2 U2 F2 R' F L2 R D B L' F'","",1773357667],[[0,15819],"R F L2 F' L2 D2 L2 U2 B2 U2 B' R2 B D' R U' L' U' B R' U","",1773357711],[[0,20241],"L2 D' L2 F2 L2 U B2 U' R2 U' B' U' L B' F D L R2 B' U2","",1773357825],[[0,18585],"D F2 R2 D2 L2 U' L2 F2 U2 F2 U F' R' U L' U2 F D L R2 F2","",1773357867],[[0,19013],"R2 F2 L2 R2 F L2 U2 B F2 L2 F U2 L' R U' L' R2 U' B' F2 R'","",1773413325],[[0,18723],"F2 L2 U2 F L2 B' U2 B' U2 F2 L2 U2 R' B F U' B2 L B2 F","",1773413383],[[0,13375],"F2 R2 D2 F2 L2 R U2 R' F2 R2 D2 B2 U' F' R D2 B R2 U2 L' D","",1773413492],[[0,18379],"L2 D2 F R2 F' D2 B L2 D2 F' L2 B2 D' R' F2 U2 R2 D' B' L2 R'","",1773413532],[[0,14553],"U L' F2 R U D' B' L U B2 R2 D' R2 D F2 B2 R2 D2 L2 D","",1773413732],[[0,16083],"F2 D2 B2 U2 L' D2 L B2 L B2 U2 F2 U B R U B R U B2 D","",1773413782],[[0,19644],"U2 R B2 R2 B2 D' F2 D' B2 F2 U' R2 D B' L' R2 D B U B2","",1773413825],[[0,13526],"F R' D L' U2 D2 F U2 B U' L2 U2 R2 B2 U' L2 U2 R2 D F2","",1773414259],[[0,15077],"D B F2 R2 U L2 U B2 D' L2 F2 L2 U2 F2 R F2 U' R B' L2 F","",1773414326],[[0,16457],"D2 R L2 B U' F2 U' F2 D2 F2 R B2 R B2 D2 L F2 D' L","",1773414360],[[0,17748],"F2 U2 R' D2 R' D2 F2 R' D2 L R' F' L F U' L2 U' L2 F' R'","",1773414400],[[0,17051],"D2 F' R U F D2 R U' B2 R2 D' F2 U' R2 L2 B2 L2 B2 D2 R'","",1773414460],[[0,16651],"U F2 U2 B2 D2 R' D2 R F2 R' F2 L2 U2 D B L D B' F' L2 R2","",1773442808],[[0,16365],"B2 U B2 R2 U2 L' B2 U2 F2 U2 R' U2 B2 L' B' R D' U' B' R' D'","",1773443027],[[0,15587],"L F' B2 D' L2 B' L' D2 R' L2 B2 U R2 U' L2 D' R2 L2 B2 D2","",1773443114],[[0,14228],"U L' U' L2 U' F2 L2 R2 D2 R2 F2 R2 U L D2 B D2 R' B L","",1773443155],[[0,16229],"F2 L' R2 D' L2 F2 D L2 B2 D2 U' L2 D2 F' L R D2 B' R B' D2","",1773443271],[[0,14045],"L2 U F2 L2 U2 R2 U' R2 F2 D' B2 F L2 D' B' R U2 B' F2 U","",1773443313],[[2000,14771],"B D L F2 D' F R B' L F2 R' B2 L' D2 R' F2 R U2 F2 B","",1773443583],[[0,16649],"F2 U2 F2 U R2 B2 D' F2 D R' B2 L D L F' D' R D2 U'","",1773443725],[[0,18895],"U2 L D R2 F2 R F' B2 L D F2 R2 D R2 U2 F2 U' L2 D B2","",1773443769]],"session2":[[[0,103466],"U' B2 R F2 R2 D R2 U2 D F2 B2 D' R U2 D' F' D F B' Uw2 Fw2 Rw2 B' D F2 Rw2 F' Uw2 F' U' F2 Rw F2 D Fw2 Rw2 F2 Fw U Rw' Fw2 L' Fw' U","",1772839802],[[0,97849],"R' U2 D' L' B' D2 F2 U' L D2 B' U2 F L2 F2 R2 L2 F' L2 U2 Rw2 F' L2 Uw2 Rw2 B L Uw2 B2 R2 L' F Uw' R' Rw2 U' R' D2 Rw' U B2 Uw' F L2 Fw2","",1772839948]]}
else:
    st.info("👆 Upload a cstimer JSON export to get started, or check **Use embedded demo data**.")
    st.stop()

# ── Demo warning banner ──────────────────────────────────────────────────────
if not uploaded and use_demo:
    st.info(
        "👀 **You're viewing demo data** — these are real solves, but they're not yours!  \n"
        "To see your own stats, export from cstimer: **Options → Export** (it's a tiny JSON file, "
        "usually under 100KB), then upload it using the panel on the left.",
        icon="ℹ️",
    )

# ── Parse ─────────────────────────────────────────────────────────────────────
sessions = parse_sessions(raw)
if not sessions:
    st.error("No sessions with solves found in this file.")
    st.stop()

# ── Session selector ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.header("🗂️ Sessions")
    session_names = list(sessions.keys())
    selected_sessions = st.multiselect(
        "Sessions to display",
        session_names,
        default=session_names,
    )

if not selected_sessions:
    st.warning("Select at least one session in the sidebar.")
    st.stop()

# ── Tab layout ────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "📈 Time Series", "📉 Distribution", "🔀 Scramble Viewer", "📋 Solve Log"])

for session_name in selected_sessions:
    df = sessions[session_name].copy()
    n_solves = len(df)
    n_dnf    = df['is_dnf'].sum()
    n_plus2  = df['is_plus2'].sum()
    valid    = df[~df['is_dnf']]['eff_ms'].dropna()

    if len(valid) == 0:
        continue

    result = compute_stats(df)
    if isinstance(result, tuple):
        df, pb_single = result
    else:
        continue

    with tabs[0]:  # ── Overview ─────────────────────────────────────────────
        st.subheader(f"Session: {session_name}")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Solves",   n_solves)
        c2.metric("DNFs",     int(n_dnf))
        c3.metric("+2s",      int(n_plus2))
        c4.metric("Best",     ms_to_str(int(pb_single)))
        c5.metric("Mean",     ms_to_str(int(valid.mean())))
        c6.metric("Std Dev",  f"{valid.std()/1000:.3f}s")

        # ── PB Motivator ─────────────────────────────────────────────────────
        if len(valid) >= 5:
            mu_v  = valid.mean()
            sig_v = valid.std()
            p5  = stats.norm.ppf(0.05, loc=mu_v, scale=sig_v)   # bottom 5%
            p1  = stats.norm.ppf(0.01, loc=mu_v, scale=sig_v)   # bottom 1%
            p50 = stats.norm.ppf(0.50, loc=mu_v, scale=sig_v)   # median

            # Only show if the predicted PB is actually better than current PB
            p5_str = ms_to_str(int(max(p5, 1)))
            p1_str = ms_to_str(int(max(p1, 1)))

            st.markdown("---")
            st.markdown("### 🎯 PB Motivator")
            st.caption("Based on your current normal distribution — what times are statistically within reach.")

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Median expected",
                ms_to_str(int(p50)),
                help="50% of your solves should land around here"
            )
            m2.metric(
                "Top 5% territory",
                p5_str,
                delta=f"{(pb_single - p5)/1000:.2f}s better than PB" if p5 < pb_single else "within current PB range",
                delta_color="inverse",
                help="There's a ~1 in 20 chance you hit this on any given solve"
            )
            m3.metric(
                "Top 1% territory",
                p1_str,
                delta=f"{(pb_single - p1)/1000:.2f}s better than PB" if p1 < pb_single else "within current PB range",
                delta_color="inverse",
                help="Rare but statistically possible — ~1 in 100 solves"
            )

            if p5 < pb_single:
                gap = (pb_single - p5) / 1000
                st.success(
                    f"🔥 Based on your {len(valid)} solves, you have a **~5% chance** of beating your PB "
                    f"({ms_to_str(int(pb_single))}) by **{gap:.2f}s** on any given attempt. "
                    f"Keep grinding — statistically it's coming!"
                )
            else:
                st.info(
                    f"📈 Your PB ({ms_to_str(int(pb_single))}) is already in the top 5% of your distribution. "
                    f"That was a special solve — but with more data the model will refine."
                )

        st.divider()

    with tabs[1]:  # ── Time Series ──────────────────────────────────────────
        st.subheader(f"Session: {session_name}")
        fig = go.Figure()

        x = df['solve_num'].tolist()

        # Individual solves
        colors = ['#e06060' if r.is_dnf else '#f0c060' if r.is_plus2 else '#88c0d0'
                  for _, r in df.iterrows()]
        hover = [f"#{r.solve_num}: {r.time_str}<br>{r.scramble}" for _,r in df.iterrows()]
        fig.add_trace(go.Scatter(
            x=x, y=df['eff_ms']/1000, mode='markers',
            marker=dict(color=colors, size=5),
            name='Solve', hovertext=hover, hoverinfo='text',
        ))

        if show_ci:
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=list(df['ci_hi']/1000) + list(df['ci_lo']/1000)[::-1],
                fill='toself', fillcolor='rgba(136,192,208,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% CI', hoverinfo='skip',
            ))

        avg_map = {"Ao5": ("#a3be8c",3), "Ao12": ("#ebcb8b",2), "Ao100": ("#b48ead",2)}
        for label, (color, width) in avg_map.items():
            if label in ao_lines:
                col = label.lower()
                vals = [v/1000 if v is not None else None for v in df[col]]
                fig.add_trace(go.Scatter(
                    x=x, y=vals, mode='lines',
                    line=dict(color=color, width=width),
                    name=label, connectgaps=False,
                ))

        if show_pb:
            fig.add_trace(go.Scatter(
                x=x, y=df['running_pb']/1000, mode='lines',
                line=dict(color='#60c060', dash='dash', width=1.5),
                name='PB progression',
            ))

        fig.update_layout(
            xaxis_title="Solve #", yaxis_title="Time (s)",
            legend=dict(orientation='h', y=1.08),
            height=420, margin=dict(t=40,b=40),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d8dee9'),
        )
        st.plotly_chart(fig, width='stretch')

    with tabs[2]:  # ── Distribution ─────────────────────────────────────────
        st.subheader(f"Session: {session_name}")

        times_s = valid.values / 1000
        mu_all, sigma_all = np.mean(times_s), np.std(times_s)
        x_global = np.linspace(max(0, mu_all - 4*sigma_all), mu_all + 4*sigma_all, 300)

        # ── Top row: histogram + final normal curve ──────────────────────────
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=times_s, nbinsx=20,
            marker_color='#88c0d0', opacity=0.65,
            name='All solves', histnorm='probability density',
        ))
        fig2.add_trace(go.Scatter(
            x=x_global, y=stats.norm.pdf(x_global, mu_all, sigma_all),
            mode='lines', line=dict(color='#ebcb8b', width=2.5),
            name=f'N(μ={mu_all:.2f}s, σ={sigma_all:.2f}s)',
        ))
        fig2.update_layout(
            xaxis_title='Time (s)', yaxis_title='Density',
            height=320, plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#d8dee9'),
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, width='stretch')

        # ── Animated evolving normal ──────────────────────────────────────────
        st.markdown("**Distribution evolution** — how your normal distribution shifts as solves accumulate")

        if n_solves >= 4:
            # Build one frame per solve (subsample for performance if large)
            step = max(1, n_solves // 60)
            checkpoints = list(range(3, n_solves + 1, step))
            if checkpoints[-1] != n_solves:
                checkpoints.append(n_solves)

            anim_frames = []
            for cp in checkpoints:
                sub_v = df.iloc[:cp][~df.iloc[:cp]['is_dnf']]['eff_ms'].dropna().values / 1000
                if len(sub_v) < 2:
                    continue
                mu_i, sig_i = np.mean(sub_v), np.std(sub_v)
                y_curve = stats.norm.pdf(x_global, mu_i, sig_i)
                # Get date of the cp-th solve if available
                date_val = df.iloc[:cp]['timestamp'].dropna()
                date_str = date_val.iloc[-1].strftime("%b %d, %Y") if len(date_val) > 0 else ""
                title_str = f"After {cp} solves — μ={mu_i:.3f}s  σ={sig_i:.3f}s" + (f"  ·  {date_str}" if date_str else "")
                anim_frames.append(go.Frame(
                    data=[
                        go.Scatter(x=x_global, y=y_curve,
                                   fill='toself', fillcolor='rgba(180,142,173,0.25)',
                                   line=dict(color='#b48ead', width=2.5)),
                        go.Scatter(x=[mu_i, mu_i],
                                   y=[0, stats.norm.pdf(mu_i, mu_i, sig_i)],
                                   mode='lines', line=dict(color='#ebcb8b', dash='dot', width=1.5)),
                    ],
                    name=str(cp),
                    layout=go.Layout(title_text=title_str),
                ))

            # Initial frame
            sub0 = df.iloc[:checkpoints[0]][~df.iloc[:checkpoints[0]]['is_dnf']]['eff_ms'].dropna().values / 1000
            mu0, sig0 = np.mean(sub0), np.std(sub0) if len(sub0) > 1 else (sub0[0], 0.1)
            y0 = stats.norm.pdf(x_global, mu0, sig0)

            fig_anim = go.Figure(
                data=[
                    go.Scatter(x=x_global, y=y0,
                               fill='toself', fillcolor='rgba(180,142,173,0.25)',
                               line=dict(color='#b48ead', width=2.5), name='Distribution'),
                    go.Scatter(x=[mu0, mu0], y=[0, stats.norm.pdf(mu0, mu0, sig0)],
                               mode='lines', line=dict(color='#ebcb8b', dash='dot', width=1.5),
                               name='Mean'),
                ],
                frames=anim_frames,
            )
            fig_anim.update_layout(
                xaxis=dict(title='Time (s)', range=[x_global[0], x_global[-1]]),
                yaxis=dict(title='Density', rangemode='tozero'),
                height=360,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#d8dee9'),
                showlegend=False,
                margin=dict(t=50, b=40),
                title=dict(text=f"After {checkpoints[0]} solves", font=dict(color='#d8dee9', size=13)),
                updatemenus=[dict(
                    type='buttons', showactive=False,
                    y=1.15, x=0.5, xanchor='center',
                    buttons=[
                        dict(label='▶  Play',
                             method='animate',
                             args=[None, dict(frame=dict(duration=180, redraw=True),
                                             fromcurrent=True, mode='immediate')]),
                        dict(label='⏸  Pause',
                             method='animate',
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                               mode='immediate')]),
                    ],
                )],
                sliders=[dict(
                    steps=[dict(method='animate', args=[[str(cp)],
                                dict(mode='immediate', frame=dict(duration=180, redraw=True))],
                                label=str(cp)) for cp in checkpoints],
                    transition=dict(duration=120),
                    x=0, y=0, len=1.0,
                    currentvalue=dict(prefix='Solve: ', font=dict(color='#d8dee9')),
                    font=dict(color='#d8dee9'),
                )],
            )
            st.plotly_chart(fig_anim, width='stretch')
        else:
            st.info("Need at least 4 solves to animate the distribution evolution.")

    with tabs[3]:  # ── Scramble Viewer ───────────────────────────────────────
        st.subheader(f"Session: {session_name}")

        # Only show for 3x3 sessions (skip wide-move sessions)
        sample_scramble = df['scramble'].iloc[0] if len(df) > 0 else ""
        is_3x3 = not any(c.islower() for c in sample_scramble.replace(" ",""))

        if not is_3x3:
            st.info("Scramble visualization is only available for 3×3 solves.")
        else:
            solve_idx = st.selectbox(
                "Pick a solve to visualize",
                options=df['solve_num'].tolist(),
                format_func=lambda i: f"#{i}  {df[df['solve_num']==i]['time_str'].values[0]}",
                key=f"scr_{session_name}",
            )
            row = df[df['solve_num'] == solve_idx].iloc[0]
            st.code(row['scramble'], language=None)

            cube = scramble_cube(row['scramble'])
            fig_cube = draw_cube_net(cube)
            st.plotly_chart(fig_cube, width='content')

            st.caption("Net layout — U on top, L/F/R/B in the middle row, D on the bottom.")

    with tabs[4]:  # ── Solve Log ────────────────────────────────────────────
        st.subheader(f"Session: {session_name}")

        display_df = df.copy()
        if not show_dnf:
            display_df = display_df[~display_df['is_dnf']]

        show_cols = ['solve_num','time_str','penalty','scramble','timestamp','is_dnf','is_plus2']
        rename = {'solve_num':'#','time_str':'Time','penalty':'Penalty',
                  'scramble':'Scramble','timestamp':'Date','is_dnf':'DNF','is_plus2':'+2'}
        table_df = display_df[show_cols].rename(columns=rename)

        def style_row(row):
            if row['DNF']:
                return ['color: #e06060'] * len(row)
            elif row['+2']:
                return ['color: #f0c060'] * len(row)
            return [''] * len(row)

        styled = table_df.style.apply(style_row, axis=1)
        st.dataframe(styled, width='stretch', height=400)
