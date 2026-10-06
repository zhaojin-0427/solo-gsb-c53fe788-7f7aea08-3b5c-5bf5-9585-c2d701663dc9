"""双版本换场施工单端到端测试：快照基准、有序步骤、逐项模拟、周转与拒绝情形。"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["DMX_DB_PATH"] = _tmp.name

from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Plan, PlanAllocation  # noqa: E402
from app.services import get_config_version, init_db  # noqa: E402


@pytest.fixture(scope="session")
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c


CATALOG_CSV = (
    "manufacturer,name,mode,channels\n"
    "ACME,Solar,Basic,10\n"
    "ACME,Solar,Full,20\n"
    "BrightCo,Par,3CH,3\n"
    "LitePro,Beam,Std,5\n"
    "HeavyCo,Wall,Fixed,256\n"
)


def _modes(client):
    return {(t["name"], m["name"]): m for t in client.get("/api/catalog").json() for m in t["modes"]}


def _fixtures(client):
    return {f["name"]: f for f in client.get("/api/fixtures").json()}


def _publish(client):
    p = client.post("/api/plans/preview").json()
    assert p["optimality_status"] == "optimal", p
    r = client.post(f"/api/plans/{p['plan_id']}/publish")
    assert r.status_code == 200, r.text
    return p["plan_id"]


def _changeover(client, from_id, to_id):
    return client.get(f"/api/plans/changeover?from_plan_id={from_id}&to_plan_id={to_id}")


def resimulate(client, from_id, to_id, steps):
    """独立重放施工步骤：每步释放/占用区间必须与模拟态一致，任何中间态不重叠、
    不占保留区，末态等于目标版本快照。"""
    src = client.get(f"/api/plans/{from_id}").json()
    dst = client.get(f"/api/plans/{to_id}").json()
    positions = {a["fixture_id"]: (a["line_id"], a["start"], a["end"]) for a in src["allocations"]}
    reservations = [
        (line["line_id"], r["start"], r["end"])
        for line in src["occupancy"]
        for r in line["reservations"]
    ]

    def check():
        by_line = {}
        for fid, (li, s, e) in positions.items():
            by_line.setdefault(li, []).append((s, e, fid))
        for li, ivs in by_line.items():
            ivs.sort()
            for (s1, e1, f1), (s2, e2, f2) in zip(ivs, ivs[1:]):
                assert s2 > e1, f"中间态重叠：线路 {li} 灯具 {f1} 与 {f2}"
            for (s, e, fid) in ivs:
                for (rl, rs, re_) in reservations:
                    if rl == li:
                        assert e < rs or s > re_, f"中间态占保留区：灯具 {fid} 线路 {li}"

    check()
    for st in steps:
        fid = st["fixture_id"]
        if st["released"] is not None:
            cur = positions.pop(fid)
            rel = st["released"]
            assert cur == (rel["line_id"], rel["start"], rel["end"]), "释放区间与当前位置不符"
        if st["occupied"] is not None:
            occ = st["occupied"]
            positions[fid] = (occ["line_id"], occ["start"], occ["end"])
        check()
    expected = {a["fixture_id"]: (a["line_id"], a["start"], a["end"]) for a in dst["allocations"]}
    assert positions == expected, "末态不等于目标版本"


def test_01_setup_and_publish_v1(client):
    r = client.post("/api/catalog/import", files={"file": ("c.csv", CATALOG_CSV, "text/csv")})
    assert r.status_code == 200, r.text
    modes = _modes(client)
    for name, key in [("Alpha", ("Solar", "Basic")), ("Beta", ("Solar", "Full")),
                      ("Gamma", ("Par", "3CH")), ("Delta", ("Beam", "Std"))]:
        r = client.post("/api/fixtures", json={"name": name, "mode_id": modes[key]["id"]})
        assert r.status_code == 201, r.text
    plan_id = _publish(client)
    payload = client.get(f"/api/plans/{plan_id}").json()
    by_name = {a["name"]: a for a in payload["allocations"]}
    # 确定性落点：全部在线路 1 低地址依次排开
    assert (by_name["Alpha"]["line_id"], by_name["Alpha"]["start"], by_name["Alpha"]["end"]) == (1, 1, 10)
    assert (by_name["Beta"]["start"], by_name["Beta"]["end"]) == (11, 30)
    assert (by_name["Gamma"]["start"], by_name["Gamma"]["end"]) == (31, 33)
    assert (by_name["Delta"]["start"], by_name["Delta"]["end"]) == (34, 38)
    pytest.P1 = plan_id


def test_02_evolve_and_publish_v2(client):
    modes = _modes(client)
    fx = _fixtures(client)
    # Alpha 迁址：锁定到新地址 100；Beta 通道数调整：20ch -> 10ch；
    # Gamma 拆除；Echo 接入（5ch）；Delta 不变
    assert client.put(f"/api/fixtures/{fx['Alpha']['id']}",
                      json={"previous_line": 1, "previous_start": 100, "locked": True}).status_code == 200
    assert client.put(f"/api/fixtures/{fx['Beta']['id']}",
                      json={"mode_id": modes[("Solar", "Basic")]["id"]}).status_code == 200
    assert client.delete(f"/api/fixtures/{fx['Gamma']['id']}").status_code == 204
    r = client.post("/api/fixtures", json={"name": "Echo", "mode_id": modes[("Beam", "Std")]["id"]})
    assert r.status_code == 201, r.text
    plan_id = _publish(client)
    payload = client.get(f"/api/plans/{plan_id}").json()
    by_name = {a["name"]: a for a in payload["allocations"]}
    assert (by_name["Alpha"]["start"], by_name["Alpha"]["end"]) == (100, 109)
    assert (by_name["Beta"]["start"], by_name["Beta"]["end"], by_name["Beta"]["channels"]) == (11, 20, 10)
    assert (by_name["Delta"]["start"], by_name["Delta"]["end"]) == (34, 38)
    assert (by_name["Echo"]["start"], by_name["Echo"]["end"]) == (1, 5)
    # P2 自身的迁移清单为空（锁定/回写抹平），不能拿它冒充换场起点
    assert payload["migrations"] == []
    pytest.P2 = plan_id


def test_03_work_order_basic(client):
    r = _changeover(client, pytest.P1, pytest.P2)
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["status"] == "ok"
    assert rep["from_plan_id"] == pytest.P1 and rep["to_plan_id"] == pytest.P2
    s = rep["summary"]
    assert (s["remove"], s["relocate"], s["resize"], s["install"]) == (1, 1, 1, 1)
    assert s["stage"] == 0 and s["unchanged"] == 1 and s["steps"] == 4

    fx = _fixtures(client)
    p1 = client.get(f"/api/plans/{pytest.P1}").json()
    p2 = client.get(f"/api/plans/{pytest.P2}").json()
    # Gamma/Echo 只存在于其中一版：ID 一律从快照取，不依赖当前灯具表
    id_p1 = {a["name"]: a["fixture_id"] for a in p1["allocations"]}
    id_p2 = {a["name"]: a["fixture_id"] for a in p2["allocations"]}
    steps = rep["steps"]
    assert [st["seq"] for st in steps] == [1, 2, 3, 4]
    # 顺序：先拆除，再迁址/通道数调整，最后接入
    assert [st["action"] for st in steps] == ["remove", "relocate", "resize", "install"]

    rm = steps[0]
    assert rm["fixture_id"] == id_p1["Gamma"] and rm["label"] == "拆除"
    assert (rm["released"]["line_id"], rm["released"]["start"], rm["released"]["end"]) == (1, 31, 33)
    assert rm["occupied"] is None and rm["to"] is None

    mv = steps[1]
    assert mv["fixture_id"] == fx["Alpha"]["id"] and mv["label"] == "迁址"
    # 起点取自 P1 快照（1-10），不是发布回写后的当前既有地址（100）
    assert (mv["from"]["start"], mv["from"]["end"]) == (1, 10)
    assert (mv["to"]["start"], mv["to"]["end"]) == (100, 109)
    assert mv["released"] == mv["from"] and mv["occupied"] == mv["to"]

    rz = steps[2]
    assert rz["fixture_id"] == fx["Beta"]["id"] and rz["label"] == "通道数调整"
    assert (rz["from"]["start"], rz["from"]["end"]) == (11, 30)
    assert (rz["to"]["start"], rz["to"]["end"]) == (11, 20)
    assert "通道数 20 → 10" in rz["note"]

    ins = steps[3]
    assert ins["fixture_id"] == id_p2["Echo"] and ins["label"] == "接入"
    assert ins["released"] is None and ins["from"] is None
    assert (ins["occupied"]["start"], ins["occupied"]["end"]) == (1, 5)

    v = rep["verification"]
    assert v["steps_simulated"] == 4
    assert v["no_overlap"] and v["reservations_respected"] and v["final_matches_target"]
    # 不变清单只有 Delta；两版快照占用随响应返回
    assert [u["fixture_id"] for u in rep["unchanged"]] == [id_p1["Delta"]]
    assert rep["from_occupancy"] == p1["occupancy"]
    assert rep["to_occupancy"] == p2["occupancy"]
    pytest.WO_BASIC = rep


def test_04_steps_resimulate_cleanly(client):
    resimulate(client, pytest.P1, pytest.P2, pytest.WO_BASIC["steps"])


def test_05_changeover_is_readonly(client):
    version_before = client.get("/api/plans/version").json()["config_version"]
    p1_before = client.get(f"/api/plans/{pytest.P1}").json()
    p2_before = client.get(f"/api/plans/{pytest.P2}").json()
    r = _changeover(client, pytest.P1, pytest.P2)
    assert r.status_code == 200
    assert client.get("/api/plans/version").json()["config_version"] == version_before
    assert client.get(f"/api/plans/{pytest.P1}").json() == p1_before
    assert client.get(f"/api/plans/{pytest.P2}").json() == p2_before


def test_06_baseline_from_snapshots_not_current_state(client):
    """发布回写、改名、改既有地址、删除灯具都不影响施工单：基准只认两份快照。"""
    fx = _fixtures(client)
    # 现状漂移源：改名、改既有地址、删除灯具（Echo 在 P2 中接入，删除当前记录）
    assert client.put(f"/api/fixtures/{fx['Alpha']['id']}", json={"name": "AlphaX"}).status_code == 200
    assert client.put(f"/api/fixtures/{fx['Delta']['id']}", json={"previous_start": 400}).status_code == 200
    assert client.delete(f"/api/fixtures/{fx['Echo']['id']}").status_code == 204
    # 当前灯具的既有地址已是 P2 落点（发布回写）：若误用当前状态，P1→P2 会得到空步骤
    r = _changeover(client, pytest.P1, pytest.P2)
    assert r.status_code == 200
    rep = r.json()
    assert rep == pytest.WO_BASIC  # 与漂移前完全一致
    assert rep["summary"]["steps"] == 4
    # 接入 Echo 的步骤仍在（来自 P2 快照，而非当前灯具——当前已被删除）
    assert any(st["action"] == "install" and st["name"] == "Echo" for st in rep["steps"])


def test_07_reverse_direction(client):
    """反向换场（P2→P1）：拆除/接入互换，迁址与通道数调整反向。"""
    r = _changeover(client, pytest.P2, pytest.P1)
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["status"] == "ok"
    s = rep["summary"]
    assert (s["remove"], s["relocate"], s["resize"], s["install"]) == (1, 1, 1, 1)
    assert [st["action"] for st in rep["steps"]] == ["remove", "relocate", "resize", "install"]
    p1 = client.get(f"/api/plans/{pytest.P1}").json()
    p2 = client.get(f"/api/plans/{pytest.P2}").json()
    id_p1 = {a["name"]: a["fixture_id"] for a in p1["allocations"]}
    id_p2 = {a["name"]: a["fixture_id"] for a in p2["allocations"]}
    rm, mv, rz, ins = rep["steps"]
    assert rm["fixture_id"] == id_p2["Echo"]  # Echo 在 P1 中不存在 -> 拆除
    assert mv["fixture_id"] == id_p1["Alpha"]
    assert (mv["from"]["start"], mv["to"]["start"]) == (100, 1)
    assert rz["fixture_id"] == id_p1["Beta"]
    assert "通道数 10 → 20" in rz["note"]
    assert ins["fixture_id"] == id_p1["Gamma"]  # Gamma 在 P1 中恢复 -> 接入
    assert (ins["occupied"]["start"], ins["occupied"]["end"]) == (31, 33)
    resimulate(client, pytest.P2, pytest.P1, rep["steps"])


def test_08_same_missing_unpublished_rejected(client):
    # 同一份方案
    r = _changeover(client, pytest.P1, pytest.P1)
    assert r.status_code == 422
    # 方案不存在
    assert _changeover(client, 999999, pytest.P1).status_code == 404
    assert _changeover(client, pytest.P1, 999999).status_code == 404
    # 草稿（未发布）
    p = client.post("/api/plans/preview").json()
    assert p["optimality_status"] == "optimal"
    r = _changeover(client, p["plan_id"], pytest.P1)
    assert r.status_code == 409 and "尚未发布" in r.json()["detail"]
    r = _changeover(client, pytest.P1, p["plan_id"])
    assert r.status_code == 409 and "尚未发布" in r.json()["detail"]


def test_09_legacy_plan_without_snapshot_rejected(client):
    """无快照旧记录：无法还原保留区与线路启停，拒绝生成貌似安全的步骤。"""
    fx = _fixtures(client)
    with SessionLocal() as db:
        plan = Plan(
            config_version=get_config_version(db),
            status="published",
            optimality_status="optimal",
            migration_count=0,
            used_lines=1,
            published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        db.add(plan)
        db.flush()
        db.add(PlanAllocation(plan_id=plan.id, fixture_id=fx["Beta"]["id"],
                              line_id=1, start=1, end=10, migrated=False))
        db.commit()
        legacy_id = plan.id
    for a, b in [(legacy_id, pytest.P1), (pytest.P1, legacy_id)]:
        r = _changeover(client, a, b)
        assert r.status_code == 409
        assert "快照" in r.json()["detail"]


def test_10_incompatible_reservations_rejected(client):
    assert client.post("/api/reservations",
                       json={"line_id": 2, "start": 1, "end": 10, "reason": "desk"}).status_code == 201
    pytest.P3 = _publish(client)
    for a, b in [(pytest.P2, pytest.P3), (pytest.P3, pytest.P2)]:
        r = _changeover(client, a, b)
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert "保留区" in detail and "线路 2" in detail
        assert "无法生成安全的换场施工单" in detail


def test_11_incompatible_line_states_rejected(client):
    assert client.patch("/api/lines/4", json={"is_active": False}).status_code == 200
    pytest.P4 = _publish(client)
    r = _changeover(client, pytest.P3, pytest.P4)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "线路启停不同" in detail and "线路 4" in detail
    r = _changeover(client, pytest.P4, pytest.P3)
    assert r.status_code == 409


def test_12_mutual_blocking_uses_staging(client):
    """两台灯具互换位置：必须先周转再归位；周转位避开全部目标落点与保留区。"""
    modes = _modes(client)
    basic = modes[("Solar", "Basic")]["id"]
    r = client.post("/api/fixtures", json={"name": "S1", "mode_id": basic,
                                           "previous_line": 1, "previous_start": 200, "locked": True})
    assert r.status_code == 201, r.text
    r = client.post("/api/fixtures", json={"name": "S2", "mode_id": basic,
                                           "previous_line": 1, "previous_start": 210, "locked": True})
    assert r.status_code == 201, r.text
    pytest.P5 = _publish(client)

    fx = _fixtures(client)
    assert client.put(f"/api/fixtures/{fx['S1']['id']}",
                      json={"previous_line": 1, "previous_start": 210, "locked": True}).status_code == 200
    assert client.put(f"/api/fixtures/{fx['S2']['id']}",
                      json={"previous_line": 1, "previous_start": 200, "locked": True}).status_code == 200
    pytest.P6 = _publish(client)

    r = _changeover(client, pytest.P5, pytest.P6)
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["status"] == "ok"
    assert rep["summary"]["stage"] == 1 and rep["summary"]["relocate"] == 2
    steps = rep["steps"]
    assert len(steps) == 3
    stage, mv1, mv2 = steps
    # 第 1 步：占位灯具 S2 先搬到周转位（S1 的 ID 更小，先等待落位）
    assert stage["action"] == "stage" and stage["temporary"] is True
    assert stage["fixture_id"] == fx["S2"]["id"]
    assert (stage["from"]["start"], stage["from"]["end"]) == (210, 219)
    assert f"为灯具 #{fx['S1']['id']} 腾出落点" in stage["note"]
    temp = stage["occupied"]
    # 周转位：避开全部目标落点与保留区（线路 1 上 21-99 为最贴合空闲段）
    p6 = client.get(f"/api/plans/{pytest.P6}").json()
    for a in p6["allocations"]:
        assert temp["line_id"] != a["line_id"] or temp["end"] < a["start"] or temp["start"] > a["end"]
    for line in p6["occupancy"]:
        for res in line["reservations"]:
            assert temp["line_id"] != line["line_id"] or temp["end"] < res["start"] or temp["start"] > res["end"]
    # 第 2 步：S1 落位；第 3 步：S2 自周转位归位
    assert mv1["action"] == "relocate" and mv1["fixture_id"] == fx["S1"]["id"]
    assert (mv1["from"]["start"], mv1["to"]["start"]) == (200, 210)
    assert mv2["action"] == "relocate" and mv2["fixture_id"] == fx["S2"]["id"]
    assert (mv2["from"]["start"], mv2["from"]["end"]) == (temp["start"], temp["end"])
    assert (mv2["to"]["start"], mv2["to"]["end"]) == (200, 209)
    assert "归位" in mv2["note"]
    resimulate(client, pytest.P5, pytest.P6, steps)


def test_13_no_safe_staging_reports_blocking(client):
    """线路被占满且其余线路停用时，找不到安全周转位：列明阻塞区间，不输出步骤。"""
    fx = _fixtures(client)
    for name in ("AlphaX", "Beta", "Delta", "S1", "S2"):  # Alpha 已在漂移测试中改名
        assert client.delete(f"/api/fixtures/{fx[name]['id']}").status_code == 204
    for lid in (2, 3):
        assert client.patch(f"/api/lines/{lid}", json={"is_active": False}).status_code == 200
    big = _modes(client)[("Wall", "Fixed")]["id"]  # 256ch
    r = client.post("/api/fixtures", json={"name": "B1", "mode_id": big})
    assert r.status_code == 201, r.text
    r = client.post("/api/fixtures", json={"name": "B2", "mode_id": big})
    assert r.status_code == 201, r.text
    pytest.P7 = _publish(client)
    p7 = client.get(f"/api/plans/{pytest.P7}").json()
    by_name = {a["name"]: a for a in p7["allocations"]}
    assert (by_name["B1"]["start"], by_name["B1"]["end"]) == (1, 256)
    assert (by_name["B2"]["start"], by_name["B2"]["end"]) == (257, 512)

    fx = _fixtures(client)
    assert client.put(f"/api/fixtures/{fx['B1']['id']}",
                      json={"previous_line": 1, "previous_start": 257, "locked": True}).status_code == 200
    assert client.put(f"/api/fixtures/{fx['B2']['id']}",
                      json={"previous_line": 1, "previous_start": 1, "locked": True}).status_code == 200
    pytest.P8 = _publish(client)

    r = _changeover(client, pytest.P7, pytest.P8)
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["status"] == "blocked"
    assert "steps" not in rep  # 不输出任何施工步骤
    assert "256 通道" in rep["message"]
    b = rep["blocking"]
    assert b["required_channels"] == 256
    assert b["waiting"]["fixture_id"] == fx["B1"]["id"]
    assert b["fixture"]["fixture_id"] == fx["B2"]["id"]
    line1 = next(l for l in b["lines"] if l["line_id"] == 1)
    assert line1["largest_free_gap"] == 0
    assert line1["free_gaps"] == []
    labels = [o["label"] for o in line1["occupied"]]
    assert any("B1" in x for x in labels) and any("B2" in x for x in labels)
    # 停用线路也在报告中，标注不可用于周转
    assert all(not l["is_active"] for l in b["lines"] if l["line_id"] != 1)


def test_14_frontend_ships_changeover_panel(client):
    html = client.get("/").text
    assert 'id="changeover-from-select"' in html
    assert 'id="changeover-to-select"' in html
    assert 'id="btn-changeover"' in html
    assert 'id="changeover-result"' in html
    js = client.get("/static/app.js").text
    assert "/api/plans/changeover" in js
    assert "renderChangeover" in js
    assert "renderPublishedTracks" in js
