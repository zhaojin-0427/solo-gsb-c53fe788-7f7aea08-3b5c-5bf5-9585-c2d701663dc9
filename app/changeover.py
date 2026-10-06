"""双版本换场施工单：从两份发布快照还原状态，按灯具 ID 比较并逐项模拟施工步骤。

本模块为纯函数实现，不触碰数据库：两份已发布方案的 payload（各自发布快照还原）
由调用方传入。起点与目标一律取自两份快照本身——绝不读取发布回写后的当前灯具
既有地址，也不使用任何单份方案的迁移清单冒充起点。

裁决与模拟规则：
- 两版保留区（按线路的闭区间集合）或线路启停不同：抛出 IncompatibleError，
  不生成貌似安全的步骤；
- 按灯具 ID 比较：仅起点有 -> 拆除；仅目标有 -> 接入；区间完全一致 -> 不变；
  通道数不同 -> 通道数调整；其余 -> 迁址；
- 步骤顺序：先拆除（只释放空间），再迁址/通道数调整（互相占位时先周转），最后接入；
- 逐项模拟：每步显式记录释放/占用区间；每个中间态都校验不重叠、不占保留区；
  末态必须等于目标版本，否则视为内部错误；
- 周转位选择：避开保留区、当前占用与全部目标落点；优先与目标落点同线、
  其次剩余空间最贴合的空闲段，段内从低地址起放以保持余量连续（兼顾后续落点）；
- 找不到安全周转位：返回 blocked 结果并列明各线路阻塞区间，不输出任何施工步骤。
"""
from __future__ import annotations

LINE_SIZE = 512


class IncompatibleError(ValueError):
    """两版保留区或线路启停不同：拒绝生成貌似安全的施工步骤。"""


class _InternalError(RuntimeError):
    """模拟过程中出现不应发生的状态（快照数据不一致）。"""


def baseline_from_payload(plan_payload: dict) -> dict:
    """从已发布方案 payload（发布快照还原）提取换场基准。

    只使用 payload 内的发布时数据：allocations 给出灯具闭区间，occupancy
    给出保留区与线路启停。此函数绝不查询当前灯具/保留区/线路。
    """
    fixtures = [
        {
            "fixture_id": a["fixture_id"],
            "name": a.get("name", f"#{a['fixture_id']}"),
            "line_id": a["line_id"],
            "start": a["start"],
            "end": a["end"],
            "channels": a["channels"],
        }
        for a in plan_payload.get("allocations", [])
    ]
    lines = [
        {
            "line_id": line["line_id"],
            "name": line.get("name", f"线路 {line['line_id']}"),
            "is_active": bool(line.get("is_active", True)),
        }
        for line in plan_payload.get("occupancy", [])
    ]
    reservations = [
        {
            "line_id": line["line_id"],
            "start": r["start"],
            "end": r["end"],
            "reason": r.get("reason", ""),
        }
        for line in plan_payload.get("occupancy", [])
        for r in (line.get("reservations") or [])
    ]
    return {"fixtures": fixtures, "lines": lines, "reservations": reservations}


def _check_compatible(src: dict, dst: dict) -> None:
    """两版保留区或线路启停必须一致，否则拒绝生成貌似安全的步骤。"""
    src_active = {l["line_id"]: l["is_active"] for l in src["lines"]}
    dst_active = {l["line_id"]: l["is_active"] for l in dst["lines"]}
    diffs = []
    for lid in sorted(set(src_active) | set(dst_active)):
        s, d = src_active.get(lid), dst_active.get(lid)
        if s != d:
            fmt = lambda v: "启用" if v else "停用"  # noqa: E731
            diffs.append(f"线路 {lid}：起点版本{fmt(s)}、目标版本{fmt(d)}")
    if diffs:
        raise IncompatibleError(
            "两版线路启停不同，无法生成安全的换场施工单："
            + "；".join(diffs)
            + "。线路启停变化会改变可用通道，请统一两版线路启停（或基于相同线路状态重新发布）后再生成施工单。"
        )

    def _res_set(base: dict) -> list[tuple[int, int, int]]:
        # 保留区按（线路, 起始, 结束）闭区间集合比较，忽略 ID 与说明文字
        return sorted({(r["line_id"], r["start"], r["end"]) for r in base["reservations"]})

    src_res, dst_res = _res_set(src), _res_set(dst)
    if src_res != dst_res:
        only_src = [r for r in src_res if r not in dst_res]
        only_dst = [r for r in dst_res if r not in src_res]
        parts = []
        if only_src:
            parts.append("仅起点版本有保留区：" + "、".join(f"线路 {l} {s}-{e}" for l, s, e in only_src))
        if only_dst:
            parts.append("仅目标版本有保留区：" + "、".join(f"线路 {l} {s}-{e}" for l, s, e in only_dst))
        raise IncompatibleError(
            "两版保留区不同，无法生成安全的换场施工单："
            + "；".join(parts)
            + "。保留区变化会改变可用通道，请统一两版保留区（或基于相同保留区重新发布）后再生成施工单。"
        )


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def _free_gaps(blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """1..512 内未被 blocked 覆盖的连续空闲段（闭区间，按起始升序）。"""
    gaps: list[tuple[int, int]] = []
    cursor = 1
    for start, end in sorted(blocked):
        if end < cursor:
            continue
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
        if cursor > LINE_SIZE:
            break
    if cursor <= LINE_SIZE:
        gaps.append((cursor, LINE_SIZE))
    return gaps


def _find_temp(
    lines: list[dict],
    positions: dict[int, dict],
    reservations_by_line: dict[int, list[dict]],
    target_intervals: list[tuple[int, int, int, int]],
    channels: int,
    target_line: int,
) -> dict | None:
    """为需要周转的灯具寻找安全的临时位置。

    周转位必须：在启用线路上、连续长度足够、避开保留区/当前占用/全部目标落点。
    选择时兼顾后续落点：优先与目标落点同线，其次剩余空间最贴合（best-fit）的
    空闲段，段内从低地址起放以保持余量连续；再按线路 ID、起始地址取字典序最小，
    保证结果确定。找不到时返回 None，由调用方列明阻塞区间。
    """
    candidates: list[tuple[int, int, int]] = []
    for line in lines:
        lid = line["line_id"]
        if not line["is_active"]:
            continue  # 停用线路没有可用输出，不能当作周转位
        blocked = [(r["start"], r["end"]) for r in reservations_by_line.get(lid, [])]
        blocked += [(p["start"], p["end"]) for p in positions.values() if p["line_id"] == lid]
        blocked += [(s, e) for (l2, s, e, _fid) in target_intervals if l2 == lid]
        for gs, ge in _free_gaps(blocked):
            if ge - gs + 1 >= channels:
                candidates.append((lid, gs, ge))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0] != target_line, (c[2] - c[1] + 1) - channels, c[0], c[1]))
    lid, gs, _ge = candidates[0]
    return {"line_id": lid, "start": gs, "end": gs + channels - 1, "channels": channels}


def _blocking_report(
    lines: list[dict],
    positions: dict[int, dict],
    reservations_by_line: dict[int, list[dict]],
    target_intervals: list[tuple[int, int, int, int]],
    waiting_move: dict,
    blocker_id: int,
    blocker_pos: dict,
) -> dict:
    """找不到安全周转位时，列明各线路的阻塞区间（当前占用/保留区/目标落点）。"""
    line_reports = []
    for line in lines:
        lid = line["line_id"]
        occupied = []
        for r in reservations_by_line.get(lid, []):
            label = f"保留区（{r['reason']}）" if r.get("reason") else "保留区"
            occupied.append({"kind": "reservation", "label": label, "start": r["start"], "end": r["end"]})
        for fid, p in sorted(positions.items()):
            if p["line_id"] == lid:
                occupied.append({
                    "kind": "fixture",
                    "fixture_id": fid,
                    "label": f"灯具 #{fid} {p['name']}（当前位置）",
                    "start": p["start"],
                    "end": p["end"],
                })
        for (l2, s, e, fid2) in target_intervals:
            if l2 == lid:
                occupied.append({
                    "kind": "target",
                    "fixture_id": fid2,
                    "label": f"灯具 #{fid2} 的目标落点",
                    "start": s,
                    "end": e,
                })
        occupied.sort(key=lambda o: (o["start"], o["end"]))
        gaps = _free_gaps([(o["start"], o["end"]) for o in occupied])
        line_reports.append({
            "line_id": lid,
            "is_active": line["is_active"],
            "largest_free_gap": max((e - s + 1 for s, e in gaps), default=0),
            "free_gaps": [{"start": s, "end": e, "channels": e - s + 1} for s, e in gaps],
            "occupied": occupied,
        })
    wt = waiting_move["target"]
    return {
        "waiting": {
            "fixture_id": waiting_move["fixture_id"],
            "name": wt["name"],
            "target": {"line_id": wt["line_id"], "start": wt["start"], "end": wt["end"], "channels": wt["channels"]},
        },
        "fixture": {
            "fixture_id": blocker_id,
            "name": blocker_pos["name"],
            "required_channels": blocker_pos["channels"],
            "current": {
                "line_id": blocker_pos["line_id"],
                "start": blocker_pos["start"],
                "end": blocker_pos["end"],
                "channels": blocker_pos["channels"],
            },
        },
        "required_channels": blocker_pos["channels"],
        "lines": line_reports,
    }


def build_work_order(from_payload: dict, to_payload: dict) -> dict:
    """生成双版本换场施工单（只读，不写任何数据）。

    from_payload/to_payload 均为已发布方案从发布快照还原的 payload。
    返回 status=ok 的有序步骤（含每步释放/占用区间与模拟校验结论），
    或 status=blocked 的阻塞说明（找不到安全周转位，不输出施工步骤）。
    两版保留区或线路启停不同抛 IncompatibleError。
    """
    src = baseline_from_payload(from_payload)
    dst = baseline_from_payload(to_payload)
    _check_compatible(src, dst)

    src_by_id = {f["fixture_id"]: f for f in src["fixtures"]}
    dst_by_id = {f["fixture_id"]: f for f in dst["fixtures"]}

    removals = [src_by_id[i] for i in sorted(src_by_id.keys() - dst_by_id.keys())]
    installs = [dst_by_id[i] for i in sorted(dst_by_id.keys() - src_by_id.keys())]
    movers: list[dict] = []
    unchanged: list[dict] = []
    for fid in sorted(src_by_id.keys() & dst_by_id.keys()):
        s, d = src_by_id[fid], dst_by_id[fid]
        if (s["line_id"], s["start"], s["end"], s["channels"]) == (
            d["line_id"], d["start"], d["end"], d["channels"],
        ):
            unchanged.append(d)
        else:
            movers.append({
                "fixture_id": fid,
                "kind": "resize" if s["channels"] != d["channels"] else "relocate",
                "source": s,
                "target": d,
            })

    lines = sorted(dst["lines"], key=lambda l: l["line_id"])
    reservations_by_line: dict[int, list[dict]] = {}
    for r in dst["reservations"]:
        reservations_by_line.setdefault(r["line_id"], []).append(r)
    # 全部目标落点：周转位必须避开它们，否则会把后续灯具的落点堵死
    target_intervals = [
        (f["line_id"], f["start"], f["end"], f["fixture_id"]) for f in dst["fixtures"]
    ]

    positions: dict[int, dict] = {fid: dict(f) for fid, f in src_by_id.items()}
    steps: list[dict] = []

    def _check_state() -> None:
        """当前模拟态必须不重叠、不占保留区（初始态与每个中间态逐步调用）。"""
        by_line: dict[int, list[tuple[int, int, str]]] = {}
        for fid, p in positions.items():
            by_line.setdefault(p["line_id"], []).append((p["start"], p["end"], f"灯具 #{fid}"))
        for lid, items in by_line.items():
            items = items + [
                (r["start"], r["end"], f"保留区 {r['start']}-{r['end']}")
                for r in reservations_by_line.get(lid, [])
            ]
            items.sort()
            for (s1, e1, n1), (s2, e2, n2) in zip(items, items[1:]):
                if s2 <= e1:
                    raise _InternalError(
                        f"中间态出现重叠：线路 {lid} 上 {n1}（{s1}-{e1}）与 {n2}（{s2}-{e2}）"
                    )

    def _iv(f: dict | None) -> dict | None:
        if f is None:
            return None
        return {"line_id": f["line_id"], "start": f["start"], "end": f["end"], "channels": f["channels"]}

    def _emit(action, label, fid, name, frm, to, note="", temporary=False):
        steps.append({
            "seq": len(steps) + 1,
            "action": action,
            "label": label,
            "fixture_id": fid,
            "name": name,
            "temporary": temporary,
            "from": _iv(frm),
            "to": _iv(to),
            "released": _iv(frm),
            "occupied": _iv(to),
            "note": note,
        })
        _check_state()

    def _is_free(line_id: int, start: int, end: int, ignore: int | None = None) -> bool:
        for r in reservations_by_line.get(line_id, []):
            if _overlaps(start, end, r["start"], r["end"]):
                return False
        for other, p in positions.items():
            if other == ignore or p["line_id"] != line_id:
                continue
            if _overlaps(start, end, p["start"], p["end"]):
                return False
        return True

    # 初始态（起点版本快照）本身也必须满足安全不变量
    _check_state()

    # 1) 拆除：只释放空间，永远不会引入冲突
    for f in removals:
        fid = f["fixture_id"]
        positions.pop(fid)
        _emit("remove", "拆除", fid, f["name"], f, None, note="目标版本不再使用该灯具")

    # 2) 迁址 / 通道数调整：目标落点空闲即落位；互相占位时先把占位灯具搬到周转位
    pending = {m["fixture_id"]: m for m in movers}
    staged: dict[int, dict] = {}  # fixture_id -> 周转位区间（尚未归位）
    while pending:
        ready_id = next(
            (
                fid for fid in sorted(pending)
                if _is_free(
                    pending[fid]["target"]["line_id"],
                    pending[fid]["target"]["start"],
                    pending[fid]["target"]["end"],
                    ignore=fid,
                )
            ),
            None,
        )
        if ready_id is not None:
            m = pending.pop(ready_id)
            cur = positions[ready_id]
            tgt = m["target"]
            note = ""
            if ready_id in staged:
                note = f"自周转位（线路 {cur['line_id']} {cur['start']}-{cur['end']}）归位"
                del staged[ready_id]
            if m["kind"] == "resize":
                chg = f"通道数 {m['source']['channels']} → {tgt['channels']}"
                note = f"{chg}；{note}" if note else chg
            positions[ready_id] = dict(tgt)
            _emit(
                m["kind"],
                "迁址" if m["kind"] == "relocate" else "通道数调整",
                ready_id, tgt["name"], cur, tgt, note=note,
            )
            continue
        # 无就绪移动：存在互相占位。取等待落位的最小 ID 灯具，把挡住其目标落点的
        # 占位灯具先搬到安全的空闲连续区间周转（占位者必为另一台待移动灯具：
        # 不变灯具与已落位灯具的位置都是目标落点，目标版本内部不会重叠）。
        wait_id = min(pending)
        wt = pending[wait_id]["target"]
        blocker_id = next(
            (
                other for other, p in positions.items()
                if other != wait_id and p["line_id"] == wt["line_id"]
                and _overlaps(wt["start"], wt["end"], p["start"], p["end"])
            ),
            None,
        )
        if blocker_id is None or blocker_id not in pending:
            raise _InternalError("目标落点被非待移动灯具占用，快照数据不一致")
        blocker = pending[blocker_id]
        bpos = positions[blocker_id]
        temp = _find_temp(
            lines, positions, reservations_by_line, target_intervals,
            bpos["channels"], blocker["target"]["line_id"],
        )
        if temp is None:
            # 找不到安全周转位：列明阻塞区间，不输出任何施工步骤
            return {
                "status": "blocked",
                "message": (
                    f"灯具 #{blocker_id}“{bpos['name']}”需要 {bpos['channels']} 通道的连续空闲周转位，"
                    "但各启用线路均找不到（周转位须避开保留区、当前占用与全部目标落点）。"
                    "未生成任何施工步骤，请调整目标排布或腾出通道后重试。"
                ),
                "blocking": _blocking_report(
                    lines, positions, reservations_by_line, target_intervals,
                    pending[wait_id], blocker_id, bpos,
                ),
                "steps_simulated_before_block": len(steps),
                "lines": lines,
                "reservations": dst["reservations"],
            }
        positions[blocker_id] = {
            **bpos, "line_id": temp["line_id"], "start": temp["start"], "end": temp["end"],
        }
        staged[blocker_id] = temp
        _emit(
            "stage", "临时周转", blocker_id, blocker["target"]["name"], bpos, temp,
            note=f"为灯具 #{wait_id} 腾出落点；周转位避开保留区与全部目标落点",
            temporary=True,
        )

    # 3) 接入：全部移动落位后，接入落点必然空闲（目标版本内部不重叠）
    for f in installs:
        fid = f["fixture_id"]
        if not _is_free(f["line_id"], f["start"], f["end"]):
            raise _InternalError("接入落点被占用，快照数据不一致")
        positions[fid] = dict(f)
        _emit("install", "接入", fid, f["name"], None, f, note="起点版本无此灯具")

    # 末态必须等于目标版本
    expected = {
        fid: (f["line_id"], f["start"], f["end"], f["channels"]) for fid, f in dst_by_id.items()
    }
    actual = {
        fid: (p["line_id"], p["start"], p["end"], p["channels"]) for fid, p in positions.items()
    }
    if staged:
        raise _InternalError("存在未归位的周转灯具")
    if actual != expected:
        raise _InternalError("模拟末态与目标版本不一致")

    return {
        "status": "ok",
        "summary": {
            "remove": len(removals),
            "relocate": sum(1 for m in movers if m["kind"] == "relocate"),
            "resize": sum(1 for m in movers if m["kind"] == "resize"),
            "install": len(installs),
            "stage": sum(1 for s in steps if s["action"] == "stage"),
            "unchanged": len(unchanged),
            "steps": len(steps),
        },
        "steps": steps,
        "unchanged": [
            {
                "fixture_id": f["fixture_id"], "name": f["name"], "line_id": f["line_id"],
                "start": f["start"], "end": f["end"], "channels": f["channels"],
            }
            for f in unchanged
        ],
        "lines": lines,
        "reservations": dst["reservations"],
        "from_occupancy": from_payload.get("occupancy", []),
        "to_occupancy": to_payload.get("occupancy", []),
        "verification": {
            "steps_simulated": len(steps),
            "intermediate_states_checked": True,
            "no_overlap": True,
            "reservations_respected": True,
            "final_matches_target": True,
        },
    }
