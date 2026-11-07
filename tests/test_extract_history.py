import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def message(
    uuid: str, kind: str, timestamp: str, content: object, **extra: object
) -> str:
    return json.dumps(
        {
            "uuid": uuid,
            "type": kind,
            "timestamp": timestamp,
            "message": {"content": content},
            **extra,
        }
    )


def run(projects: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "extract_history.py"),
            "--projects",
            str(projects),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )


def write_history(projects: Path) -> None:
    main = projects / "-Users-j-Desktop-transformer-lab"
    main.mkdir(parents=True)
    worktree = projects / "-Users-j-Desktop-transformer-lab--claude-worktrees-x"
    worktree.mkdir()
    other = projects / "-Users-j-Desktop-other"
    other.mkdir()
    (main / "a.jsonl").write_text(
        "\n".join(
            [
                message("1", "user", "2026-11-01T10:00:00Z", "第一句"),
                message(
                    "2",
                    "assistant",
                    "2026-11-01T10:01:00Z",
                    [
                        {"type": "text", "text": "回复"},
                        {"type": "tool_use", "name": "Bash", "input": {}},
                    ],
                ),
                # 工具结果、子 agent、通知、系统提示都不是对话内容
                message("3", "user", "2026-11-01T10:02:00Z", [{"type": "tool_result"}]),
                message(
                    "4", "user", "2026-11-01T10:03:00Z", "子任务", isSidechain=True
                ),
                message(
                    "5",
                    "user",
                    "2026-11-01T10:04:00Z",
                    "<task-notification>x</task-notification>",
                ),
                message(
                    "6",
                    "user",
                    "2026-11-01T10:05:00Z",
                    "<system-reminder>y</system-reminder>第二句",
                ),
                message(
                    "10",
                    "user",
                    "2026-11-01T10:06:00Z",
                    "Base directory for this skill: /x\n技能说明",
                ),
                message(
                    "11",
                    "user",
                    "2026-11-01T10:07:00Z",
                    "Another Claude session sent a message:\n子 agent 的报告",
                ),
                "这一行不是 json",
            ]
        ),
        encoding="utf-8",
    )
    # 压缩或恢复后的新文件会带着旧消息，要去重；时间在前的消息也要排到前面
    (main / "b.jsonl").write_text(
        "\n".join(
            [
                message(
                    "2",
                    "assistant",
                    "2026-11-01T10:01:00Z",
                    [{"type": "text", "text": "回复"}],
                ),
                message("7", "user", "2026-11-02T09:00:00Z", "第三句"),
            ]
        ),
        encoding="utf-8",
    )
    (worktree / "c.jsonl").write_text(
        message("8", "user", "2026-10-31T09:00:00Z", "worktree 里的话"),
        encoding="utf-8",
    )
    (other / "d.jsonl").write_text(
        message("9", "user", "2026-11-01T09:00:00Z", "别的项目"), encoding="utf-8"
    )


def test_history_is_deduplicated_sorted_and_split_by_day(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    write_history(projects)
    out = tmp_path / "history"

    done = run(projects, out)

    assert done.returncode == 0, done.stderr
    assert sorted(path.name for path in out.iterdir()) == [
        "2026-10-31.txt",
        "2026-11-01.txt",
        "2026-11-02.txt",
        "user_messages.txt",
    ]
    day = (out / "2026-11-01.txt").read_text(encoding="utf-8")
    assert day.index("第一句") < day.index("回复") < day.index("第二句")
    assert day.count("回复") == 1
    for not_conversation in (
        "子任务",
        "task-notification",
        "system-reminder",
        "别的项目",
        "技能说明",
    ):
        assert not_conversation not in day


def test_user_messages_are_collected_in_time_order(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    write_history(projects)
    out = tmp_path / "history"

    run(projects, out)

    users = (out / "user_messages.txt").read_text(encoding="utf-8")
    assert (
        users.index("worktree 里的话") < users.index("第一句") < users.index("第三句")
    )
    assert "回复" not in users
    # 子 agent 的报告不是用户说的话，只留在每天的文件里，并标明角色
    assert "子 agent 的报告" not in users
    day = (out / "2026-11-01.txt").read_text(encoding="utf-8")
    assert "===== 子 agent 2026-11-01T10:07 =====" in day
