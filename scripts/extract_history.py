"""把这个项目的全部 Claude Code 对话提取成纯文本，给新对话接手时读。

只留用户消息和助手的文字回复，去掉工具调用、工具结果、子 agent、后台通知、系统提示
和技能说明。
同一段对话压缩或恢复后会写进新文件，内容有重复，所以按消息 uuid 去重，再按时间排序。
每天写一个文件，用户消息另外单独写一份。

用法：
    python3 -I scripts/extract_history.py --out <输出目录>
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_GLOB = "-Users-j-Desktop-transformer-lab*"  # 也包括 worktree 里的对话


def message_text(record: dict[str, Any]) -> str:
    """取出一条记录里给人看的文字；不是对话内容时返回空字符串。"""
    if record.get("type") not in ("user", "assistant") or record.get("isSidechain"):
        return ""
    content = record.get("message", {}).get("content")
    if isinstance(content, str):
        parts = [content]
    else:
        parts = [
            part.get("text", "")
            for part in content or []
            if isinstance(part, dict) and part.get("type") == "text"
        ]
    text = "\n".join(part for part in parts if part and part.strip())
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = text.strip()
    if text.startswith(
        ("<task-notification", "[Request interrupted", "Base directory for this skill")
    ):
        return ""
    return text


def load_messages(projects: Path) -> list[tuple[str, str, str]]:
    """读所有对话文件，去重后按时间排序，返回 [(时间, 角色, 文字)]。"""
    messages: dict[Any, tuple[str, str, str]] = {}
    for directory in projects.glob(PROJECT_GLOB):
        for path in directory.glob("*.jsonl"):
            for line in path.open(encoding="utf-8", errors="replace"):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                text = message_text(record)
                timestamp = record.get("timestamp")
                if not text or not timestamp:
                    continue
                role = "用户" if record["type"] == "user" else "助手"
                if record.get("isCompactSummary") or text.startswith(
                    "This session is being continued"
                ):
                    role = "压缩摘要"
                elif text.startswith("Another Claude session sent a message"):
                    role = "子 agent"
                key = record.get("uuid") or (timestamp, role, text[:50])
                messages[key] = (timestamp, role, text)
    return sorted(messages.values(), key=lambda message: message[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects", type=Path, default=Path.home() / ".claude" / "projects"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    by_day: dict[str, list[str]] = {}
    users: list[str] = []
    for timestamp, role, text in load_messages(args.projects):
        entry = f"\n===== {role} {timestamp[:16]} =====\n{text}\n"
        by_day.setdefault(timestamp[:10], []).append(entry)
        if role == "用户":
            users.append(entry)

    for day, entries in by_day.items():
        (args.out / f"{day}.txt").write_text("".join(entries), encoding="utf-8")
    (args.out / "user_messages.txt").write_text("".join(users), encoding="utf-8")

    print("文件 条数 字符数")
    for day, entries in sorted(by_day.items()):
        print(f"{day}.txt {len(entries)} {sum(map(len, entries))}")
    print(f"user_messages.txt {len(users)} {sum(map(len, users))}")


if __name__ == "__main__":
    main()
