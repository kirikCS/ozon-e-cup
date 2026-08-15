"""PreToolUse guard: the last line of defence once permissions are bypassed.

Bypass mode switches off the permission system, so allow/deny rules in settings.json stop
being enforced. Hooks are not part of that system and still run, which makes this the only
thing standing between an autonomous agent and the two files it cannot recreate:

  submission/stack6.csv   the current best result, 1.6496638 on the leaderboard
  data/                   the 180 MB raw dataset

Everything else in the project is reproducible from these two plus code, so nothing else is
protected — the agent is meant to be free.

Reads the hook payload on stdin, prints a deny decision when a call would write to, move or
delete a protected path, and stays silent otherwise. Reading is never blocked.
"""
import json
import re
import sys

PROTECTED = ("submission/stack6.csv", "data/train.parquet", "data/sample_submit.csv")
PROTECTED_DIRS = ("data/",)

# shell verbs that modify rather than read, plus redirection into a path
DESTRUCTIVE = re.compile(
    r"(^|[;&|]\s*)(rm|mv|cp|truncate|shred|dd)\s|>\s*\S*(stack6\.csv|data/)", re.I)


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason}}, ensure_ascii=False))
    sys.exit(0)


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception:
        # Unparseable payload: we cannot tell what the call does, so fall back to scanning the
        # raw text. Failing open here would mean a payload-format change silently disarms the
        # only guard left once permissions are bypassed.
        if any(x in raw.replace("\\\\", "/").replace("\\", "/") for x in PROTECTED):
            deny("Payload хука не разобрался, но в нём упомянут защищённый путь. "
                 "Отказано на всякий случай — проверь формат вызова.")
        sys.exit(0)

    tool = payload.get("tool_name") or ""
    ti = payload.get("tool_input") or {}

    if tool in ("Write", "Edit", "NotebookEdit"):
        p = str(ti.get("file_path") or "").replace("\\", "/")
        hit = any(p.endswith(x) for x in PROTECTED) or any(f"/{d}" in p for d in PROTECTED_DIRS)
        if hit:
            deny(f"{p} защищён: это либо текущий лучший результат (stack6.csv, LB 1,6496638), "
                 f"либо исходные данные. Новые сабмиты клади в submission/queue/.")

    if tool in ("Bash", "PowerShell"):
        cmd = str(ti.get("command") or "").replace("\\", "/")
        touches = any(x in cmd for x in PROTECTED) or "data/" in cmd
        if touches and DESTRUCTIVE.search(cmd):
            deny("Команда изменяет или удаляет защищённый путь (submission/stack6.csv или data/). "
                 "Чтение разрешено, запись — нет. Новые файлы клади в submission/queue/.")

    sys.exit(0)


if __name__ == "__main__":
    main()
