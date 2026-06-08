"""用户账号存储与认证（多用户登陆用）。

设计：
- 所有用户记录保存在单个 JSON 文件（config.USERS_FILE）里，以用户名为键。
- 密码不存明文，使用标准库的 PBKDF2-HMAC-SHA256 + 每用户随机盐。
- user_id 由用户名做文件系统安全化得到，用作每用户会话目录名
  （conversations/users/<user_id>/）。

只依赖标准库（hashlib / secrets / hmac），不引入新依赖。
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from pathlib import Path

# PBKDF2 迭代次数。越大越慢越安全；对本地测试工具来说这个量级足够。
_PBKDF2_ITERATIONS = 200_000


def _safe_user_id(raw: str) -> str:
    """把用户名清洗成文件系统安全的目录名（与 conversation 的清洗同款思路）。"""
    cleaned = re.sub(r"[^A-Za-z0-9_\-.]", "_", (raw or "").strip())
    return cleaned[:64]


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        _PBKDF2_ITERATIONS,
    )
    return dk.hex()


class UserStore:
    """基于单 JSON 文件的极简用户表，进程内加锁保证读写安全。"""

    def __init__(self, users_file: str | Path):
        self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- 持久化 ----
    def _load(self) -> dict:
        if not self.users_file.exists():
            return {}
        try:
            data = json.loads(self.users_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        self.users_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- 对外接口 ----
    def exists(self, username: str) -> bool:
        return (username or "").strip() in self._load()

    def register(self, username: str, password: str) -> tuple[bool, str]:
        """注册新用户。返回 (成功?, user_id 或错误信息)。"""
        username = (username or "").strip()
        if not username:
            return False, "用户名不能为空"
        if len(username) > 64:
            return False, "用户名过长（最多 64 字符）"
        if not password:
            return False, "密码不能为空"
        if len(password) < 4:
            return False, "密码太短（至少 4 位）"
        user_id = _safe_user_id(username)
        if not user_id:
            return False, "用户名不合法"
        with self._lock:
            data = self._load()
            if username in data:
                return False, "该用户名已被注册"
            # 防止两个不同用户名清洗后撞到同一个 user_id（目录冲突）。
            if any(rec.get("user_id") == user_id for rec in data.values()):
                return False, "该用户名与已有用户冲突，请换一个"
            salt = secrets.token_hex(16)
            data[username] = {
                "user_id": user_id,
                "salt": salt,
                "pwd_hash": _hash_password(password, salt),
                "created_at": time.time(),
            }
            self._save(data)
        return True, user_id

    def verify(self, username: str, password: str) -> str | None:
        """校验用户名+密码。成功返回 user_id，失败返回 None。"""
        username = (username or "").strip()
        rec = self._load().get(username)
        if not rec:
            return None
        expected = rec.get("pwd_hash", "")
        actual = _hash_password(password or "", rec.get("salt", ""))
        if secrets.compare_digest(expected, actual):
            return rec.get("user_id")
        return None
