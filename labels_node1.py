from __future__ import annotations

NODE1_CLASSES = ["BENIGN", "ATTACK"]
NODE1_MAP = {name: i for i, name in enumerate(NODE1_CLASSES)}

NODE1_ALIASES: dict[str, str] = {
    "BENIGN": "BENIGN", "NORMAL": "BENIGN", "BENIGN_TRAFFIC": "BENIGN",
    "SYN": "ATTACK", "SYN_FLOOD": "ATTACK", "DRDOS_SYN": "ATTACK",
    "UDP": "ATTACK", "UDP_FLOOD": "ATTACK", "DRDOS_UDP": "ATTACK",
    "LDAP": "ATTACK", "DRDOS_LDAP": "ATTACK",
    "DNS": "ATTACK", "DRDOS_DNS": "ATTACK",
    "NTP": "ATTACK", "DRDOS_NTP": "ATTACK",
    "SSDP": "ATTACK", "DRDOS_SSDP": "ATTACK",
    "MSSQL": "ATTACK", "DRDOS_MSSQL": "ATTACK",
}
NODE1_ALIAS_TO_IDX = {k: NODE1_MAP[v] for k, v in NODE1_ALIASES.items()}

CLASS_LABELS_RU = {"BENIGN": "Нормальный трафик", "ATTACK": "Атака"}
CLASS_NAMES_RU = [CLASS_LABELS_RU[c] for c in NODE1_CLASSES]