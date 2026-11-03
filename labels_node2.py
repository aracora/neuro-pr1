from __future__ import annotations

# Узел 2 распознает только типы атак (без BENIGN)
NODE1_CLASSES = ["SYN", "UDP", "LDAP", "DNS", "OTHER"]
NODE1_MAP = {name: i for i, name in enumerate(NODE1_CLASSES)}

NODE1_ALIASES: dict[str, str] = {
    "SYN": "SYN", "SYN_FLOOD": "SYN", "DRDOS_SYN": "SYN",
    "UDP": "UDP", "UDP_FLOOD": "UDP", "DRDOS_UDP": "UDP",
    "LDAP": "LDAP", "DRDOS_LDAP": "LDAP",
    "DNS": "DNS", "DRDOS_DNS": "DNS",
    "NTP": "OTHER", "DRDOS_NTP": "OTHER",
    "SSDP": "OTHER", "DRDOS_SSDP": "OTHER",
    "MSSQL": "OTHER", "DRDOS_MSSQL": "OTHER",
}
NODE1_ALIAS_TO_IDX = {k: NODE1_MAP[v] for k, v in NODE1_ALIASES.items()}

CLASS_LABELS_RU = {
    "SYN": "SYN", "UDP": "UDP", "LDAP": "LDAP", "DNS": "DNS", "OTHER": "Другая атака"
}
CLASS_NAMES_RU = [CLASS_LABELS_RU[c] for c in NODE1_CLASSES]