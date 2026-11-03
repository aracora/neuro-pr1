from __future__ import annotations

NODE1_CLASSES = ["SYN", "UDP", "LDAP", "DNS", "OTHER", "BENIGN"]
NODE1_MAP = {name: i for i, name in enumerate(NODE1_CLASSES)}

NODE1_ALIASES: dict[str, str] = {
    "SYN": "SYN",
    "SYN_FLOOD": "SYN",
    "DRDOS_SYN": "SYN",
    "UDP": "UDP",
    "UDP_FLOOD": "UDP",
    "DRDOS_UDP": "UDP",
    "LDAP": "LDAP",
    "DRDOS_LDAP": "LDAP",
    "DNS": "DNS",
    "DRDOS_DNS": "DNS",
    
    # Класс "Другая атака" (сюда попадут NTP, SSDP, MSSQL и т.д.)
    "NTP": "OTHER",
    "DRDOS_NTP": "OTHER",
    "SSDP": "OTHER",
    "DRDOS_SSDP": "OTHER",
    "MSSQL": "OTHER",
    "DRDOS_MSSQL": "OTHER",
    "NETBIOS": "OTHER",
    "DRDOS_NETBIOS": "OTHER",
    "SNMP": "OTHER",
    "DRDOS_SNMP": "OTHER",
    "TFTP": "OTHER",
    "UDPLAG": "OTHER",
    
    "BENIGN": "BENIGN",
    "NORMAL": "BENIGN",
    "BENIGN_TRAFFIC": "BENIGN",
}

NODE1_ALIAS_TO_IDX: dict[str, int] = {k: NODE1_MAP[v] for k, v in NODE1_ALIASES.items()}

CLASS_LABELS_RU: dict[str, str] = {
    "SYN": "SYN",
    "UDP": "UDP",
    "LDAP": "LDAP",
    "DNS": "DNS",
    "OTHER": "Другая атака",
    "BENIGN": "Нормальный трафик",
}

CLASS_NAMES_RU: list[str] = [CLASS_LABELS_RU[c] for c in NODE1_CLASSES]