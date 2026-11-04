import itertools
import random
import threading
from pathlib import Path
from typing import List, Optional, Tuple

_ASSIGN_LOCK = threading.Lock()
_ASSIGN_COUNTER = itertools.count()


class EndpointMixin:
    def _strip_wrapping_quotes(self, value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1].strip()
        return value

    def _load_server_endpoints(self) -> Tuple[List[str], List[int], str]:
        hostname_list = self.cfg.infra.hostname_list
        port_list = self.cfg.infra.ports
        endpoint_source = "config"

        config_file = self.cfg.infra.server_endpoints_file
        if config_file:
            config_path = Path(config_file).expanduser()
            if config_path.is_file():
                try:
                    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = self._strip_wrapping_quotes(value)
                        if key == "HOSTNAME_LIST" and value:
                            hostname_list = value
                        elif key == "PORTS" and value:
                            port_list = value
                    endpoint_source = f"file:{config_path}"
                except Exception as e:
                    print(f"Warning: failed to read endpoint config {config_path}: {e}. Falling back to configured values.")
            else:
                print(f"Warning: SERVER_ENDPOINTS_FILE not found: {config_path}. Falling back to configured values.")

        hosts = [h.strip() for h in hostname_list.split(',') if h.strip()]
        if not hosts:
            hosts = ['localhost']

        ports: List[int] = []
        invalid_ports: List[str] = []
        for raw_port in port_list.split(','):
            raw_port = raw_port.strip()
            if not raw_port:
                continue
            try:
                ports.append(int(raw_port))
            except ValueError:
                invalid_ports.append(raw_port)

        if invalid_ports:
            print(f"Warning: invalid ports ignored: {', '.join(invalid_ports)}")

        if not ports:
            ports = [8000]

        return hosts, ports, endpoint_source

    def _select_endpoint(self, available_endpoints: List[Tuple[str, int]]) -> Tuple[str, int]:
        tl = self._get_thread_state()

        if tl.preferred_endpoint and tl.preferred_endpoint in available_endpoints:
            return tl.preferred_endpoint

        with _ASSIGN_LOCK:
            index = next(_ASSIGN_COUNTER)
        selected = available_endpoints[index % len(available_endpoints)]
        tl.preferred_endpoint = selected
        tl.endpoint_failure_count = 0
        return selected

    def _failover_endpoint(self, current_endpoint: Tuple[str, int],
                           all_endpoints: List[Tuple[str, int]],
                           hosts: List[str], ports: List[int]) -> Optional[Tuple[str, int]]:
        current_host, current_port = current_endpoint

        other_host_endpoints = [ep for ep in all_endpoints
                                if ep[0] != current_host and ep != current_endpoint]
        if other_host_endpoints:
            return random.choice(other_host_endpoints)

        same_host_other_ports = [ep for ep in all_endpoints
                                 if ep[0] == current_host and ep[1] != current_port]
        if same_host_other_ports:
            return random.choice(same_host_other_ports)

        return None
