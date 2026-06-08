from __future__ import annotations

from unittest.mock import patch

from silhouette_outliner.gui.network import check_online


def test_check_online_returns_true_when_host_reachable() -> None:
    with patch("silhouette_outliner.gui.network.socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value = None
        assert check_online(hosts=[("example.com", 443)]) is True


def test_check_online_returns_false_when_all_hosts_fail() -> None:
    with patch(
        "silhouette_outliner.gui.network.socket.create_connection",
        side_effect=OSError("offline"),
    ):
        assert check_online(hosts=[("example.com", 443)]) is False
