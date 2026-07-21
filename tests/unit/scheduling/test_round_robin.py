from fedapfa.scheduling.round_robin import round_robin


def test_round_robin_is_selected_position_modulo_process_count():
    clients = [f"client_{value}" for value in range(7)]
    assignments, loads = round_robin(clients, 4)
    assert [value.client_id for value in assignments] == clients
    assert [value.process_rank for value in assignments] == [0, 1, 2, 3, 0, 1, 2]
    assert loads == {"0": 2.0, "1": 2.0, "2": 2.0, "3": 1.0}
