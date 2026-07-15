from fedapfa.federated.client_sampling import ClientSelectionSchedule


def test_selection_count_uniqueness_and_paired_nesting():
    clients = [f"client_{index:02d}" for index in range(20)]
    lower = ClientSelectionSchedule(clients, seed=712)
    higher = ClientSelectionSchedule(clients, seed=712)
    for round_number in range(1, 8):
        selected_lower = lower.select(round_number, 5)
        selected_higher = higher.select(round_number, 10)
        assert len(selected_lower) == len(set(selected_lower)) == 5
        assert len(selected_higher) == len(set(selected_higher)) == 10
        assert selected_lower == selected_higher[:5]


def test_selection_state_continues_the_same_schedule():
    clients = [f"client_{index:02d}" for index in range(20)]
    uninterrupted = ClientSelectionSchedule(clients, seed=17)
    first = uninterrupted.select(1, 5)
    state = uninterrupted.state_dict()
    expected = uninterrupted.select(2, 5)
    resumed = ClientSelectionSchedule(clients, seed=999)
    resumed.load_state_dict(state)
    assert first != expected
    assert resumed.select(2, 5) == expected
