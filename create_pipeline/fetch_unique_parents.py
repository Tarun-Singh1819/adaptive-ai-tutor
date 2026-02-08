def fetch_unique_parents(child_hits, parent_map, max_parents=3):
    seen = set()
    parents = []

    for hit in child_hits:
        pid = hit["parent_id"]
        if pid not in seen:
            parents.append(parent_map[pid])
            seen.add(pid)

        if len(parents) >= max_parents:
            break

    return parents
