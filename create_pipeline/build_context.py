def build_context(parents, max_sections=3):
    ctx = []
    for p in parents[:max_sections]:
        ctx.append(f"[{p.title}]\n{p.content}")
    return "\n\n".join(ctx)
