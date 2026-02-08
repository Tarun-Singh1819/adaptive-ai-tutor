"""
Debug script to inspect store and state during runtime.
Add this to your run loop or call it when needed.
"""
from memory.v_fire_memory import get_ltm_store


def inspect_store(user_id: str = "default_user"):
    """Inspect what's stored in the InMemoryStore"""
    store = get_ltm_store()
    
    print("\n" + "=" * 60)
    print("🔍 STORE INSPECTION")
    print("=" * 60)
    
    # 1. Check STM summary
    print("\n📝 STM Summary:")
    try:
        stm_item = store.get(("stm", user_id), "summary")
        if stm_item and hasattr(stm_item, "value"):
            print(f"   {stm_item.value}")
        else:
            print("   (empty)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Check LTM memories
    print("\n🧠 LTM Memories:")
    try:
        ltm_items = store.search(("user", user_id, "details"))
        if ltm_items:
            for i, item in enumerate(ltm_items, 1):
                mem = item.value if hasattr(item, "value") else item
                print(f"   {i}. {mem.get('data', 'N/A')}")
                print(f"      Importance: {mem.get('importance', 'N/A')}")
                print(f"      Created: {mem.get('created_at', 'N/A')}")
        else:
            print("   (empty)")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)


def inspect_state(state: dict):
    """Inspect current state"""
    print("\n" + "=" * 60)
    print("📊 STATE INSPECTION")
    print("=" * 60)
    
    print(f"\n👤 User ID: {state.get('user_id', 'N/A')}")
    print(f"📄 Paper: {state.get('paper_name', 'N/A')}")
    print(f"❓ Query: {state.get('user_query', 'N/A')}")
    print(f"🔀 Routed to: {state.get('routed_to', 'N/A')}")
    
    print("\n💬 Messages count:", len(state.get("messages", [])))
    
    print("\n📝 STM Raw:")
    stm_raw = state.get("stm_raw", [])
    for m in stm_raw[-4:]:  # Last 4
        print(f"   {m['role']}: {m['content'][:50]}...")
    
    print("\n📋 STM Context:")
    stm_context = state.get("stm_context", [])
    for m in stm_context[:3]:  # First 3
        print(f"   {m['role']}: {m['content'][:50]}...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Quick test
    inspect_store("default_user")
