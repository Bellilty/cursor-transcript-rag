#!/usr/bin/env python3
"""
Quick verification of transcript adapter parsing.

Tests if the adapter can read real Cursor transcripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "adapters"))

from cursor_transcripts import CursorTranscriptAdapter

def main():
    print("=" * 60)
    print("Transcript Adapter Verification")
    print("=" * 60)
    
    # Find a real transcript file
    cursor_dir = Path.home() / ".cursor" / "projects"
    
    if not cursor_dir.exists():
        print(f"\nError: {cursor_dir} not found")
        return
    
    # Find first .jsonl file
    transcript_files = list(cursor_dir.rglob("agent-transcripts/**/*.jsonl"))
    
    # Exclude subagents
    transcript_files = [f for f in transcript_files if "subagents" not in f.parts]
    
    if not transcript_files:
        print(f"\nNo transcript files found in {cursor_dir}")
        return
    
    # Test first file
    test_file = transcript_files[0]
    print(f"\nTesting file: {test_file}")
    print(f"Size: {test_file.stat().st_size / 1024:.1f} KB")
    
    print("\nParsing...")
    adapter = CursorTranscriptAdapter(test_file)
    messages = adapter.read_messages()
    
    print(f"\n✓ Extracted {len(messages)} messages")
    
    if messages:
        print("\nFirst message:")
        msg = messages[0]
        print(f"  Role: {msg.role}")
        print(f"  Content length: {len(msg.content)} chars")
        print(f"  Content preview: {msg.content[:200]}...")
        print(f"  Timestamp: {msg.timestamp}")
        print(f"  Conversation ID: {msg.conversation_id}")
    
    print("\n" + "=" * 60)
    print("✓ Adapter verification complete!")
    print("=" * 60)
    
    if messages:
        print(f"\nAdapter is working! Found {len(messages)} messages.")
        print("Ready to run: python scripts/ingest_transcripts.py")
    else:
        print("\n⚠ Warning: No messages extracted from transcript.")
        print("Check the adapter logic or transcript format.")

if __name__ == "__main__":
    main()
